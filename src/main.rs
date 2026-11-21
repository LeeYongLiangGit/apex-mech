mod combat;
mod nav;
mod protocol;
mod world;

use combat::*;
use nav::NavEngine;
use protocol::*;
use serde_json::Value;
use std::f64::consts::PI;
use std::io::{self, BufWriter, Write};
use world::World;

const UPGRADE_THRESHOLDS: [u32; 6] = [1, 3, 6, 10, 15, 21];

struct BotBrain {
    cfg: RuleConfig,
    world: World,
    nav: NavEngine,
    mode: MatchMode,
    last_combat_t: f64,
    last_dmg_bearing: Option<(f64, f64)>, // (bearing, timestamp)
    prev_alive: u32,
    our_kills: u32,
    last_aim: [f64; 2],
    climb_stuck_ticks: u32,
}

impl BotBrain {
    fn new(cfg: RuleConfig, world: World) -> Self {
        let nav = NavEngine::build(&world, cfg.step_up, cfg.max_slope_deg);
        Self {
            cfg,
            world,
            nav,
            mode: MatchMode::Unknown,
            last_combat_t: -10.0,
            last_dmg_bearing: None,
            prev_alive: 16,
            our_kills: 0,
            last_aim: [0.0, 0.0],
            climb_stuck_ticks: 0,
        }
    }

    fn decide(&mut self, obs: &ObsMsg) -> Action {
        self.update_state(obs);
        let mut act = Action::idle(obs.tick, obs.me.aim);

        // 1. Instant upgrade resolution using live hello.config.upgrades
        if let Some(offer) = &obs.offer {
            act.pick = Some(self.pick_upgrade(offer, obs));
        }

        // 2. Optimal 2-slot weapon pickup / swap
        self.handle_weapon_pickup(obs, &mut act);

        // 3. Dodge field (bullets, parabolic mortar shells, enemy charging lasers)
        let (dodge, urgent_dodge) = compute_dodge(obs, &self.cfg, &self.world);

        // 4. Select best threat/farm target
        let target = select_target(obs, &self.cfg, &self.world, self.mode);

        // 5. 5-second out-of-combat regen discipline
        let hp_frac = obs.me.hp / obs.me.max_hp.max(1.0);
        let close_threat = obs.mechs.iter().any(|m| {
            m.kind == "mech" && dist_2d(obs.me.pos, m.pos) < 42.0
        });
        let regen_mode = hp_frac < 0.72 && obs.me.in_zone && !close_threat;

        // 6. Combat micro (all 4 weapons + rear armor guard)
        let fallback_bearing = self.best_unseen_threat_bearing(obs);
        decide_combat(
            obs,
            &self.cfg,
            &self.world,
            self.mode,
            regen_mode,
            urgent_dodge,
            target.as_ref(),
            fallback_bearing,
            self.last_aim,
            &mut act,
        );

        // 7. Movement: 2.5D A* macro navigation + hard-cover corner kiting + jet-climb
        self.decide_movement(obs, target.as_ref(), regen_mode, dodge, urgent_dodge, &mut act);

        act.sanitize(self.cfg.min_pitch, self.cfg.max_pitch);
        self.last_aim = act.aim;
        act
    }

    fn update_state(&mut self, obs: &ObsMsg) {
        let mut kills_this_tick = 0_u32;
        for ev in &obs.events {
            match ev.kind.as_str() {
                "damaged" => {
                    self.last_combat_t = obs.t;
                    if let Some(b) = ev.bearing {
                        self.last_dmg_bearing = Some((b, obs.t));
                    }
                }
                "hit" => self.last_combat_t = obs.t,
                "kill" => {
                    kills_this_tick += 1;
                    self.our_kills += 1;
                }
                _ => {}
            }
        }

        if self.mode == MatchMode::Unknown {
            let other_deaths = self
                .prev_alive
                .saturating_sub(obs.alive)
                .saturating_sub(kills_this_tick);
            let mech_vs_mech_beam = obs.beams.iter().any(|b| {
                b.by != obs.me.id
                    && b.path
                        .last()
                        .map_or(false, |end| dist_3d(*end, obs.me.pos) > 18.0)
            });
            if (other_deaths > 0 && obs.zone.dps <= 35.0) || mech_vs_mech_beam {
                self.mode = MatchMode::FfaSurvival;
            } else if obs.t > 40.0 && obs.alive + self.our_kills == 16 {
                self.mode = MatchMode::PracticeHivemind;
            }
        }
        self.prev_alive = obs.alive;
    }

    fn best_unseen_threat_bearing(&self, obs: &ObsMsg) -> Option<f64> {
        if let Some((b, t)) = self.last_dmg_bearing {
            if obs.t - t <= 2.5 {
                return Some(b);
            }
        }
        // Nearest threatening audio cue (weapon fire / laser charge / boost)
        obs.sounds
            .iter()
            .filter(|s| {
                s.dist == "near"
                    && matches!(
                        s.kind.as_str(),
                        "laser_charge" | "lightning" | "gun" | "laser" | "mortar" | "boost" | "step"
                    )
            })
            .min_by(|a, b| a.age.total_cmp(&b.age))
            .map(|s| s.bearing)
    }

    fn pick_upgrade(&self, offer: &Offer, obs: &ObsMsg) -> usize {
        let held_laser = obs.me.weapons.iter().any(|w| w.as_ref().map_or(false, |s| s.kind == "laser"));
        let held_lightning = obs.me.weapons.iter().any(|w| w.as_ref().map_or(false, |s| s.kind == "lightning"));
        let held_mortar = obs.me.weapons.iter().any(|w| w.as_ref().map_or(false, |s| s.kind == "mortar"));

        let score_choice = |c: &OfferChoice| -> f64 {
            let id = c.id.to_lowercase();
            let desc = c.desc.to_lowercase();
            let mut s = 30.0;

            // Inspect live hello.config.upgrades metadata if available
            if let Some(arr) = self.cfg.raw_upgrades.as_array() {
                if let Some(meta) = arr.iter().find(|u| u.get("id").and_then(Value::as_str) == Some(c.id.as_str())) {
                    let meta_str = meta.to_string().to_lowercase();
                    if meta_str.contains("armor") || meta_str.contains("damage_mult") {
                        s += 105.0;
                    }
                    if meta_str.contains("max_hp") || meta_str.contains("regen") {
                        s += 95.0;
                    }
                    if meta_str.contains("walk_speed") || meta_str.contains("boost") || meta_str.contains("energy") {
                        s += 86.0;
                    }
                }
            }

            if id.contains("armor") || desc.contains("damage taken") {
                s += 110.0;
            }
            if id.contains("hp") || desc.contains("max hp") || desc.contains("regen") {
                s += 96.0;
            }
            if id.contains("sensor") || desc.contains("vision") || desc.contains("proximity") {
                s += if self.mode == MatchMode::FfaSurvival { 92.0 } else { 72.0 };
            }
            if id.contains("speed") || id.contains("energy") || id.contains("boost") || id.contains("jet") {
                s += 84.0;
            }
            if id.contains("laser") {
                s += if held_laser { 98.0 } else { 40.0 };
            }
            if id.contains("lightning") {
                s += if held_lightning { 96.0 } else { 40.0 };
            }
            if id.contains("mortar") {
                s += if held_mortar || obs.zone.radius <= 140.0 { 88.0 } else { 35.0 };
            }
            if id.contains("gun") {
                s += if !held_laser && !held_lightning { 65.0 } else { 35.0 };
            }
            s
        };

        offer
            .choices
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| score_choice(a).total_cmp(&score_choice(b)))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn weapon_priority(&self, kind: &str, obs: &ObsMsg) -> i32 {
        match kind {
            "laser" => 100,
            "lightning" => 96,
            "mortar" => {
                // Avoid carrying mortar early (laser +60% cook-off risk), love it in final circles
                if obs.zone.radius <= 140.0 {
                    99
                } else if self.mode == MatchMode::PracticeHivemind {
                    78
                } else {
                    42
                }
            }
            "gun" => 50,
            _ => 0,
        }
    }

    fn handle_weapon_pickup(&self, obs: &ObsMsg, act: &mut Action) {
        let held: Vec<Option<&str>> = obs
            .me
            .weapons
            .iter()
            .map(|w| w.as_ref().map(|s| s.kind.as_str()))
            .collect();
        let has_kind = |k: &str| held.iter().any(|w| *w == Some(k));
        let empty_slot = held.iter().position(|w| w.is_none());

        let mut best: Option<(i32, u64, Option<usize>)> = None;
        for p in &obs.pickups {
            let dh = dist_2d(obs.me.pos, p.pos);
            let dz = (obs.me.pos[2] - p.pos[2]).abs();
            if dh > self.cfg.pickup_radius || dz > 4.0 || has_kind(&p.kind) {
                continue;
            }
            let new_prio = self.weapon_priority(&p.kind, obs);
            if empty_slot.is_some() {
                if new_prio > 0 && best.map_or(true, |b| new_prio > b.0) {
                    best = Some((new_prio, p.id, None));
                }
            } else {
                for (slot_idx, slot) in obs.me.weapons.iter().enumerate() {
                    let Some(w) = slot else { continue };
                    if w.kind == "laser" && w.charge > 0.0 {
                        continue;
                    }
                    let cur_prio = self.weapon_priority(&w.kind, obs);
                    if new_prio > cur_prio + 12 && best.map_or(true, |b| new_prio > b.0) {
                        best = Some((new_prio, p.id, Some(slot_idx)));
                    }
                }
            }
        }

        if let Some((_, pid, drop_slot)) = best {
            act.take = Some(pid);
            act.drop = drop_slot;
        }
    }

    fn is_in_safe_next_zone(&self, x: f64, y: f64, obs: &ObsMsg, margin: f64) -> bool {
        let d = ((x - obs.zone.next_center[0]).powi(2) + (y - obs.zone.next_center[1]).powi(2)).sqrt();
        d <= (obs.zone.next_radius - margin).max(4.0)
    }

    /// Evaluates 24 local tactical spots around the mech to find 1-on-1 peek corners
    /// or full hard-cover breaks against the 1v15 squad / FFA threats.
    fn pick_tactical_cover_spot(
        &self,
        obs: &ObsMsg,
        regen_mode: bool,
        primary: &TargetInfo,
    ) -> Option<[f64; 3]> {
        let pos = obs.me.pos;
        let radius = obs.me.radius();
        let height = obs.me.height();
        let my_eye_off = obs.me.eye_height();

        let enemy_mechs: Vec<&VisibleMech> = obs
            .mechs
            .iter()
            .filter(|m| m.kind == "mech" && dist_2d(pos, m.pos) < 160.0)
            .collect();
        if enemy_mechs.is_empty() {
            return None;
        }

        let mut best_spot: Option<(f64, [f64; 3])> = None;
        for &r in &[10.0, 20.0] {
            for k in 0..12 {
                let ang = (k as f64) * (2.0 * PI / 12.0);
                let cx = pos[0] + ang.cos() * r;
                let cy = pos[1] + ang.sin() * r;
                if !self
                    .world
                    .is_walkable_step(pos, cx, cy, radius, height, self.cfg.step_up)
                {
                    continue;
                }
                // Check zone safety
                let dist_cur_z = ((cx - obs.zone.center[0]).powi(2) + (cy - obs.zone.center[1]).powi(2)).sqrt();
                if dist_cur_z > obs.zone.radius - 6.0 {
                    continue;
                }

                let cz = self.world.terrain_height(cx, cy).max(pos[2] - 2.0);
                let cand_eye = [cx, cy, cz + my_eye_off];

                let mut los_count = 0_usize;
                let mut sees_primary = false;
                let mut min_enemy_d = f64::INFINITY;

                for em in &enemy_mechs {
                    let eh = 9.0 * (1.0 + 0.05 * em.cubes as f64);
                    let e_eye = [em.pos[0], em.pos[1], em.pos[2] + eh * 0.85];
                    let d = dist_2d([cx, cy, cz], em.pos);
                    min_enemy_d = min_enemy_d.min(d);
                    if self.world.raycast_solid(e_eye, cand_eye).is_none() {
                        los_count += 1;
                        if em.id as u64 == primary.id {
                            sees_primary = true;
                        }
                    }
                }

                let mut score = 0.0;
                if regen_mode {
                    // Break ALL sightlines so 5-second regen finishes untouched
                    score += if los_count == 0 { 160.0 } else { -(los_count as f64) * 90.0 };
                    score += min_enemy_d.min(90.0) * 1.2;
                } else {
                    // Isolate 1-on-1 duels! Huge advantage against 1v15 hivemind and FFA third-parties
                    if los_count == 1 && sees_primary {
                        score += 140.0;
                    } else if los_count == 0 {
                        score += 75.0; // Safe peek/reload spot
                    } else {
                        score -= (los_count as f64 - 1.0) * 110.0;
                    }
                    let ideal_d = if obs.me.weapons.iter().any(|w| w.as_ref().map_or(false, |s| s.kind == "lightning"))
                        && enemy_mechs.len() == 1
                        && !primary.has_mortar
                    {
                        32.0
                    } else {
                        58.0
                    };
                    score -= (dist_2d([cx, cy, cz], primary.pos) - ideal_d).abs() * 1.1;
                }

                if self.nav.is_near_cover(cx, cy) {
                    score += 22.0;
                }
                if self.is_in_safe_next_zone(cx, cy, obs, 12.0) {
                    score += 30.0;
                }
                score += (cz - pos[2]).clamp(-4.0, 8.0) * 3.0;

                if best_spot.map_or(true, |(bs, _)| score > bs) {
                    best_spot = Some((score, [cx, cy, cz]));
                }
            }
        }
        best_spot.map(|(_, p)| p)
    }

    fn choose_macro_goal(&self, obs: &ObsMsg, target: Option<&TargetInfo>, regen_mode: bool) -> [f64; 3] {
        let pos = obs.me.pos;
        let next_z = [
            obs.zone.next_center[0],
            obs.zone.next_center[1],
            self.world
                .terrain_height(obs.zone.next_center[0], obs.zone.next_center[1]),
        ];

        // 1. Zone survival priority
        let dist_next = dist_2d(pos, next_z);
        let time_to_shrink = (obs.zone.shrink_start - obs.t).max(0.0);
        let must_rotate = !obs.me.in_zone
            || (dist_next > (obs.zone.next_radius - 18.0).max(5.0)
                && (time_to_shrink < 18.0 || obs.zone.dps >= 60.0));
        if must_rotate {
            return next_z;
        }

        // 2. Tactical cover / 1v1 corner kiting when engaged by enemy mechs
        if let Some(t) = target {
            if t.is_mech && (t.dist < 95.0 || regen_mode) {
                if let Some(cover_spot) = self.pick_tactical_cover_spot(obs, regen_mode, t) {
                    return cover_spot;
                }
            }
        }

        // 3. Weapon pickup priority (especially BEFORE grabbing a cube that hits an upgrade threshold)
        let next_hits_upgrade = UPGRADE_THRESHOLDS.contains(&(obs.me.cubes + 1));
        let empty_slot = obs.me.weapons.iter().any(|w| w.is_none());
        let best_weapon_pickup = obs
            .pickups
            .iter()
            .filter(|p| {
                let already_have = obs
                    .me
                    .weapons
                    .iter()
                    .any(|w| w.as_ref().map_or(false, |s| s.kind == p.kind));
                if already_have || !self.is_in_safe_next_zone(p.pos[0], p.pos[1], obs, 8.0) {
                    return false;
                }
                let prio = self.weapon_priority(&p.kind, obs);
                if empty_slot {
                    prio >= 75
                } else {
                    let min_held = obs
                        .me
                        .weapons
                        .iter()
                        .flatten()
                        .map(|w| self.weapon_priority(&w.kind, obs))
                        .min()
                        .unwrap_or(0);
                    prio > min_held + 15
                }
            })
            .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)));

        if let Some(p) = best_weapon_pickup {
            let max_range = if empty_slot || next_hits_upgrade { 95.0 } else { 55.0 };
            if dist_2d(pos, p.pos) < max_range {
                return p.pos;
            }
        }

        // 4. Supply drops (3 cubes + full HP + full energy)
        let visible_mechs = obs.mechs.iter().filter(|m| m.kind == "mech").count();
        for d in &obs.drops {
            let dist = dist_2d(pos, d.pos);
            if d.lands_in <= 14.0 && dist < 130.0 {
                let safe_to_cap = self.mode == MatchMode::PracticeHivemind || visible_mechs == 0;
                if safe_to_cap {
                    return d.pos;
                } else if self.mode == MatchMode::FfaSurvival && dist > 45.0 {
                    // Hold overwatch ~40m off the drop to ambush capturers
                    let ux = (pos[0] - d.pos[0]) / dist.max(1.0);
                    let uy = (pos[1] - d.pos[1]) / dist.max(1.0);
                    return [d.pos[0] + ux * 40.0, d.pos[1] + uy * 40.0, d.pos[2]];
                }
            }
        }

        // 5. Fast 10-cube rush: loose cubes, nearby crates, and isolated NPCs
        let want_cubes = obs.me.cubes < self.cfg.cube_stat_cap
            || (obs.me.cubes == 14 && obs.me.hp / obs.me.max_hp.max(1.0) > 0.7);
        if want_cubes {
            if let Some(c) = obs
                .cubes
                .iter()
                .filter(|c| self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 6.0))
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, c.pos) < 80.0 {
                    return c.pos;
                }
            }
            if let Some(cr) = obs
                .crates
                .iter()
                .filter(|c| self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 8.0))
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, cr.pos) < 70.0 {
                    return cr.pos;
                }
            }
            if let Some(npc) = obs
                .mechs
                .iter()
                .filter(|m| {
                    m.kind == "npc"
                        && self.is_in_safe_next_zone(m.pos[0], m.pos[1], obs, 10.0)
                        && dist_2d(pos, m.pos) < 75.0
                })
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                return npc.pos;
            }
        }

        // 6. Claim high-ground rooftop inside next safe zone
        if let Some(roof) = self.nav.find_safe_rooftop(
            pos,
            obs.zone.next_center,
            obs.zone.next_radius,
            110.0,
        ) {
            return roof;
        }

        next_z
    }

    fn decide_movement(
        &mut self,
        obs: &ObsMsg,
        target: Option<&TargetInfo>,
        regen_mode: bool,
        dodge: [f64; 2],
        urgent_dodge: bool,
        act: &mut Action,
    ) {
        let pos = obs.me.pos;
        let radius = obs.me.radius();
        let height = obs.me.height();

        let goal = self.choose_macro_goal(obs, target, regen_mode);
        let allow_jet = obs.me.energy >= 45.0 && !urgent_dodge;
        let wp = self.nav.next_waypoint(obs.tick, pos, goal, allow_jet);

        // Jet-climb controller when next waypoint is on a higher rooftop
        let climbing = wp.z > pos[2] + self.cfg.step_up + 0.3 && obs.me.energy > 12.0 && !urgent_dodge;
        let mut desired = [wp.x - pos[0], wp.y - pos[1]];

        if climbing {
            let dh = (desired[0] * desired[0] + desired[1] * desired[1]).sqrt();
            if dh < 9.0 {
                if obs.me.grounded {
                    act.jump = true;
                }
                act.boost = true;
                // Detect eave snag while jetting upward; nudge back briefly to clear eave
                if !obs.me.grounded && obs.me.vel[2] < 1.2 {
                    self.climb_stuck_ticks += 1;
                } else {
                    self.climb_stuck_ticks = 0;
                }
                if self.climb_stuck_ticks >= 2 && self.climb_stuck_ticks <= 5 {
                    desired = [-desired[0], -desired[1]];
                } else if pos[2] < wp.z + 0.25 {
                    // Rise cleanly before pushing horizontally onto the roof edge
                    desired = [desired[0] * 0.35, desired[1] * 0.35];
                }
            }
        } else {
            self.climb_stuck_ticks = 0;
        }

        // Blend A* direction with active dodge vector
        let d_len = (desired[0] * desired[0] + desired[1] * desired[1]).sqrt();
        let nav_u = if d_len > 0.5 {
            [desired[0] / d_len, desired[1] / d_len]
        } else {
            [0.0, 0.0]
        };

        let combined = if urgent_dodge {
            [dodge[0] * 1.5 + nav_u[0] * 0.4, dodge[1] * 1.5 + nav_u[1] * 0.4]
        } else {
            [nav_u[0] + dodge[0], nav_u[1] + dodge[1]]
        };
        let c_len = (combined[0] * combined[0] + combined[1] * combined[1]).sqrt();
        if c_len < 1e-4 {
            return;
        }

        let base_yaw = combined[1].atan2(combined[0]);
        // Local collision smoothing around corners (skip during vertical roof climb)
        let best_yaw = if climbing {
            base_yaw
        } else {
            let mut byaw = base_yaw;
            let mut bscore = f64::NEG_INFINITY;
            for step in -5..=5 {
                let yaw = base_yaw + (step as f64) * (PI / 12.0);
                let ux = yaw.cos();
                let uy = yaw.sin();
                let ok1 = self
                    .world
                    .is_walkable_step(pos, pos[0] + ux * 3.0, pos[1] + uy * 3.0, radius, height, self.cfg.step_up);
                let ok2 = ok1
                    && self.world.is_walkable_step(
                        pos,
                        pos[0] + ux * 6.5,
                        pos[1] + uy * 6.5,
                        radius,
                        height,
                        self.cfg.step_up,
                    );
                let align = (yaw - base_yaw).cos();
                let score = align * 10.0 + if ok2 { 18.0 } else if ok1 { 6.0 } else { -40.0 };
                if score > bscore {
                    bscore = score;
                    byaw = yaw;
                }
            }
            byaw
        };

        // Silent-walk stealth in late FFA when unengaged and safely inside next zone
        let mut mag = if d_len < 2.0 && !urgent_dodge { 0.35 } else { 1.0 };
        if self.mode == MatchMode::FfaSurvival
            && target.is_none()
            && !urgent_dodge
            && self.is_in_safe_next_zone(pos[0], pos[1], obs, 15.0)
            && obs.zone.radius <= 160.0
        {
            mag = ((self.cfg.sound_step_speed - 0.35) / obs.me.walk_speed()).clamp(0.25, 1.0);
        }

        act.move_dir = [best_yaw.cos() * mag, best_yaw.sin() * mag];
        if !climbing {
            let need_sprint = urgent_dodge || !obs.me.in_zone || (regen_mode && target.is_some());
            act.boost = need_sprint && obs.me.grounded && obs.me.energy > 22.0;
        }
    }
}

fn main() {
    let mailbox = StdinMailbox::spawn();
    let Some(hello) = mailbox.wait_hello() else {
        return;
    };

    let cfg = RuleConfig::from_json(&hello.config);
    let world = match World::load(&hello.map.dir, &hello.map.files) {
        Ok(w) => w,
        Err(err) => {
            eprintln!("world load error: {err}");
            return;
        }
    };

    let mut brain = BotBrain::new(cfg, world);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let ready = ReadyMsg {
        kind: "ready",
        name: "apex-mech",
    };
    if serde_json::to_writer(&mut out, &ready).is_err()
        || writeln!(out).is_err()
        || out.flush().is_err()
    {
        return;
    }

    while let Some(obs) = mailbox.next_obs() {
        let action = brain.decide(&obs);
        if serde_json::to_writer(&mut out, &action).is_err()
            || writeln!(out).is_err()
            || out.flush().is_err()
        {
            break;
        }
    }
}