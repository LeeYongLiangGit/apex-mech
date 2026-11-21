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
    last_dmg_bearing: Option<(f64, f64)>,
    prev_alive: u32,
    our_kills: u32,
    last_aim: [f64; 2],
    strafe_sign: f64,
    last_strafe_flip: u64,
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
            strafe_sign: 1.0,
            last_strafe_flip: 0,
            climb_stuck_ticks: 0,
        }
    }

    fn decide(&mut self, obs: &ObsMsg) -> Action {
        self.update_state(obs);
        let mut act = Action::idle(obs.tick, obs.me.aim);

        // 1. Instant upgrade pick
        if let Some(offer) = &obs.offer {
            act.pick = Some(self.pick_upgrade(offer, obs));
        }

        // 2. Optimal 2-slot weapon pickup & swap (protects sustained + burst synergy)
        self.handle_weapon_pickup(obs, &mut act);

        // 3. Active dodge field (bullets, mortar parabolas, charging lasers)
        let (dodge, urgent_dodge) = compute_dodge(obs, &self.cfg, &self.world);

        // 4. Select best target & count active LOS attackers
        let los_enemies = count_los_enemy_mechs(obs, &self.world);
        let target = select_target(obs, &self.cfg, &self.world, self.mode);

        // 5. True regen discipline:
        //    - seek_heal: pull behind cover when wounded
        //    - ceasefire_for_regen: ONLY hold fire once 0 enemy mechs actually have LOS!
        let hp_frac = obs.me.hp / obs.me.max_hp.max(1.0);
        let seek_heal = hp_frac < 0.55 && obs.me.in_zone;
        let ceasefire_for_regen = hp_frac < 0.78
            && obs.me.in_zone
            && los_enemies == 0
            && (obs.t - self.last_combat_t) >= 1.2;

        // 6. Combat micro (single-weapon arbitration + rear-armor guard)
        let fallback_bearing = self.best_unseen_threat_bearing(obs);
        decide_combat(
            obs,
            &self.cfg,
            &self.world,
            self.mode,
            ceasefire_for_regen,
            urgent_dodge,
            target.as_ref(),
            fallback_bearing,
            self.last_aim,
            &mut act,
        );

        // 7. Continuous 16-heading strafe + corner-isolation + 2.5D A* movement
        self.decide_movement(
            obs,
            target.as_ref(),
            los_enemies,
            seek_heal,
            dodge,
            urgent_dodge,
            &mut act,
        );

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

        // Strict Practice vs FFA detection (no false triggers from missed laser beams)
        if self.mode == MatchMode::Unknown {
            let other_deaths = self
                .prev_alive
                .saturating_sub(obs.alive)
                .saturating_sub(kills_this_tick);

            let verified_mech_vs_mech_beam = obs.beams.iter().any(|b| {
                b.by != obs.me.id
                    && b.path.last().map_or(false, |end| {
                        obs.mechs.iter().any(|m| {
                            m.kind == "mech"
                                && m.id != b.by
                                && dist_2d(*end, m.pos) < 5.0
                        })
                    })
            });

            if (other_deaths > 0 && obs.zone.dps <= 35.0 && obs.t < 160.0)
                || verified_mech_vs_mech_beam
            {
                self.mode = MatchMode::FfaSurvival;
            } else if obs.t > 35.0 && obs.alive + self.our_kills == 16 {
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

            if let Some(arr) = self.cfg.raw_upgrades.as_array() {
                if let Some(meta) = arr
                    .iter()
                    .find(|u| u.get("id").and_then(Value::as_str) == Some(c.id.as_str()))
                {
                    let m = meta.to_string().to_lowercase();
                    if m.contains("armor") || m.contains("damage_mult") {
                        s += 105.0;
                    }
                    if m.contains("max_hp") || m.contains("regen") {
                        s += 98.0;
                    }
                    if m.contains("walk_speed") || m.contains("energy") || m.contains("boost") {
                        s += 86.0;
                    }
                }
            }

            if id.contains("armor") || desc.contains("damage taken") {
                s += 115.0;
            }
            if id.contains("hp") || desc.contains("max hp") || desc.contains("regen") {
                s += 100.0;
            }
            if id.contains("lightning") {
                s += if held_lightning { 108.0 } else { 40.0 };
            }
            if id.contains("laser") {
                s += if held_laser { 102.0 } else { 40.0 };
            }
            if id.contains("speed") || id.contains("energy") || id.contains("boost") {
                s += 88.0;
            }
            if id.contains("sensor") || desc.contains("vision") || desc.contains("proximity") {
                s += if self.mode == MatchMode::FfaSurvival { 94.0 } else { 65.0 };
            }
            if id.contains("mortar") {
                s += if held_mortar { 86.0 } else { 30.0 };
            }
            if id.contains("gun") {
                s += if !held_laser && !held_lightning { 72.0 } else { 32.0 };
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
            "lightning" => 100, // 240 DPS instant, AoE chain, shorts enemy lasers, fast crate clear
            "laser" => 97,      // 315 hitscan burst + cook-off
            "mortar" => {
                if self.mode == MatchMode::FfaSurvival && obs.zone.radius <= 140.0 {
                    98
                } else {
                    45 // Avoid early mortar (laser +60% cook-off vulnerability)
                }
            }
            "gun" => 60,
            _ => 0,
        }
    }

    /// Prevents creating the slow `laser + mortar` trap (always keep lightning or gun).
    fn valid_loadout_pair(a: &str, b: &str) -> bool {
        if a == b {
            return false;
        }
        let has_sustained = a == "lightning" || b == "lightning" || a == "gun" || b == "gun";
        has_sustained
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
                for (slot_idx, slot) in obs.me.weapons.iter().enumerate().take(2) {
                    let Some(w) = slot else { continue };
                    if w.kind == "laser" && w.charge > 0.0 {
                        continue;
                    }
                    let other_kind = obs
                        .me
                        .weapons
                        .get(1 - slot_idx)
                        .and_then(|s| s.as_ref())
                        .map(|s| s.kind.as_str())
                        .unwrap_or("");
                    if !other_kind.is_empty() && !Self::valid_loadout_pair(&p.kind, other_kind) {
                        continue;
                    }
                    let cur_prio = self.weapon_priority(&w.kind, obs);
                    if new_prio > cur_prio + 10 && best.map_or(true, |b| new_prio > b.0) {
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

    fn choose_macro_goal(&self, obs: &ObsMsg, target: Option<&TargetInfo>) -> [f64; 3] {
        let pos = obs.me.pos;
        let next_z = [
            obs.zone.next_center[0],
            obs.zone.next_center[1],
            self.world
                .terrain_height(obs.zone.next_center[0], obs.zone.next_center[1]),
        ];

        // 1. Zone rotation safety
        let dist_next = dist_2d(pos, next_z);
        let time_to_shrink = (obs.zone.shrink_start - obs.t).max(0.0);
        let must_rotate = !obs.me.in_zone
            || (dist_next > (obs.zone.next_radius - 16.0).max(5.0)
                && (time_to_shrink < 16.0 || obs.zone.dps >= 60.0));
        if must_rotate {
            return next_z;
        }

        // 2. High-value weapon pickup (especially lightning / laser, or before upgrade threshold)
        let next_hits_upgrade = UPGRADE_THRESHOLDS.contains(&(obs.me.cubes + 1));
        let empty_slot = obs.me.weapons.iter().any(|w| w.is_none());
        let best_pickup = obs
            .pickups
            .iter()
            .filter(|p| {
                let already_have = obs
                    .me
                    .weapons
                    .iter()
                    .any(|w| w.as_ref().map_or(false, |s| s.kind == p.kind));
                if already_have || !self.is_in_safe_next_zone(p.pos[0], p.pos[1], obs, 6.0) {
                    return false;
                }
                let prio = self.weapon_priority(&p.kind, obs);
                if empty_slot {
                    prio >= 90
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

        if let Some(p) = best_pickup {
            let max_r = if empty_slot || next_hits_upgrade { 95.0 } else { 55.0 };
            if dist_2d(pos, p.pos) < max_r {
                return p.pos;
            }
        }

        // 3. Supply drops (3 cubes + FULL HP + FULL ENERGY!)
        let near_mechs = obs
            .mechs
            .iter()
            .filter(|m| m.kind == "mech" && dist_2d(pos, m.pos) < 50.0)
            .count();
        for d in &obs.drops {
            let dist = dist_2d(pos, d.pos);
            if d.lands_in <= 14.0 && dist < 140.0 {
                if self.mode != MatchMode::FfaSurvival || near_mechs == 0 {
                    return d.pos;
                } else if dist > 45.0 {
                    let ux = (pos[0] - d.pos[0]) / dist.max(1.0);
                    let uy = (pos[1] - d.pos[1]) / dist.max(1.0);
                    return [d.pos[0] + ux * 42.0, d.pos[1] + uy * 42.0, d.pos[2]];
                }
            }
        }

        // 4. Fast 10-cube snowball: loose cubes > nearby crates > isolated NPCs
        let want_cubes = obs.me.cubes < self.cfg.cube_stat_cap || obs.me.cubes == 14;
        if want_cubes {
            if let Some(c) = obs
                .cubes
                .iter()
                .filter(|c| self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 5.0))
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, c.pos) < 85.0 {
                    return c.pos;
                }
            }
            if let Some(cr) = obs
                .crates
                .iter()
                .filter(|c| self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 8.0))
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, cr.pos) < 75.0 {
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

        // 5. Mode-specific mid/late objective:
        //    - Practice: actively hunt tracked mechs when healthy to snowball placement
        //    - FFA: claim safe rooftop high ground inside next circle
        if self.mode == MatchMode::PracticeHivemind && obs.me.hp / obs.me.max_hp.max(1.0) > 0.65 {
            if let Some(t) = target {
                if t.is_mech {
                    return t.pos;
                }
            }
            if let Some(s) = obs.sounds.iter().min_by(|a, b| a.age.total_cmp(&b.age)) {
                return [
                    pos[0] + s.bearing.cos() * 65.0,
                    pos[1] + s.bearing.sin() * 65.0,
                    pos[2],
                ];
            }
        } else if self.mode == MatchMode::FfaSurvival && near_mechs == 0 {
            if let Some(roof) = self.nav.find_safe_rooftop(
                pos,
                obs.zone.next_center,
                obs.zone.next_radius,
                100.0,
            ) {
                return roof;
            }
        }

        next_z
    }

    fn decide_movement(
        &mut self,
        obs: &ObsMsg,
        target: Option<&TargetInfo>,
        los_enemies: usize,
        seek_heal: bool,
        dodge: [f64; 2],
        urgent_dodge: bool,
        act: &mut Action,
    ) {
        let pos = obs.me.pos;
        let radius = obs.me.radius();
        let height = obs.me.height();
        let my_eye_off = obs.me.eye_height();

        let has_lightning = obs
            .me
            .weapons
            .iter()
            .any(|w| w.as_ref().map_or(false, |s| s.kind == "lightning"));
        let has_laser = obs
            .me
            .weapons
            .iter()
            .any(|w| w.as_ref().map_or(false, |s| s.kind == "laser"));

        // Flip strafe direction periodically so bots can't predict constant orbits
        if obs.tick.saturating_sub(self.last_strafe_flip) >= 18 {
            self.strafe_sign = -self.strafe_sign;
            self.last_strafe_flip = obs.tick;
        }

        let goal = self.choose_macro_goal(obs, target);
        // Only allow rooftop jet-climbing when unengaged and high energy
        let allow_jet = self.mode == MatchMode::FfaSurvival
            && los_enemies == 0
            && obs.me.energy >= 65.0
            && !urgent_dodge;
        let wp = self.nav.next_waypoint(obs.tick, pos, goal, allow_jet);

        // Rooftop jet-climb (only when safe & high energy)
        let climbing = allow_jet
            && wp.z > pos[2] + self.cfg.step_up + 0.4
            && ((wp.x - pos[0]).powi(2) + (wp.y - pos[1]).powi(2)).sqrt() < 9.0;
        if climbing {
            if obs.me.grounded {
                act.jump = true;
            }
            act.boost = true;
            if !obs.me.grounded && obs.me.vel[2] < 1.2 {
                self.climb_stuck_ticks += 1;
            } else {
                self.climb_stuck_ticks = 0;
            }
            let mut dx = wp.x - pos[0];
            let mut dy = wp.y - pos[1];
            if self.climb_stuck_ticks >= 2 && self.climb_stuck_ticks <= 5 {
                dx = -dx;
                dy = -dy;
            } else if pos[2] < wp.z + 0.25 {
                dx *= 0.35;
                dy *= 0.35;
            }
            let l = (dx * dx + dy * dy).sqrt().max(1e-4);
            act.move_dir = [dx / l, dy / l];
            return;
        }
        self.climb_stuck_ticks = 0;

        // Nav unit vector
        let nav_dx = wp.x - pos[0];
        let nav_dy = wp.y - pos[1];
        let nav_len = (nav_dx * nav_dx + nav_dy * nav_dy).sqrt();
        let nav_u = if nav_len > 0.5 {
            [nav_dx / nav_len, nav_dy / nav_len]
        } else {
            [0.0, 0.0]
        };

        let dodge_len = (dodge[0] * dodge[0] + dodge[1] * dodge[1]).sqrt();
        let dodge_u = if dodge_len > 0.05 {
            [dodge[0] / dodge_len, dodge[1] / dodge_len]
        } else {
            [0.0, 0.0]
        };

        // Gather nearby enemy mechs for real-time 16-heading tactical scoring
        let enemy_mechs: Vec<&VisibleMech> = obs
            .mechs
            .iter()
            .filter(|m| m.kind == "mech" && dist_2d(pos, m.pos) < 150.0)
            .collect();
        let in_combat = !enemy_mechs.is_empty();

        // Ideal combat spacing based on our loadout vs primary threat
        let primary_mech = target.filter(|t| t.is_mech);
        let ideal_dist = if let Some(pm) = primary_mech {
            if seek_heal || los_enemies >= 2 {
                90.0 // Pull back around corners when outnumbered or wounded
            } else if has_lightning && obs.me.energy >= 20.0 && !pm.has_mortar {
                29.0 // Melt inside 45m lightning range while circle-strafing
            } else if pm.has_lightning && !has_lightning {
                68.0 // Kite outside enemy lightning range
            } else if has_laser {
                72.0
            } else {
                54.0
            }
        } else {
            55.0
        };

        // Evaluate 16 continuous headings: NEVER stop moving in combat!
        let mut best_yaw = nav_dy.atan2(nav_dx);
        let mut best_score = f64::NEG_INFINITY;

        for k in 0..16 {
            let yaw = (k as f64) * (2.0 * PI / 16.0);
            let ux = yaw.cos();
            let uy = yaw.sin();

            // Exact cylinder wall check at 3.2 m and 7.0 m
            let ok_near = self.world.is_walkable_step(
                pos,
                pos[0] + ux * 3.2,
                pos[1] + uy * 3.2,
                radius + 0.2,
                height,
                self.cfg.step_up,
            );
            let ok_far = ok_near
                && self.world.is_walkable_step(
                    pos,
                    pos[0] + ux * 7.0,
                    pos[1] + uy * 7.0,
                    radius + 0.2,
                    height,
                    self.cfg.step_up,
                );

            let mut score = if ok_far {
                35.0
            } else if ok_near {
                8.0
            } else {
                -180.0
            };

            // Zone safety at 10m lookahead
            let lx = pos[0] + ux * 10.0;
            let ly = pos[1] + uy * 10.0;
            let d_cur_z = ((lx - obs.zone.center[0]).powi(2) + (ly - obs.zone.center[1]).powi(2)).sqrt();
            if d_cur_z > obs.zone.radius - 4.0 {
                score -= 220.0;
            } else if !obs.me.in_zone {
                let d_next_z = ((lx - obs.zone.next_center[0]).powi(2) + (ly - obs.zone.next_center[1]).powi(2)).sqrt();
                score -= d_next_z * 2.5;
            }

            // Active projectile / laser dodge alignment (top priority)
            if dodge_len > 0.05 {
                let dodge_dot = ux * dodge_u[0] + uy * dodge_u[1];
                score += dodge_dot * if urgent_dodge { 150.0 } else { 75.0 };
            }

            // Macro A* alignment
            let nav_dot = ux * nav_u[0] + uy * nav_u[1];
            let nav_weight = if !obs.me.in_zone {
                95.0
            } else if in_combat {
                22.0
            } else {
                65.0
            };
            score += nav_dot * nav_weight;

            // Real-time combat shaping: continuous circle-strafing + 1v1 corner isolation
            if in_combat && ok_near {
                let cand_eye = [lx, ly, pos[2] + my_eye_off];
                let mut future_los = 0_usize;
                for em in &enemy_mechs {
                    let eh = 9.0 * (1.0 + 0.05 * em.cubes as f64);
                    let e_eye = [em.pos[0], em.pos[1], em.pos[2] + 0.78 * eh];
                    if self.world.raycast_solid(e_eye, cand_eye).is_none() {
                        future_los += 1;
                    }
                }

                if seek_heal || los_enemies >= 2 {
                    // Break extra sightlines around building corners!
                    if future_los == 0 {
                        score += if seek_heal { 125.0 } else { 70.0 };
                    } else if future_los == 1 && !seek_heal {
                        score += 105.0; // Isolate 1v1 duel!
                    } else {
                        score -= (future_los as f64) * 65.0;
                    }
                    if self.nav.is_near_cover(lx, ly) {
                        score += 18.0;
                    }
                } else if let Some(pm) = primary_mech {
                    // Clean 1v1: maintain optimal spacing + strong continuous lateral strafe
                    let to_x = pm.pos[0] - pos[0];
                    let to_y = pm.pos[1] - pos[1];
                    let d = (to_x * to_x + to_y * to_y).sqrt().max(1.0);
                    let rx = to_x / d;
                    let ry = to_y / d;
                    // Perpendicular strafe vector
                    let sx = -ry * self.strafe_sign;
                    let sy = rx * self.strafe_sign;

                    let strafe_dot = ux * sx + uy * sy;
                    score += strafe_dot * 42.0;

                    let new_d = ((pm.pos[0] - lx).powi(2) + (pm.pos[1] - ly).powi(2)).sqrt();
                    let cur_err = (d - ideal_dist).abs();
                    let new_err = (new_d - ideal_dist).abs();
                    score += (cur_err - new_err) * 4.5;

                    // Keep LOS in a favorable 1v1 unless reloading/healing
                    if future_los == 1 {
                        score += 18.0;
                    }
                }
            }

            if score > best_score {
                best_score = score;
                best_yaw = yaw;
            }
        }

        // Flip strafe direction immediately if our chosen path is close to a wall
        if !self.world.is_walkable_step(
            pos,
            pos[0] + best_yaw.cos() * 4.5,
            pos[1] + best_yaw.sin() * 4.5,
            radius + 0.2,
            height,
            self.cfg.step_up,
        ) {
            self.strafe_sign = -self.strafe_sign;
            self.last_strafe_flip = obs.tick;
        }

        // Silent-walk ONLY in late FFA when unengaged and safely inside next zone
        let mut mag = 1.0;
        if self.mode == MatchMode::FfaSurvival
            && !in_combat
            && !urgent_dodge
            && self.is_in_safe_next_zone(pos[0], pos[1], obs, 15.0)
            && obs.zone.radius <= 160.0
        {
            mag = ((self.cfg.sound_step_speed - 0.35) / obs.me.walk_speed()).clamp(0.25, 1.0);
        }

        act.move_dir = [best_yaw.cos() * mag, best_yaw.sin() * mag];

        // Smart energy hysteresis: preserve >= 35 energy if holding lightning!
        let min_boost_energy = if has_lightning { 38.0 } else { 20.0 };
        let need_sprint = urgent_dodge
            || !obs.me.in_zone
            || (seek_heal && los_enemies > 0)
            || (los_enemies >= 2 && has_lightning && obs.me.energy > 55.0);
        act.boost = need_sprint && obs.me.grounded && obs.me.energy >= min_boost_energy;
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