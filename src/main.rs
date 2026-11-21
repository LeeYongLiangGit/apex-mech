mod belief;
mod combat;
mod nav;
mod protocol;
mod world;

use belief::{BeliefMap, MatchMode};
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
    belief: BeliefMap,
    last_aim: [f64; 2],
    last_target_id: Option<u64>,
    strafe_sign: f64,
    last_strafe_flip: u64,
    climb_stuck_ticks: u32,
    last_pos: [f64; 3],
    stuck_ticks: u32,
    escape_cache: Option<([f64; 3], u64)>,
}

impl BotBrain {
    fn new(cfg: RuleConfig, world: World) -> Self {
        let nav = NavEngine::build(&world, cfg.step_up, cfg.max_slope_deg);
        Self {
            cfg,
            world,
            nav,
            belief: BeliefMap::new(),
            last_aim: [0.0, 0.0],
            last_target_id: None,
            strafe_sign: 1.0,
            last_strafe_flip: 0,
            climb_stuck_ticks: 0,
            last_pos: [0.0; 3],
            stuck_ticks: 0,
            escape_cache: None,
        }
    }

    fn decide(&mut self, obs: &ObsMsg) -> Action {
        self.belief.update(obs);
        let mut act = Action::idle(obs.tick, obs.me.aim);

        // Stuck detection (for wall-corner snags)
        let moved = dist_2d(obs.me.pos, self.last_pos);
        if moved < 0.25 && obs.tick > 10 {
            self.stuck_ticks = self.stuck_ticks.saturating_add(1);
            if self.stuck_ticks == 7 {
                self.nav.invalidate_cache();
                self.strafe_sign = -self.strafe_sign;
            }
        } else {
            self.stuck_ticks = 0;
        }
        self.last_pos = obs.me.pos;

        // 1. Instant upgrade resolution
        if let Some(offer) = &obs.offer {
            act.pick = Some(self.pick_upgrade(offer, obs));
        }

        // 2. Optimal 2-slot weapon pickup & swap
        self.handle_weapon_pickup(obs, &mut act);

        // 3. Active dodge field
        let (dodge, urgent_dodge) = compute_dodge(obs, &self.cfg, &self.world);

        // 4. Target selection with kill-finish hysteresis
        let los_enemies = count_los_enemy_mechs(obs, &self.world);
        let target = select_target(
            obs,
            &self.cfg,
            &self.world,
            self.belief.mode,
            self.last_target_id,
        );
        self.last_target_id = target.as_ref().filter(|t| t.is_mech).map(|t| t.id);

        // 5. True regen discipline
        let hp_frac = obs.me.hp / obs.me.max_hp.max(1.0);
        let gun_reloading = obs
            .me
            .weapons
            .iter()
            .flatten()
            .any(|w| w.kind == "gun" && w.reload > 0.35);
        let seek_heal = (hp_frac < 0.58 || (hp_frac < 0.72 && los_enemies >= 2)) && obs.me.in_zone;
        let ceasefire_for_regen = hp_frac < 0.80
            && obs.me.in_zone
            && los_enemies == 0
            && (obs.t - self.belief.last_combat_t) >= 1.2;

        // 6. Combat micro (single-weapon arbitration + rear-armor guard)
        let fallback_bearing = self.belief.best_unseen_bearing(obs.me.pos, obs.t);
        decide_combat(
            obs,
            &self.cfg,
            &self.world,
            self.belief.mode,
            ceasefire_for_regen,
            urgent_dodge,
            target.as_ref(),
            fallback_bearing,
            self.last_aim,
            &mut act,
        );

        // 7. Movement: escape-cover A* + 12-heading fast strafe/isolation + jet climb
        self.decide_movement(
            obs,
            target.as_ref(),
            los_enemies,
            seek_heal,
            gun_reloading,
            dodge,
            urgent_dodge,
            &mut act,
        );

        act.sanitize(self.cfg.min_pitch, self.cfg.max_pitch);
        self.last_aim = act.aim;
        act
    }

    fn pick_upgrade(&self, offer: &Offer, obs: &ObsMsg) -> usize {
        let held_laser = obs
            .me
            .weapons
            .iter()
            .any(|w| w.as_ref().map_or(false, |s| s.kind == "laser"));
        let held_lightning = obs
            .me
            .weapons
            .iter()
            .any(|w| w.as_ref().map_or(false, |s| s.kind == "lightning"));
        let held_mortar = obs
            .me
            .weapons
            .iter()
            .any(|w| w.as_ref().map_or(false, |s| s.kind == "mortar"));

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
                        s += 108.0;
                    }
                    if m.contains("max_hp") || m.contains("regen") {
                        s += 100.0;
                    }
                    if m.contains("walk_speed") || m.contains("energy") || m.contains("boost") {
                        s += 88.0;
                    }
                }
            }

            if id.contains("armor") || desc.contains("damage taken") {
                s += 118.0;
            }
            if id.contains("hp") || desc.contains("max hp") || desc.contains("regen") {
                s += 104.0;
            }
            if id.contains("lightning") {
                s += if held_lightning { 112.0 } else { 40.0 };
            }
            if id.contains("laser") {
                s += if held_laser { 104.0 } else { 40.0 };
            }
            if id.contains("speed") || id.contains("energy") || id.contains("boost") {
                s += 90.0;
            }
            if id.contains("sensor") || desc.contains("vision") || desc.contains("proximity") {
                s += if self.belief.mode == MatchMode::FfaSurvival { 95.0 } else { 64.0 };
            }
            if id.contains("mortar") {
                s += if held_mortar { 86.0 } else { 30.0 };
            }
            if id.contains("gun") {
                s += if !held_laser && !held_lightning { 88.0 } else { 32.0 };
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
            "lightning" => 100,
            "laser" => 97,
            "mortar" => {
                if self.belief.mode == MatchMode::FfaSurvival && obs.zone.radius <= 140.0 {
                    98
                } else {
                    45
                }
            }
            "gun" => 60,
            _ => 0,
        }
    }

    fn valid_loadout_pair(a: &str, b: &str) -> bool {
        if a == b {
            return false;
        }
        a == "lightning" || b == "lightning" || a == "gun" || b == "gun"
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

    fn elevation_reachable(&self, my_z: f64, target_z: f64, energy: f64) -> bool {
        let dz = target_z - my_z;
        dz <= self.cfg.step_up + 1.0 || (dz <= 30.0 && energy >= 48.0)
    }

    fn choose_macro_goal(
        &mut self,
        obs: &ObsMsg,
        target: Option<&TargetInfo>,
        threat_eyes: &[[f64; 3]],
        los_enemies: usize,
        seek_heal: bool,
        gun_reloading: bool,
    ) -> ([f64; 3], bool) {
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
            || (dist_next > (obs.zone.next_radius - 16.0).max(5.0)
                && (time_to_shrink < 16.0 || obs.zone.dps >= 60.0));
        if must_rotate {
            return (next_z, false);
        }

        // 2. TRUE COVER ESCAPE / 1v1 CORNER ISOLATION when wounded, outnumbered, or reloading!
        let need_cover_route = !threat_eyes.is_empty()
            && (seek_heal || los_enemies >= 2 || (gun_reloading && los_enemies >= 1));
        if need_cover_route {
            let reuse = self
                .escape_cache
                .filter(|&(pt, tick)| obs.tick.saturating_sub(tick) < 8 && dist_2d(pos, pt) > 6.0);
            if let Some((pt, _)) = reuse {
                let want_jet = pt[2] > pos[2] + 3.0 && obs.me.energy >= 42.0;
                return (pt, want_jet);
            }
            let esc_dir = self.belief.escape_vector(pos, obs.t);
            let allow_roof = obs.me.energy >= 55.0;
            if let Some(cover_pt) = self.nav.find_escape_cover(
                pos,
                obs.me.eye_height(),
                threat_eyes,
                esc_dir,
                seek_heal || gun_reloading,
                allow_roof,
                &self.world,
                obs.zone.center,
                obs.zone.radius,
            ) {
                self.escape_cache = Some((cover_pt, obs.tick));
                let want_jet = cover_pt[2] > pos[2] + 3.0;
                return (cover_pt, want_jet);
            }
        } else {
            self.escape_cache = None;
        }

        // 3. High-value weapon pickup (lightning / laser, or before upgrade threshold)
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
                if already_have
                    || !self.is_in_safe_next_zone(p.pos[0], p.pos[1], obs, 6.0)
                    || !self.elevation_reachable(pos[2], p.pos[2], obs.me.energy)
                {
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
                return (p.pos, p.pos[2] > pos[2] + 2.5);
            }
        }

        // 4. Supply drops (3 cubes + full HP + full energy)
        let near_mechs = obs
            .mechs
            .iter()
            .filter(|m| m.kind == "mech" && dist_2d(pos, m.pos) < 50.0)
            .count();
        for d in &obs.drops {
            let dist = dist_2d(pos, d.pos);
            if d.lands_in <= 14.0 && dist < 140.0 {
                if self.belief.mode != MatchMode::FfaSurvival || near_mechs == 0 {
                    return (d.pos, false);
                } else if dist > 45.0 {
                    let ux = (pos[0] - d.pos[0]) / dist.max(1.0);
                    let uy = (pos[1] - d.pos[1]) / dist.max(1.0);
                    return ([d.pos[0] + ux * 42.0, d.pos[1] + uy * 42.0, d.pos[2]], false);
                }
            }
        }

        // 5. Fast 10-cube snowball (respecting reachable elevation!)
        let want_cubes = obs.me.cubes < self.cfg.cube_stat_cap || obs.me.cubes == 14;
        if want_cubes {
            if let Some(c) = obs
                .cubes
                .iter()
                .filter(|c| {
                    self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 5.0)
                        && self.elevation_reachable(pos[2], c.pos[2], obs.me.energy)
                })
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, c.pos) < 85.0 {
                    return (c.pos, c.pos[2] > pos[2] + 2.5);
                }
            }
            if let Some(cr) = obs
                .crates
                .iter()
                .filter(|c| {
                    self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 8.0)
                        && self.elevation_reachable(pos[2], c.pos[2], obs.me.energy)
                })
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, cr.pos) < 75.0 {
                    return (cr.pos, cr.pos[2] > pos[2] + 2.5);
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
                return (npc.pos, false);
            }
        }

        // 6. Mode-specific mid/late objective
        if self.belief.mode == MatchMode::PracticeHivemind
            && obs.me.hp / obs.me.max_hp.max(1.0) > 0.65
        {
            if let Some(t) = target {
                if t.is_mech {
                    return (t.pos, false);
                }
            }
            if let Some(hunt) = self.belief.nearest_hunt_point(pos, obs.t) {
                return (hunt, false);
            }
        } else if self.belief.mode == MatchMode::FfaSurvival && near_mechs == 0 {
            if let Some(roof) = self.nav.find_safe_rooftop(
                pos,
                obs.zone.next_center,
                obs.zone.next_radius,
                100.0,
            ) {
                return (roof, obs.me.energy >= 55.0);
            }
        }

        (next_z, false)
    }

    fn decide_movement(
        &mut self,
        obs: &ObsMsg,
        target: Option<&TargetInfo>,
        los_enemies: usize,
        seek_heal: bool,
        gun_reloading: bool,
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

        if obs.tick.saturating_sub(self.last_strafe_flip) >= 18 {
            self.strafe_sign = -self.strafe_sign;
            self.last_strafe_flip = obs.tick;
        }

        // Cap raycast threats to top 4 closest active enemies for guaranteed < 3 ms ticks!
        let threat_eyes = self.belief.top_threat_eyes(pos, obs.t, 4);
        let escape_dir = self.belief.escape_vector(pos, obs.t);

        let (goal, goal_wants_jet) =
            self.choose_macro_goal(obs, target, &threat_eyes, los_enemies, seek_heal, gun_reloading);
        let allow_jet = goal_wants_jet && obs.me.energy >= 35.0 && !urgent_dodge;
        let wp = self.nav.next_waypoint(obs.tick, pos, goal, allow_jet);

        // Jet-climb controller when next waypoint is elevated
        let climbing = allow_jet
            && wp.z > pos[2] + self.cfg.step_up + 0.4
            && ((wp.x - pos[0]).powi(2) + (wp.y - pos[1]).powi(2)).sqrt() < 9.5;
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

        let in_combat = !threat_eyes.is_empty();
        let retreating = seek_heal || los_enemies >= 2 || (gun_reloading && los_enemies >= 1);
        let primary_mech = target.filter(|t| t.is_mech);

        let ideal_dist = if let Some(pm) = primary_mech {
            if retreating {
                92.0
            } else if has_lightning && obs.me.energy >= 20.0 && !pm.has_mortar {
                28.0
            } else if pm.has_lightning && !has_lightning {
                68.0
            } else if has_laser {
                72.0
            } else {
                54.0
            }
        } else {
            55.0
        };

        // Fast 12-heading tactical steering (12 * <=4 threats = <=48 DDA raycasts -> <0.2ms)
        let mut best_yaw = nav_dy.atan2(nav_dx);
        let mut best_score = f64::NEG_INFINITY;

        for k in 0..12 {
            let yaw = (k as f64) * (2.0 * PI / 12.0);
            let ux = yaw.cos();
            let uy = yaw.sin();

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

            let lx = pos[0] + ux * 10.0;
            let ly = pos[1] + uy * 10.0;
            let d_cur_z = ((lx - obs.zone.center[0]).powi(2) + (ly - obs.zone.center[1]).powi(2)).sqrt();
            if d_cur_z > obs.zone.radius - 4.0 {
                score -= 220.0;
            } else if !obs.me.in_zone {
                let d_next_z = ((lx - obs.zone.next_center[0]).powi(2) + (ly - obs.zone.next_center[1]).powi(2)).sqrt();
                score -= d_next_z * 2.5;
            }

            if dodge_len > 0.05 {
                let dodge_dot = ux * dodge_u[0] + uy * dodge_u[1];
                score += dodge_dot * if urgent_dodge { 155.0 } else { 78.0 };
            }

            // Strongly follow A* escape-cover path when retreating or rotating!
            let nav_dot = ux * nav_u[0] + uy * nav_u[1];
            let nav_weight = if !obs.me.in_zone {
                95.0
            } else if retreating && self.escape_cache.is_some() {
                72.0
            } else if in_combat {
                24.0
            } else {
                68.0
            };
            score += nav_dot * nav_weight;

            if in_combat && ok_near {
                let cand_eye = [lx, ly, pos[2] + my_eye_off];
                let mut future_los = 0_usize;
                for &te in &threat_eyes {
                    if self.world.raycast_solid(te, cand_eye).is_none() {
                        future_los += 1;
                    }
                }

                // Anti-flank bias: avoid walking into the tracked/heard threat cluster
                let esc_dot = ux * escape_dir[0] + uy * escape_dir[1];
                if retreating {
                    score += esc_dot * 38.0;
                    if future_los == 0 {
                        score += if seek_heal || gun_reloading { 130.0 } else { 75.0 };
                    } else if future_los == 1 && !seek_heal && !gun_reloading {
                        score += 110.0;
                    } else {
                        score -= (future_los as f64) * 65.0;
                    }
                    if self.nav.is_near_cover(lx, ly) {
                        score += 18.0;
                    }
                } else if let Some(pm) = primary_mech {
                    score += esc_dot * 12.0; // Subtle anti-flank awareness even in 1v1s
                    let to_x = pm.pos[0] - pos[0];
                    let to_y = pm.pos[1] - pos[1];
                    let d = (to_x * to_x + to_y * to_y).sqrt().max(1.0);
                    let rx = to_x / d;
                    let ry = to_y / d;
                    let sx = -ry * self.strafe_sign;
                    let sy = rx * self.strafe_sign;

                    score += (ux * sx + uy * sy) * 44.0;

                    let new_d = ((pm.pos[0] - lx).powi(2) + (pm.pos[1] - ly).powi(2)).sqrt();
                    score += ((d - ideal_dist).abs() - (new_d - ideal_dist).abs()) * 4.5;

                    if future_los == 1 {
                        score += 20.0;
                    } else if future_los >= 2 {
                        score -= (future_los as f64 - 1.0) * 55.0;
                    }
                }
            }

            if score > best_score {
                best_score = score;
                best_yaw = yaw;
            }
        }

        // Unstuck kick if pinned against a corner
        if self.stuck_ticks >= 8 && obs.me.grounded {
            act.jump = true;
            best_yaw += 0.9 * self.strafe_sign;
        }

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

        let mut mag = 1.0;
        if self.belief.mode == MatchMode::FfaSurvival
            && !in_combat
            && !urgent_dodge
            && self.is_in_safe_next_zone(pos[0], pos[1], obs, 15.0)
            && obs.zone.radius <= 160.0
        {
            mag = ((self.cfg.sound_step_speed - 0.35) / obs.me.walk_speed()).clamp(0.25, 1.0);
        }

        act.move_dir = [best_yaw.cos() * mag, best_yaw.sin() * mag];

        // Energy management: preserve lightning reserve, sprint early to snowball loot/escape
        let min_boost_energy = if has_lightning { 38.0 } else { 20.0 };
        let early_loot_sprint = !in_combat && obs.me.cubes < self.cfg.cube_stat_cap && obs.me.energy > 68.0;
        let need_sprint = urgent_dodge
            || !obs.me.in_zone
            || (retreating && los_enemies > 0)
            || early_loot_sprint;
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