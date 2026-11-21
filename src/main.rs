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

        if let Some(offer) = &obs.offer {
            act.pick = Some(self.pick_upgrade(offer, obs));
        }
        self.handle_weapon_pickup(obs, &mut act);

        let (dodge, urgent_dodge) = compute_dodge(obs, &self.cfg, &self.world);
        let los_enemies = count_los_enemy_mechs(obs, &self.world);
        let target = select_target(
            obs,
            &self.cfg,
            &self.world,
            self.belief.mode,
            self.last_target_id,
        );
        self.last_target_id = target.as_ref().filter(|t| t.is_mech).map(|t| t.id);

        let hp_frac = obs.me.hp / obs.me.max_hp.max(1.0);
        let seek_heal = hp_frac < 0.48 && obs.me.in_zone;
        let ceasefire_for_regen = hp_frac < 0.80
            && obs.me.in_zone
            && los_enemies == 0
            && (obs.t - self.belief.last_combat_t) >= 1.2;

        let fallback_bearing = self.belief.best_unseen_bearing(obs.me.pos, obs.t);
        decide_combat(
            obs,
            &self.cfg,
            &self.world,
            self.belief.mode,
            los_enemies,
            ceasefire_for_regen,
            urgent_dodge,
            target.as_ref(),
            fallback_bearing,
            self.last_aim,
            &mut act,
        );

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
                        s += 110.0;
                    }
                    if m.contains("max_hp") || m.contains("regen") {
                        s += 102.0;
                    }
                    if m.contains("walk_speed") || m.contains("energy") || m.contains("boost") {
                        s += 88.0;
                    }
                }
            }

            if id.contains("armor") || desc.contains("damage taken") {
                s += 120.0;
            }
            if id.contains("hp") || desc.contains("max hp") || desc.contains("regen") {
                s += 105.0;
            }
            if id.contains("lightning") {
                s += if held_lightning { 114.0 } else { 40.0 };
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

    fn choose_macro_goal(
        &mut self,
        obs: &ObsMsg,
        target: Option<&TargetInfo>,
        visible_threat_eyes: &[[f64; 3]],
        los_enemies: usize,
        seek_heal: bool,
    ) -> [f64; 3] {
        let pos = obs.me.pos;
        let next_z = [
            obs.zone.next_center[0],
            obs.zone.next_center[1],
            self.world
                .terrain_height(obs.zone.next_center[0], obs.zone.next_center[1]),
        ];

        // 1. Zone rotation
        let dist_next = dist_2d(pos, next_z);
        let time_to_shrink = (obs.zone.shrink_start - obs.t).max(0.0);
        let must_rotate = !obs.me.in_zone
            || (dist_next > (obs.zone.next_radius - 16.0).max(5.0)
                && (time_to_shrink < 16.0 || obs.zone.dps >= 60.0));
        if must_rotate {
            return next_z;
        }

        // 2. Ground-only building-corner escape ONLY when critically low HP under LOS fire
        if seek_heal && los_enemies > 0 && !visible_threat_eyes.is_empty() {
            let reuse = self
                .escape_cache
                .filter(|&(pt, tick)| obs.tick.saturating_sub(tick) < 6 && dist_2d(pos, pt) > 6.0);
            if let Some((pt, _)) = reuse {
                return pt;
            }
            let esc_dir = self.belief.escape_vector(pos, obs.t);
            if let Some(cover_pt) = self.nav.find_escape_cover(
                pos,
                obs.me.eye_height(),
                visible_threat_eyes,
                esc_dir,
                true,
                false, // Never jet-climb mid-combat
                &self.world,
                obs.zone.center,
                obs.zone.radius,
            ) {
                self.escape_cache = Some((cover_pt, obs.tick));
                return cover_pt;
            }
        } else {
            self.escape_cache = None;
        }

        // 3. Ground-level high-value weapon pickups (lightning / laser)
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
                    || (p.pos[2] - pos[2]) > self.cfg.step_up + 1.5
                    || !self.is_in_safe_next_zone(p.pos[0], p.pos[1], obs, 6.0)
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
                return p.pos;
            }
        }

        // 4. Supply drops (skip if contested by >=2 LOS mechs)
        let near_mechs = obs
            .mechs
            .iter()
            .filter(|m| m.kind == "mech" && dist_2d(pos, m.pos) < 50.0)
            .count();
        for d in &obs.drops {
            let dist = dist_2d(pos, d.pos);
            if d.lands_in <= 14.0 && dist < 140.0 && los_enemies <= 1 {
                if self.belief.mode != MatchMode::FfaSurvival || near_mechs == 0 {
                    return d.pos;
                } else if dist > 45.0 {
                    let ux = (pos[0] - d.pos[0]) / dist.max(1.0);
                    let uy = (pos[1] - d.pos[1]) / dist.max(1.0);
                    return [d.pos[0] + ux * 42.0, d.pos[1] + uy * 42.0, d.pos[2]];
                }
            }
        }

        // 5. Fast 10-cube snowball: loose cubes > NPCs (2 cubes) > ground crates
        let want_cubes = obs.me.cubes < self.cfg.cube_stat_cap || obs.me.cubes == 14;
        if want_cubes {
            if let Some(c) = obs
                .cubes
                .iter()
                .filter(|c| {
                    (c.pos[2] - pos[2]) <= self.cfg.step_up + 1.5
                        && self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 5.0)
                })
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, c.pos) < 85.0 {
                    return c.pos;
                }
            }
            if let Some(npc) = obs
                .mechs
                .iter()
                .filter(|m| {
                    m.kind == "npc"
                        && self.is_in_safe_next_zone(m.pos[0], m.pos[1], obs, 10.0)
                        && dist_2d(pos, m.pos) < 80.0
                })
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                return npc.pos;
            }
            if let Some(cr) = obs
                .crates
                .iter()
                .filter(|c| {
                    (c.pos[2] - pos[2]) <= self.cfg.step_up + 1.5
                        && self.is_in_safe_next_zone(c.pos[0], c.pos[1], obs, 8.0)
                })
                .min_by(|a, b| dist_2d(pos, a.pos).total_cmp(&dist_2d(pos, b.pos)))
            {
                if dist_2d(pos, cr.pos) < 75.0 {
                    return cr.pos;
                }
            }
        }

        // 6. Mode-specific mid/late objective
        if self.belief.mode == MatchMode::PracticeHivemind
            && obs.me.hp / obs.me.max_hp.max(1.0) > 0.62
        {
            if let Some(t) = target {
                if t.is_mech {
                    return t.pos;
                }
            }
            if let Some(hunt) = self.belief.nearest_hunt_point(pos, obs.t) {
                return hunt;
            }
        } else if self.belief.mode == MatchMode::FfaSurvival && near_mechs == 0 {
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

        let has_lightning = obs.me.weapons.iter().any(|w| w.as_ref().map_or(false, |s| s.kind == "lightning"));
        let has_laser = obs.me.weapons.iter().any(|w| w.as_ref().map_or(false, |s| s.kind == "laser"));

        if obs.tick.saturating_sub(self.last_strafe_flip) >= 18 {
            self.strafe_sign = -self.strafe_sign;
            self.last_strafe_flip = obs.tick;
        }

        // Strictly currently-visible enemy mechs (top 4 closest) for combat steering
        let mut vis_enemies: Vec<(f64, [f64; 3])> = obs
            .mechs
            .iter()
            .filter(|m| m.kind == "mech")
            .map(|m| {
                let d2 = (m.pos[0] - pos[0]).powi(2) + (m.pos[1] - pos[1]).powi(2);
                let h = 9.0 * (1.0 + 0.05 * m.cubes as f64);
                (d2, [m.pos[0], m.pos[1], m.pos[2] + 0.78 * h])
            })
            .filter(|(d2, _)| *d2 <= 155.0 * 155.0)
            .collect();
        vis_enemies.sort_by(|a, b| a.0.total_cmp(&b.0));
        vis_enemies.truncate(4);
        let visible_threat_eyes: Vec<[f64; 3]> = vis_enemies.into_iter().map(|(_, p)| p).collect();

        let goal = self.choose_macro_goal(obs, target, &visible_threat_eyes, los_enemies, seek_heal);
        let allow_jet = self.belief.mode == MatchMode::FfaSurvival
            && los_enemies == 0
            && obs.me.energy >= 65.0
            && !urgent_dodge;
        let wp = self.nav.next_waypoint(obs.tick, pos, goal, allow_jet);

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

        let in_combat = !visible_threat_eyes.is_empty();
        let primary_mech = target.filter(|t| t.is_mech);
        let escape_dir = self.belief.escape_vector(pos, obs.t);

        // Stay at the edge of lightning range (37m) for max bullet-dodging space
        let ideal_dist = if let Some(pm) = primary_mech {
            if seek_heal || los_enemies >= 2 {
                90.0
            } else if has_lightning && obs.me.energy >= 20.0 && !pm.has_mortar {
                37.0
            } else if pm.has_lightning && !has_lightning {
                68.0
            } else if has_laser {
                74.0
            } else {
                56.0
            }
        } else {
            56.0
        };

        let mut best_yaw = nav_dy.atan2(nav_dx);
        let mut best_score = f64::NEG_INFINITY;

        for k in 0..16 {
            let yaw = (k as f64) * (2.0 * PI / 16.0);
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
                score += (ux * dodge_u[0] + uy * dodge_u[1]) * if urgent_dodge { 155.0 } else { 78.0 };
            }

            let nav_dot = ux * nav_u[0] + uy * nav_u[1];
            let nav_weight = if !obs.me.in_zone {
                95.0
            } else if seek_heal && self.escape_cache.is_some() {
                48.0
            } else if in_combat {
                22.0
            } else {
                68.0
            };
            score += nav_dot * nav_weight;

            if in_combat && ok_near {
                let cand_eye = [lx, ly, pos[2] + my_eye_off];
                let mut future_los = 0_usize;
                for &te in &visible_threat_eyes {
                    if self.world.raycast_solid(te, cand_eye).is_none() {
                        future_los += 1;
                    }
                }

                if seek_heal || los_enemies >= 2 {
                    score += (ux * escape_dir[0] + uy * escape_dir[1]) * 26.0;
                    if future_los == 0 {
                        score += if seek_heal { 130.0 } else { 72.0 };
                    } else if future_los == 1 && !seek_heal {
                        score += 110.0;
                    } else {
                        score -= (future_los as f64) * 65.0;
                    }
                    if self.nav.is_near_cover(lx, ly) {
                        score += 18.0;
                    }
                } else if let Some(pm) = primary_mech {
                    let to_x = pm.pos[0] - pos[0];
                    let to_y = pm.pos[1] - pos[1];
                    let d = (to_x * to_x + to_y * to_y).sqrt().max(1.0);
                    let sx = -(to_y / d) * self.strafe_sign;
                    let sy = (to_x / d) * self.strafe_sign;

                    score += (ux * sx + uy * sy) * 46.0;

                    let new_d = ((pm.pos[0] - lx).powi(2) + (pm.pos[1] - ly).powi(2)).sqrt();
                    score += ((d - ideal_dist).abs() - (new_d - ideal_dist).abs()) * 4.5;

                    if future_los == 1 {
                        score += 20.0;
                    }
                }
            }

            if score > best_score {
                best_score = score;
                best_yaw = yaw;
            }
        }

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

        let min_boost_energy = if has_lightning { 38.0 } else { 20.0 };
        let early_loot_sprint = !in_combat && obs.me.cubes < self.cfg.cube_stat_cap && obs.me.energy > 72.0;
        let need_sprint = urgent_dodge
            || !obs.me.in_zone
            || (seek_heal && los_enemies > 0)
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