mod belief;
mod combat;
mod nav;
mod protocol;
mod world;

use belief::{BeliefMap, MatchMode, TrackedMech};
use combat::*;
use nav::NavEngine;
use protocol::*;
use std::f64::consts::PI;
use std::io::{self, BufWriter, Write};
use world::World;

const UPGRADE_PREF: &[&str] = &[
    "plating",
    "armor",
    "nanites",
    "lightning_efficiency",
    "laser_damage",
    "laser_charge",
    "gun_damage",
    "gun_rate",
    "lightning_range",
    "lightning_chain",
    "gyros",
    "servos",
    "gun_mag",
    "capacitor",
    "reactor",
    "jets",
    "gun_velocity",
    "sensors",
    "mortar_reload",
    "mortar_velocity",
    "mortar_splash",
    "laser_pierce",
];

#[inline]
fn weapon_pref(kind: &str) -> f64 {
    match kind {
        "laser" => 3.2,
        "lightning" => 2.5,
        "mortar" => 1.0,
        "gun" => 0.0,
        _ => 0.0,
    }
}

struct CoverCache {
    goal: [f64; 2],
    path: Vec<[f64; 2]>,
    tick: u64,
    orig_los: usize,
}

struct BotBrain {
    cfg: RuleConfig,
    world: World,
    nav: NavEngine,
    belief: BeliefMap,
    rng_state: u32,
    strafe_dir: f64,
    strafe_until: f64,
    last_pos: [f64; 3],
    stuck_t: f64,
    scan_phase: f64,
    hide_until: f64,
    target_id: Option<u32>,
    target_t: f64,
    cover_cache: Option<CoverCache>,
}

impl BotBrain {
    fn new(seed: u64, cfg: RuleConfig, world: World) -> Self {
        let nav = NavEngine::build(&world, cfg.step_up, cfg.max_slope_deg);
        let rng_state = ((seed as u32) ^ ((seed >> 32) as u32)).max(1);
        Self {
            cfg,
            world,
            nav,
            belief: BeliefMap::new(),
            rng_state,
            strafe_dir: 1.0,
            strafe_until: 0.0,
            last_pos: [0.0; 3],
            stuck_t: 0.0,
            scan_phase: 0.0,
            hide_until: 0.0,
            target_id: None,
            target_t: -99.0,
            cover_cache: None,
        }
    }

    fn next_rng(&mut self) -> f64 {
        let mut s = self.rng_state;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.rng_state = s.max(1);
        (s as f64) / 4_294_967_296.0
    }

    fn pick_upgrade(&self, offer: &Offer, obs: &ObsMsg) -> usize {
        let held_kind = |k: &str| obs.me.weapons.iter().flatten().any(|w| w.kind == k);
        let mut best_idx = 0_usize;
        let mut best_rank = usize::MAX;
        for (i, c) in offer.choices.iter().enumerate() {
            let mut rank = UPGRADE_PREF
                .iter()
                .position(|&id| id == c.id)
                .unwrap_or(50);
            if let Some(def) = self.cfg.upgrades.iter().find(|u| u.id == c.id) {
                if let Some(req) = &def.requires {
                    if !held_kind(req) {
                        rank += 100;
                    }
                }
            }
            if rank < best_rank {
                best_rank = rank;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Returns:
    /// - `None`: pickup not wanted
    /// - `Some(None)`: take into empty slot
    /// - `Some(Some(slot))`: swap out `slot` (never drops `"gun"` or a charging `"laser"`)
    fn want_kind(&self, obs: &ObsMsg, kind: &str) -> Option<Option<usize>> {
        let w0 = obs.me.weapons.first().and_then(|s| s.as_ref());
        let w1 = obs.me.weapons.get(1).and_then(|s| s.as_ref());
        if w0.map_or(false, |w| w.kind == kind) || w1.map_or(false, |w| w.kind == kind) {
            return None;
        }
        if w0.is_none() || w1.is_none() {
            return Some(None);
        }
        let mut worst_slot: Option<usize> = None;
        let mut worst_val = f64::INFINITY;
        for (i, w_opt) in [w0, w1].iter().enumerate() {
            let Some(w) = w_opt else { continue };
            if w.kind == "gun" || (w.kind == "laser" && w.charge > 0.0) {
                continue;
            }
            let v = weapon_pref(&w.kind);
            if v < worst_val {
                worst_val = v;
                worst_slot = Some(i);
            }
        }
        if let Some(slot) = worst_slot {
            if weapon_pref(kind) > worst_val + 0.5 {
                return Some(Some(slot));
            }
        }
        None
    }

    fn dual_height_los(&self, muzzle: [f64; 3], tr: &TrackedMech) -> ([f64; 3], bool) {
        let h = 9.0 * (1.0 + self.cfg.cube_size_per_cube * tr.cubes as f64);
        let center = [tr.pos[0], tr.pos[1], tr.pos[2] + 0.50 * h];
        let upper = [tr.pos[0], tr.pos[1], tr.pos[2] + 0.78 * h];
        if self.world.raycast_solid(muzzle, center).is_none() {
            (center, true)
        } else if self.world.raycast_solid(muzzle, upper).is_none() {
            (upper, true)
        } else {
            (center, false)
        }
    }

    fn escape_vector(&self, x: f64, y: f64, now: f64) -> [f64; 2] {
        let mut rx = 0.0;
        let mut ry = 0.0;
        for tr in self.belief.tracks.values() {
            let age = (now - tr.t).max(0.0);
            if age > 7.0 {
                continue;
            }
            let dx = x - tr.pos[0];
            let dy = y - tr.pos[1];
            let d = dx.hypot(dy).max(4.0);
            if d > 155.0 {
                continue;
            }
            let w = (1.0 - age / 8.0) * (90.0 / d).min(3.5);
            rx += (dx / d) * w;
            ry += (dy / d) * w;
        }
        let len = rx.hypot(ry);
        if len > 1e-4 {
            [rx / len, ry / len]
        } else {
            [0.0, 0.0]
        }
    }

    /// Cached path-verified 2 m BFS cover step around building corners (ready on tick 0).
    fn get_bfs_cover_step(
        &mut self,
        obs: &ObsMsg,
        threat_eyes: &[[f64; 3]],
        escape_dir: [f64; 2],
    ) -> Option<[f64; 2]> {
        if threat_eyes.is_empty() {
            self.cover_cache = None;
            return None;
        }
        let [x, y, _] = obs.me.pos;
        let eye_off = obs.me.eye_height();
        let safe_r = (obs.zone.radius - 5.0).max(5.0);
        let world = &self.world;
        let los_at = |pt: [f64; 2]| -> usize {
            let gz = world.terrain_height(pt[0], pt[1]);
            let pt_eye = [pt[0], pt[1], gz + eye_off];
            threat_eyes
                .iter()
                .filter(|&&te| world.raycast_solid(te, pt_eye).is_none())
                .count()
        };

        let reuse_ok = self.cover_cache.as_ref().map_or(false, |cc| {
            obs.tick.saturating_sub(cc.tick) < 6
                && !cc.path.is_empty()
                && (cc.goal[0] - x).hypot(cc.goal[1] - y) > 4.0
                && (cc.goal[0] - obs.zone.center[0]).hypot(cc.goal[1] - obs.zone.center[1])
                    <= safe_r
                && los_at(cc.goal) <= cc.orig_los
        });

        if !reuse_ok {
            self.cover_cache = self
                .nav
                .find_bfs_cover(
                    x,
                    y,
                    eye_off,
                    threat_eyes,
                    escape_dir,
                    obs.zone.center,
                    obs.zone.radius,
                    obs.zone.next_center,
                    obs.zone.next_radius,
                    &self.world,
                )
                .map(|(goal, path, orig_los)| CoverCache {
                    goal,
                    path,
                    tick: obs.tick,
                    orig_los,
                });
        }

        let nav = &self.nav;
        if let Some(cc) = self.cover_cache.as_mut() {
            while cc.path.len() > 1
                && (cc.path[0][0] - x).hypot(cc.path[0][1] - y) < 3.5
            {
                cc.path.remove(0);
            }
            let max_look = cc.path.len().min(8);
            for idx in (1..max_look).rev() {
                let wp = cc.path[idx];
                if nav.walk_segment(x, y, wp[0], wp[1]) {
                    return Some(wp);
                }
            }
            cc.path.first().copied().or(Some(cc.goal))
        } else {
            None
        }
    }

    fn decide(&mut self, obs: &ObsMsg) -> Action {
        self.belief.update(obs, &self.world);

        let me = &obs.me;
        let [x, y, z] = me.pos;
        let t = obs.t;
        let radius = me.radius();
        let height = me.height();
        let muzzle_z = z + me.muzzle_height();
        let muzzle = [x, y, muzzle_z];
        let turn_rate = me.turn_rate();
        let pitch_rate = me.stat_f64("pitch_rate", 3.0);
        let hp_frac = me.hp / me.max_hp.max(1.0);
        let big_mech = me.cubes >= 6;
        let cube_mult =
            1.0 + self.cfg.cube_damage_per_cube * (me.cubes.min(self.cfg.cube_stat_cap) as f64);

        let mut act = Action::idle(obs.tick, me.aim);

        // 1. Upgrades
        if let Some(offer) = &obs.offer {
            act.pick = Some(self.pick_upgrade(offer, obs));
        }

        // 2. Held weapon slots & incoming dodge / heavy threats
        let find_slot = |k: &str| -> Option<(usize, &WeaponSlot)> {
            me.weapons
                .iter()
                .enumerate()
                .take(2)
                .find_map(|(i, s)| s.as_ref().filter(|w| w.kind == k).map(|w| (i, w)))
        };
        let gun_slot = find_slot("gun");
        let laser_slot = find_slot("laser");
        let light_slot = find_slot("lightning");
        let mortar_slot = find_slot("mortar");
        let only_gun = me.weapons.iter().take(2).any(|w| w.is_none());

        let (dodge, urgent_dodge, heavy_threat) = compute_dodge(obs, &self.cfg, &self.world);
        let dodge_len = dodge[0].hypot(dodge[1]);

        // 3. Nearest wanted pickup & cube from persistent memory
        let mut near_pick: Option<[f64; 3]> = None;
        let mut near_pick_d = f64::INFINITY;
        for p in self.belief.pick_mem.values() {
            if self.want_kind(obs, &p.kind).is_none() {
                continue;
            }
            let d = (p.pos[0] - x).hypot(p.pos[1] - y)
                - if only_gun { 0.0 } else { weapon_pref(&p.kind) * 5.0 };
            if d < near_pick_d {
                near_pick_d = d;
                near_pick = Some(p.pos);
            }
        }

        // Immediate weapon pickup (avoid mid-brawl full-slot lockout when an enemy is close)
        let close_mech_threat = obs
            .mechs
            .iter()
            .any(|m| m.kind == "mech" && (m.pos[0] - x).hypot(m.pos[1] - y) < 55.0);
        let reach = (self.cfg.pickup_radius - 0.3).max(2.0);
        for p in &obs.pickups {
            if (p.pos[0] - x).hypot(p.pos[1] - y) > reach || (p.pos[2] - z).abs() > 3.5 {
                continue;
            }
            let Some(swap_slot) = self.want_kind(obs, &p.kind) else {
                continue;
            };
            if swap_slot.is_some() && close_mech_threat && p.kind != "laser" {
                continue;
            }
            act.take = Some(p.id);
            act.drop = swap_slot;
            break;
        }

        let mut near_cube: Option<[f64; 3]> = None;
        let mut near_cube_d = f64::INFINITY;
        for c in self.belief.cube_mem.values() {
            let d = (c.pos[0] - x).hypot(c.pos[1] - y);
            if d < near_cube_d {
                near_cube_d = d;
                near_cube = Some(c.pos);
            }
        }

        // 4. Select primary enemy mech target (dual-height LOS, kill-focus & signature bonuses)
        let is_ffa = self.belief.mode == MatchMode::FfaSurvival;
        let in_hide_window = t < self.hide_until && hp_frac < 0.85;
        let undergeared_with_loot =
            (only_gun || me.cubes < 6) && (near_pick_d < 85.0 || near_cube_d < 70.0);
        let mut target: Option<(TrackedMech, [f64; 3], bool)> = None;
        let mut best_t_score = f64::INFINITY;
        for tr in self.belief.tracks.values() {
            let age = (t - tr.t).max(0.0);
            if age > 2.5 {
                continue;
            }
            let (aim_pt, has_los) = self.dual_height_los(muzzle, tr);
            let visible_now = age < 0.15;

            if in_hide_window && (!visible_now || !has_los) {
                continue;
            }
            let allowed = visible_now
                || (has_los && age <= 1.2)
                || (!has_los && mortar_slot.is_some() && age <= 2.2);
            if !allowed {
                continue;
            }

            let d = (tr.pos[0] - x).hypot(tr.pos[1] - y);
            let hit_us = self
                .belief
                .attackers
                .get(&tr.id)
                .map_or(false, |&ht| t - ht < 3.0);
            if is_ffa && obs.alive > 5 && d > 105.0 && tr.hp > 1000.0 && !hit_us {
                continue;
            }
            if undergeared_with_loot && d > 115.0 && !hit_us && tr.hp > 900.0 {
                continue;
            }

            let enemy_mortar = tr.weapons.iter().any(|w| w == "mortar");
            let bearing_from_enemy = (y - tr.pos[1]).atan2(x - tr.pos[0]);
            let rear_exposed = angle_diff(tr.aim[0], bearing_from_enemy).abs() > PI * 0.52;

            let hit_bonus = if hit_us { 50.0 } else { 0.0 };
            let stick_bonus = if self.target_id == Some(tr.id) && t - self.target_t < 2.0 {
                45.0
            } else {
                0.0
            };
            let los_bonus = if has_los { 35.0 } else { 0.0 };
            let cookoff_bonus = if laser_slot.is_some() && enemy_mortar && has_los {
                45.0
            } else {
                0.0
            };
            let charge_bonus = if tr.charging && has_los { 35.0 } else { 0.0 };
            let rear_bonus = if rear_exposed && has_los { 25.0 } else { 0.0 };
            let hp_term = (tr.hp / tr.max_hp.max(1.0)) * 135.0;

            let s = d + hp_term + age * 30.0
                - hit_bonus
                - stick_bonus
                - los_bonus
                - cookoff_bonus
                - charge_bonus
                - rear_bonus;
            if s < best_t_score {
                best_t_score = s;
                target = Some((tr.clone(), aim_pt, has_los));
            }
        }
        if let Some((tr, _, _)) = &target {
            self.target_id = Some(tr.id);
            self.target_t = t;
        }

        // Active threats & secondary-enemy repulsion vector
        let mut threats = 0_usize;
        let mut active_threat_eyes: Vec<[f64; 3]> = Vec::with_capacity(6);
        let mut sec_repel = [0.0_f64, 0.0_f64];
        let primary_id = target.as_ref().map(|(tr, _, _)| tr.id);
        for tr in self.belief.tracks.values() {
            let d = (tr.pos[0] - x).hypot(tr.pos[1] - y);
            if t - tr.t < 3.0 && d < 160.0 {
                threats += 1;
                let h = 9.0 * (1.0 + self.cfg.cube_size_per_cube * tr.cubes as f64);
                active_threat_eyes.push([tr.pos[0], tr.pos[1], tr.pos[2] + 0.78 * h]);
                if Some(tr.id) != primary_id && d < 120.0 {
                    let w = (80.0 / d.max(8.0)).min(2.0);
                    sec_repel[0] += ((x - tr.pos[0]) / d.max(1.0)) * w;
                    sec_repel[1] += ((y - tr.pos[1]) / d.max(1.0)) * w;
                }
            }
        }
        let sec_len = sec_repel[0].hypot(sec_repel[1]);
        if sec_len > 1e-4 {
            sec_repel[0] /= sec_len;
            sec_repel[1] /= sec_len;
        }
        let escape_dir = self.escape_vector(x, y, t);

        // 5. Zone rotation goal (near-edge, travel-time triggered)
        let zn = &obs.zone;
        let shrinking = t >= zn.shrink_start;
        let time_to_shrink = zn.shrink_start - t;
        let d_next = (x - zn.next_center[0]).hypot(y - zn.next_center[1]);
        let d_cur = (x - zn.center[0]).hypot(y - zn.center[1]);
        let walk_speed = me.walk_speed().max(1.0);
        let need_next = d_next > zn.next_radius * 0.8
            && (shrinking
                || time_to_shrink < (d_next - zn.next_radius * 0.6) / walk_speed + 12.0);
        let zone_goal: Option<[f64; 2]> = if !me.in_zone || d_cur > zn.radius * 0.9 || need_next {
            let ang = (y - zn.next_center[1]).atan2(x - zn.next_center[0]);
            let r = zn.next_radius * 0.45;
            Some([
                zn.next_center[0] + ang.cos() * r,
                zn.next_center[1] + ang.sin() * r,
            ])
        } else {
            None
        };

        // Check if we are actively holding/capturing an uncontested ground supply drop
        let active_drop = if me.in_zone {
            obs.drops.iter().find(|dr| {
                let gz = self.world.terrain_height(dr.pos[0], dr.pos[1]);
                if (dr.pos[2] - gz).abs() > 2.5 {
                    return false;
                }
                if dr.holder.map_or(false, |h| h != me.id) {
                    return false;
                }
                let d = (dr.pos[0] - x).hypot(dr.pos[1] - y);
                dr.holder == Some(me.id) || (d <= dr.radius + 12.0 && dr.lands_in <= 3.0)
            })
        } else {
            None
        };

        // 6. Combat & macro decision
        let mut move_goal: Option<[f64; 2]> = None;
        let mut combat_mv: Option<[f64; 2]> = None;
        let mut using_bfs_cover = false;
        let mut aim_yaw = me.aim[0];
        let mut aim_pitch = 0.0_f64;
        let mut want_aim = false;
        let mut in_fight = false;
        let mut has_los_to_target = false;
        let mut laser_handled = false;
        let mut is_losing = false;
        let mut navigating = false;
        let mut high_prio_loot = false;
        let mut boost = false;
        let mut jump = false;
        let mut slow = false;
        let laser_charging = laser_slot.map_or(false, |(_, w)| w.charge > 0.0);

        if let Some((tr, aim_pt, has_los)) = &target {
            in_fight = true;
            has_los_to_target = *has_los;
            let tp = tr.pos;
            let d = (tp[0] - x).hypot(tp[1] - y);
            let t_scale = 1.0 + self.cfg.cube_size_per_cube * tr.cubes as f64;
            let age = (t - tr.t).max(0.0);
            let seen = age < 0.15;

            // Threat-scaled retreat with hide_until hysteresis (computed before Laser charge)
            let their_frac = tr.hp / tr.max_hp.max(1.0);
            let finishing = their_frac < 0.20 && hp_frac > 0.20 && threats <= 2;
            let losing = !finishing
                && ((hp_frac < 0.35 && their_frac > hp_frac + 0.10)
                    || (t < self.hide_until && hp_frac < 0.65 && their_frac > hp_frac + 0.15)
                    || (threats >= 2 && hp_frac < 0.55)
                    || (threats >= 3 && hp_frac < 0.80));
            is_losing = losing;
            if losing {
                self.hide_until = t + 6.0;
            }

            let short_lead_pt = [
                aim_pt[0] + tr.vel[0] * 0.05,
                aim_pt[1] + tr.vel[1] * 0.05,
                aim_pt[2] + tr.vel[2] * 0.05,
            ];
            let instant_aim = aim_to(muzzle, short_lead_pt);
            let gun_speed = gun_slot.map_or(252.0, |(_, g)| g.stat_f64("speed", 252.0));
            let gun_lead = lead_aim(muzzle, *aim_pt, tr.vel, age, gun_speed, &self.world);

            let laser_snap_aim = !losing
                && laser_slot.map_or(false, |(_, lw)| {
                    if lw.charge <= 0.0 {
                        return false;
                    }
                    let full = lw.stat_f64("charge", 0.75);
                    let min_chg = lw.stat_f64("min_charge", 0.15);
                    let min_frac = lw.stat_f64("min_frac", 0.35);
                    let base_dmg = lw.stat_f64("damage", 315.0);
                    let hold_s = lw.stat_f64("hold_s", 2.0);
                    let frac =
                        min_frac + (1.0 - min_frac) * (lw.charge / full.max(0.05)).clamp(0.0, 1.0);
                    let enemy_mortar = tr.weapons.iter().any(|w| w == "mortar");
                    let expected_dmg =
                        base_dmg * frac * if enemy_mortar { 1.6 } else { 1.0 } * cube_mult;
                    lw.charge >= full - 0.20
                        || (lw.charge >= min_chg && expected_dmg >= tr.hp * 1.02)
                        || lw.charge >= full + hold_s - 0.35
                });

            let gun_ready_to_shoot =
                gun_slot.map_or(false, |(_, gw)| gw.reload <= 0.0 && gw.ammo != Some(0));

            let chosen_aim = if let Some((_, lw)) = light_slot {
                let l_range = lw.stat_f64("range", 45.0);
                if d < l_range * 1.1 {
                    let cone = lw.stat_f64("cone_deg", 30.0).to_radians() * 0.45;
                    if gun_ready_to_shoot
                        && angle_diff(gun_lead[0], instant_aim[0]).abs() <= cone * 0.80
                    {
                        gun_lead
                    } else {
                        instant_aim
                    }
                } else if laser_snap_aim || (!gun_ready_to_shoot && laser_charging) {
                    instant_aim
                } else {
                    gun_lead
                }
            } else if laser_snap_aim || (!gun_ready_to_shoot && laser_charging) {
                instant_aim
            } else {
                gun_lead
            };

            aim_yaw = chosen_aim[0];
            aim_pitch = chosen_aim[1];
            want_aim = true;

            let pred_aim =
                predict_aim_after_step(me.aim, [aim_yaw, aim_pitch], turn_rate, pitch_rate);
            let tol = (2.2 * t_scale).atan2(d.max(1.0));

            if seen && *has_los {
                if let Some((gi, gw)) = gun_slot {
                    let g_range = gw.stat_f64("range", 230.0);
                    if gw.reload <= 0.0 && gw.ammo == Some(0) {
                        act.reload = true;
                    } else if gw.reload <= 0.0 && d < g_range {
                        let g_yaw_err = angle_diff(pred_aim[0], gun_lead[0]).abs();
                        let g_pitch_err = (pred_aim[1] - gun_lead[1]).abs();
                        let dir = aim_dir(pred_aim);
                        let barrel_tip = [
                            muzzle[0] + dir[0] * (radius + 0.5),
                            muzzle[1] + dir[1] * (radius + 0.5),
                            muzzle[2] + dir[2] * (radius + 0.5),
                        ];
                        if g_yaw_err < tol * 1.25
                            && g_pitch_err < tol * 2.0 + 0.03
                            && self.world.raycast_solid(muzzle, barrel_tip).is_none()
                        {
                            act.fire[gi] = true;
                        }
                    }
                }

                if let Some((li, lw)) = light_slot {
                    let l_range = lw.stat_f64("range", 45.0);
                    let l_cone = lw.stat_f64("cone_deg", 30.0).to_radians() * 0.47;
                    let l_eng = lw.stat_f64("energy", 3.0);
                    let l_yaw_err = angle_diff(pred_aim[0], instant_aim[0]).abs();
                    if d < l_range - 0.5 && l_yaw_err < l_cone && me.energy > l_eng + 2.0 {
                        act.fire[li] = true;
                    }
                }
            }

            // Laser: charge on LOS or verified corner peek; release on full, lethal, retreat, or heavy dodge
            if let Some((li, lw)) = laser_slot {
                laser_handled = true;
                let range = lw.stat_f64("range", 280.0);
                let full = lw.stat_f64("charge", 0.75);
                let min_chg = lw.stat_f64("min_charge", 0.15);
                let min_frac = lw.stat_f64("min_frac", 0.35);
                let base_dmg = lw.stat_f64("damage", 315.0);
                let hold_s = lw.stat_f64("hold_s", 2.0);
                let ch = lw.charge;

                if ch > 0.0 {
                    let frac =
                        min_frac + (1.0 - min_frac) * (ch / full.max(0.05)).clamp(0.0, 1.0);
                    let enemy_mortar = tr.weapons.iter().any(|w| w == "mortar");
                    let expected_dmg =
                        base_dmg * frac * if enemy_mortar { 1.6 } else { 1.0 } * cube_mult;

                    let cur_yaw_err = angle_diff(me.aim[0], instant_aim[0]).abs();
                    let cur_pitch_err = (me.aim[1] - instant_aim[1]).abs();
                    let dir = aim_dir(me.aim);
                    let beam_dist = (d - 3.0).max(radius + 0.5);
                    let beam_end = [
                        muzzle[0] + dir[0] * beam_dist,
                        muzzle[1] + dir[1] * beam_dist,
                        muzzle[2] + dir[2] * beam_dist,
                    ];
                    let beam_clear = self.world.raycast_solid(muzzle, beam_end).is_none();

                    let ready = ch >= full - 0.05
                        || (ch >= min_chg && expected_dmg >= tr.hp * 1.02);
                    let aligned = seen
                        && *has_los
                        && cur_yaw_err < tol * 0.95
                        && cur_pitch_err < tol * 1.5 + 0.02
                        && beam_clear;
                    let hold_expired = ch >= full + hold_s - 0.20;
                    let abort_for_dodge = heavy_threat && ch < full - 0.12;

                    if (ready && aligned) || hold_expired || losing || abort_for_dodge {
                        act.fire[li] = false;
                    } else {
                        act.fire[li] = true;
                    }
                } else if !losing && !heavy_threat && lw.cooldown <= 0.05 && d < range {
                    let enemy_peeking_soon = !*has_los && age <= 1.0 && d <= 140.0 && {
                        let pred_t = [
                            aim_pt[0] + tr.vel[0] * 0.55,
                            aim_pt[1] + tr.vel[1] * 0.55,
                            aim_pt[2],
                        ];
                        self.world.raycast_solid(muzzle, pred_t).is_none()
                    };
                    if (seen && *has_los) || enemy_peeking_soon {
                        act.fire[li] = true;
                    }
                }
            }

            // Dual-arc Mortar (only when not firing primary arms or holding laser charge)
            let holding_laser_now = laser_slot.map_or(false, |(li, _)| act.fire[li]);
            if let Some((mi, mw)) = mortar_slot {
                if !act.fire[0]
                    && !act.fire[1]
                    && !holding_laser_now
                    && mw.cooldown <= 0.05
                    && d > 16.0
                    && d < 225.0
                {
                    let splash_r = mw.stat_f64("splash_radius", 7.0);
                    let speed = mw.stat_f64("speed", self.cfg.mortar_speed);
                    let (m_pos, m_vel) =
                        compute_mortar_aim_point(obs, tr.id, tr.pos, tr.vel, splash_r);
                    if let Some(sol) =
                        solve_mortar(obs, &self.cfg, &self.world, m_pos, m_vel, speed, splash_r)
                    {
                        aim_yaw = sol.aim[0];
                        aim_pitch = sol.aim[1];
                        let pred =
                            predict_aim_after_step(me.aim, sol.aim, turn_rate, pitch_rate);
                        if angle_diff(pred[0], sol.aim[0]).abs() < 0.06
                            && (pred[1] - sol.aim[1]).abs() < 0.06
                        {
                            act.fire[mi] = true;
                        }
                    }
                }
            }

            // Combat movement, BFS cover retreat, Laser kiting, or drop holding
            let prefer = if let Some((_, lw)) = light_slot {
                (lw.stat_f64("range", 45.0) - 10.0).max(26.0)
            } else if laser_slot.is_some() {
                125.0
            } else {
                75.0
            };

            let hold_drop_in_fight = active_drop.filter(|dr| {
                !losing
                    && ((!*has_los || d > 48.0)
                        || (dr.progress >= dr.capture_s - 1.5 && hp_frac > 0.35))
            });

            if losing {
                let retreat_threats = if !active_threat_eyes.is_empty() {
                    active_threat_eyes.clone()
                } else {
                    vec![[tp[0], tp[1], tp[2] + 8.0 * t_scale]]
                };
                if let Some(step_wp) =
                    self.get_bfs_cover_step(obs, &retreat_threats, escape_dir)
                {
                    move_goal = Some(step_wp);
                    using_bfs_cover = true;
                } else {
                    move_goal = Some([x + escape_dir[0] * 14.0, y + escape_dir[1] * 14.0]);
                }
                boost = me.energy > 15.0;
            } else if let Some(dr) = hold_drop_in_fight {
                self.cover_cache = None;
                let dd = (dr.pos[0] - x).hypot(dr.pos[1] - y);
                let inner_r = (dr.radius - 3.0).max(2.0);
                if dd > inner_r {
                    move_goal = Some([dr.pos[0], dr.pos[1]]);
                } else {
                    if t > self.strafe_until {
                        self.strafe_dir = if self.next_rng() < 0.5 { -1.0 } else { 1.0 };
                        self.strafe_until = t + 0.28 + self.next_rng() * 0.37;
                    }
                    let dirx = (tp[0] - x) / d.max(1.0);
                    let diry = (tp[1] - y) / d.max(1.0);
                    let sx = x - diry * self.strafe_dir * 3.5;
                    let sy = y + dirx * self.strafe_dir * 3.5;
                    if (sx - dr.pos[0]).hypot(sy - dr.pos[1]) <= inner_r {
                        move_goal = Some([sx, sy]);
                    } else {
                        move_goal = Some([dr.pos[0], dr.pos[1]]);
                    }
                }
            } else {
                self.cover_cache = None;
                let dirx = (tp[0] - x) / d.max(1.0);
                let diry = (tp[1] - y) / d.max(1.0);
                let enemy_lightning = tr.weapons.iter().any(|w| w == "lightning");

                let mut radial = 0.0;
                if d > prefer + 15.0 {
                    radial = 1.0;
                    if light_slot.is_some() && d < 90.0 && me.energy > 65.0 {
                        boost = true;
                    }
                } else if d < prefer - 15.0 {
                    if light_slot.is_none() {
                        if enemy_lightning && d < 60.0 {
                            radial = -1.35;
                            if me.energy > 30.0 {
                                boost = true;
                            }
                        } else if d < 70.0 {
                            radial = -1.05;
                            if d < 55.0 && me.energy > 50.0 {
                                boost = true;
                            }
                        } else {
                            radial = -0.75;
                        }
                    } else {
                        radial = -0.7;
                    }
                }
                if !*has_los && !seen {
                    radial = 1.0;
                }

                if t > self.strafe_until {
                    self.strafe_dir = if self.next_rng() < 0.5 { -1.0 } else { 1.0 };
                    self.strafe_until = t + 0.28 + self.next_rng() * 0.37;
                }

                if tr.charging {
                    let to_me = (y - tp[1]).atan2(x - tp[0]);
                    if angle_diff(tr.aim[0], to_me).abs() < 0.25 {
                        boost = me.energy > if light_slot.is_some() { 45.0 } else { 10.0 };
                    }
                }

                let cube_thresh = if hp_frac > 0.5 { 68.0 } else { 38.0 };
                let pick_thresh = if only_gun { 65.0 } else { 40.0 };
                if let Some(nc) = near_cube.filter(|_| near_cube_d < cube_thresh) {
                    move_goal = Some([nc[0], nc[1]]);
                } else if let Some(np) = near_pick.filter(|_| near_pick_d < pick_thresh) {
                    move_goal = Some([np[0], np[1]]);
                } else if radial > 0.0 && !*has_los {
                    move_goal = Some([tp[0], tp[1]]);
                } else {
                    // Wall-safe continuous strafe/kite vector with gentle secondary-flank drift
                    let walk_ok = |ux: f64, uy: f64| -> bool {
                        self.world.is_walkable_step(
                            me.pos,
                            x + ux * 4.0,
                            y + uy * 4.0,
                            radius + 0.15,
                            height,
                            self.cfg.step_up,
                            self.cfg.max_slope_deg,
                        )
                    };
                    let flank_w = if threats >= 2 && sec_len > 0.1 { 0.38 } else { 0.0 };
                    let make_u = |r_val: f64, s_val: f64| -> [f64; 2] {
                        let vx = dirx * r_val - diry * s_val * 0.9 + sec_repel[0] * flank_w;
                        let vy = diry * r_val + dirx * s_val * 0.9 + sec_repel[1] * flank_w;
                        let vl = vx.hypot(vy).max(1e-4);
                        [vx / vl, vy / vl]
                    };

                    let c1 = make_u(radial, self.strafe_dir);
                    if walk_ok(c1[0], c1[1]) {
                        combat_mv = Some(c1);
                    } else {
                        let c2 = make_u(radial, -self.strafe_dir);
                        if walk_ok(c2[0], c2[1]) {
                            self.strafe_dir = -self.strafe_dir;
                            self.strafe_until = t + 0.40;
                            combat_mv = Some(c2);
                        } else {
                            let l1 = [-diry * self.strafe_dir, dirx * self.strafe_dir];
                            let l2 = [diry * self.strafe_dir, -dirx * self.strafe_dir];
                            if walk_ok(l1[0], l1[1]) {
                                combat_mv = Some(l1);
                            } else if walk_ok(l2[0], l2[1]) {
                                self.strafe_dir = -self.strafe_dir;
                                self.strafe_until = t + 0.40;
                                combat_mv = Some(l2);
                            } else {
                                combat_mv = Some(c1);
                            }
                        }
                    }
                }
            }

            if zone_goal.is_some() && !me.in_zone {
                move_goal = None;
                combat_mv = None;
                using_bfs_cover = false;
            }
        }

        // 7. Active cover-healing between waves (maintains separation behind cover so regen finishes)
        let healing = move_goal.is_none()
            && combat_mv.is_none()
            && target.is_none()
            && zone_goal.is_none()
            && active_drop.is_none()
            && near_cube_d > 40.0
            && (hp_frac < 0.55 || (t < self.hide_until && hp_frac < 0.90));
        if healing {
            navigating = true;
            if let Some(last_tr) = self
                .belief
                .tracks
                .values()
                .filter(|tr| t - tr.t < 8.0)
                .min_by(|a, b| {
                    (a.pos[0] - x)
                        .hypot(a.pos[1] - y)
                        .total_cmp(&(b.pos[0] - x).hypot(b.pos[1] - y))
                })
            {
                let l_dist = (last_tr.pos[0] - x).hypot(last_tr.pos[1] - y);
                let l_eye = [last_tr.pos[0], last_tr.pos[1], last_tr.pos[2] + 8.0];
                let my_body = [x, y, z + 6.0];
                let exposed = self.world.raycast_solid(l_eye, my_body).is_none();
                // Keep moving behind cover if exposed OR if pursuer is close enough to round our corner
                if exposed || l_dist < 45.0 {
                    let heal_threats = if !active_threat_eyes.is_empty() {
                        active_threat_eyes.clone()
                    } else {
                        vec![l_eye]
                    };
                    if let Some(step_wp) =
                        self.get_bfs_cover_step(obs, &heal_threats, escape_dir)
                    {
                        move_goal = Some(step_wp);
                        using_bfs_cover = true;
                        slow = !exposed && l_dist > 30.0;
                    }
                }
            }
        } else if !is_losing {
            self.cover_cache = None;
        }

        // 8. Zone rotation / Economy-first looting / Safe cover-anchored roaming
        if move_goal.is_none() && combat_mv.is_none() && !healing {
            if let Some(zg) = zone_goal {
                navigating = true;
                high_prio_loot = true;
                move_goal = Some(zg);
                boost = !me.in_zone && me.energy > 15.0;
            } else if !in_fight {
                navigating = true;
                let mut best: Option<[f64; 2]> = None;
                let mut bs = f64::INFINITY;

                for p in self.belief.pick_mem.values() {
                    if self.want_kind(obs, &p.kind).is_none() {
                        continue;
                    }
                    let d = (p.pos[0] - x).hypot(p.pos[1] - y);
                    let s = d - weapon_pref(&p.kind) * 20.0 - if only_gun { 50.0 } else { 0.0 };
                    if s < bs {
                        bs = s;
                        best = Some([p.pos[0], p.pos[1]]);
                        high_prio_loot = true;
                    }
                }

                for c in self.belief.cube_mem.values() {
                    let s = (c.pos[0] - x).hypot(c.pos[1] - y) - 30.0;
                    if s < bs {
                        bs = s;
                        best = Some([c.pos[0], c.pos[1]]);
                        high_prio_loot = true;
                    }
                }

                let est_sprint_speed = (walk_speed * 1.25).max(14.0);
                for dr in &obs.drops {
                    let gz = self.world.terrain_height(dr.pos[0], dr.pos[1]);
                    if (dr.pos[2] - gz).abs() > 2.5 {
                        continue;
                    }
                    if dr.holder.map_or(false, |h| h != me.id) {
                        continue;
                    }
                    let d = (dr.pos[0] - x).hypot(dr.pos[1] - y);
                    if d > 260.0 || dr.lands_in > d / est_sprint_speed + 4.0 {
                        continue;
                    }
                    let zr = (dr.pos[0] - zn.next_center[0]).hypot(dr.pos[1] - zn.next_center[1]);
                    if zr > zn.next_radius && time_to_shrink < 15.0 {
                        continue;
                    }
                    let sc = d - 125.0;
                    if sc < bs {
                        bs = sc;
                        high_prio_loot = true;
                        best = Some(if d < (dr.radius - 3.0).max(2.0) {
                            [x, y]
                        } else {
                            [dr.pos[0], dr.pos[1]]
                        });
                    }
                }

                if hp_frac > 0.50 {
                    for nm in self.belief.npc_mem.values() {
                        let d = (nm.pos[0] - x).hypot(nm.pos[1] - y);
                        if d > 170.0 {
                            continue;
                        }
                        let sc = d - 20.0;
                        if sc < bs {
                            bs = sc;
                            high_prio_loot = false;
                            if d > 36.0 {
                                best = Some([nm.pos[0], nm.pos[1]]);
                            } else {
                                // Strafe gently around drone instead of standing still eating plasma
                                let dirx = (nm.pos[0] - x) / d.max(1.0);
                                let diry = (nm.pos[1] - y) / d.max(1.0);
                                best = Some([
                                    x - diry * self.strafe_dir * 6.0,
                                    y + dirx * self.strafe_dir * 6.0,
                                ]);
                            }
                        }
                    }
                }

                // Persistent ground crates (never forgotten when turning)
                if hp_frac > 0.45 {
                    for c in self.belief.crate_mem.values() {
                        let d = (c.pos[0] - x).hypot(c.pos[1] - y);
                        if d > 150.0 {
                            continue;
                        }
                        let sc = d + 5.0;
                        if sc < bs {
                            bs = sc;
                            high_prio_loot = false;
                            best = Some(if d > 30.0 {
                                [c.pos[0], c.pos[1]]
                            } else {
                                [x, y]
                            });
                        }
                    }
                }

                if let Some(bg) = best {
                    move_goal = Some(bg);
                } else if let Some(anchor) = self.nav.find_cover_anchor(
                    x,
                    y,
                    zn.next_center,
                    zn.next_radius,
                    85.0,
                ) {
                    move_goal = Some(anchor);
                } else {
                    let ang = (y - zn.next_center[1]).atan2(x - zn.next_center[0]);
                    let r = (zn.next_radius * 0.6).min((d_next - 30.0).max(20.0));
                    move_goal = Some([
                        zn.next_center[0] + ang.cos() * r,
                        zn.next_center[1] + ang.sin() * r,
                    ]);
                }
            }
        }

        // Shoot visible NPCs / crates whenever out of mech fight (even while rotating to zone!)
        if !in_fight && !healing {
            let unseen_attacker = t - self.belief.mech_hit_t < 1.5;
            if hp_frac > 0.40 && !unseen_attacker {
                let best_npc = obs
                    .mechs
                    .iter()
                    .filter(|m| {
                        m.kind == "npc"
                            && (m.pos[0] - x).hypot(m.pos[1] - y) < 110.0
                            && self
                                .world
                                .raycast_solid(muzzle, [m.pos[0], m.pos[1], m.pos[2] + 1.75])
                                .is_none()
                    })
                    .min_by(|a, b| {
                        (a.pos[0] - x)
                            .hypot(a.pos[1] - y)
                            .total_cmp(&(b.pos[0] - x).hypot(b.pos[1] - y))
                    });
                let best_crate = obs
                    .crates
                    .iter()
                    .filter(|c| {
                        (c.pos[0] - x).hypot(c.pos[1] - y) < 120.0
                            && self
                                .world
                                .raycast_solid(muzzle, [c.pos[0], c.pos[1], c.pos[2] + 1.2])
                                .is_none()
                    })
                    .min_by(|a, b| {
                        (a.pos[0] - x)
                            .hypot(a.pos[1] - y)
                            .total_cmp(&(b.pos[0] - x).hypot(b.pos[1] - y))
                    });

                let farm_tgt: Option<([f64; 3], [f64; 3], f64, bool)> = if let Some(n) = best_npc {
                    Some((n.pos, n.vel, 1.75, false))
                } else {
                    best_crate.map(|c| (c.pos, [0.0, 0.0, 0.0], 1.2, true))
                };

                if let Some((fpos, fvel, fh, is_crate)) = farm_tgt {
                    let dd = (fpos[0] - x).hypot(fpos[1] - y);
                    let g_spd = gun_slot.map_or(252.0, |(_, g)| g.stat_f64("speed", 252.0));
                    let tt = dd / g_spd.max(1.0);
                    let px = fpos[0] + fvel[0] * tt;
                    let py = fpos[1] + fvel[1] * tt;
                    aim_yaw = (py - y).atan2(px - x);
                    aim_pitch = (fpos[2] + fh - muzzle_z).atan2(dd.max(1e-4));
                    want_aim = true;

                    let pred =
                        predict_aim_after_step(me.aim, [aim_yaw, aim_pitch], turn_rate, pitch_rate);
                    let tol = 1.3_f64.atan2(dd.max(1.0));
                    if angle_diff(pred[0], aim_yaw).abs() < tol
                        && (pred[1] - aim_pitch).abs() < tol * 2.0 + 0.02
                    {
                        if let Some((gi, gw)) = gun_slot {
                            if gw.reload <= 0.0 {
                                act.fire[gi] = true;
                            }
                        }
                        if let Some((li, lw)) = light_slot {
                            let l_rng = lw.stat_f64("range", 45.0);
                            let l_eng = lw.stat_f64("energy", 3.0);
                            if dd < l_rng - 3.0 && me.energy > l_eng + 2.0 {
                                act.fire[li] = true;
                            }
                        }
                    }
                    if is_crate && dd < 25.0 && active_drop.is_none() && zone_goal.is_none() {
                        move_goal = Some([fpos[0], fpos[1]]);
                    }
                }
            }
        }

        // 9. Movement execution: direct wall-safe combat vector, BFS cover step, or instant 2 m A*
        let mut mv = [0.0_f64, 0.0_f64];
        if let Some(cmv) = combat_mv {
            mv = cmv;
        } else if let Some(mg) = move_goal {
            let gd = (mg[0] - x).hypot(mg[1] - y);
            if gd > 2.5 {
                let gz = self.world.terrain_height(mg[0], mg[1]);
                let direct = self
                    .world
                    .raycast_solid([x, y, z + 2.0], [mg[0], mg[1], gz + 2.0])
                    .is_none();
                let check_ground =
                    navigating && me.grounded && (z - self.world.terrain_height(x, y)).abs() < 2.0;
                if using_bfs_cover {
                    mv = [(mg[0] - x) / gd, (mg[1] - y) / gd];
                } else if direct
                    && gd < 60.0
                    && (!check_ground || self.nav.walk_segment(x, y, mg[0], mg[1]))
                {
                    mv = [(mg[0] - x) / gd, (mg[1] - y) / gd];
                } else {
                    let wp = self.nav.next_waypoint(obs.tick, x, y, mg, big_mech);
                    let wdx = wp[0] - x;
                    let wdy = wp[1] - y;
                    let wd = wdx.hypot(wdy);
                    if wd > 0.25 {
                        mv = [wdx / wd, wdy / wd];
                    } else {
                        mv = [(mg[0] - x) / gd, (mg[1] - y) / gd];
                    }
                }
            }
        }

        let ffa_stealth = is_ffa
            && !in_fight
            && d_next <= (zn.next_radius - 15.0).max(4.0)
            && zn.radius <= 160.0;

        if slow {
            let silent_scale = ((self.cfg.sound_step_speed - 0.35) / walk_speed).clamp(0.25, 0.42);
            mv = [mv[0] * silent_scale, mv[1] * silent_scale];
        } else if ffa_stealth {
            let silent_scale = ((self.cfg.sound_step_speed - 0.35) / walk_speed).clamp(0.25, 1.0);
            mv = [mv[0] * silent_scale, mv[1] * silent_scale];
        } else if !in_fight && !healing && me.grounded && mv[0].hypot(mv[1]) > 0.5 {
            let sprint_thresh = if is_ffa {
                72.0
            } else if high_prio_loot {
                50.0
            } else {
                62.0
            };
            if me.energy > sprint_thresh {
                boost = true;
            }
        }

        // 10. Wall-safe relative-velocity bullet, mortar, and laser dodge
        if dodge_len > 0.05 {
            let dx = dodge[0] / dodge_len;
            let dy = dodge[1] / dodge_len;
            mv = [mv[0] * 0.3 + dx * 0.95, mv[1] * 0.3 + dy * 0.95];
            let ml = mv[0].hypot(mv[1]);
            if ml > 1.0 {
                mv = [mv[0] / ml, mv[1] / ml];
            }
            if urgent_dodge || heavy_threat {
                let min_e = if heavy_threat {
                    8.0
                } else if light_slot.is_some() && target.is_some() {
                    50.0
                } else {
                    18.0
                };
                if me.energy > min_e {
                    boost = true;
                }
            }
        }

        // Keep strafing / dodging inside the active storm ring
        if me.in_zone && mv[0].hypot(mv[1]) > 0.05 {
            let nx = x + mv[0] * 4.5;
            let ny = y + mv[1] * 4.5;
            if (nx - zn.center[0]).hypot(ny - zn.center[1]) > (zn.radius - 2.5).max(3.0) {
                let ix = zn.center[0] - x;
                let iy = zn.center[1] - y;
                let il = ix.hypot(iy).max(1e-4);
                mv = [ix / il, iy / il];
            }
        }

        // 11. Stuck handling: 0.6 s jump + jet recovery
        let moved = (x - self.last_pos[0]).hypot(y - self.last_pos[1]);
        if (mv[0] != 0.0 || mv[1] != 0.0) && moved < 0.25 && me.grounded {
            self.stuck_t += 0.1;
        } else if moved > 0.6 {
            self.stuck_t = 0.0;
        }
        if self.stuck_t > 0.6 {
            self.nav.invalidate_cache();
            jump = true;
            boost = me.energy > 10.0;
            if self.stuck_t > 2.5 {
                mv = [-mv[1] * self.strafe_dir, mv[0] * self.strafe_dir];
            }
            if self.stuck_t > 4.0 {
                self.stuck_t = 0.0;
            }
        }
        if !me.grounded && self.stuck_t > 0.0 {
            boost = me.energy > 5.0;
        }
        self.last_pos = [x, y, z];

        // 12. Active idle scanning + front-armor facing toward recent pursuers
        if !want_aim {
            let recent_pursuer = self
                .belief
                .tracks
                .values()
                .filter(|tr| t - tr.t < 4.5)
                .min_by(|a, b| {
                    (a.pos[0] - x)
                        .hypot(a.pos[1] - y)
                        .total_cmp(&(b.pos[0] - x).hypot(b.pos[1] - y))
                });
            if t < 3.0 {
                aim_yaw = me.aim[0] + 1.6;
            } else if t - self.belief.mech_hit_t < 2.5 {
                aim_yaw = self.belief.mech_hit_bearing;
            } else if let Some(tr) = recent_pursuer {
                aim_yaw = (tr.pos[1] - y).atan2(tr.pos[0] - x);
            } else if let Some(tb) = self.belief.threat_bearing {
                if t - self.belief.threat_t < 3.0 {
                    aim_yaw = tb;
                } else if mv[0] != 0.0 || mv[1] != 0.0 {
                    self.scan_phase += 0.1;
                    aim_yaw = mv[1].atan2(mv[0]) + (self.scan_phase * 1.3).sin() * 0.9;
                } else {
                    self.scan_phase += 0.1;
                    aim_yaw = me.aim[0] + 0.6;
                }
            } else if mv[0] != 0.0 || mv[1] != 0.0 {
                self.scan_phase += 0.1;
                aim_yaw = mv[1].atan2(mv[0]) + (self.scan_phase * 1.3).sin() * 0.9;
            } else {
                self.scan_phase += 0.1;
                aim_yaw = me.aim[0] + 0.6;
            }
            aim_pitch = 0.0;
        }

        // Keep laser charging ONLY across brief target blinks when not handled/retreating/healing/dodging
        if !laser_handled
            && !is_losing
            && !healing
            && !heavy_threat
            && me.in_zone
            && (t - self.target_t) <= 0.6
        {
            if let Some((li, lw)) = laser_slot {
                if laser_charging && lw.charge < lw.stat_f64("charge", 0.75) - 0.05 {
                    act.fire[li] = true;
                }
            }
        }

        // Safe cover reload whenever out of fight or target is behind cover (or empty mag)
        if let Some((gi, gw)) = gun_slot {
            let mag = gw.stat_f64("mag", 12.0).max(1.0);
            let ammo = gw.ammo.unwrap_or(mag as u32) as f64;
            if gw.reload <= 0.0 && !act.fire[gi] {
                if ammo <= 0.0 || ((!in_fight || !has_los_to_target) && ammo < mag * 0.65) {
                    act.reload = true;
                }
            }
        }

        let laser_holding = laser_slot.map_or(false, |(li, _)| act.fire[li]);
        act.move_dir = mv;
        act.aim = [wrap_angle(aim_yaw), aim_pitch];
        act.boost = boost && !laser_holding;
        act.jump = jump;
        act.sanitize(self.cfg.min_pitch, self.cfg.max_pitch);

        act
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

    let mut brain = BotBrain::new(hello.seed, cfg, world);

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