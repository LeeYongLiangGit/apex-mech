mod belief;
mod combat;
mod nav;
mod protocol;
mod tactics;
mod world;

use belief::{BeliefMap, MatchMode};
use combat::*;
use nav::NavEngine;
use protocol::*;
use std::f64::consts::PI;
use std::io::{self, BufWriter, Write};
use world::World;

struct BotBrain {
    cfg: RuleConfig,
    world: World,
    nav: NavEngine,
    belief: BeliefMap,
    last_aim: [f64; 2],
    last_target_id: Option<u64>,
    laser_last_los_t: f64,
    strafe_sign: f64,
    last_strafe_flip: u64,
    climb_stuck_ticks: u32,
    last_pos: [f64; 3],
    stuck_ticks: u32,
    escape_cache: Option<([f64; 3], u64, usize)>,
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
            laser_last_los_t: -10.0,
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

        let (dodge, urgent_dodge) = compute_dodge(obs, &self.cfg, &self.world);
        let los_enemies = count_los_enemy_mechs(obs, &self.cfg, &self.world);
        self.handle_weapon_pickup(obs, los_enemies, &mut act);

        let visible_target = select_target(
            obs,
            &self.cfg,
            &self.world,
            self.belief.mode,
            self.last_target_id,
        );
        self.last_target_id = visible_target.as_ref().filter(|t| t.is_mech).map(|t| t.id);

        // Fall back to through-foliage / over-wall mortar suppression on recently tracked mechs
        let combat_target = visible_target.clone().or_else(|| {
            tactics::find_tracked_blind_target(obs, &self.cfg, &self.world, &self.belief)
        });

        let hp_frac = obs.me.hp / obs.me.max_hp.max(1.0);
        let seek_heal = hp_frac < 0.48 && obs.me.in_zone;
        let hold_regen = tactics::should_hold_regen(obs, los_enemies, self.belief.last_combat_t);
        let ceasefire_for_regen = hp_frac < 0.80
            && obs.me.in_zone
            && los_enemies == 0
            && (obs.t - self.belief.last_combat_t) >= 1.2;

        let fallback_bearing = self.belief.best_unseen_bearing(obs.me.pos, obs.t);
        let mut peek_point = None;
        decide_combat(
            obs,
            &self.cfg,
            &self.world,
            self.belief.mode,
            los_enemies,
            ceasefire_for_regen,
            urgent_dodge,
            combat_target.as_ref(),
            fallback_bearing,
            self.last_aim,
            &mut self.laser_last_los_t,
            &mut peek_point,
            &mut act,
        );

        self.decide_movement(
            obs,
            visible_target.as_ref(),
            los_enemies,
            seek_heal,
            hold_regen,
            peek_point,
            dodge,
            urgent_dodge,
            &mut act,
        );

        act.sanitize(self.cfg.min_pitch, self.cfg.max_pitch);
        self.last_aim = act.aim;
        act
    }

    fn pick_upgrade(&self, offer: &Offer, obs: &ObsMsg) -> usize {
        let held_slot = |k: &str| -> Option<&WeaponSlot> {
            obs.me.weapons.iter().flatten().find(|w| w.kind == k)
        };
        let held_laser = held_slot("laser").is_some();
        let held_lightning = held_slot("lightning").is_some();
        let held_mortar = held_slot("mortar").is_some();
        let is_ffa = self.belief.mode == MatchMode::FfaSurvival;

        let apply_effect = |mut val: f64, eff: &UpgradeEffect| -> f64 {
            if let Some(m) = eff.mul {
                val *= m;
            }
            if let Some(a) = eff.add {
                val += a;
            }
            val
        };

        let score_choice = |c: &OfferChoice| -> f64 {
            let Some(def) = self.cfg.upgrades.iter().find(|u| u.id == c.id) else {
                let id = c.id.to_lowercase();
                let desc = c.desc.to_lowercase();
                let mut s = 30.0;
                if id.contains("armor") || desc.contains("damage taken") {
                    s += 120.0;
                }
                if id.contains("hp")
                    || id.contains("plating")
                    || desc.contains("max hp")
                    || desc.contains("regen")
                {
                    s += 115.0;
                }
                if id.contains("lightning") && held_lightning {
                    s += 110.0;
                }
                if id.contains("laser") && held_laser {
                    s += 105.0;
                }
                if id.contains("mortar") && held_mortar {
                    s += 85.0;
                }
                return s;
            };

            let base_hp = obs.me.stat_f64("hp", obs.me.max_hp.max(1400.0));
            let base_armor = obs.me.stat_f64("armor", 0.0).clamp(0.0, 0.85);
            let base_regen = obs.me.stat_f64("regen_per_s", 45.0).max(1.0);
            let base_delay = obs.me.stat_f64("regen_delay", self.cfg.regen_delay).max(0.5);
            let base_walk = obs.me.stat_f64("walk_speed", 11.7).max(1.0);
            let base_boost = obs.me.stat_f64("boost_speed", 19.5).max(1.0);
            let base_energy = obs.me.stat_f64("energy", obs.me.max_energy.max(100.0)).max(1.0);
            let base_eregen = obs.me.stat_f64("energy_regen", 18.0).max(1.0);
            let base_bdrain = obs.me.stat_f64("boost_drain", 17.0).max(1.0);
            let base_jdrain = obs.me.stat_f64("jet_drain", 15.0).max(1.0);
            let base_turn = obs.me.stat_f64("turn_rate", 4.0).max(0.1);
            let base_pitch = obs.me.stat_f64("pitch_rate", 3.0).max(0.1);
            let base_near = obs.me.stat_f64("near_range_bonus", 0.0);
            let base_vis = obs.me.stat_f64("vision_bonus", 0.0);
            let base_hear = obs.me.stat_f64("hearing", 1.0).max(0.05);

            let mut new_hp = base_hp;
            let mut new_armor = base_armor;
            let mut new_regen = base_regen;
            let mut new_delay = base_delay;
            let mut new_walk = base_walk;
            let mut new_boost = base_boost;
            let mut new_energy = base_energy;
            let mut new_eregen = base_eregen;
            let mut new_bdrain = base_bdrain;
            let mut new_jdrain = base_jdrain;
            let mut new_turn = base_turn;
            let mut new_pitch = base_pitch;
            let mut new_near = base_near;
            let mut new_vis = base_vis;
            let mut new_hear = base_hear;

            let mut s = 0.0;

            for eff in &def.effects {
                match eff.path.as_str() {
                    "mech.hp" => new_hp = apply_effect(new_hp, eff),
                    "mech.armor" => new_armor = apply_effect(new_armor, eff).clamp(0.0, 0.85),
                    "mech.regen_per_s" => new_regen = apply_effect(new_regen, eff),
                    "mech.regen_delay" => new_delay = apply_effect(new_delay, eff).max(0.5),
                    "mech.walk_speed" => new_walk = apply_effect(new_walk, eff),
                    "mech.boost_speed" => new_boost = apply_effect(new_boost, eff),
                    "mech.energy" => new_energy = apply_effect(new_energy, eff),
                    "mech.energy_regen" => new_eregen = apply_effect(new_eregen, eff),
                    "mech.boost_drain" => new_bdrain = apply_effect(new_bdrain, eff).max(0.5),
                    "mech.jet_drain" => new_jdrain = apply_effect(new_jdrain, eff).max(0.5),
                    "mech.turn_rate" => new_turn = apply_effect(new_turn, eff),
                    "mech.pitch_rate" => new_pitch = apply_effect(new_pitch, eff),
                    "mech.near_range_bonus" => new_near = apply_effect(new_near, eff),
                    "mech.vision_bonus" => new_vis = apply_effect(new_vis, eff),
                    "mech.hearing" => new_hear = apply_effect(new_hear, eff).max(0.05),
                    _ => {}
                }
            }

            // 1. EHP & armor regen-amplification delta
            let cur_ehp = base_hp / (1.0 - base_armor).max(0.15);
            let new_ehp = new_hp / (1.0 - new_armor).max(0.15);
            let ehp_rel = (new_ehp - cur_ehp) / cur_ehp;
            let armor_rel = (1.0 - base_armor) / (1.0 - new_armor).max(0.15) - 1.0;
            s += ehp_rel * 860.0 + armor_rel * 550.0;

            // 2. Out-of-combat regen delta
            let regen_rel = (new_regen - base_regen) / base_regen;
            let delay_red = (base_delay - new_delay).max(0.0) / base_delay;
            s += regen_rel * 130.0 + delay_red * 145.0;

            // 3. Mobility, energy & torso slew
            s += ((new_walk - base_walk) / base_walk) * 620.0;
            s += ((new_boost - base_boost) / base_boost) * 480.0;
            s += ((new_energy - base_energy) / base_energy)
                * if held_lightning { 270.0 } else { 210.0 };
            s += ((new_eregen - base_eregen) / base_eregen)
                * if held_lightning { 310.0 } else { 240.0 };
            s += ((base_bdrain - new_bdrain) / base_bdrain) * 190.0;
            s += ((base_jdrain - new_jdrain) / base_jdrain) * 120.0;
            s += ((new_turn - base_turn) / base_turn) * 150.0;
            s += ((new_pitch - base_pitch) / base_pitch) * 80.0;

            // 4. Perception / sensors
            let sensor_mult = if is_ffa { 1.45 } else { 0.85 };
            s += ((new_near - base_near) * 2.8
                + (new_vis - base_vis) * 1.1
                + ((base_hear - new_hear) / base_hear) * 35.0)
                * sensor_mult;

            // 5. Weapon DPS & utility deltas against live weapon slot stats
            if let Some(w) = held_slot("lightning") {
                let b_dmg = w.stat_f64("damage", 60.0);
                let b_cd = w.stat_f64("cooldown", 0.25).max(0.05);
                let b_rng = w.stat_f64("range", 45.0).max(1.0);
                let b_cone = w.stat_f64("cone_deg", 30.0).max(1.0);
                let b_eng = w.stat_f64("energy", 3.0).max(0.1);
                let b_chain = w.stat_f64("chain", 2.0);
                let (mut a_dmg, mut a_cd, mut a_rng, mut a_cone, mut a_eng, mut a_chain) =
                    (b_dmg, b_cd, b_rng, b_cone, b_eng, b_chain);
                for eff in &def.effects {
                    match eff.path.as_str() {
                        "weapons.lightning.damage" => a_dmg = apply_effect(a_dmg, eff),
                        "weapons.lightning.cooldown" => a_cd = apply_effect(a_cd, eff).max(0.05),
                        "weapons.lightning.range" => a_rng = apply_effect(a_rng, eff),
                        "weapons.lightning.cone_deg" => a_cone = apply_effect(a_cone, eff),
                        "weapons.lightning.energy" => a_eng = apply_effect(a_eng, eff).max(0.1),
                        "weapons.lightning.chain" => a_chain = apply_effect(a_chain, eff),
                        _ => {}
                    }
                }
                let dps_rel = ((a_dmg / a_cd) - (b_dmg / b_cd)) / (b_dmg / b_cd);
                s += dps_rel * 480.0
                    + ((a_rng - b_rng) / b_rng) * 520.0
                    + ((a_cone - b_cone) / b_cone) * 130.0
                    + ((b_eng - a_eng) / b_eng) * 210.0
                    + (a_chain - b_chain) * if is_ffa { 42.0 } else { 92.0 };
            }

            if let Some(w) = held_slot("laser") {
                let b_dmg = w.stat_f64("damage", 315.0);
                let b_chg = w.stat_f64("charge", 0.75).max(0.05);
                let b_cd = w.stat_f64("cooldown", 1.65).max(0.05);
                let b_pierce = w.stat_f64("pierce", 0.0);
                let (mut a_dmg, mut a_chg, mut a_cd, mut a_pierce) =
                    (b_dmg, b_chg, b_cd, b_pierce);
                for eff in &def.effects {
                    match eff.path.as_str() {
                        "weapons.laser.damage" => a_dmg = apply_effect(a_dmg, eff),
                        "weapons.laser.charge" => a_chg = apply_effect(a_chg, eff).max(0.05),
                        "weapons.laser.cooldown" => a_cd = apply_effect(a_cd, eff).max(0.05),
                        "weapons.laser.pierce" => a_pierce = apply_effect(a_pierce, eff),
                        _ => {}
                    }
                }
                let dps_b = b_dmg / (b_chg + b_cd);
                let dps_a = a_dmg / (a_chg + a_cd);
                s += ((dps_a - dps_b) / dps_b) * 640.0
                    + ((b_chg - a_chg) / b_chg) * 260.0
                    + (a_pierce - b_pierce) * if is_ffa { 55.0 } else { 85.0 };
            }

            if let Some(w) = held_slot("mortar") {
                let b_dmg = w.stat_f64("damage", 200.0);
                let b_spl = w.stat_f64("splash", 160.0);
                let b_cd = w.stat_f64("cooldown", 2.6).max(0.1);
                let b_rad = w.stat_f64("splash_radius", 7.0).max(1.0);
                let b_spd = w.stat_f64("speed", self.cfg.mortar_speed).max(1.0);
                let (mut a_dmg, mut a_spl, mut a_cd, mut a_rad, mut a_spd) =
                    (b_dmg, b_spl, b_cd, b_rad, b_spd);
                for eff in &def.effects {
                    match eff.path.as_str() {
                        "weapons.mortar.damage" => a_dmg = apply_effect(a_dmg, eff),
                        "weapons.mortar.splash" => a_spl = apply_effect(a_spl, eff),
                        "weapons.mortar.cooldown" => a_cd = apply_effect(a_cd, eff).max(0.1),
                        "weapons.mortar.splash_radius" => a_rad = apply_effect(a_rad, eff),
                        "weapons.mortar.speed" => a_spd = apply_effect(a_spd, eff),
                        _ => {}
                    }
                }
                let dps_b = (b_dmg + 0.5 * b_spl) / b_cd;
                let dps_a = (a_dmg + 0.5 * a_spl) / a_cd;
                s += ((dps_a - dps_b) / dps_b) * 460.0
                    + ((a_rad - b_rad) / b_rad) * 300.0
                    + ((a_spd - b_spd) / b_spd) * 380.0;
            }

            if let Some(w) = held_slot("gun") {
                let b_dmg = w.stat_f64("damage", 45.0);
                let b_cd = w.stat_f64("cooldown", 0.24).max(0.05);
                let b_mag = w.stat_f64("mag", 12.0).max(1.0);
                let b_rel = w.stat_f64("reload", 2.4).max(0.1);
                let b_spd = w.stat_f64("speed", 252.0).max(1.0);
                let b_rng = w.stat_f64("range", 230.0).max(1.0);
                let (mut a_dmg, mut a_cd, mut a_mag, mut a_rel, mut a_spd, mut a_rng) =
                    (b_dmg, b_cd, b_mag, b_rel, b_spd, b_rng);
                for eff in &def.effects {
                    match eff.path.as_str() {
                        "weapons.gun.damage" => a_dmg = apply_effect(a_dmg, eff),
                        "weapons.gun.cooldown" => a_cd = apply_effect(a_cd, eff).max(0.05),
                        "weapons.gun.mag" => a_mag = apply_effect(a_mag, eff).max(1.0),
                        "weapons.gun.reload" => a_rel = apply_effect(a_rel, eff).max(0.1),
                        "weapons.gun.speed" => a_spd = apply_effect(a_spd, eff),
                        "weapons.gun.range" => a_rng = apply_effect(a_rng, eff),
                        _ => {}
                    }
                }
                let burst_b = b_dmg / b_cd;
                let burst_a = a_dmg / a_cd;
                let sust_b = (b_mag * b_dmg) / (b_mag * b_cd + b_rel);
                let sust_a = (a_mag * a_dmg) / (a_mag * a_cd + a_rel);
                let dps_rel =
                    0.55 * ((burst_a - burst_b) / burst_b) + 0.45 * ((sust_a - sust_b) / sust_b);
                let gun_mult = if held_lightning || held_laser { 0.42 } else { 0.72 };
                s += (dps_rel * 580.0
                    + ((a_spd - b_spd) / b_spd) * 240.0
                    + ((a_rng - b_rng) / b_rng) * 100.0)
                    * gun_mult;
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
                if obs.zone.dps >= 100.0 || obs.zone.radius <= 140.0 {
                    98
                } else {
                    84
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

    fn handle_weapon_pickup(&self, obs: &ObsMsg, los_enemies: usize, act: &mut Action) {
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
            } else if los_enemies == 0 {
                // Never trigger a 1.0–1.65 s equip lockout swap while under LOS fire
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
        let d = ((x - obs.zone.next_center[0]).powi(2) + (y - obs.zone.next_center[1]).powi(2))
            .sqrt();
        d <= (obs.zone.next_radius - margin).max(4.0)
    }

    fn choose_macro_goal(
        &mut self,
        obs: &ObsMsg,
        target: Option<&TargetInfo>,
        visible_threat_eyes: &[[f64; 3]],
        los_enemies: usize,
        seek_heal: bool,
        hold_regen: bool,
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

        // 2. Ground-only building-corner escape ONLY when critically low HP (<48%) under LOS fire
        if seek_heal && los_enemies > 0 && !visible_threat_eyes.is_empty() {
            let safe_z_r = (obs.zone.radius - 5.0).max(5.0);
            let world = &self.world;
            let eye_off = obs.me.eye_height();
            let cover_los = |pt: [f64; 3]| -> usize {
                let pt_eye = [pt[0], pt[1], pt[2] + eye_off];
                visible_threat_eyes
                    .iter()
                    .filter(|&&te| world.raycast_solid(te, pt_eye).is_none())
                    .count()
            };

            let reuse = self.escape_cache.filter(|&(pt, tick, orig_los)| {
                if obs.tick.saturating_sub(tick) >= 6 || dist_2d(pos, pt) <= 6.0 {
                    return false;
                }
                let d_zone = ((pt[0] - obs.zone.center[0]).powi(2)
                    + (pt[1] - obs.zone.center[1]).powi(2))
                .sqrt();
                d_zone <= safe_z_r && cover_los(pt) <= orig_los
            });
            if let Some((pt, _, _)) = reuse {
                return pt;
            }
            self.escape_cache = None;
            let esc_dir = self.belief.escape_vector(pos, obs.t);
            if let Some(cover_pt) = self.nav.find_escape_cover(
                pos,
                eye_off,
                visible_threat_eyes,
                esc_dir,
                true,
                false, // Never jet-climb mid-combat
                &self.world,
                obs.zone.center,
                obs.zone.radius,
                obs.zone.next_center,
                obs.zone.next_radius,
            ) {
                let los_now = cover_los(cover_pt);
                self.escape_cache = Some((cover_pt, obs.tick, los_now));
                return cover_pt;
            }
        } else {
            self.escape_cache = None;
        }

        // 3. Ground-level high-value weapon pickups (lightning / laser / mortar into empty slot)
        let next_hits_upgrade = self.cfg.upgrade_thresholds.contains(&(obs.me.cubes + 1));
        let empty_slot = obs.me.weapons.iter().any(|w| w.is_none());
        let best_pickup = obs
            .pickups
            .iter()
            .filter(|p| {
                if !empty_slot && los_enemies > 0 {
                    return false;
                }
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
                    prio >= 80
                } else {
                    obs.me.weapons.iter().enumerate().take(2).any(|(idx, slot)| {
                        let Some(w) = slot else { return false };
                        let other_kind = obs
                            .me
                            .weapons
                            .get(1 - idx)
                            .and_then(|s| s.as_ref())
                            .map(|s| s.kind.as_str())
                            .unwrap_or("");
                        let pair_ok =
                            other_kind.is_empty() || Self::valid_loadout_pair(&p.kind, other_kind);
                        pair_ok && prio > self.weapon_priority(&w.kind, obs) + 12
                    })
                }
            })
            .min_by(|a, b| {
                let sa = dist_2d(pos, a.pos) - self.weapon_priority(&a.kind, obs) as f64 * 0.6;
                let sb = dist_2d(pos, b.pos) - self.weapon_priority(&b.kind, obs) as f64 * 0.6;
                sa.total_cmp(&sb)
            });

        if let Some(p) = best_pickup {
            let max_r = if empty_slot || next_hits_upgrade { 95.0 } else { 55.0 };
            if dist_2d(pos, p.pos) < max_r {
                return p.pos;
            }
        }

        // 4. Supply drops (exact Iter 8 logic: skip only if contested by >=2 LOS mechs)
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
        let next_threshold_close = self
            .cfg
            .upgrade_thresholds
            .iter()
            .copied()
            .find(|&th| th > obs.me.cubes)
            .map_or(false, |th| th - obs.me.cubes <= 2 && th <= 21);
        let want_cubes = obs.me.cubes < self.cfg.cube_stat_cap || next_threshold_close;
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

        // 6. Mode-specific mid/late objective (skip blind hunting while actively finishing regen)
        if self.belief.mode == MatchMode::PracticeHivemind
            && obs.me.hp / obs.me.max_hp.max(1.0) > 0.62
            && !hold_regen
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

        // 7. Anchor near a building edge inside next_zone instead of the open Oval center
        if let Some(anchor) = self.nav.find_cover_anchor(
            pos,
            obs.zone.next_center,
            obs.zone.next_radius,
            90.0,
        ) {
            return anchor;
        }

        next_z
    }

    fn decide_movement(
        &mut self,
        obs: &ObsMsg,
        target: Option<&TargetInfo>,
        los_enemies: usize,
        seek_heal: bool,
        hold_regen: bool,
        peek_point: Option<[f64; 3]>,
        dodge: [f64; 2],
        urgent_dodge: bool,
        act: &mut Action,
    ) {
        let pos = obs.me.pos;
        let radius = obs.me.radius();
        let height = obs.me.height();
        let my_eye_off = obs.me.eye_height();
        let on_roof = pos[2] > self.world.terrain_height(pos[0], pos[1]) + 2.5;

        let find_w = |k: &str| obs.me.weapons.iter().flatten().find(|w| w.kind == k);
        let lightning_w = find_w("lightning");
        let laser_w = find_w("laser");
        let mortar_w = find_w("mortar");
        let gun_w = find_w("gun");

        let has_lightning = lightning_w.is_some();
        let has_laser = laser_w.is_some();
        let laser_charging = laser_w.map_or(false, |w| w.charge > 0.0);

        if obs.tick.saturating_sub(self.last_strafe_flip) >= 18 {
            self.strafe_sign = -self.strafe_sign;
            self.last_strafe_flip = obs.tick;
        }

        // Strictly currently-visible enemy mechs within 155 m (top 4 closest) for combat steering
        let mut vis_enemies: Vec<(f64, [f64; 3])> = obs
            .mechs
            .iter()
            .filter(|m| m.kind == "mech")
            .map(|m| {
                let d2 = (m.pos[0] - pos[0]).powi(2) + (m.pos[1] - pos[1]).powi(2);
                let h = 9.0 * (1.0 + self.cfg.cube_size_per_cube * m.cubes as f64);
                (d2, [m.pos[0], m.pos[1], m.pos[2] + 0.78 * h])
            })
            .filter(|(d2, _)| *d2 <= 155.0 * 155.0)
            .collect();
        vis_enemies.sort_by(|a, b| a.0.total_cmp(&b.0));
        vis_enemies.truncate(4);
        let visible_threat_eyes: Vec<[f64; 3]> = vis_enemies.into_iter().map(|(_, p)| p).collect();

        let goal = self.choose_macro_goal(
            obs,
            target,
            &visible_threat_eyes,
            los_enemies,
            seek_heal,
            hold_regen,
        );
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

        let laser_threats = tactics::active_laser_threats(obs, &self.cfg, &self.world);
        let secondary_eyes = if !seek_heal && los_enemies == 1 {
            primary_mech
                .map(|pm| tactics::secondary_threat_eyes(obs, &self.cfg, &self.belief, pm.id))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Corner peek-out unit vector when pre-charging laser behind cover
        let peek_u = peek_point.and_then(|pk| {
            let dx = pk[0] - pos[0];
            let dy = pk[1] - pos[1];
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.5 { Some([dx / len, dy / len]) } else { None }
        });

        // Duck briefly behind cover during zero-DPS burst cooldowns
        let lightning_active = lightning_w.map_or(false, |w| {
            let rng = w.stat_f64("range", 45.0);
            obs.me.energy >= 7.0 && primary_mech.map_or(false, |pm| pm.los && pm.dist <= rng + 4.0)
        });
        let gun_active = gun_w.map_or(false, |w| w.reload <= 0.25 && w.ammo.unwrap_or(1) > 0);
        let burst_on_cd = laser_w.map_or(false, |w| w.cooldown > 0.35 && w.charge <= 0.0)
            || mortar_w.map_or(false, |w| w.cooldown > 0.5);
        let duck_on_cooldown = !laser_charging
            && los_enemies >= 1
            && burst_on_cd
            && !lightning_active
            && !gun_active
            && !seek_heal;

        // Stay at the edge of lightning range (37 m) for max bullet-dodging space;
        // only hold laser distance against mortar bots if we actually carry a laser
        let ideal_dist = if let Some(pm) = primary_mech {
            if seek_heal || los_enemies >= 2 {
                90.0
            } else if has_lightning
                && obs.me.energy >= 20.0
                && (!pm.has_mortar || !has_laser || pm.dist <= 48.0)
            {
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
                self.cfg.max_slope_deg,
            );
            let ok_far = ok_near
                && self.world.is_walkable_step(
                    pos,
                    pos[0] + ux * 7.0,
                    pos[1] + uy * 7.0,
                    radius + 0.2,
                    height,
                    self.cfg.step_up,
                    self.cfg.max_slope_deg,
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
            let d_cur_z = ((lx - obs.zone.center[0]).powi(2) + (ly - obs.zone.center[1]).powi(2))
                .sqrt();
            if d_cur_z > obs.zone.radius - 4.0 {
                score -= 220.0;
            } else if !obs.me.in_zone {
                let d_next_z = ((lx - obs.zone.next_center[0]).powi(2)
                    + (ly - obs.zone.next_center[1]).powi(2))
                .sqrt();
                score -= d_next_z * 2.5;
            }

            if dodge_len > 0.05 {
                score += (ux * dodge_u[0] + uy * dodge_u[1])
                    * if urgent_dodge { 155.0 } else { 78.0 };
            }

            if let Some(pu) = peek_u {
                score += (ux * pu[0] + uy * pu[1]) * 95.0;
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

            if (in_combat || !laser_threats.is_empty()) && ok_near {
                let cand_z = if on_roof {
                    pos[2]
                } else {
                    self.world.terrain_height(lx, ly)
                };
                let cand_eye = [lx, ly, cand_z + my_eye_off];

                // Step behind solid geometry to blank incoming enemy laser charges
                if !laser_threats.is_empty() {
                    let blanked = laser_threats
                        .iter()
                        .filter(|&&lm| self.world.raycast_solid(lm, cand_eye).is_some())
                        .count();
                    if blanked > 0 {
                        score += (blanked as f64) * 110.0;
                    }
                }

                if in_combat {
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

                        // Soft corner slicing: bias strafe side to keep flankers behind walls
                        if !secondary_eyes.is_empty() && !duck_on_cooldown {
                            let flank_los = secondary_eyes
                                .iter()
                                .filter(|&&se| self.world.raycast_solid(se, cand_eye).is_none())
                                .count();
                            if flank_los == 0 && future_los == 1 {
                                score += 28.0;
                            } else if flank_los > 0 {
                                score -= (flank_los as f64) * 20.0;
                            }
                        }

                        if duck_on_cooldown {
                            if future_los == 0 {
                                score += 75.0;
                            }
                            if self.nav.is_near_cover(lx, ly) {
                                score += 18.0;
                            }
                        } else if future_los == 1 {
                            score += 20.0;
                        }
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
            self.cfg.max_slope_deg,
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

        let min_boost_energy = if urgent_dodge || !obs.me.in_zone {
            6.0
        } else if has_lightning {
            38.0
        } else {
            20.0
        };
        let early_loot_sprint =
            !in_combat && obs.me.cubes < self.cfg.cube_stat_cap && obs.me.energy > 72.0;
        let need_sprint = urgent_dodge
            || !obs.me.in_zone
            || (seek_heal && los_enemies > 0)
            || early_loot_sprint;
        act.boost = need_sprint
            && !laser_charging
            && obs.me.grounded
            && obs.me.energy >= min_boost_energy;
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