use crate::belief::BeliefMap;
use crate::combat::{aim_dir, dist_2d, dist_3d, TargetInfo};
use crate::protocol::*;
use crate::world::World;

/// Enemy muzzles actively charging a laser aimed near our mech with solid LOS.
pub fn active_laser_threats(obs: &ObsMsg, cfg: &RuleConfig, world: &World) -> Vec<[f64; 3]> {
    let pos = obs.me.pos;
    let my_center = [pos[0], pos[1], pos[2] + 0.5 * obs.me.height()];
    let radius = obs.me.radius();
    let mut threats = Vec::with_capacity(4);

    for m in &obs.mechs {
        if m.kind != "mech" || !m.charging {
            continue;
        }
        let h = 9.0 * (1.0 + cfg.cube_size_per_cube * m.cubes as f64);
        let muzzle = [m.pos[0], m.pos[1], m.pos[2] + 0.78 * h];
        let d = dist_3d(muzzle, my_center);
        if d > 275.0 || world.raycast_solid(muzzle, my_center).is_some() {
            continue;
        }
        let dir = aim_dir(m.aim);
        let rx = my_center[0] - muzzle[0];
        let ry = my_center[1] - muzzle[1];
        let rz = my_center[2] - muzzle[2];
        let proj = rx * dir[0] + ry * dir[1] + rz * dir[2];
        if proj <= 0.0 {
            continue;
        }
        let closest = [
            muzzle[0] + dir[0] * proj,
            muzzle[1] + dir[1] * proj,
            muzzle[2] + dir[2] * proj,
        ];
        if dist_3d(my_center, closest) <= radius + 9.0 {
            threats.push(muzzle);
        }
    }
    threats
}

/// Recently-seen or visible secondary enemy mechs (excluding `primary_id`) within 120 m,
/// used to bias 1v1 strafing so buildings block flanks without abandoning the fight.
pub fn secondary_threat_eyes(
    obs: &ObsMsg,
    cfg: &RuleConfig,
    belief: &BeliefMap,
    primary_id: u64,
) -> Vec<[f64; 3]> {
    let pos = obs.me.pos;
    let mut items: Vec<(f64, [f64; 3])> = belief
        .tracked
        .iter()
        .filter(|m| {
            m.id as u64 != primary_id && (m.visible_now || (obs.t - m.last_seen_t) <= 3.0)
        })
        .map(|m| {
            let d2 = (m.pos[0] - pos[0]).powi(2) + (m.pos[1] - pos[1]).powi(2);
            let h = 9.0 * (1.0 + cfg.cube_size_per_cube * m.cubes as f64);
            (d2, [m.pos[0], m.pos[1], m.pos[2] + 0.78 * h])
        })
        .filter(|(d2, _)| *d2 <= 120.0 * 120.0)
        .collect();
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    items.truncate(3);
    items.into_iter().map(|(_, p)| p).collect()
}

/// True when we are ALREADY out of LOS and close to / actively regenerating HP,
/// so we shouldn't rush out into the open to hunt until healed (~78% HP).
#[inline]
pub fn should_hold_regen(obs: &ObsMsg, los_enemies: usize, last_combat_t: f64) -> bool {
    let hp_frac = obs.me.hp / obs.me.max_hp.max(1.0);
    hp_frac < 0.78 && obs.me.in_zone && los_enemies == 0 && (obs.t - last_combat_t) >= 2.8
}

/// Fallback target from `belief.tracked` when no visible target is available:
/// - Shoots through tree canopy (`raycast_solid == None`) for <= 1.2 s after losing sight
/// - Lobs Mortar over solid cover for <= 2.8 s after losing sight to deny 5 s regen
pub fn find_tracked_blind_target(
    obs: &ObsMsg,
    cfg: &RuleConfig,
    world: &World,
    belief: &BeliefMap,
) -> Option<TargetInfo> {
    let muzzle = [
        obs.me.pos[0],
        obs.me.pos[1],
        obs.me.pos[2] + obs.me.muzzle_height(),
    ];
    let has_mortar = obs
        .me
        .weapons
        .iter()
        .any(|w| w.as_ref().map_or(false, |s| s.kind == "mortar"));

    let mut best: Option<TargetInfo> = None;

    for tm in &belief.tracked {
        if tm.visible_now {
            continue;
        }
        let age = (obs.t - tm.last_seen_t).max(0.0);
        if age > 2.8 {
            continue;
        }

        let h = 9.0 * (1.0 + cfg.cube_size_per_cube * tm.cubes as f64);
        let base_mid = [tm.pos[0], tm.pos[1], tm.pos[2] + 0.52 * h];
        let ext_t = age.min(0.85);
        let raw_mid = [
            base_mid[0] + tm.vel[0] * ext_t,
            base_mid[1] + tm.vel[1] * ext_t,
            base_mid[2],
        ];
        let pred_mid = if let Some(ht) = world.raycast_solid(base_mid, raw_mid) {
            let f = (ht - 0.1).max(0.0);
            [
                base_mid[0] + (raw_mid[0] - base_mid[0]) * f,
                base_mid[1] + (raw_mid[1] - base_mid[1]) * f,
                base_mid[2],
            ]
        } else {
            raw_mid
        };

        let d = dist_3d(muzzle, pred_mid);
        if d > 195.0 {
            continue;
        }

        let solid_clear = world.raycast_solid(muzzle, pred_mid).is_none();
        // Shoot through foliage only when track is very fresh; mortar over walls up to 2.8 s
        let allowed = (solid_clear && age <= 1.2 && dist_2d(obs.me.pos, pred_mid) <= 165.0)
            || (!solid_clear && has_mortar && age <= 2.8 && d >= 14.0);
        if !allowed {
            continue;
        }

        let feet_z = world.terrain_height(pred_mid[0], pred_mid[1]).max(tm.pos[2] - 1.5);
        let score = 900.0 - d * 2.5 - age * 140.0 + (1400.0 - tm.hp).max(0.0) * 0.5;
        if best.as_ref().map_or(true, |b| score > b.score) {
            best = Some(TargetInfo {
                id: tm.id as u64,
                pos: pred_mid,
                feet_z,
                vel: tm.vel,
                dist: d,
                hp: tm.hp,
                los: solid_clear,
                is_mech: true,
                is_crate: false,
                has_mortar: false,
                has_lightning: false,
                enemy_charging: false,
                score,
            });
        }
    }

    best
}