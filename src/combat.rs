use crate::protocol::*;
use crate::world::World;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMode {
    Unknown,
    PracticeHivemind,
    FfaSurvival,
}

#[derive(Debug, Clone)]
pub struct TargetInfo {
    pub id: u64,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub yaw: f64,
    pub dist: f64,
    pub hp: f64,
    pub los: bool,
    pub is_mech: bool,
    pub is_crate: bool,
    pub has_mortar: bool,
    pub has_lightning: bool,
    pub enemy_charging: bool,
    pub score: f64,
}

pub struct MortarSolution {
    pub aim: [f64; 2],
    pub high_arc: bool,
}

#[inline]
pub fn dist_2d(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

#[inline]
pub fn dist_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[inline]
pub fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = (b - a) % (2.0 * PI);
    if d > PI {
        d -= 2.0 * PI;
    } else if d < -PI {
        d += 2.0 * PI;
    }
    d
}

pub fn aim_to(from: [f64; 3], to: [f64; 3]) -> [f64; 2] {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    let horiz = (dx * dx + dy * dy).sqrt().max(1e-6);
    [dy.atan2(dx), dz.atan2(horiz)]
}

pub fn aim_dir(aim: [f64; 2]) -> [f64; 3] {
    let cp = aim[1].cos();
    [cp * aim[0].cos(), cp * aim[0].sin(), aim[1].sin()]
}

/// 3D gun lead solver with velocity damping if target is running into a wall.
pub fn lead_aim(
    from: [f64; 3],
    target_pos: [f64; 3],
    target_vel: [f64; 3],
    proj_speed: f64,
    world: &World,
) -> [f64; 2] {
    let mut t = (dist_3d(from, target_pos) / proj_speed.max(1.0)).clamp(0.0, 1.2);
    let mut pred = target_pos;
    for _ in 0..3 {
        let cand = [
            target_pos[0] + target_vel[0] * t,
            target_pos[1] + target_vel[1] * t,
            target_pos[2] + target_vel[2] * t,
        ];
        // If predicted path goes into a wall, clamp lead to the wall hit point
        pred = if let Some(hit_t) = world.raycast_solid(target_pos, cand) {
            let f = (hit_t - 0.1).max(0.0);
            [
                target_pos[0] + (cand[0] - target_pos[0]) * f,
                target_pos[1] + (cand[1] - target_pos[1]) * f,
                target_pos[2] + (cand[2] - target_pos[2]) * f,
            ]
        } else {
            cand
        };
        t = (dist_3d(from, pred) / proj_speed.max(1.0)).clamp(0.0, 1.2);
    }
    aim_to(from, pred)
}

/// Dual-arc parabolic mortar solver (tries fast low arc, then high lob over buildings).
pub fn solve_mortar(
    obs: &ObsMsg,
    cfg: &RuleConfig,
    world: &World,
    target_pos: [f64; 3],
    target_vel: [f64; 3],
    speed: f64,
) -> Option<MortarSolution> {
    let g = cfg.mortar_gravity;
    let tube_z = obs.me.pos[2] + obs.me.height() + 0.5;
    let tube_base = [obs.me.pos[0], obs.me.pos[1], tube_z];
    let barrel_off = obs.me.radius() + 0.5;

    for high_arc in [false, true] {
        let mut pred = target_pos;
        let mut valid_pitch = None;
        let mut flight_t = 0.0;

        for _ in 0..3 {
            let dx = pred[0] - tube_base[0];
            let dy = pred[1] - tube_base[1];
            let horiz = (dx * dx + dy * dy).sqrt();
            if !(11.0..=230.0).contains(&horiz) {
                valid_pitch = None;
                break;
            }
            let r = (horiz - barrel_off).max(4.0);
            let dz = pred[2] - tube_base[2];
            let v2 = speed * speed;
            let disc = v2 * v2 - g * (g * r * r + 2.0 * dz * v2);
            if disc < 0.0 {
                valid_pitch = None;
                break;
            }
            let root = disc.sqrt();
            let tan_theta = if high_arc {
                (v2 + root) / (g * r)
            } else {
                (v2 - root) / (g * r)
            };
            let pitch = tan_theta.atan();
            if pitch < cfg.min_pitch || pitch > cfg.max_pitch {
                valid_pitch = None;
                break;
            }
            let vx = speed * pitch.cos();
            if vx < 1.0 {
                valid_pitch = None;
                break;
            }
            flight_t = (r / vx).clamp(0.1, 6.0);
            let lead_t = flight_t.min(2.2); // Damp long-range velocity extrapolation
            let cand = [
                target_pos[0] + target_vel[0] * lead_t,
                target_pos[1] + target_vel[1] * lead_t,
                target_pos[2],
            ];
            pred = if let Some(ht) = world.raycast_solid(target_pos, cand) {
                let f = (ht - 0.1).max(0.0);
                [
                    target_pos[0] + (cand[0] - target_pos[0]) * f,
                    target_pos[1] + (cand[1] - target_pos[1]) * f,
                    target_pos[2],
                ]
            } else {
                cand
            };
            valid_pitch = Some(pitch);
        }

        let Some(pitch) = valid_pitch else { continue };
        // Never self-splash (7m radius + safety buffer)
        if dist_2d(obs.me.pos, pred) < 9.5 {
            continue;
        }
        let yaw = (pred[1] - tube_base[1]).atan2(pred[0] - tube_base[0]);
        let dir = aim_dir([yaw, pitch]);
        let start = [
            tube_base[0] + dir[0] * barrel_off,
            tube_base[1] + dir[1] * barrel_off,
            tube_base[2] + dir[2] * barrel_off,
        ];
        // Check barrel doesn't poke into a wall
        if world.raycast_solid(tube_base, start).is_some() {
            continue;
        }

        // Ray-march 10 chords along the parabola to verify clear flight over/around buildings
        let v0 = [dir[0] * speed, dir[1] * speed, dir[2] * speed];
        let mut prev = start;
        let mut clear = true;
        const SEGMENTS: usize = 10;
        for s in 1..=SEGMENTS {
            let t = (s as f64 / SEGMENTS as f64) * flight_t * 0.94;
            let cur = [
                start[0] + v0[0] * t,
                start[1] + v0[1] * t,
                start[2] + v0[2] * t - 0.5 * g * t * t,
            ];
            if world.raycast_solid(prev, cur).is_some() {
                clear = false;
                break;
            }
            prev = cur;
        }
        if clear {
            return Some(MortarSolution {
                aim: [yaw, pitch],
                high_arc,
            });
        }
    }
    None
}

/// Evaluates all visible mechs, NPCs, and crates, preserving 5-second wallhacked mechs.
pub fn select_target(
    obs: &ObsMsg,
    cfg: &RuleConfig,
    world: &World,
    mode: MatchMode,
) -> Option<TargetInfo> {
    let eye = [
        obs.me.pos[0],
        obs.me.pos[1],
        obs.me.pos[2] + obs.me.eye_height(),
    ];
    let has_mortar = obs
        .me
        .weapons
        .iter()
        .any(|w| w.as_ref().map_or(false, |s| s.kind == "mortar"));
    let has_laser = obs
        .me
        .weapons
        .iter()
        .any(|w| w.as_ref().map_or(false, |s| s.kind == "laser"));
    let has_lightning = obs
        .me
        .weapons
        .iter()
        .any(|w| w.as_ref().map_or(false, |s| s.kind == "lightning"));

    let mut best: Option<TargetInfo> = None;

    for m in &obs.mechs {
        let is_mech = m.kind == "mech";
        let h = if is_mech {
            9.0 * (1.0 + 0.05 * m.cubes as f64)
        } else {
            3.5
        };
        let center = [m.pos[0], m.pos[1], m.pos[2] + 0.5 * h];
        let d = dist_3d(eye, center);
        let los = world.raycast_solid(eye, center).is_none();
        if !los && d > 160.0 {
            continue;
        }

        let hp = m.hp.unwrap_or(if is_mech { 1400.0 } else { 350.0 });
        let enemy_mortar = m.weapons.iter().any(|w| w == "mortar");
        let enemy_lightning = m.weapons.iter().any(|w| w == "lightning");

        // Check if we can hit their 1.5x rear armor arc
        let bearing_from_enemy = (obs.me.pos[1] - m.pos[1]).atan2(obs.me.pos[0] - m.pos[0]);
        let rear_exposed = angle_diff(m.aim[0], bearing_from_enemy).abs() > PI * 0.52;

        let mut score = if is_mech { 1000.0 } else { 440.0 } - d * 2.2;
        score += (1400.0 - hp).max(0.0) * 0.55; // Finish wounded mechs!
        if rear_exposed && is_mech {
            score += 180.0;
        }
        if m.charging {
            score += if has_lightning && d <= 45.0 { 420.0 } else { 220.0 };
        }
        if has_laser && enemy_mortar {
            score += 200.0; // +60% cook-off bonus
        }
        if !los {
            // Still very valuable if we hold mortar (lob over buildings) or close for wallhack peek
            score -= if has_mortar && d <= 200.0 { 120.0 } else { 420.0 };
        }

        // FFA discipline: don't obsess over healthy far mechs unless we can third-party/rear-burst
        if mode == MatchMode::FfaSurvival && is_mech && d > 75.0 && hp > 900.0 && !rear_exposed {
            score -= 380.0;
        }
        // NPC farming value drops once at 10-cube stat cap
        if !is_mech && obs.me.cubes >= cfg.cube_stat_cap {
            score -= 350.0;
        }

        if best.as_ref().map_or(true, |b| score > b.score) {
            best = Some(TargetInfo {
                id: m.id as u64,
                pos: center,
                vel: m.vel,
                yaw: m.aim[0],
                dist: d,
                hp,
                los,
                is_mech,
                is_crate: false,
                has_mortar: enemy_mortar,
                has_lightning: enemy_lightning,
                enemy_charging: m.charging,
                score,
            });
        }
    }

    // Crate farming when under cube cap and no immediate LOS mech fight
    let want_crates = obs.me.cubes < cfg.cube_stat_cap;
    let under_mech_fire = best.as_ref().map_or(false, |b| b.is_mech && b.los && b.dist < 75.0);
    if want_crates && !under_mech_fire {
        for c in &obs.crates {
            let pt = [c.pos[0], c.pos[1], c.pos[2] + 1.5];
            let d = dist_3d(eye, pt);
            if d > 55.0 || world.raycast_solid(eye, pt).is_some() {
                continue;
            }
            let score = 620.0 - d * 3.5 + (300.0 - c.hp) * 0.4;
            if best.as_ref().map_or(true, |b| score > b.score) {
                best = Some(TargetInfo {
                    id: c.id,
                    pos: pt,
                    vel: [0.0, 0.0, 0.0],
                    yaw: 0.0,
                    dist: d,
                    hp: c.hp,
                    los: true,
                    is_mech: false,
                    is_crate: true,
                    has_mortar: false,
                    has_lightning: false,
                    enemy_charging: false,
                    score,
                });
            }
        }
    }

    best
}

/// Decides aim, weapon triggers, laser flick-peek, mortar lobbing, and gun reload.
pub fn decide_combat(
    obs: &ObsMsg,
    cfg: &RuleConfig,
    world: &World,
    mode: MatchMode,
    regen_mode: bool,
    urgent_dodge: bool,
    target: Option<&TargetInfo>,
    fallback_bearing: Option<f64>,
    last_aim: [f64; 2],
    act: &mut Action,
) {
    let muzzle = [
        obs.me.pos[0],
        obs.me.pos[1],
        obs.me.pos[2] + obs.me.muzzle_height(),
    ];
    let cube_mult = 1.0 + 0.04 * (obs.me.cubes.min(cfg.cube_stat_cap) as f64);

    let Some(t) = target else {
        // No target: face incoming damage/sound bearing to guard 180° front armor, else scan
        if let Some(yaw) = fallback_bearing {
            act.aim = [yaw, -0.02];
        } else {
            act.aim = [last_aim[0] + 0.14, -0.02];
        }
        // Opportunistic reload when unengaged
        if let Some(Some(w0)) = obs.me.weapons.first() {
            if w0.kind == "gun" && w0.reload <= 0.0 && w0.ammo.unwrap_or(12) <= 5 {
                act.reload = true;
            }
        }
        return;
    };

    // Always face highest-priority target to protect 1.5x rear armor
    let direct_aim = aim_to(muzzle, t.pos);
    act.aim = direct_aim;

    // Ceasefire during regen discipline unless target is right on top of us or near-dead
    let must_fight = t.is_mech && (t.dist < 38.0 || t.hp <= 220.0);
    if regen_mode && !must_fight {
        return;
    }

    // In FFA, avoid poking healthy distant mechs that aren't facing away unless we can burst
    let ffa_hold_fire = mode == MatchMode::FfaSurvival
        && t.is_mech
        && t.dist > 85.0
        && t.hp > 950.0
        && obs.alive > 4;

    // 1. Check lightning interrupt / close-range shred first
    let mut fired_primary = false;
    for (i, slot) in obs.me.weapons.iter().enumerate().take(2) {
        let Some(w) = slot else { continue };
        if w.kind != "lightning" || !t.los {
            continue;
        }
        let range = w.stat_f64("range", 45.0);
        let cone = w.stat_f64("cone_deg", 30.0).to_radians() * 0.45;
        let yaw_err = angle_diff(obs.me.aim[0], direct_aim[0]).abs();
        let energy_ok = obs.me.energy >= if t.enemy_charging || t.hp < 180.0 { 3.5 } else { 10.0 };
        if t.dist <= range - 0.8 && energy_ok {
            act.aim = direct_aim;
            if yaw_err <= cone {
                act.fire[i] = true;
                fired_primary = true;
                if t.enemy_charging {
                    return; // Immediate laser short-circuit
                }
            }
        }
    }

    // 2. Laser charge-peek, deception flick, and early lethal release
    for (i, slot) in obs.me.weapons.iter().enumerate().take(2) {
        let Some(w) = slot else { continue };
        if w.kind != "laser" || ffa_hold_fire || t.is_crate {
            continue;
        }
        let range = w.stat_f64("range", 280.0);
        let full_charge = w.stat_f64("charge", 0.75);
        let min_charge = w.stat_f64("min_charge", 0.15);
        let min_frac = w.stat_f64("min_frac", 0.35);
        let base_dmg = w.stat_f64("damage", 315.0);
        let hold_s = w.stat_f64("hold_s", 2.0);

        if t.dist > range {
            continue;
        }

        // Compute current expected beam damage (including cookoff & rear bonus)
        let frac = min_frac + (1.0 - min_frac) * (w.charge / full_charge).clamp(0.0, 1.0);
        let cook_mult = if t.has_mortar { 1.6 } else { 1.0 };
        let expected_dmg = base_dmg * frac * cook_mult * cube_mult;

        let yaw_err = angle_diff(obs.me.aim[0], direct_aim[0]).abs();
        let pitch_err = (obs.me.aim[1] - direct_aim[1]).abs();
        let hit_tol = (obs.me.radius().max(2.2) / t.dist.max(8.0)).clamp(0.025, 0.14);
        let aligned = yaw_err <= hit_tol && pitch_err <= hit_tol;

        // Barrel clearance check
        let dir = aim_dir(obs.me.aim);
        let barrel_tip = [
            muzzle[0] + dir[0] * (obs.me.radius() + 0.5),
            muzzle[1] + dir[1] * (obs.me.radius() + 0.5),
            muzzle[2] + dir[2] * (obs.me.radius() + 0.5),
        ];
        let barrel_clear = world.raycast_solid(muzzle, barrel_tip).is_none();

        if w.charge > 0.0 {
            let ready_to_release = (w.charge >= full_charge - 0.03)
                || (w.charge >= min_charge && expected_dmg >= t.hp * 1.05)
                || (urgent_dodge && w.charge >= min_charge)
                || (w.charge >= full_charge + hold_s - 0.25);

            if ready_to_release && t.los && aligned && barrel_clear {
                act.aim = direct_aim;
                act.fire[i] = false; // Release fires hitscan laser!
                fired_primary = true;
            } else if ready_to_release && (urgent_dodge || w.charge >= full_charge + hold_s - 0.25) {
                act.aim = direct_aim;
                act.fire[i] = false; // Drop charge to restore full sprint speed
            } else {
                // Keep charging: flick offset while charging in open LOS to deceive dodging enemies
                act.fire[i] = true;
                if t.los && w.charge < full_charge - 0.14 && t.is_mech {
                    let sign = if (obs.tick / 10) % 2 == 0 { 1.0 } else { -1.0 };
                    act.aim = [direct_aim[0] + 0.26 * sign, direct_aim[1]];
                } else {
                    act.aim = direct_aim;
                }
                fired_primary = true;
            }
        } else if w.cooldown <= 0.0 && !urgent_dodge {
            // Start charging if we have LOS, or pre-charge when a wallhacked mech is close behind a corner
            if t.los || (t.is_mech && t.dist < 55.0) {
                act.fire[i] = true;
                act.aim = direct_aim;
                fired_primary = true;
            }
        }
    }

    // 3. Dual-arc Mortar (shines against hidden 5s-tracked mechs and lightning rushers)
    for (i, slot) in obs.me.weapons.iter().enumerate().take(2) {
        let Some(w) = slot else { continue };
        if w.kind != "mortar" || w.cooldown > 0.08 || t.is_crate || ffa_hold_fire {
            continue;
        }
        // Prefer mortar whenever target is behind cover (!los), holds lightning, or we aren't mid-laser
        let prefer_mortar = !t.los || t.has_lightning || !fired_primary;
        if !prefer_mortar {
            continue;
        }
        let speed = w.stat_f64("speed", cfg.mortar_speed);
        if let Some(sol) = solve_mortar(obs, cfg, world, t.pos, t.vel, speed) {
            act.aim = sol.aim;
            let yaw_err = angle_diff(obs.me.aim[0], sol.aim[0]).abs();
            let pitch_err = (obs.me.aim[1] - sol.aim[1]).abs();
            if yaw_err < 0.06 && pitch_err < 0.06 {
                act.fire[i] = true;
                fired_primary = true;
            }
        }
    }

    // 4. Arm Gun (3D lead + barrel check + smart reload)
    for (i, slot) in obs.me.weapons.iter().enumerate().take(2) {
        let Some(w) = slot else { continue };
        if w.kind != "gun" || !t.los || ffa_hold_fire {
            continue;
        }
        let range = w.stat_f64("range", 230.0);
        let speed = w.stat_f64("speed", 252.0);
        if t.dist > range {
            continue;
        }
        let lead = lead_aim(muzzle, t.pos, t.vel, speed, world);
        if !fired_primary {
            act.aim = lead;
        }
        let yaw_err = angle_diff(obs.me.aim[0], lead[0]).abs();
        let pitch_err = (obs.me.aim[1] - lead[1]).abs();
        let hit_tol = (3.2 / t.dist.max(6.0)).clamp(0.03, 0.24);
        let dir = aim_dir(obs.me.aim);
        let barrel_tip = [
            muzzle[0] + dir[0] * (obs.me.radius() + 0.5),
            muzzle[1] + dir[1] * (obs.me.radius() + 0.5),
            muzzle[2] + dir[2] * (obs.me.radius() + 0.5),
        ];
        if w.reload <= 0.0
            && yaw_err <= hit_tol
            && pitch_err <= hit_tol
            && world.raycast_solid(muzzle, barrel_tip).is_none()
        {
            act.fire[i] = true;
        }
    }
}

/// Computes active evasion vector against bullets, parabolic mortar shells, and charging lasers.
pub fn compute_dodge(obs: &ObsMsg, cfg: &RuleConfig, world: &World) -> ([f64; 2], bool) {
    let pos = obs.me.pos;
    let my_center = [pos[0], pos[1], pos[2] + 0.5 * obs.me.height()];
    let radius = obs.me.radius();
    let mut dodge = [0.0, 0.0];
    let mut urgent = false;

    // 1. Projectiles (gun/npc rays + exact parabolic mortar landing spots)
    for p in &obs.projectiles {
        if p.kind == "mortar" {
            // Simulate parabola until ground/roof impact
            let g = cfg.mortar_gravity;
            let mut cur = p.pos;
            let mut vel = p.vel;
            let dt = 0.06;
            for step in 1..=55 {
                let next = [
                    cur[0] + vel[0] * dt,
                    cur[1] + vel[1] * dt,
                    cur[2] + vel[2] * dt - 0.5 * g * dt * dt,
                ];
                vel[2] -= g * dt;
                let hit = next[2] <= world.terrain_height(next[0], next[1]) + 0.2
                    || (step % 2 == 0 && world.raycast_solid(cur, next).is_some());
                cur = next;
                if hit {
                    let t_imp = step as f64 * dt;
                    let dx = pos[0] - cur[0];
                    let dy = pos[1] - cur[1];
                    let d = (dx * dx + dy * dy).sqrt();
                    if d < 9.5 && t_imp < 2.2 {
                        let ux = if d > 0.1 { dx / d } else { 1.0 };
                        let uy = if d > 0.1 { dy / d } else { 0.0 };
                        let w = (10.0 - d) * 0.8;
                        dodge[0] += ux * w;
                        dodge[1] += uy * w;
                        urgent = true;
                    }
                    break;
                }
            }
        } else {
            let rx = my_center[0] - p.pos[0];
            let ry = my_center[1] - p.pos[1];
            let v2 = p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1];
            if v2 < 4.0 {
                continue;
            }
            let t_closest = (rx * p.vel[0] + ry * p.vel[1]) / v2;
            if !(0.0..=0.85).contains(&t_closest) {
                continue;
            }
            let cx = p.pos[0] + p.vel[0] * t_closest;
            let cy = p.pos[1] + p.vel[1] * t_closest;
            let cz = p.pos[2] + p.vel[2] * t_closest;
            if (cz - my_center[2]).abs() > obs.me.height() * 0.65 {
                continue;
            }
            let mx = my_center[0] - cx;
            let my = my_center[1] - cy;
            let miss_d = (mx * mx + my * my).sqrt();
            if miss_d < radius + 2.2 {
                let spd = v2.sqrt();
                let px = -p.vel[1] / spd;
                let py = p.vel[0] / spd;
                let sign = if mx * px + my * py >= 0.0 { 1.0 } else { -1.0 };
                dodge[0] += px * sign * 3.2;
                dodge[1] += py * sign * 3.2;
                if t_closest < 0.45 {
                    urgent = true;
                }
            }
        }
    }

    // 2. Enemy charging lasers: step perpendicularly out of their live aim ray
    for m in &obs.mechs {
        if m.kind != "mech" || !m.charging {
            continue;
        }
        let h = 9.0 * (1.0 + 0.05 * m.cubes as f64);
        let muzzle = [m.pos[0], m.pos[1], m.pos[2] + h * 0.78];
        let d_to_me = dist_3d(muzzle, my_center);
        if d_to_me > 280.0 || world.raycast_solid(muzzle, my_center).is_some() {
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
        let px = muzzle[0] + dir[0] * proj;
        let py = muzzle[1] + dir[1] * proj;
        let pz = muzzle[2] + dir[2] * proj;
        let perp_d = dist_3d(my_center, [px, py, pz]);
        if perp_d < radius + 3.0 {
            let hlen = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-4);
            let side_x = -dir[1] / hlen;
            let side_y = dir[0] / hlen;
            let sign = if (my_center[0] - px) * side_x + (my_center[1] - py) * side_y >= 0.0 {
                1.0
            } else {
                -1.0
            };
            dodge[0] += side_x * sign * 5.0;
            dodge[1] += side_y * sign * 5.0;
            urgent = true;
        }
    }

    (dodge, urgent)
}