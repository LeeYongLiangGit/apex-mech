use crate::protocol::*;
use crate::world::World;
use std::f64::consts::PI;

pub struct MortarSolution {
    pub aim: [f64; 2],
}

#[inline]
pub fn dist_2d(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[inline]
pub fn dist_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[inline]
pub fn wrap_angle(mut a: f64) -> f64 {
    a = (a + PI).rem_euclid(2.0 * PI);
    a - PI
}

#[inline]
pub fn angle_diff(a: f64, b: f64) -> f64 {
    wrap_angle(b - a)
}

pub fn aim_to(from: [f64; 3], to: [f64; 3]) -> [f64; 2] {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    let horiz = dx.hypot(dy).max(1e-6);
    [dy.atan2(dx), dz.atan2(horiz)]
}

pub fn aim_dir(aim: [f64; 2]) -> [f64; 3] {
    let cp = aim[1].cos();
    [cp * aim[0].cos(), cp * aim[0].sin(), aim[1].sin()]
}

pub fn predict_aim_after_step(
    cur: [f64; 2],
    desired: [f64; 2],
    turn_rate: f64,
    pitch_rate: f64,
) -> [f64; 2] {
    let max_yaw = turn_rate * 0.05;
    let max_pitch = pitch_rate * 0.05;
    let dy = angle_diff(cur[0], desired[0]).clamp(-max_yaw, max_yaw);
    let dp = (desired[1] - cur[1]).clamp(-max_pitch, max_pitch);
    [wrap_angle(cur[0] + dy), cur[1] + dp]
}

/// 3-iteration wall-clamped projectile lead (accounts for track `age`).
pub fn lead_aim(
    from: [f64; 3],
    target_pos: [f64; 3],
    target_vel: [f64; 3],
    age: f64,
    proj_speed: f64,
    world: &World,
) -> [f64; 2] {
    let mut pred = [
        target_pos[0] + target_vel[0] * age,
        target_pos[1] + target_vel[1] * age,
        target_pos[2],
    ];
    for _ in 0..3 {
        let tt = (dist_3d(from, pred) / proj_speed.max(1.0)).clamp(0.0, 1.2);
        let cand = [
            target_pos[0] + target_vel[0] * (tt + age),
            target_pos[1] + target_vel[1] * (tt + age),
            target_pos[2] + target_vel[2] * tt.min(0.3),
        ];
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
    }
    aim_to(from, pred)
}

/// Clump-averaging & zone-edge knock-out mortar aim point.
pub fn compute_mortar_aim_point(
    obs: &ObsMsg,
    target_id: u32,
    target_pos: [f64; 3],
    target_vel: [f64; 3],
    splash_radius: f64,
) -> ([f64; 3], [f64; 3]) {
    let mut sum_x = target_pos[0];
    let mut sum_y = target_pos[1];
    let mut sum_z = target_pos[2] + 1.0;
    let mut sum_vx = target_vel[0];
    let mut sum_vy = target_vel[1];
    let mut count = 1.0;

    let clump_r = splash_radius * 1.45;
    for m in &obs.mechs {
        if m.kind != "mech" || m.id == target_id {
            continue;
        }
        let d = (m.pos[0] - target_pos[0]).hypot(m.pos[1] - target_pos[1]);
        if d <= clump_r && (m.pos[2] - target_pos[2]).abs() <= 5.0 {
            sum_x += m.pos[0];
            sum_y += m.pos[1];
            sum_z += m.pos[2] + 1.0;
            sum_vx += m.vel[0];
            sum_vy += m.vel[1];
            count += 1.0;
        }
    }

    let mut aim_pt = [sum_x / count, sum_y / count, sum_z / count];
    let aim_vel = [sum_vx / count, sum_vy / count, 0.0];

    if count < 2.0 && (obs.zone.dps >= 60.0 || obs.zone.radius <= 140.0) {
        let zx = aim_pt[0] - obs.zone.center[0];
        let zy = aim_pt[1] - obs.zone.center[1];
        let dz = zx.hypot(zy);
        if dz > 5.0 && dz >= (obs.zone.radius - 22.0).max(0.0) {
            aim_pt[0] -= (zx / dz) * 2.4;
            aim_pt[1] -= (zy / dz) * 2.4;
        }
    }

    (aim_pt, aim_vel)
}

/// Dual low/high-arc ballistic mortar solver with 8-segment 3D obstacle clearance.
pub fn solve_mortar(
    obs: &ObsMsg,
    cfg: &RuleConfig,
    world: &World,
    target_pos: [f64; 3],
    target_vel: [f64; 3],
    speed: f64,
    splash_radius: f64,
) -> Option<MortarSolution> {
    let g = cfg.mortar_gravity;
    let tube_z = obs.me.pos[2] + obs.me.height() + 0.5;
    let tube_base = [obs.me.pos[0], obs.me.pos[1], tube_z];
    let barrel_off = obs.me.radius() + 0.5;
    let min_safe_dist = (splash_radius + 4.0).max(11.0);

    let base_terrain_z = world.terrain_height(target_pos[0], target_pos[1]);
    let target_on_roof = target_pos[2] > base_terrain_z + 3.5;
    let z_above_ground = (target_pos[2] - base_terrain_z).clamp(0.5, 2.5);

    for high_arc in [false, true] {
        let mut pred = target_pos;
        let mut valid_pitch = None;
        let mut flight_t = 0.0;

        for _ in 0..3 {
            let dx = pred[0] - tube_base[0];
            let dy = pred[1] - tube_base[1];
            let horiz = dx.hypot(dy);
            if !(min_safe_dist..=228.0).contains(&horiz) {
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
            let lead_t = flight_t.min(1.85);
            let cx = target_pos[0] + target_vel[0] * lead_t;
            let cy = target_pos[1] + target_vel[1] * lead_t;
            let cz = if target_on_roof {
                target_pos[2]
            } else {
                world.terrain_height(cx, cy) + z_above_ground
            };
            let cand = [cx, cy, cz];
            pred = if let Some(ht) = world.raycast_solid(target_pos, cand) {
                let f = (ht - 0.1).max(0.0);
                let px = target_pos[0] + (cand[0] - target_pos[0]) * f;
                let py = target_pos[1] + (cand[1] - target_pos[1]) * f;
                let pz = if target_on_roof {
                    target_pos[2]
                } else {
                    world.terrain_height(px, py) + z_above_ground
                };
                [px, py, pz]
            } else {
                cand
            };
            valid_pitch = Some(pitch);
        }

        let Some(pitch) = valid_pitch else { continue };
        if dist_2d(obs.me.pos, pred) < min_safe_dist {
            continue;
        }
        let yaw = (pred[1] - tube_base[1]).atan2(pred[0] - tube_base[0]);
        let dir = aim_dir([yaw, pitch]);
        let start = [
            tube_base[0] + dir[0] * barrel_off,
            tube_base[1] + dir[1] * barrel_off,
            tube_base[2] + dir[2] * barrel_off,
        ];
        if world.raycast_solid(tube_base, start).is_some() {
            continue;
        }

        let v0 = [dir[0] * speed, dir[1] * speed, dir[2] * speed];
        let mut prev = start;
        let mut clear = true;
        const SEGMENTS: usize = 8;
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
            return Some(MortarSolution { aim: [yaw, pitch] });
        }
    }
    None
}

/// Iter 38 bullet/mortar/laser evasion: static + relative-velocity lead-shot detection,
/// scaled mortar splash radius, and enemy laser-line perpendicular dodge.
pub fn compute_dodge(obs: &ObsMsg, cfg: &RuleConfig, world: &World) -> ([f64; 2], bool) {
    let pos = obs.me.pos;
    let my_vel = obs.me.vel;
    let my_center = [pos[0], pos[1], pos[2] + 0.5 * obs.me.height()];
    let radius = obs.me.radius();
    let mut dodge = [0.0, 0.0];
    let mut urgent = false;

    for p in &obs.projectiles {
        if p.kind == "mortar" {
            let g = cfg.mortar_gravity;
            let mut cur = p.pos;
            let mut vel = p.vel;
            let dt = 0.08;
            let danger_r = (radius + 7.5).max(10.5);
            for step in 1..=40 {
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
                    let d = dx.hypot(dy);
                    if d < danger_r && t_imp < 2.4 {
                        let ux = if d > 0.1 { dx / d } else { 1.0 };
                        let uy = if d > 0.1 { dy / d } else { 0.0 };
                        let w = (danger_r + 0.8 - d) * 0.9;
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
            let spd = v2.sqrt();
            let px = -p.vel[1] / spd;
            let py = p.vel[0] / spd;
            let thresh = radius + 2.3;

            let t_stat = (rx * p.vel[0] + ry * p.vel[1]) / v2;
            let mx_stat = my_center[0] - (p.pos[0] + p.vel[0] * t_stat);
            let my_stat = my_center[1] - (p.pos[1] + p.vel[1] * t_stat);
            let cz_stat = p.pos[2] + p.vel[2] * t_stat;
            let miss_d_stat = mx_stat.hypot(my_stat);
            let stat_threat = (0.0..=0.85).contains(&t_stat)
                && (cz_stat - my_center[2]).abs() <= obs.me.height() * 0.65
                && miss_d_stat < thresh;

            let rvx = p.vel[0] - my_vel[0];
            let rvy = p.vel[1] - my_vel[1];
            let rv2 = rvx * rvx + rvy * rvy;
            let (rel_threat, t_rel, mx_rel, my_rel) = if rv2 >= 4.0 {
                let tr = (rx * rvx + ry * rvy) / rv2;
                let mxr = rx - rvx * tr;
                let myr = ry - rvy * tr;
                let czr = p.pos[2] + p.vel[2] * tr;
                let my_zr = my_center[2] + my_vel[2] * tr;
                let miss_r = mxr.hypot(myr);
                let hit = (0.0..=0.85).contains(&tr)
                    && (czr - my_zr).abs() <= obs.me.height() * 0.65
                    && miss_r < thresh;
                (hit, tr, mxr, myr)
            } else {
                (false, t_stat, mx_stat, my_stat)
            };

            if stat_threat || rel_threat {
                let side_stat = mx_stat * px + my_stat * py;
                let side_rel = mx_rel * px + my_rel * py;
                let sign = if side_stat.abs() >= 0.4 {
                    if side_stat >= 0.0 { 1.0 } else { -1.0 }
                } else if side_rel >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                dodge[0] += px * sign * 3.4;
                dodge[1] += py * sign * 3.4;
                let t_hit = if rel_threat { t_rel } else { t_stat };
                if t_hit < 0.45 && (rel_threat || miss_d_stat < radius + 1.1) {
                    urgent = true;
                }
            }
        }
    }

    for m in &obs.mechs {
        if m.kind != "mech" || !m.charging {
            continue;
        }
        let h = 9.0 * (1.0 + cfg.cube_size_per_cube * m.cubes as f64);
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
        if perp_d < radius + 3.2 {
            let hlen = dir[0].hypot(dir[1]).max(1e-4);
            let side_x = -dir[1] / hlen;
            let side_y = dir[0] / hlen;
            let sign = if (my_center[0] - px) * side_x + (my_center[1] - py) * side_y >= 0.0 {
                1.0
            } else {
                -1.0
            };
            dodge[0] += side_x * sign * 5.2;
            dodge[1] += side_y * sign * 5.2;
            urgent = true;
        }
    }

    (dodge, urgent)
}