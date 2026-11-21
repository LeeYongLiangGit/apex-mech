mod protocol;
mod world;

use protocol::*;
use std::f64::consts::PI;
use std::io::{self, BufWriter, Write};
use world::World;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchMode {
    Unknown,
    PracticeHivemind,
    FfaSurvival,
}

struct BotBrain {
    cfg: RuleConfig,
    world: World,
    mode: MatchMode,
    last_combat_t: f64,
    prev_alive: u32,
    our_kills: u32,
    last_aim: [f64; 2],
}

impl BotBrain {
    fn new(cfg: RuleConfig, world: World) -> Self {
        Self {
            cfg,
            world,
            mode: MatchMode::Unknown,
            last_combat_t: -10.0,
            prev_alive: 16,
            our_kills: 0,
            last_aim: [0.0, 0.0],
        }
    }

    fn decide(&mut self, obs: &ObsMsg) -> Action {
        self.update_state(obs);

        let mut act = Action::idle(obs.tick, obs.me.aim);

        // 1. Resolve upgrade offers immediately
        if let Some(offer) = &obs.offer {
            act.pick = Some(self.pick_upgrade(offer, obs));
        }

        // 2. Weapon looting & optimal 2-slot swapping
        self.handle_weapon_pickup(obs, &mut act);

        // 3. Choose best visible threat / target
        let target = self.select_target(obs);

        // 4. Regen discipline: hold fire when wounded and safe to finish 5s regen
        let hp_frac = obs.me.hp / obs.me.max_hp.max(1.0);
        let regen_mode = hp_frac < 0.72
            && obs.me.in_zone
            && (obs.t - self.last_combat_t) > 1.5
            && target.as_ref().map_or(true, |t| t.dist > 55.0);

        // 5. Aiming, rear-armor guard, and weapon firing
        if let Some(t) = &target {
            act.aim = t.aim;
            if !regen_mode {
                self.decide_fire(obs, t, &mut act);
            }
        } else if let Some(dmg_yaw) = self.last_damage_bearing(obs) {
            // Turn front armor (180° arc) immediately toward incoming unseen fire
            act.aim = [dmg_yaw, 0.0];
        } else {
            // Scan slowly toward movement direction
            act.aim = [self.last_aim[0] + 0.12, -0.02];
        }

        // 6. Reload gun proactively when safe
        if let Some(Some(w0)) = obs.me.weapons.first() {
            if w0.kind == "gun" && w0.reload <= 0.0 && w0.ammo.unwrap_or(12) <= 4 && target.is_none() {
                act.reload = true;
            }
        }

        // 7. Movement: zone safety + projectile dodge + looting/kiting + wall steering
        self.decide_movement(obs, target.as_ref(), regen_mode, &mut act);

        act.sanitize(self.cfg.min_pitch, self.cfg.max_pitch);
        self.last_aim = act.aim;
        act
    }

    fn update_state(&mut self, obs: &ObsMsg) {
        let mut kills_this_tick = 0_u32;
        for ev in &obs.events {
            match ev.kind.as_str() {
                "damaged" | "hit" => self.last_combat_t = obs.t,
                "kill" => {
                    kills_this_tick += 1;
                    self.our_kills += 1;
                }
                _ => {}
            }
        }

        // Detect FFA vs 1v15 Practice Hivemind
        if self.mode == MatchMode::Unknown {
            let other_deaths = self.prev_alive.saturating_sub(obs.alive).saturating_sub(kills_this_tick);
            let mech_vs_mech_beam = obs.beams.iter().any(|b| {
                b.by != obs.me.id
                    && b.path.last().map_or(false, |end| dist_3d(*end, obs.me.pos) > 18.0)
            });
            if (other_deaths > 0 && obs.zone.dps <= 35.0) || mech_vs_mech_beam {
                self.mode = MatchMode::FfaSurvival;
            } else if obs.t > 45.0 && obs.alive + self.our_kills == 16 {
                self.mode = MatchMode::PracticeHivemind;
            }
        }
        self.prev_alive = obs.alive;
    }

    fn last_damage_bearing(&self, obs: &ObsMsg) -> Option<f64> {
        obs.events
            .iter()
            .rev()
            .find(|e| e.kind == "damaged")
            .and_then(|e| e.bearing)
    }

    fn pick_upgrade(&self, offer: &Offer, obs: &ObsMsg) -> usize {
        let score_choice = |c: &OfferChoice| -> f64 {
            let id = c.id.as_str();
            let desc = c.desc.to_lowercase();
            let mut s = 0.0;
            if id.contains("armor") || desc.contains("damage taken") {
                s += 100.0;
            }
            if id.contains("hp") || desc.contains("max hp") || desc.contains("regen") {
                s += 90.0;
            }
            if id.contains("sensor") || desc.contains("vision") {
                s += if self.mode == MatchMode::FfaSurvival { 88.0 } else { 70.0 };
            }
            if id.contains("speed") || id.contains("energy") || id.contains("boost") {
                s += 82.0;
            }
            if id.contains("laser") || id.contains("lightning") {
                s += 85.0;
            }
            if id.contains("mortar") && obs.zone.radius <= 140.0 {
                s += 75.0;
            }
            if id.contains("gun") {
                s += 45.0;
            }
            s
        };
        offer
            .choices
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| score_choice(a).partial_cmp(&score_choice(b)).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
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

        let want_priority = |k: &str| -> i32 {
            match k {
                "laser" => 100,
                "lightning" => 95,
                "mortar" => if obs.zone.radius <= 140.0 { 98 } else { 40 },
                "gun" => 50,
                _ => 0,
            }
        };

        let mut best_pickup: Option<(i32, u64, Option<usize>)> = None;
        for p in &obs.pickups {
            let dh = dist_2d(obs.me.pos, p.pos);
            let dz = (obs.me.pos[2] - p.pos[2]).abs();
            if dh > self.cfg.pickup_radius || dz > 4.0 || has_kind(&p.kind) {
                continue;
            }
            let new_prio = want_priority(&p.kind);
            if empty_slot.is_some() {
                if new_prio > 0 && best_pickup.map_or(true, |b| new_prio > b.0) {
                    best_pickup = Some((new_prio, p.id, None));
                }
            } else {
                // Consider swapping out our lowest-priority non-charging weapon
                for (slot_idx, slot) in obs.me.weapons.iter().enumerate() {
                    let Some(w) = slot else { continue };
                    if w.kind == "laser" && w.charge > 0.0 {
                        continue;
                    }
                    let cur_prio = want_priority(&w.kind);
                    if new_prio > cur_prio + 10 && best_pickup.map_or(true, |b| new_prio > b.0) {
                        best_pickup = Some((new_prio, p.id, Some(slot_idx)));
                    }
                }
            }
        }

        if let Some((_, pid, drop_slot)) = best_pickup {
            act.take = Some(pid);
            act.drop = drop_slot;
        }
    }

    fn select_target(&self, obs: &ObsMsg) -> Option<TargetInfo> {
        let muzzle_z = obs.me.pos[2] + obs.me.muzzle_height();
        let eye_pos = [obs.me.pos[0], obs.me.pos[1], obs.me.pos[2] + obs.me.eye_height()];
        let mut best: Option<TargetInfo> = None;

        // 1. Evaluate visible enemy mechs and NPCs
        for m in &obs.mechs {
            let is_mech = m.kind == "mech";
            let h = if is_mech { 9.0 * (1.0 + 0.05 * m.cubes as f64) } else { 3.5 };
            let center = [m.pos[0], m.pos[1], m.pos[2] + 0.5 * h];
            let d = dist_3d(eye_pos, center);
            let los = self.world.raycast_solid(eye_pos, center).is_none();
            if !los && d > 120.0 {
                continue;
            }

            let mut score = if is_mech { 1000.0 } else { 420.0 } - d * 2.5;
            if let Some(hp) = m.hp {
                score += (1400.0 - hp).max(0.0) * 0.4;
            }
            if m.charging {
                score += 250.0; // Priority interrupt with lightning or dodge
            }
            if !los {
                score -= 500.0; // Tracked via 5s wallhack; keep facing them without wasting gun ammo
            }
            // In FFA, deprioritize far healthy mechs
            if self.mode == MatchMode::FfaSurvival && is_mech && d > 85.0 && m.hp.unwrap_or(1400.0) > 900.0 {
                score -= 350.0;
            }

            let aim = lead_aim([obs.me.pos[0], obs.me.pos[1], muzzle_z], center, m.vel, 252.0);
            if best.as_ref().map_or(true, |b| score > b.score) {
                best = Some(TargetInfo {
                    pos: center,
                    vel: m.vel,
                    dist: d,
                    aim,
                    los,
                    is_mech,
                    enemy_charging: m.charging,
                    score,
                });
            }
        }

        // 2. If no immediate mech threat, farm nearby crates while under cube cap
        let want_crates = obs.me.cubes < self.cfg.cube_stat_cap;
        if want_crates && best.as_ref().map_or(true, |b| !b.is_mech || !b.los) {
            for c in &obs.crates {
                let target_pt = [c.pos[0], c.pos[1], c.pos[2] + 1.5];
                let d = dist_3d(eye_pos, target_pt);
                if d > 45.0 || self.world.raycast_solid(eye_pos, target_pt).is_some() {
                    continue;
                }
                let score = 520.0 - d * 4.0;
                if best.as_ref().map_or(true, |b| score > b.score) {
                    let aim = aim_to([obs.me.pos[0], obs.me.pos[1], muzzle_z], target_pt);
                    best = Some(TargetInfo {
                        pos: target_pt,
                        vel: [0.0, 0.0, 0.0],
                        dist: d,
                        aim,
                        los: true,
                        is_mech: false,
                        enemy_charging: false,
                        score,
                    });
                }
            }
        }

        best
    }

    fn decide_fire(&self, obs: &ObsMsg, t: &TargetInfo, act: &mut Action) {
        if !t.los {
            return;
        }
        let yaw_err = angle_diff(obs.me.aim[0], t.aim[0]).abs();
        let pitch_err = (obs.me.aim[1] - t.aim[1]).abs();

        for (i, slot) in obs.me.weapons.iter().enumerate().take(2) {
            let Some(w) = slot else { continue };
            match w.kind.as_str() {
                "lightning" => {
                    let range = w.stat_f64("range", 45.0);
                    let cone = w.stat_f64("cone_deg", 30.0).to_radians() * 0.5;
                    if t.dist <= range - 1.0 && yaw_err <= cone && obs.me.energy >= 8.0 {
                        act.fire[i] = true;
                        // Interrupt charging laser immediately
                        if t.enemy_charging {
                            return;
                        }
                    }
                }
                "laser" => {
                    let range = w.stat_f64("range", 280.0);
                    let full_charge = w.stat_f64("charge", 0.75);
                    if t.dist <= range {
                        // Hold trigger to charge; release when charged and aligned
                        let aligned = yaw_err < 0.05 && pitch_err < 0.05;
                        if w.charge >= full_charge && aligned {
                            act.fire[i] = false; // Release fires hitscan beam
                        } else if w.cooldown <= 0.0 {
                            act.fire[i] = true; // Keep charging
                        }
                    }
                }
                "gun" => {
                    let range = w.stat_f64("range", 230.0);
                    let hit_tol = (3.0 / t.dist.max(6.0)).clamp(0.03, 0.25);
                    if t.dist <= range && w.reload <= 0.0 && yaw_err < hit_tol && pitch_err < hit_tol {
                        act.fire[i] = true;
                    }
                }
                _ => {}
            }
        }
    }

    fn decide_movement(&self, obs: &ObsMsg, target: Option<&TargetInfo>, regen_mode: bool, act: &mut Action) {
        let pos = obs.me.pos;
        let radius = obs.me.radius();
        let height = obs.me.height();

        // 1. Active projectile sidestepping (highest priority)
        let mut dodge = [0.0, 0.0];
        for p in &obs.projectiles {
            let rx = pos[0] - p.pos[0];
            let ry = pos[1] - p.pos[1];
            let v2 = p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1];
            if v2 < 1.0 {
                continue;
            }
            let t_closest = (rx * p.vel[0] + ry * p.vel[1]) / v2;
            if !(0.0..=0.9).contains(&t_closest) {
                continue;
            }
            let cx = p.pos[0] + p.vel[0] * t_closest;
            let cy = p.pos[1] + p.vel[1] * t_closest;
            let miss_x = pos[0] - cx;
            let miss_y = pos[1] - cy;
            let miss_d = (miss_x * miss_x + miss_y * miss_y).sqrt();
            let danger_r = if p.kind == "mortar" { 9.0 } else { radius + 2.2 };
            if miss_d < danger_r {
                let speed = v2.sqrt();
                let px = -p.vel[1] / speed;
                let py = p.vel[0] / speed;
                let sign = if miss_x * px + miss_y * py >= 0.0 { 1.0 } else { -1.0 };
                dodge[0] += px * sign * 3.0;
                dodge[1] += py * sign * 3.0;
            }
        }

        // 2. Strategic goal vector: zone safety > nearby cubes/weapons > kiting
        let mut goal = [obs.zone.next_center[0], obs.zone.next_center[1]];
        let dist_to_next = dist_2d(pos, [obs.zone.next_center[0], obs.zone.next_center[1], pos[2]]);
        let must_rotate = !obs.me.in_zone || dist_to_next > (obs.zone.next_radius - 20.0).max(5.0);

        if !must_rotate {
            // Grab nearby cubes while below stat cap
            if obs.me.cubes < self.cfg.cube_stat_cap {
                if let Some(c) = obs
                    .cubes
                    .iter()
                    .min_by(|a, b| dist_2d(pos, a.pos).partial_cmp(&dist_2d(pos, b.pos)).unwrap())
                {
                    if dist_2d(pos, c.pos) < 55.0 {
                        goal = [c.pos[0], c.pos[1]];
                    }
                }
            }
            // Grab an initial second weapon if slot 1 is empty
            if obs.me.weapons.get(1).map_or(true, |w| w.is_none()) {
                if let Some(p) = obs
                    .pickups
                    .iter()
                    .filter(|p| p.kind == "laser" || p.kind == "lightning")
                    .min_by(|a, b| dist_2d(pos, a.pos).partial_cmp(&dist_2d(pos, b.pos)).unwrap())
                {
                    if dist_2d(pos, p.pos) < 75.0 {
                        goal = [p.pos[0], p.pos[1]];
                    }
                }
            }
        }

        let mut desired = [goal[0] - pos[0], goal[1] - pos[1]];
        // Combat spacing / kiting when engaged
        if let Some(t) = target {
            if t.is_mech && !must_rotate {
                let away_x = pos[0] - t.pos[0];
                let away_y = pos[1] - t.pos[1];
                let d = (away_x * away_x + away_y * away_y).max(1.0);
                let pref_dist = if regen_mode { 95.0 } else { 52.0 };
                let radial = if t.dist < pref_dist { 1.0 } else { -0.4 };
                let strafe_sign = if (obs.tick / 20) % 2 == 0 { 1.0 } else { -1.0 };
                desired = [
                    (away_x / d) * radial + (-away_y / d) * 0.85 * strafe_sign,
                    (away_y / d) * radial + (away_x / d) * 0.85 * strafe_sign,
                ];
            }
        }

        let raw_dir = [desired[0] + dodge[0], desired[1] + dodge[1]];
        let raw_len = (raw_dir[0] * raw_dir[0] + raw_dir[1] * raw_dir[1]).sqrt();
        if raw_len < 1e-4 {
            return;
        }
        let base_yaw = raw_dir[1].atan2(raw_dir[0]);

        // 3. Local wall-avoidance fan (sample 13 headings using exact prism/terrain collision)
        let mut best_yaw = base_yaw;
        let mut best_score = f64::NEG_INFINITY;
        for step in -6..=6 {
            let yaw = base_yaw + (step as f64) * (PI / 12.0);
            let ux = yaw.cos();
            let uy = yaw.sin();
            let ok_near = self.world.is_walkable_step(pos, pos[0] + ux * 3.0, pos[1] + uy * 3.0, radius, height, self.cfg.step_up);
            let ok_far = ok_near
                && self.world.is_walkable_step(pos, pos[0] + ux * 7.0, pos[1] + uy * 7.0, radius, height, self.cfg.step_up);
            let align = (yaw - base_yaw).cos();
            let score = align * 10.0 + if ok_far { 20.0 } else if ok_near { 8.0 } else { -50.0 };
            if score > best_score {
                best_score = score;
                best_yaw = yaw;
            }
        }

        let mut mag = 1.0;
        // Silent stealth rotation in late FFA when unengaged
        if self.mode == MatchMode::FfaSurvival && target.is_none() && !must_rotate && obs.zone.radius <= 140.0 {
            mag = ((self.cfg.sound_step_speed - 0.3) / obs.me.walk_speed()).clamp(0.25, 1.0);
        }

        act.move_dir = [best_yaw.cos() * mag, best_yaw.sin() * mag];
        let dodging = dodge[0].abs() + dodge[1].abs() > 0.1;
        act.boost = (dodging || !obs.me.in_zone) && obs.me.grounded && obs.me.energy > 25.0;
    }
}

struct TargetInfo {
    pos: [f64; 3],
    vel: [f64; 3],
    dist: f64,
    aim: [f64; 2],
    los: bool,
    is_mech: bool,
    enemy_charging: bool,
    score: f64,
}

fn dist_2d(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn dist_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = (b - a) % (2.0 * PI);
    if d > PI {
        d -= 2.0 * PI;
    } else if d < -PI {
        d += 2.0 * PI;
    }
    d
}

fn aim_to(from: [f64; 3], to: [f64; 3]) -> [f64; 2] {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    let horiz = (dx * dx + dy * dy).sqrt().max(1e-6);
    [dy.atan2(dx), dz.atan2(horiz)]
}

fn lead_aim(from: [f64; 3], target_pos: [f64; 3], target_vel: [f64; 3], proj_speed: f64) -> [f64; 2] {
    let mut t = dist_3d(from, target_pos) / proj_speed.max(1.0);
    for _ in 0..3 {
        let pred = [
            target_pos[0] + target_vel[0] * t,
            target_pos[1] + target_vel[1] * t,
            target_pos[2] + target_vel[2] * t,
        ];
        t = (dist_3d(from, pred) / proj_speed.max(1.0)).clamp(0.0, 1.5);
    }
    let pred = [
        target_pos[0] + target_vel[0] * t,
        target_pos[1] + target_vel[1] * t,
        target_pos[2] + target_vel[2] * t,
    ];
    aim_to(from, pred)
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

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let ready = ReadyMsg {
        kind: "ready",
        name: "apex-mech",
    };
    if serde_json::to_writer(&mut out, &ready).is_err() || writeln!(out).is_err() || out.flush().is_err() {
        return;
    }

    let mut brain = BotBrain::new(cfg, world);
    while let Some(obs) = mailbox.next_obs() {
        let action = brain.decide(&obs);
        if serde_json::to_writer(&mut out, &action).is_err() || writeln!(out).is_err() || out.flush().is_err() {
            break;
        }
    }
}