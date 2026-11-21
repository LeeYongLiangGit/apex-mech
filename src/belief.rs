use crate::protocol::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMode {
    Unknown,
    PracticeHivemind,
    FfaSurvival,
}

#[derive(Debug, Clone)]
pub struct TrackedMech {
    pub id: u32,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub hp: f64,
    pub cubes: u32,
    pub last_seen_t: f64,
    pub visible_now: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ThreatPing {
    pub pos: [f64; 3],
    pub bearing: f64,
    pub t: f64,
    pub weight: f64,
}

pub struct BeliefMap {
    pub mode: MatchMode,
    pub last_combat_t: f64,
    pub last_damaged_t: f64,
    pub last_dmg_bearing: Option<(f64, f64)>,
    pub prev_alive: u32,
    pub our_kills: u32,
    pub tracked: Vec<TrackedMech>,
    pub pings: Vec<ThreatPing>,
}

impl BeliefMap {
    pub fn new() -> Self {
        Self {
            mode: MatchMode::Unknown,
            last_combat_t: -10.0,
            last_damaged_t: -10.0,
            last_dmg_bearing: None,
            prev_alive: 16,
            our_kills: 0,
            tracked: Vec::with_capacity(16),
            pings: Vec::with_capacity(24),
        }
    }

    pub fn update(&mut self, obs: &ObsMsg) {
        let mut kills_this_tick = 0_u32;
        for ev in &obs.events {
            match ev.kind.as_str() {
                "damaged" => {
                    self.last_combat_t = obs.t;
                    self.last_damaged_t = obs.t;
                    if let Some(b) = ev.bearing {
                        self.last_dmg_bearing = Some((b, obs.t));
                        let est = [
                            obs.me.pos[0] + b.cos() * 45.0,
                            obs.me.pos[1] + b.sin() * 45.0,
                            obs.me.pos[2] + 6.0,
                        ];
                        self.pings.push(ThreatPing {
                            pos: est,
                            bearing: b,
                            t: obs.t,
                            weight: 3.0,
                        });
                    }
                }
                "hit" => self.last_combat_t = obs.t,
                "kill" => {
                    kills_this_tick += 1;
                    self.our_kills += 1;
                    if let Some(tid) = ev.target {
                        self.tracked.retain(|m| m.id as u64 != tid);
                    }
                }
                _ => {}
            }
        }

        // Mark all currently tracked mechs as stale before refreshing visible ones
        for tm in &mut self.tracked {
            tm.visible_now = false;
        }
        for m in &obs.mechs {
            if m.kind != "mech" {
                continue;
            }
            if let Some(tm) = self.tracked.iter_mut().find(|x| x.id == m.id) {
                tm.pos = m.pos;
                tm.vel = m.vel;
                tm.hp = m.hp.unwrap_or(tm.hp);
                tm.cubes = m.cubes;
                tm.last_seen_t = obs.t;
                tm.visible_now = true;
            } else {
                self.tracked.push(TrackedMech {
                    id: m.id,
                    pos: m.pos,
                    vel: m.vel,
                    hp: m.hp.unwrap_or(1400.0),
                    cubes: m.cubes,
                    last_seen_t: obs.t,
                    visible_now: true,
                });
            }
        }
        // Expire stale mech tracks after 12 s
        self.tracked.retain(|m| obs.t - m.last_seen_t <= 12.0);

        // Ingest threatening audio cues
        for s in &obs.sounds {
            let (est_dist, weight) = match s.dist.as_str() {
                "near" => (32.0, 2.2),
                "mid" => (95.0, 1.0),
                _ => continue,
            };
            let combat_sound = matches!(
                s.kind.as_str(),
                "laser_charge" | "lightning" | "gun" | "laser" | "mortar" | "boost" | "step" | "jet"
            );
            if !combat_sound {
                continue;
            }
            let pos = [
                obs.me.pos[0] + s.bearing.cos() * est_dist,
                obs.me.pos[1] + s.bearing.sin() * est_dist,
                obs.me.pos[2] + 6.0,
            ];
            self.pings.push(ThreatPing {
                pos,
                bearing: s.bearing,
                t: obs.t - s.age,
                weight,
            });
        }
        self.pings.retain(|p| obs.t - p.t <= 3.0);
        if self.pings.len() > 24 {
            let drain = self.pings.len() - 24;
            self.pings.drain(0..drain);
        }

        // Strict Practice vs. FFA mode detection (allow PracticeHivemind -> FfaSurvival override)
        if self.mode != MatchMode::FfaSurvival {
            let other_deaths = if obs.tick > 1 {
                self.prev_alive
                    .saturating_sub(obs.alive)
                    .saturating_sub(kills_this_tick)
            } else {
                0
            };

            let verified_mech_vs_mech_beam = obs.beams.iter().any(|b| {
                b.by != obs.me.id
                    && b.path.last().map_or(false, |end| {
                        obs.mechs.iter().any(|m| {
                            m.kind == "mech"
                                && m.id != b.by
                                && ((end[0] - m.pos[0]).powi(2) + (end[1] - m.pos[1]).powi(2))
                                    .sqrt()
                                    < 5.0
                        })
                    })
            });

            if (other_deaths > 0 && obs.zone.dps <= 35.0 && obs.t < 160.0)
                || verified_mech_vs_mech_beam
            {
                self.mode = MatchMode::FfaSurvival;
            } else if self.mode == MatchMode::Unknown
                && obs.t > 32.0
                && obs.alive + self.our_kills == 16
            {
                self.mode = MatchMode::PracticeHivemind;
            }
        }
        self.prev_alive = obs.alive;
    }

    /// Returns up to `max_n` closest active enemy eye positions (visible or seen < 4s ago).
    pub fn top_threat_eyes(&self, pos: [f64; 3], now: f64, max_n: usize) -> Vec<[f64; 3]> {
        let mut items: Vec<(f64, [f64; 3])> = self
            .tracked
            .iter()
            .filter(|m| m.visible_now || (now - m.last_seen_t) <= 3.5)
            .map(|m| {
                let d2 = (m.pos[0] - pos[0]).powi(2) + (m.pos[1] - pos[1]).powi(2);
                let h = 9.0 * (1.0 + 0.05 * m.cubes as f64);
                (d2, [m.pos[0], m.pos[1], m.pos[2] + 0.78 * h])
            })
            .filter(|(d2, _)| *d2 <= 170.0 * 170.0)
            .collect();
        items.sort_by(|a, b| a.0.total_cmp(&b.0));
        items.truncate(max_n);
        items.into_iter().map(|(_, p)| p).collect()
    }

    /// Returns a unit vector pointing AWAY from the weighted threat cluster (tracked + recent pings).
    pub fn escape_vector(&self, pos: [f64; 3], now: f64) -> [f64; 2] {
        let mut rx = 0.0;
        let mut ry = 0.0;
        for m in &self.tracked {
            let age = (now - m.last_seen_t).max(0.0);
            if age > 6.0 {
                continue;
            }
            let dx = pos[0] - m.pos[0];
            let dy = pos[1] - m.pos[1];
            let d = (dx * dx + dy * dy).sqrt().max(4.0);
            if d > 140.0 {
                continue;
            }
            let w = (1.0 - age / 7.0) * (90.0 / d).min(3.5);
            rx += (dx / d) * w;
            ry += (dy / d) * w;
        }
        for p in &self.pings {
            let age = (now - p.t).max(0.0);
            let dx = pos[0] - p.pos[0];
            let dy = pos[1] - p.pos[1];
            let d = (dx * dx + dy * dy).sqrt().max(6.0);
            let w = p.weight * (1.0 - age / 3.2).max(0.0) * (50.0 / d).min(2.0);
            rx += (dx / d) * w;
            ry += (dy / d) * w;
        }
        let len = (rx * rx + ry * ry).sqrt();
        if len > 1e-4 {
            [rx / len, ry / len]
        } else {
            [0.0, 0.0]
        }
    }

    /// Best yaw to face when no visible target exists (protects 180° front armor).
    pub fn best_unseen_bearing(&self, pos: [f64; 3], now: f64) -> Option<f64> {
        if let Some((b, t)) = self.last_dmg_bearing {
            if now - t <= 2.5 {
                return Some(b);
            }
        }
        // Closest recently-tracked mech
        if let Some(m) = self
            .tracked
            .iter()
            .filter(|m| now - m.last_seen_t <= 5.0)
            .min_by(|a, b| {
                let da = (a.pos[0] - pos[0]).powi(2) + (a.pos[1] - pos[1]).powi(2);
                let db = (b.pos[0] - pos[0]).powi(2) + (b.pos[1] - pos[1]).powi(2);
                da.total_cmp(&db)
            })
        {
            return Some((m.pos[1] - pos[1]).atan2(m.pos[0] - pos[0]));
        }
        // Most recent nearby sound
        self.pings
            .iter()
            .filter(|p| now - p.t <= 2.0)
            .max_by(|a, b| a.t.total_cmp(&b.t))
            .map(|p| p.bearing)
    }

    /// Nearest known unseen hunt target (for Practice snowballing when healthy).
    pub fn nearest_hunt_point(&self, pos: [f64; 3], now: f64) -> Option<[f64; 3]> {
        if let Some(m) = self
            .tracked
            .iter()
            .filter(|m| now - m.last_seen_t <= 10.0)
            .min_by(|a, b| {
                let da = (a.pos[0] - pos[0]).powi(2) + (a.pos[1] - pos[1]).powi(2);
                let db = (b.pos[0] - pos[0]).powi(2) + (b.pos[1] - pos[1]).powi(2);
                da.total_cmp(&db)
            })
        {
            return Some(m.pos);
        }
        self.pings
            .iter()
            .max_by(|a, b| a.t.total_cmp(&b.t))
            .map(|p| p.pos)
    }
}