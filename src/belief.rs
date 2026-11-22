use crate::protocol::*;
use crate::world::World;
use std::collections::HashMap;

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
    pub aim: [f64; 2],
    pub hp: f64,
    pub max_hp: f64,
    pub cubes: u32,
    pub weapons: Vec<String>,
    pub charging: bool,
    pub t: f64,
}

#[derive(Debug, Clone)]
pub struct RememberedNpc {
    pub id: u32,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub hp: f64,
    pub t: f64,
}

#[derive(Debug, Clone)]
pub struct RememberedCrate {
    pub id: u64,
    pub pos: [f64; 3],
    pub hp: f64,
    pub t: f64,
}

#[derive(Debug, Clone)]
pub struct RememberedCube {
    pub id: u64,
    pub pos: [f64; 3],
    pub t: f64,
}

#[derive(Debug, Clone)]
pub struct RememberedPickup {
    pub id: u64,
    pub pos: [f64; 3],
    pub kind: String,
    pub t: f64,
}

pub struct BeliefMap {
    pub mode: MatchMode,
    pub prev_alive: u32,
    pub our_kills: u32,
    pub tracks: HashMap<u32, TrackedMech>,
    pub attackers: HashMap<u32, f64>,
    pub npc_mem: HashMap<u32, RememberedNpc>,
    pub crate_mem: HashMap<u64, RememberedCrate>,
    pub cube_mem: HashMap<u64, RememberedCube>,
    pub pick_mem: HashMap<u64, RememberedPickup>,
    pub last_damage_t: f64,
    pub last_damage_bearing: f64,
    pub mech_hit_t: f64,
    pub mech_hit_bearing: f64,
    pub threat_bearing: Option<f64>,
    pub threat_t: f64,
}

impl BeliefMap {
    pub fn new() -> Self {
        Self {
            mode: MatchMode::Unknown,
            prev_alive: 16,
            our_kills: 0,
            tracks: HashMap::with_capacity(16),
            attackers: HashMap::with_capacity(16),
            npc_mem: HashMap::with_capacity(16),
            crate_mem: HashMap::with_capacity(32),
            cube_mem: HashMap::with_capacity(32),
            pick_mem: HashMap::with_capacity(32),
            last_damage_t: -99.0,
            last_damage_bearing: 0.0,
            mech_hit_t: -99.0,
            mech_hit_bearing: 0.0,
            threat_bearing: None,
            threat_t: -99.0,
        }
    }

    pub fn update(&mut self, obs: &ObsMsg, world: &World) {
        let t = obs.t;
        let [x, y, _] = obs.me.pos;
        let my_eye = [x, y, obs.me.pos[2] + obs.me.eye_height()];
        let mut kills_this_tick = 0_u32;

        // Visible mechs and NPCs
        for m in &obs.mechs {
            if m.kind == "mech" {
                let prev_hp = self.tracks.get(&m.id).map(|tr| tr.hp).unwrap_or(1400.0);
                let prev_max = self.tracks.get(&m.id).map(|tr| tr.max_hp).unwrap_or(1400.0);
                self.tracks.insert(
                    m.id,
                    TrackedMech {
                        id: m.id,
                        pos: m.pos,
                        vel: m.vel,
                        aim: m.aim,
                        hp: m.hp.unwrap_or(prev_hp),
                        max_hp: m.max_hp.unwrap_or(prev_max).max(1.0),
                        cubes: m.cubes,
                        weapons: m.weapons.clone(),
                        charging: m.charging,
                        t,
                    },
                );
            } else if m.kind == "npc" {
                let prev_hp = self.npc_mem.get(&m.id).map(|n| n.hp).unwrap_or(350.0);
                self.npc_mem.insert(
                    m.id,
                    RememberedNpc {
                        id: m.id,
                        pos: m.pos,
                        vel: m.vel,
                        hp: m.hp.unwrap_or(prev_hp),
                        t,
                    },
                );
            }
        }
        self.tracks.retain(|_, tr| t - tr.t <= 12.0);

        // Events
        for ev in &obs.events {
            match ev.kind.as_str() {
                "damaged" => {
                    let is_npc = ev.weapon.as_deref() == Some("npc");
                    let is_zone = ev.weapon.as_deref() == Some("zone");
                    if let Some(by) = ev.by {
                        if !is_npc && !is_zone {
                            self.attackers.insert(by, t);
                            if by < 16 {
                                if let Some(b) = ev.bearing {
                                    self.mech_hit_t = t;
                                    self.mech_hit_bearing = b;
                                }
                            }
                        }
                    }
                    if !is_zone {
                        if let Some(b) = ev.bearing {
                            self.last_damage_t = t;
                            self.last_damage_bearing = b;
                            self.threat_bearing = Some(b);
                            self.threat_t = t;
                        }
                    }
                }
                "hit" => {
                    if let (Some(tid), Some(amt)) = (ev.target, ev.amount) {
                        if let Ok(nid) = u32::try_from(tid) {
                            if let Some(nm) = self.npc_mem.get_mut(&nid) {
                                nm.hp -= amt;
                            }
                        }
                        if let Some(cm) = self.crate_mem.get_mut(&tid) {
                            cm.hp -= amt;
                        }
                    }
                }
                "kill" => {
                    kills_this_tick += 1;
                    self.our_kills += 1;
                    if let Some(tid) = ev.target {
                        if let Ok(id32) = u32::try_from(tid) {
                            self.tracks.remove(&id32);
                            self.npc_mem.remove(&id32);
                        }
                    }
                }
                "took" | "take_failed" => {
                    if let Some(pid) = ev.id {
                        self.pick_mem.remove(&pid);
                    }
                }
                _ => {}
            }
        }

        // Sounds
        for s in &obs.sounds {
            if s.dist == "far" {
                continue;
            }
            let combat_sound = matches!(
                s.kind.as_str(),
                "gun" | "laser" | "laser_charge" | "lightning" | "mortar" | "boost" | "step" | "jet"
            );
            if combat_sound && t - self.threat_t > 1.5 {
                self.threat_bearing = Some(s.bearing);
                self.threat_t = t;
            }
        }

        // Expire stale NPCs
        self.npc_mem.retain(|_, nm| t - nm.t <= 25.0 && nm.hp > 0.0);

        // Persistent ground crate memory (35 s TTL)
        for cr in &obs.crates {
            let gz = world.terrain_height(cr.pos[0], cr.pos[1]);
            if (cr.pos[2] - gz).abs() > 2.0 {
                continue;
            }
            self.crate_mem.insert(
                cr.id,
                RememberedCrate {
                    id: cr.id,
                    pos: cr.pos,
                    hp: cr.hp,
                    t,
                },
            );
        }
        self.crate_mem.retain(|_, c| {
            if c.hp <= 0.0 || t - c.t > 35.0 {
                return false;
            }
            if c.t == t {
                return true;
            }
            let d = (c.pos[0] - x).hypot(c.pos[1] - y);
            if d < 6.0 {
                return false;
            }
            if d < 18.0 {
                let pt = [c.pos[0], c.pos[1], c.pos[2] + 1.2];
                if world.raycast_solid(my_eye, pt).is_none() {
                    return false;
                }
            }
            true
        });

        // Ground cube memory (40 s TTL)
        for c in &obs.cubes {
            let gz = world.terrain_height(c.pos[0], c.pos[1]);
            if (c.pos[2] - gz).abs() > 2.0 {
                continue;
            }
            self.cube_mem.insert(
                c.id,
                RememberedCube {
                    id: c.id,
                    pos: c.pos,
                    t,
                },
            );
        }
        self.cube_mem.retain(|_, c| {
            if c.t == t {
                return true;
            }
            if t - c.t > 40.0 {
                return false;
            }
            let d = (c.pos[0] - x).hypot(c.pos[1] - y);
            if d < 5.0 {
                return false;
            }
            if d < 18.0 {
                let pt = [c.pos[0], c.pos[1], c.pos[2] + 1.0];
                if world.raycast_solid(my_eye, pt).is_none() {
                    return false;
                }
            }
            true
        });

        // Ground pickup memory (60 s TTL)
        for p in &obs.pickups {
            let gz = world.terrain_height(p.pos[0], p.pos[1]);
            if (p.pos[2] - gz).abs() > 2.0 {
                continue;
            }
            self.pick_mem.insert(
                p.id,
                RememberedPickup {
                    id: p.id,
                    pos: p.pos,
                    kind: p.kind.clone(),
                    t,
                },
            );
        }
        self.pick_mem.retain(|_, p| {
            if p.t == t {
                return true;
            }
            if t - p.t > 60.0 {
                return false;
            }
            let d = (p.pos[0] - x).hypot(p.pos[1] - y);
            if d < 5.0 {
                return false;
            }
            if d < 18.0 {
                let pt = [p.pos[0], p.pos[1], p.pos[2] + 1.0];
                if world.raycast_solid(my_eye, pt).is_none() {
                    return false;
                }
            }
            true
        });

        // Practice vs. FFA mode detection (no t < 160 cap)
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
                                && (end[0] - m.pos[0]).hypot(end[1] - m.pos[1]) < 5.0
                        })
                    })
            });

            if (other_deaths > 0 && obs.zone.dps <= 35.0) || verified_mech_vs_mech_beam {
                self.mode = MatchMode::FfaSurvival;
            } else if self.mode == MatchMode::Unknown
                && t > 32.0
                && obs.alive + self.our_kills == 16
            {
                self.mode = MatchMode::PracticeHivemind;
            }
        }
        self.prev_alive = obs.alive;
    }
}