use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, BufRead};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Hello(HelloMsg),
    Obs(Box<ObsMsg>),
    End(EndMsg),
}

#[derive(Debug, Deserialize)]
pub struct HelloMsg {
    pub protocol: String,
    pub slot: u32,
    pub seed: u64,
    pub map: MapInfo,
    pub config: Value,
}

#[derive(Debug, Deserialize)]
pub struct MapInfo {
    pub dir: String,
    pub files: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct EndMsg {
    pub place: u32,
    pub kills: u32,
}

#[derive(Debug, Deserialize)]
pub struct ObsMsg {
    pub tick: u64,
    pub t: f64,
    pub alive: u32,
    #[serde(rename = "self")]
    pub me: SelfState,
    #[serde(default)]
    pub mechs: Vec<VisibleMech>,
    #[serde(default)]
    pub projectiles: Vec<Projectile>,
    #[serde(default)]
    pub crates: Vec<Crate>,
    #[serde(default)]
    pub cubes: Vec<Cube>,
    #[serde(default)]
    pub pickups: Vec<Pickup>,
    #[serde(default)]
    pub sounds: Vec<Sound>,
    #[serde(default)]
    pub beams: Vec<Beam>,
    #[serde(default)]
    pub events: Vec<GameEvent>,
    #[serde(default)]
    pub drops: Vec<SupplyDrop>,
    pub zone: Zone,
    #[serde(default)]
    pub offer: Option<Offer>,
}

#[derive(Debug, Deserialize)]
pub struct SelfState {
    pub id: u32,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub aim: [f64; 2],
    pub hp: f64,
    pub max_hp: f64,
    pub energy: f64,
    pub max_energy: f64,
    pub grounded: bool,
    pub cubes: u32,
    #[serde(default)]
    pub upgrades: Vec<String>,
    pub in_zone: bool,
    pub weapons: Vec<Option<WeaponSlot>>,
    #[serde(default)]
    pub stats: Value,
}

impl SelfState {
    pub fn stat_f64(&self, key: &str, default: f64) -> f64 {
        self.stats.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }

    pub fn eye_height(&self) -> f64 {
        self.stat_f64("eye_height", 8.0 * (1.0 + 0.05 * self.cubes as f64))
    }

    pub fn muzzle_height(&self) -> f64 {
        self.stat_f64("muzzle_height", 7.0 * (1.0 + 0.05 * self.cubes as f64))
    }

    pub fn radius(&self) -> f64 {
        self.stat_f64("radius", 2.5 * (1.0 + 0.05 * self.cubes as f64))
    }

    pub fn height(&self) -> f64 {
        self.stat_f64("height", 9.0 * (1.0 + 0.05 * self.cubes as f64))
    }

    pub fn walk_speed(&self) -> f64 {
        self.stat_f64("walk_speed", 11.7)
    }

    pub fn turn_rate(&self) -> f64 {
        self.stat_f64("turn_rate", 4.0)
    }
}

#[derive(Debug, Deserialize)]
pub struct WeaponSlot {
    pub kind: String,
    #[serde(default)]
    pub cooldown: f64,
    #[serde(default)]
    pub ammo: Option<u32>,
    #[serde(default)]
    pub reload: f64,
    #[serde(default)]
    pub charge: f64,
    #[serde(default)]
    pub stats: Value,
}

impl WeaponSlot {
    pub fn stat_f64(&self, key: &str, default: f64) -> f64 {
        self.stats.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }
}

#[derive(Debug, Deserialize)]
pub struct VisibleMech {
    pub id: u32,
    pub kind: String, // "mech" | "npc"
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub aim: [f64; 2],
    #[serde(default)]
    pub hp: Option<f64>,
    #[serde(default)]
    pub max_hp: Option<f64>,
    #[serde(default)]
    pub cubes: u32,
    #[serde(default)]
    pub weapons: Vec<String>,
    #[serde(default)]
    pub charging: bool,
}

#[derive(Debug, Deserialize)]
pub struct Projectile {
    pub id: u64,
    pub kind: String,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

#[derive(Debug, Deserialize)]
pub struct Crate {
    pub id: u64,
    pub pos: [f64; 3],
    pub hp: f64,
}

#[derive(Debug, Deserialize)]
pub struct Cube {
    pub id: u64,
    pub pos: [f64; 3],
}

#[derive(Debug, Deserialize)]
pub struct Pickup {
    pub id: u64,
    pub kind: String,
    pub pos: [f64; 3],
}

#[derive(Debug, Deserialize)]
pub struct Sound {
    pub kind: String,
    pub bearing: f64,
    pub dist: String, // "near" | "mid" | "far"
    pub age: f64,
}

#[derive(Debug, Deserialize)]
pub struct Beam {
    pub by: u32,
    pub kind: String,
    pub path: Vec<[f64; 3]>,
    pub age: f64,
}

#[derive(Debug, Deserialize)]
pub struct GameEvent {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub amount: Option<f64>,
    #[serde(default)]
    pub weapon: Option<String>,
    #[serde(default)]
    pub bearing: Option<f64>,
    #[serde(default)]
    pub by: Option<u32>,
    #[serde(default)]
    pub rear: Option<bool>,
    #[serde(default)]
    pub target: Option<u64>,
    #[serde(default)]
    pub effect: Option<String>,
    #[serde(default)]
    pub id: Option<u64>,
    #[serde(default)]
    pub cubes: Option<u32>,
    #[serde(default)]
    pub slot: Option<usize>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SupplyDrop {
    pub id: u64,
    pub pos: [f64; 3],
    pub radius: f64,
    pub lands_in: f64,
    pub capture_s: f64,
    pub progress: f64,
    pub holder: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct Zone {
    pub center: [f64; 2],
    pub radius: f64,
    pub next_center: [f64; 2],
    pub next_radius: f64,
    pub shrink_start: f64,
    pub shrink_end: f64,
    pub dps: f64,
}

#[derive(Debug, Deserialize)]
pub struct Offer {
    pub id: u64,
    pub expires: f64,
    pub choices: Vec<OfferChoice>,
}

#[derive(Debug, Deserialize)]
pub struct OfferChoice {
    pub id: String,
    pub name: String,
    pub desc: String,
}

#[derive(Debug, Serialize)]
pub struct ReadyMsg<'a> {
    #[serde(rename = "type")]
    pub kind: &'a str,
    pub name: &'a str,
}

#[derive(Debug, Serialize, Clone)]
pub struct Action {
    pub tick: u64,
    #[serde(rename = "move")]
    pub move_dir: [f64; 2],
    pub boost: bool,
    pub jump: bool,
    pub aim: [f64; 2],
    pub fire: [bool; 2],
    pub reload: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pick: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop: Option<usize>,
}

impl Action {
    pub fn idle(tick: u64, aim: [f64; 2]) -> Self {
        Self {
            tick,
            move_dir: [0.0, 0.0],
            boost: false,
            jump: false,
            aim,
            fire: [false, false],
            reload: false,
            pick: None,
            take: None,
            drop: None,
        }
    }

    pub fn sanitize(&mut self, min_pitch: f64, max_pitch: f64) {
        for v in &mut self.move_dir {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        let len = (self.move_dir[0].powi(2) + self.move_dir[1].powi(2)).sqrt();
        if len > 1.0 {
            self.move_dir[0] /= len;
            self.move_dir[1] /= len;
        }
        if !self.aim[0].is_finite() {
            self.aim[0] = 0.0;
        }
        if !self.aim[1].is_finite() {
            self.aim[1] = 0.0;
        }
        self.aim[1] = self.aim[1].clamp(min_pitch, max_pitch);
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpgradeDef {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub desc: String,
    #[serde(default)]
    pub requires: Option<String>,
    #[serde(default)]
    pub max: Option<u32>,
    #[serde(default)]
    pub effects: Vec<UpgradeEffect>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpgradeEffect {
    pub path: String,
    #[serde(default)]
    pub mul: Option<f64>,
    #[serde(default)]
    pub add: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CubesConfig {
    pub pickup_radius: f64,
    pub hp_per_cube: f64,
    pub damage_per_cube: f64,
    pub stat_cap: u32,
    pub size_per_cube: f64,
    pub upgrade_thresholds: Vec<u32>,
}

/// Extracted live constants from `hello.config` with protocol defaults.
#[derive(Debug, Clone)]
pub struct RuleConfig {
    pub min_pitch: f64,
    pub max_pitch: f64,
    pub step_up: f64,
    pub max_slope_deg: f64,
    pub regen_delay: f64,
    pub mortar_gravity: f64,
    pub mortar_speed: f64,
    pub pickup_radius: f64,
    pub cube_stat_cap: u32,
    pub sound_step_speed: f64,
    pub raw_upgrades: Value,
    pub cubes: CubesConfig,
    pub upgrades: Vec<UpgradeDef>,
}

impl RuleConfig {
    pub fn from_json(cfg: &Value) -> Self {
        let get = |path: &[&str], def: f64| -> f64 {
            let mut cur = cfg;
            for k in path {
                match cur.get(*k) {
                    Some(next) => cur = next,
                    None => return def,
                }
            }
            cur.as_f64().unwrap_or(def)
        };

        let stat_cap = get(&["cubes", "stat_cap"], 10.0) as u32;
        let upgrade_thresholds = cfg
            .get("cubes")
            .and_then(|c| c.get("upgrade_thresholds"))
            .and_then(|t| t.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect()
            })
            .unwrap_or_else(|| vec![1, 3, 6, 10, 15, 21, 28, 36, 45]);

        let cubes = CubesConfig {
            pickup_radius: get(&["cubes", "pickup_radius"], get(&["loot", "pickup_radius"], 3.5)),
            hp_per_cube: get(&["cubes", "hp_per_cube"], 0.04),
            damage_per_cube: get(&["cubes", "damage_per_cube"], 0.04),
            stat_cap,
            size_per_cube: get(&["cubes", "size_per_cube"], 0.05),
            upgrade_thresholds,
        };

        let upgrades: Vec<UpgradeDef> = cfg
            .get("upgrades")
            .and_then(|u| serde_json::from_value(u.clone()).ok())
            .unwrap_or_default();

        Self {
            min_pitch: get(&["mech", "min_pitch"], -1.3),
            max_pitch: get(&["mech", "max_pitch"], 1.3),
            step_up: get(&["mech", "step_up"], 1.2),
            max_slope_deg: get(&["mech", "max_slope_deg"], 50.0),
            regen_delay: get(&["mech", "regen_delay"], 5.0),
            mortar_gravity: get(&["weapons", "mortar", "gravity"], 22.0),
            mortar_speed: get(&["weapons", "mortar", "speed"], 72.0),
            pickup_radius: get(&["loot", "pickup_radius"], 4.0),
            cube_stat_cap: stat_cap,
            sound_step_speed: get(&["sound", "step_speed"], 5.0),
            raw_upgrades: cfg.get("upgrades").cloned().unwrap_or(Value::Null),
            cubes,
            upgrades,
        }
    }
}

/// Single-slot mailbox: reader thread continuously drains stdin and overwrites
/// stale observations so the main thread always answers only the newest tick.
struct MailboxState {
    hello: Option<HelloMsg>,
    latest_obs: Option<ObsMsg>,
    end: Option<EndMsg>,
    eof: bool,
}

pub struct StdinMailbox {
    shared: Arc<(Mutex<MailboxState>, Condvar)>,
}

impl StdinMailbox {
    pub fn spawn() -> Self {
        let shared = Arc::new((
            Mutex::new(MailboxState {
                hello: None,
                latest_obs: None,
                end: None,
                eof: false,
            }),
            Condvar::new(),
        ));
        let worker_shared = Arc::clone(&shared);
        thread::spawn(move || {
            let stdin = io::stdin();
            let mut reader = stdin.lock();
            let mut line = String::with_capacity(8192);
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let (lock, cvar) = &*worker_shared;
                        let mut st = lock.lock().unwrap();
                        st.eof = true;
                        cvar.notify_all();
                        break;
                    }
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<ServerMessage>(trimmed) {
                            Ok(msg) => {
                                let (lock, cvar) = &*worker_shared;
                                let mut st = lock.lock().unwrap();
                                match msg {
                                    ServerMessage::Hello(h) => st.hello = Some(h),
                                    ServerMessage::Obs(o) => st.latest_obs = Some(*o),
                                    ServerMessage::End(e) => {
                                        st.end = Some(e);
                                        cvar.notify_all();
                                        break;
                                    }
                                }
                                cvar.notify_one();
                            }
                            Err(err) => {
                                eprintln!("parse error: {err}");
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("stdin error: {err}");
                        let (lock, cvar) = &*worker_shared;
                        let mut st = lock.lock().unwrap();
                        st.eof = true;
                        cvar.notify_all();
                        break;
                    }
                }
            }
        });
        Self { shared }
    }

    pub fn wait_hello(&self) -> Option<HelloMsg> {
        let (lock, cvar) = &*self.shared;
        let mut st = lock.lock().unwrap();
        while st.hello.is_none() && !st.eof && st.end.is_none() {
            st = cvar.wait(st).unwrap();
        }
        st.hello.take()
    }

    pub fn next_obs(&self) -> Option<ObsMsg> {
        let (lock, cvar) = &*self.shared;
        let mut st = lock.lock().unwrap();
        loop {
            if let Some(obs) = st.latest_obs.take() {
                return Some(obs);
            }
            if st.end.is_some() || st.eof {
                return None;
            }
            st = cvar.wait(st).unwrap();
        }
    }
}