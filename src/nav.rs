use crate::world::{point_in_prism_2d, World};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

const NAV_STEP: f64 = 4.0;
const MAX_CLIMB_M: f64 = 46.0;

const DIRS: [(isize, isize, f32); 8] = [
    (1, 0, 4.0),
    (-1, 0, 4.0),
    (0, 1, 4.0),
    (0, -1, 4.0),
    (1, 1, 5.657),
    (1, -1, 5.657),
    (-1, 1, 5.657),
    (-1, -1, 5.657),
];

#[derive(Debug, Clone, Copy)]
pub struct Waypoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub layer: usize,
}

#[derive(Clone, Copy)]
struct NavCell {
    ground_z: f32,
    roof_z: f32,
    blocked_ground: bool,
    near_cover: bool,
}

#[derive(Clone, Copy)]
struct OpenNode {
    f: f32,
    idx: usize,
}

impl PartialEq for OpenNode {
    fn eq(&self, other: &Self) -> bool {
        self.f.to_bits() == other.f.to_bits() && self.idx == other.idx
    }
}
impl Eq for OpenNode {}

impl Ord for OpenNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}
impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct NavEngine {
    min_x: f64,
    min_y: f64,
    nx: usize,
    ny: usize,
    step_up: f32,
    cells: Vec<NavCell>,
    g_score: Vec<f32>,
    parent: Vec<u32>,
    stamp: Vec<u32>,
    cur_stamp: u32,
    cached_goal: [f64; 3],
    cached_path: Vec<Waypoint>,
    last_plan_tick: u64,
}

impl NavEngine {
    pub fn build(world: &World, step_up: f64, max_slope_deg: f64) -> Self {
        let nx = (((world.max_x - world.min_x) / NAV_STEP).ceil() as usize).max(1);
        let ny = (((world.max_y - world.min_y) / NAV_STEP).ceil() as usize).max(1);
        let total = nx * ny;
        let max_slope_dz = (max_slope_deg.to_radians().tan() * NAV_STEP) as f32;

        let mut cells = vec![
            NavCell {
                ground_z: 0.0,
                roof_z: f32::NEG_INFINITY,
                blocked_ground: false,
                near_cover: false,
            };
            total
        ];
        let mut max_prism_top = vec![f32::NEG_INFINITY; total];

        for gy in 0..ny {
            let cy = world.min_y + (gy as f64 + 0.5) * NAV_STEP;
            for gx in 0..nx {
                let cx = world.min_x + (gx as f64 + 0.5) * NAV_STEP;
                cells[gy * nx + gx].ground_z = world.terrain_height(cx, cy) as f32;
            }
        }

        for gy in 0..ny {
            for gx in 0..nx {
                let idx = gy * nx + gx;
                if gx == 0 || gy == 0 || gx + 1 == nx || gy + 1 == ny {
                    cells[idx].blocked_ground = true;
                    continue;
                }
                let z = cells[idx].ground_z;
                let max_diff = (cells[idx - 1].ground_z - z)
                    .abs()
                    .max((cells[idx + 1].ground_z - z).abs())
                    .max((cells[idx - nx].ground_z - z).abs())
                    .max((cells[idx + nx].ground_z - z).abs());
                if max_diff > max_slope_dz {
                    cells[idx].blocked_ground = true;
                }
            }
        }

        let clearance = 3.8;
        let diag = clearance * 0.7071;
        for p in &world.prisms {
            let gx0 = (((p.min_x - clearance - world.min_x) / NAV_STEP).floor() as isize)
                .clamp(0, nx as isize - 1) as usize;
            let gx1 = (((p.max_x + clearance - world.min_x) / NAV_STEP).ceil() as isize)
                .clamp(0, nx as isize - 1) as usize;
            let gy0 = (((p.min_y - clearance - world.min_y) / NAV_STEP).floor() as isize)
                .clamp(0, ny as isize - 1) as usize;
            let gy1 = (((p.max_y + clearance - world.min_y) / NAV_STEP).ceil() as isize)
                .clamp(0, ny as isize - 1) as usize;

            for gy in gy0..=gy1 {
                let cy = world.min_y + (gy as f64 + 0.5) * NAV_STEP;
                for gx in gx0..=gx1 {
                    let cx = world.min_x + (gx as f64 + 0.5) * NAV_STEP;
                    let center_in = point_in_prism_2d(cx, cy, p);
                    let margin_in = center_in
                        || point_in_prism_2d(cx + clearance, cy, p)
                        || point_in_prism_2d(cx - clearance, cy, p)
                        || point_in_prism_2d(cx, cy + clearance, p)
                        || point_in_prism_2d(cx, cy - clearance, p)
                        || point_in_prism_2d(cx + diag, cy + diag, p)
                        || point_in_prism_2d(cx + diag, cy - diag, p)
                        || point_in_prism_2d(cx - diag, cy + diag, p)
                        || point_in_prism_2d(cx - diag, cy - diag, p);
                    if !margin_in {
                        continue;
                    }

                    let idx = gy * nx + gx;
                    let gz = cells[idx].ground_z as f64;
                    if center_in && (p.top as f32) > max_prism_top[idx] {
                        max_prism_top[idx] = p.top as f32;
                    }

                    if p.top <= gz + step_up && p.top > gz {
                        cells[idx].ground_z = p.top as f32;
                    } else if p.base < gz + 11.5 {
                        cells[idx].blocked_ground = true;
                    }

                    if center_in && p.top > gz + step_up && (p.top - gz) <= MAX_CLIMB_M {
                        if (p.top as f32) > cells[idx].roof_z {
                            cells[idx].roof_z = p.top as f32;
                        }
                    }
                }
            }
        }

        for gy in 1..(ny - 1) {
            for gx in 1..(nx - 1) {
                let idx = gy * nx + gx;
                if cells[idx].roof_z.is_finite()
                    && max_prism_top[idx] > cells[idx].roof_z + step_up as f32
                {
                    cells[idx].roof_z = f32::NEG_INFINITY;
                }
                if !cells[idx].blocked_ground {
                    cells[idx].near_cover = cells[idx - 1].blocked_ground
                        || cells[idx + 1].blocked_ground
                        || cells[idx - nx].blocked_ground
                        || cells[idx + nx].blocked_ground;
                }
            }
        }

        let num_states = total * 2;
        Self {
            min_x: world.min_x,
            min_y: world.min_y,
            nx,
            ny,
            step_up: step_up as f32,
            cells,
            g_score: vec![f32::INFINITY; num_states],
            parent: vec![u32::MAX; num_states],
            stamp: vec![0; num_states],
            cur_stamp: 0,
            cached_goal: [0.0; 3],
            cached_path: Vec::new(),
            last_plan_tick: 0,
        }
    }

    pub fn invalidate_cache(&mut self) {
        self.cached_path.clear();
    }

    #[inline]
    fn world_to_grid(&self, x: f64, y: f64) -> (usize, usize) {
        let gx = (((x - self.min_x) / NAV_STEP).floor() as isize)
            .clamp(0, self.nx as isize - 1) as usize;
        let gy = (((y - self.min_y) / NAV_STEP).floor() as isize)
            .clamp(0, self.ny as isize - 1) as usize;
        (gx, gy)
    }

    #[inline]
    fn grid_to_world(&self, gx: usize, gy: usize) -> (f64, f64) {
        (
            self.min_x + (gx as f64 + 0.5) * NAV_STEP,
            self.min_y + (gy as f64 + 0.5) * NAV_STEP,
        )
    }

    #[inline]
    fn state_idx(&self, gx: usize, gy: usize, layer: usize) -> usize {
        layer * (self.nx * self.ny) + gy * self.nx + gx
    }

    #[inline]
    fn decode_state(&self, idx: usize) -> (usize, usize, usize) {
        let total = self.nx * self.ny;
        let layer = idx / total;
        let rem = idx % total;
        (rem % self.nx, rem / self.nx, layer)
    }

    fn cell_z(&self, gx: usize, gy: usize, layer: usize) -> f32 {
        let c = self.cells[gy * self.nx + gx];
        if layer == 1 { c.roof_z } else { c.ground_z }
    }

    fn is_valid_state(&self, gx: usize, gy: usize, layer: usize) -> bool {
        let c = self.cells[gy * self.nx + gx];
        if layer == 0 {
            !c.blocked_ground
        } else {
            c.roof_z.is_finite()
        }
    }

    pub fn is_near_cover(&self, x: f64, y: f64) -> bool {
        let (gx, gy) = self.world_to_grid(x, y);
        self.cells[gy * self.nx + gx].near_cover
    }

    fn snap_valid(&self, gx: usize, gy: usize, z: f64) -> (usize, usize, usize) {
        let c = self.cells[gy * self.nx + gx];
        if c.roof_z.is_finite() && (z as f32 - c.roof_z).abs() < 3.0 {
            return (gx, gy, 1);
        }
        if !c.blocked_ground && (z as f32 - c.ground_z).abs() < 4.5 {
            return (gx, gy, 0);
        }
        let mut best = (gx, gy, 0);
        let mut best_cost = f32::INFINITY;
        for r in 1..=3_isize {
            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = gx as isize + dx;
                    let ny = gy as isize + dy;
                    if nx < 0 || ny < 0 || nx >= self.nx as isize || ny >= self.ny as isize {
                        continue;
                    }
                    let (ux, uy) = (nx as usize, ny as usize);
                    for layer in 0..2 {
                        if !self.is_valid_state(ux, uy, layer) {
                            continue;
                        }
                        let cz = self.cell_z(ux, uy, layer);
                        let cost =
                            ((dx * dx + dy * dy) as f32).sqrt() * 4.0 + (cz - z as f32).abs();
                        if cost < best_cost {
                            best_cost = cost;
                            best = (ux, uy, layer);
                        }
                    }
                }
            }
            if best_cost.is_finite() {
                break;
            }
        }
        best
    }

    pub fn next_waypoint(
        &mut self,
        tick: u64,
        pos: [f64; 3],
        goal: [f64; 3],
        allow_jet: bool,
    ) -> Waypoint {
        let goal_shift = ((goal[0] - self.cached_goal[0]).powi(2)
            + (goal[1] - self.cached_goal[1]).powi(2)
            + (goal[2] - self.cached_goal[2]).powi(2))
        .sqrt();
        let need_replan = self.cached_path.is_empty()
            || tick.saturating_sub(self.last_plan_tick) >= 8
            || goal_shift > 12.0;

        if need_replan {
            self.cached_path = self.find_path(pos, goal, allow_jet);
            self.cached_goal = goal;
            self.last_plan_tick = tick;
        }

        while self.cached_path.len() > 1 {
            let wp = self.cached_path[0];
            let dh = ((wp.x - pos[0]).powi(2) + (wp.y - pos[1]).powi(2)).sqrt();
            let dz = (wp.z - pos[2]).abs();
            if dh < 5.5 && dz < 3.0 {
                self.cached_path.remove(0);
            } else {
                break;
            }
        }

        if let Some(&wp) = self.cached_path.first() {
            if self.cached_path.len() >= 2 {
                let w2 = self.cached_path[1];
                if w2.layer == wp.layer && (w2.z - wp.z).abs() <= self.step_up as f64 {
                    return w2;
                }
            }
            wp
        } else {
            Waypoint {
                x: goal[0],
                y: goal[1],
                z: goal[2],
                layer: 0,
            }
        }
    }

    fn find_path(
        &mut self,
        start_pos: [f64; 3],
        goal_pos: [f64; 3],
        allow_jet: bool,
    ) -> Vec<Waypoint> {
        let (sgx, sgy) = self.world_to_grid(start_pos[0], start_pos[1]);
        let (ggx, ggy) = self.world_to_grid(goal_pos[0], goal_pos[1]);

        let (sx, sy, sl) = self.snap_valid(sgx, sgy, start_pos[2]);
        let (tx, ty, tl) = self.snap_valid(ggx, ggy, goal_pos[2]);

        let start_idx = self.state_idx(sx, sy, sl);
        let goal_idx = self.state_idx(tx, ty, tl);
        if start_idx == goal_idx {
            return vec![Waypoint {
                x: goal_pos[0],
                y: goal_pos[1],
                z: goal_pos[2],
                layer: tl,
            }];
        }

        self.cur_stamp = self.cur_stamp.wrapping_add(1);
        if self.cur_stamp == 0 {
            self.stamp.fill(0);
            self.cur_stamp = 1;
        }
        let stamp = self.cur_stamp;

        let mut open = BinaryHeap::with_capacity(512);
        self.stamp[start_idx] = stamp;
        self.g_score[start_idx] = 0.0;
        self.parent[start_idx] = u32::MAX;

        let heuristic = |gx: usize, gy: usize| -> f32 {
            let dx = (gx as f32 - tx as f32) * NAV_STEP as f32;
            let dy = (gy as f32 - ty as f32) * NAV_STEP as f32;
            (dx * dx + dy * dy).sqrt()
        };

        open.push(OpenNode {
            f: heuristic(sx, sy),
            idx: start_idx,
        });

        let mut best_idx = start_idx;
        let mut best_h = heuristic(sx, sy);
        let mut expansions = 0_usize;
        const MAX_EXPANSIONS: usize = 1200;

        while let Some(OpenNode { idx, .. }) = open.pop() {
            if idx == goal_idx {
                best_idx = idx;
                break;
            }
            expansions += 1;
            if expansions >= MAX_EXPANSIONS {
                break;
            }

            let cur_g = self.g_score[idx];
            let (cx, cy, cl) = self.decode_state(idx);
            let cz = self.cell_z(cx, cy, cl);

            let h_cur = heuristic(cx, cy);
            if h_cur < best_h {
                best_h = h_cur;
                best_idx = idx;
            }

            for &(dx, dy, dist) in &DIRS {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx < 0 || ny < 0 || nx >= self.nx as isize || ny >= self.ny as isize {
                    continue;
                }
                let (ux, uy) = (nx as usize, ny as usize);

                for nl in 0..2 {
                    if !self.is_valid_state(ux, uy, nl) {
                        continue;
                    }
                    let nz = self.cell_z(ux, uy, nl);
                    let dz = nz - cz;

                    let step_cost = if dz <= self.step_up {
                        if cl == 1 && nl == 0 {
                            dist + 2.0
                        } else {
                            dist
                        }
                    } else if allow_jet && nl == 1 && dz <= MAX_CLIMB_M as f32 {
                        dist + 20.0 + dz * 1.6
                    } else {
                        continue;
                    };

                    let nidx = self.state_idx(ux, uy, nl);
                    let ng = cur_g + step_cost;
                    let visited = self.stamp[nidx] == stamp;
                    if !visited || ng < self.g_score[nidx] {
                        self.stamp[nidx] = stamp;
                        self.g_score[nidx] = ng;
                        self.parent[nidx] = idx as u32;
                        open.push(OpenNode {
                            f: ng + heuristic(ux, uy) * 1.2,
                            idx: nidx,
                        });
                    }
                }
            }
        }

        let mut rev = Vec::with_capacity(64);
        let mut cur = best_idx;
        while cur != u32::MAX as usize {
            let (gx, gy, layer) = self.decode_state(cur);
            let (wx, wy) = self.grid_to_world(gx, gy);
            let wz = self.cell_z(gx, gy, layer) as f64;
            rev.push(Waypoint {
                x: wx,
                y: wy,
                z: wz,
                layer,
            });
            let p = self.parent[cur];
            if p == u32::MAX {
                break;
            }
            cur = p as usize;
        }
        rev.reverse();
        if rev.len() > 1 {
            rev.remove(0);
        }
        rev
    }

    /// Bounded 360-degree Dijkstra over walkable cells (up to 58 m walk distance).
    /// Buckets reachable cover spots into 12 angular sectors x 2 distance rings so
    /// every direction gets evaluated and no through-wall unreachable spot is picked.
    pub fn find_escape_cover(
        &mut self,
        pos: [f64; 3],
        my_eye_off: f64,
        threat_eyes: &[[f64; 3]],
        escape_dir: [f64; 2],
        want_zero_los: bool,
        allow_roof: bool,
        world: &World,
        zone_center: [f64; 2],
        zone_radius: f64,
        next_center: [f64; 2],
        next_radius: f64,
    ) -> Option<[f64; 3]> {
        if threat_eyes.is_empty() {
            return None;
        }

        let (sgx, sgy) = self.world_to_grid(pos[0], pos[1]);
        let (sx, sy, sl) = self.snap_valid(sgx, sgy, pos[2]);
        let start_idx = self.state_idx(sx, sy, sl);

        self.cur_stamp = self.cur_stamp.wrapping_add(1);
        if self.cur_stamp == 0 {
            self.stamp.fill(0);
            self.cur_stamp = 1;
        }
        let stamp = self.cur_stamp;

        let mut open = BinaryHeap::with_capacity(256);
        self.stamp[start_idx] = stamp;
        self.g_score[start_idx] = 0.0;
        open.push(OpenNode {
            f: 0.0,
            idx: start_idx,
        });

        let safe_r = (zone_radius - 6.0).max(6.0);
        let safe_next_r = (next_radius - 8.0).max(4.0);
        const MAX_WALK_DIST: f32 = 58.0;
        const MAX_EXPANSIONS: usize = 260;

        // 12 sectors x 2 rings (near: 4..22 m walk, mid: 22..58 m walk)
        // Tuple: (pre_score, walk_dist, cand_pt, use_roof)
        let mut buckets: [Option<(f64, f64, [f64; 3], bool)>; 24] = [None; 24];
        let mut expansions = 0_usize;

        while let Some(OpenNode { f: cur_g, idx }) = open.pop() {
            if cur_g > self.g_score[idx] + 1e-3 {
                continue;
            }
            if cur_g > MAX_WALK_DIST {
                break;
            }
            expansions += 1;
            if expansions > MAX_EXPANSIONS {
                break;
            }

            let (cx, cy, cl) = self.decode_state(idx);
            let cz = self.cell_z(cx, cy, cl);

            if cur_g >= 4.0 && cx > 1 && cy > 1 && cx + 1 < self.nx && cy + 1 < self.ny {
                let c = self.cells[cy * self.nx + cx];
                let use_roof = allow_roof
                    && cl == 1
                    && c.roof_z.is_finite()
                    && (c.roof_z as f64 - pos[2]) > 4.0
                    && (c.roof_z as f64 - pos[2]) <= 24.0;
                let is_cand = (cl == 0 && c.near_cover) || use_roof;

                if is_cand {
                    let (wx, wy) = self.grid_to_world(cx, cy);
                    let dist_z =
                        ((wx - zone_center[0]).powi(2) + (wy - zone_center[1]).powi(2)).sqrt();
                    if dist_z <= safe_r {
                        let dist_me =
                            ((wx - pos[0]).powi(2) + (wy - pos[1]).powi(2)).sqrt().max(1.0);
                        let dir_dot = ((wx - pos[0]) * escape_dir[0]
                            + (wy - pos[1]) * escape_dir[1])
                            / dist_me;
                        let pre_score = dir_dot * 30.0 - (cur_g as f64) * 1.4;

                        let ang = (wy - pos[1]).atan2(wx - pos[0]);
                        let sec = (((ang + PI) / (2.0 * PI) * 12.0).floor() as isize)
                            .clamp(0, 11) as usize;
                        let ring = if cur_g < 22.0 { 0 } else { 1 };
                        let bidx = sec * 2 + ring;

                        if buckets[bidx].map_or(true, |b| pre_score > b.0) {
                            buckets[bidx] =
                                Some((pre_score, cur_g as f64, [wx, wy, cz as f64], use_roof));
                        }
                    }
                }
            }

            for &(dx, dy, dist) in &DIRS {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx <= 0 || ny <= 0 || nx >= self.nx as isize - 1 || ny >= self.ny as isize - 1 {
                    continue;
                }
                let (ux, uy) = (nx as usize, ny as usize);

                let max_layer = if allow_roof || cl == 1 { 2 } else { 1 };
                for nl in 0..max_layer {
                    if !self.is_valid_state(ux, uy, nl) {
                        continue;
                    }
                    let nz = self.cell_z(ux, uy, nl);
                    let dz = nz - cz;

                    let step_cost = if dz <= self.step_up {
                        if cl == 1 && nl == 0 {
                            dist + 2.0
                        } else {
                            dist
                        }
                    } else if allow_roof && nl == 1 && dz <= 24.0 {
                        dist + 18.0 + dz * 1.5
                    } else {
                        continue;
                    };

                    let ng = cur_g + step_cost;
                    if ng > MAX_WALK_DIST {
                        continue;
                    }

                    let nidx = self.state_idx(ux, uy, nl);
                    let visited = self.stamp[nidx] == stamp;
                    if !visited || ng < self.g_score[nidx] {
                        self.stamp[nidx] = stamp;
                        self.g_score[nidx] = ng;
                        open.push(OpenNode { f: ng, idx: nidx });
                    }
                }
            }
        }

        let mut best: Option<(f64, [f64; 3])> = None;
        for slot in buckets.iter().flatten() {
            let &(_, walk_dist, cand_pt, use_roof) = slot;
            let cand_eye = [cand_pt[0], cand_pt[1], cand_pt[2] + my_eye_off];

            let mut los_count = 0_usize;
            let mut min_threat_d = f64::INFINITY;
            for &te in threat_eyes {
                let td = ((cand_pt[0] - te[0]).powi(2) + (cand_pt[1] - te[1]).powi(2)).sqrt();
                min_threat_d = min_threat_d.min(td);
                if world.raycast_solid(te, cand_eye).is_none() {
                    los_count += 1;
                }
            }

            let mut score = if want_zero_los {
                if los_count == 0 {
                    240.0
                } else if los_count == 1 {
                    -35.0
                } else {
                    -(los_count as f64) * 130.0
                }
            } else if los_count == 1 {
                210.0
            } else if los_count == 0 {
                175.0
            } else {
                -(los_count as f64) * 110.0
            };

            let dist_me = ((cand_pt[0] - pos[0]).powi(2) + (cand_pt[1] - pos[1]).powi(2))
                .sqrt()
                .max(1.0);
            let dir_dot = ((cand_pt[0] - pos[0]) * escape_dir[0]
                + (cand_pt[1] - pos[1]) * escape_dir[1])
                / dist_me;
            score += dir_dot * 42.0;
            score += min_threat_d.min(85.0) * 1.15;
            score -= walk_dist * 1.85;

            let d_next = ((cand_pt[0] - next_center[0]).powi(2)
                + (cand_pt[1] - next_center[1]).powi(2))
            .sqrt();
            if d_next > safe_next_r {
                score -= 22.0;
            }
            if use_roof {
                score += 35.0;
            }

            if best.map_or(true, |(bs, _)| score > bs) {
                best = Some((score, cand_pt));
            }
        }

        best.map(|(_, p)| p)
    }

    /// Nearest walkable ground cover-edge (`near_cover`) cell inside `next_zone`,
    /// keeping the mech anchored beside building corners instead of the open Oval center.
    pub fn find_cover_anchor(
        &self,
        pos: [f64; 3],
        next_center: [f64; 2],
        next_radius: f64,
        max_dist: f64,
    ) -> Option<[f64; 3]> {
        let (cgx, cgy) = self.world_to_grid(pos[0], pos[1]);
        let r_cells = (max_dist / NAV_STEP).ceil() as isize;
        let safe_r = (next_radius - 10.0).max(5.0);

        let mut best: Option<(f64, [f64; 3])> = None;
        for dy in (-r_cells..=r_cells).step_by(2) {
            let ny = cgy as isize + dy;
            if ny <= 1 || ny >= (self.ny as isize - 1) {
                continue;
            }
            for dx in (-r_cells..=r_cells).step_by(2) {
                let nx = cgx as isize + dx;
                if nx <= 1 || nx >= (self.nx as isize - 1) {
                    continue;
                }
                let (ux, uy) = (nx as usize, ny as usize);
                let c = self.cells[uy * self.nx + ux];
                if c.blocked_ground || !c.near_cover {
                    continue;
                }
                let (wx, wy) = self.grid_to_world(ux, uy);
                let d_next =
                    ((wx - next_center[0]).powi(2) + (wy - next_center[1]).powi(2)).sqrt();
                if d_next > safe_r {
                    continue;
                }
                let d_me = ((wx - pos[0]).powi(2) + (wy - pos[1]).powi(2)).sqrt();
                if d_me > max_dist {
                    continue;
                }
                let score = -d_me * 1.0 - d_next * 0.25;
                if best.map_or(true, |(bs, _)| score > bs) {
                    best = Some((score, [wx, wy, c.ground_z as f64]));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    pub fn find_safe_rooftop(
        &self,
        pos: [f64; 3],
        zone_center: [f64; 2],
        zone_radius: f64,
        max_dist: f64,
    ) -> Option<[f64; 3]> {
        let (cgx, cgy) = self.world_to_grid(pos[0], pos[1]);
        let r_cells = (max_dist / NAV_STEP).ceil() as isize;
        let safe_r = (zone_radius - 16.0).max(8.0);

        let mut best: Option<(f64, [f64; 3])> = None;
        for dy in (-r_cells..=r_cells).step_by(2) {
            let ny = cgy as isize + dy;
            if ny <= 1 || ny >= (self.ny as isize - 1) {
                continue;
            }
            for dx in (-r_cells..=r_cells).step_by(2) {
                let nx = cgx as isize + dx;
                if nx <= 1 || nx >= (self.nx as isize - 1) {
                    continue;
                }
                let (ux, uy) = (nx as usize, ny as usize);
                let c = self.cells[uy * self.nx + ux];
                if !c.roof_z.is_finite() {
                    continue;
                }
                let (wx, wy) = self.grid_to_world(ux, uy);
                let dist_center =
                    ((wx - zone_center[0]).powi(2) + (wy - zone_center[1]).powi(2)).sqrt();
                if dist_center > safe_r {
                    continue;
                }
                let climb = (c.roof_z as f64 - c.ground_z as f64).max(0.0);
                if !(5.0..=30.0).contains(&climb) {
                    continue;
                }
                let d_me = ((wx - pos[0]).powi(2) + (wy - pos[1]).powi(2)).sqrt();
                if d_me > max_dist {
                    continue;
                }
                let score = climb * 1.3 - d_me * 0.65 - dist_center * 0.25;
                if best.map_or(true, |(bs, _)| score > bs) {
                    best = Some((score, [wx, wy, c.roof_z as f64]));
                }
            }
        }
        best.map(|(_, p)| p)
    }
}