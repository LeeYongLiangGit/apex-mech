use crate::world::World;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

const INF: f32 = 1e9;

#[derive(Clone, Copy)]
struct HeapNode {
    f: f32,
    idx: usize,
}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.f.to_bits() == other.f.to_bits() && self.idx == other.idx
    }
}
impl Eq for HeapNode {}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}
impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct NavEngine {
    min_x: f64,
    min_y: f64,
    nnx: usize,
    nny: usize,
    walk: Vec<u8>,
    near_cover: Vec<u8>,
    tight: Vec<u8>,
    g_score: Vec<f32>,
    parent: Vec<u32>,
    stamp: Vec<u32>,
    cur_stamp: u32,
    cached_goal: [f64; 2],
    cached_path: Vec<[f64; 2]>,
    last_plan_tick: u64,
}

impl NavEngine {
    pub fn build(world: &World, step_up: f64, max_slope_deg: f64) -> Self {
        let min_x = world.min_x;
        let min_y = world.min_y;
        let nx = ((world.max_x - min_x).ceil() as usize).max(2) + 1;
        let ny = ((world.max_y - min_y).ceil() as usize).max(2) + 1;
        let n1 = nx * ny;

        let mut ground = vec![0.0_f32; n1];
        for j in 0..ny {
            let y = min_y + j as f64;
            for i in 0..nx {
                let x = min_x + i as f64;
                ground[j * nx + i] = world.terrain_height(x, y) as f32;
            }
        }

        let mut top = vec![-INF; n1];
        let mut base = vec![INF; n1];
        let mut xs: Vec<f64> = Vec::with_capacity(16);

        for p in &world.prisms {
            let j0 = ((p.min_y - min_y).floor() as isize).clamp(0, ny as isize - 1) as usize;
            let j1 = ((p.max_y - min_y).ceil() as isize).clamp(0, ny as isize - 1) as usize;
            let top_f = p.top as f32;
            let base_f = p.base as f32;

            for j in j0..=j1 {
                let y = min_y + j as f64 + 0.0001;
                xs.clear();

                let mut add_edges = |ring: &[[f64; 2]]| {
                    let m = ring.len();
                    if m < 3 {
                        return;
                    }
                    for a in 0..m {
                        let b = (a + 1) % m;
                        let (ax, ay) = (ring[a][0], ring[a][1]);
                        let (bx, by) = (ring[b][0], ring[b][1]);
                        if (ay <= y && by > y) || (by <= y && ay > y) {
                            xs.push(ax + ((y - ay) / (by - ay)) * (bx - ax));
                        }
                    }
                };

                add_edges(&p.ring);
                for hole in &p.holes {
                    add_edges(hole);
                }

                xs.sort_by(|a, b| a.total_cmp(b));
                let mut k = 0;
                while k + 1 < xs.len() {
                    let raw_i0 = (xs[k] - min_x).ceil() as isize;
                    let raw_i1 = (xs[k + 1] - min_x).floor() as isize;
                    if raw_i1 >= 0 && raw_i0 < nx as isize && raw_i0 <= raw_i1 {
                        let i0 = raw_i0.max(0) as usize;
                        let i1 = raw_i1.min(nx as isize - 1) as usize;
                        for i in i0..=i1 {
                            let idx = j * nx + i;
                            if top_f > top[idx] {
                                top[idx] = top_f;
                            }
                            if base_f < base[idx] {
                                base[idx] = base_f;
                            }
                        }
                    }
                    k += 2;
                }
            }
        }

        let nnx = nx.div_ceil(2);
        let nny = ny.div_ceil(2);
        let total = nnx * nny;
        let mut walk = vec![0_u8; total];
        let step_up_f = step_up as f32;
        let max_slope_dz = ((max_slope_deg.to_radians().tan() * 2.0) as f32).min(2.2);

        for jj in 0..nny {
            for ii in 0..nnx {
                let mut ok = 1_u8;
                'window: for dj in -2_isize..=3 {
                    for di in -2_isize..=3 {
                        let i = ii as isize * 2 + di;
                        let j = jj as isize * 2 + dj;
                        if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize {
                            ok = 0;
                            break 'window;
                        }
                        let k = j as usize * nx + i as usize;
                        let g = ground[k];
                        if top[k] > g + step_up_f && base[k] < g + 9.5 {
                            ok = 0;
                            break 'window;
                        }
                    }
                }
                if ok == 1 && nx >= 3 && ny >= 3 {
                    let i = (ii * 2).min(nx - 3);
                    let j = (jj * 2).min(ny - 3);
                    let g = ground[j * nx + i];
                    if (ground[j * nx + i + 2] - g).abs() > max_slope_dz
                        || (ground[(j + 2) * nx + i] - g).abs() > max_slope_dz
                    {
                        ok = 0;
                    }
                }
                walk[jj * nnx + ii] = ok;
            }
        }

        let mut near_cover = vec![0_u8; total];
        let mut tight = vec![0_u8; total];
        if nnx > 6 && nny > 6 {
            for jj in 3..(nny - 3) {
                for ii in 3..(nnx - 3) {
                    let k = jj * nnx + ii;
                    if walk[k] == 0 {
                        continue;
                    }
                    let mut adj1 = false;
                    let mut adj2 = false;
                    for dj in -2_isize..=2 {
                        for di in -2_isize..=2 {
                            if di == 0 && dj == 0 {
                                continue;
                            }
                            let nk = (jj as isize + dj) as usize * nnx + (ii as isize + di) as usize;
                            if walk[nk] == 0 {
                                adj2 = true;
                                if di.abs() <= 1 && dj.abs() <= 1 {
                                    adj1 = true;
                                }
                            }
                        }
                    }
                    if adj1 {
                        tight[k] = 1;
                    }
                    if adj2 {
                        near_cover[k] = 1;
                    }
                }
            }
        }

        Self {
            min_x,
            min_y,
            nnx,
            nny,
            walk,
            near_cover,
            tight,
            g_score: vec![INF; total],
            parent: vec![u32::MAX; total],
            stamp: vec![0; total],
            cur_stamp: 0,
            cached_goal: [0.0, 0.0],
            cached_path: Vec::new(),
            last_plan_tick: 0,
        }
    }

    pub fn invalidate_cache(&mut self) {
        self.cached_path.clear();
    }

    #[inline]
    pub fn cell_of(&self, x: f64, y: f64) -> usize {
        let i = (((x - self.min_x) * 0.5).round() as isize)
            .clamp(0, self.nnx as isize - 1) as usize;
        let j = (((y - self.min_y) * 0.5).round() as isize)
            .clamp(0, self.nny as isize - 1) as usize;
        j * self.nnx + i
    }

    #[inline]
    fn cell_world(&self, idx: usize) -> [f64; 2] {
        let i = idx % self.nnx;
        let j = idx / self.nnx;
        [self.min_x + (i as f64) * 2.0, self.min_y + (j as f64) * 2.0]
    }

    #[inline]
    fn next_stamp(&mut self) -> u32 {
        self.cur_stamp = self.cur_stamp.wrapping_add(1);
        if self.cur_stamp == 0 {
            self.stamp.fill(0);
            self.cur_stamp = 1;
        }
        self.cur_stamp
    }

    fn snap_walkable(&self, idx: usize, max_r: isize) -> Option<usize> {
        if self.walk[idx] != 0 {
            return Some(idx);
        }
        let w = self.nnx as isize;
        let h = self.nny as isize;
        let ci = (idx % self.nnx) as isize;
        let cj = (idx / self.nnx) as isize;
        let mut best: Option<usize> = None;
        let mut best_d2 = isize::MAX;
        for r in 1..=max_r {
            for dj in -r..=r {
                for di in -r..=r {
                    let ni = ci + di;
                    let nj = cj + dj;
                    if ni < 0 || nj < 0 || ni >= w || nj >= h {
                        continue;
                    }
                    let nk = nj as usize * self.nnx + ni as usize;
                    let d2 = di * di + dj * dj;
                    if self.walk[nk] != 0 && d2 < best_d2 {
                        best_d2 = d2;
                        best = Some(nk);
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        best
    }

    pub fn walk_segment(&self, x: f64, y: f64, gx: f64, gy: f64) -> bool {
        let mut prev = self.cell_of(x, y);
        if self.walk[prev] == 0 {
            return true;
        }
        let steps = ((gx - x).hypot(gy - y).ceil() as usize).max(1);
        for step in 1..=steps {
            let f = step as f64 / steps as f64;
            let cell = self.cell_of(x + (gx - x) * f, y + (gy - y) * f);
            if self.walk[cell] == 0 {
                return false;
            }
            let dx = (cell % self.nnx) as isize - (prev % self.nnx) as isize;
            let dy = (cell / self.nnx) as isize - (prev / self.nnx) as isize;
            if dx != 0 && dy != 0 && dx.abs() == 1 && dy.abs() == 1 {
                let c1 = (prev as isize + dx) as usize;
                let c2 = (prev as isize + dy * self.nnx as isize) as usize;
                if self.walk[c1] == 0 || self.walk[c2] == 0 {
                    return false;
                }
            }
            prev = cell;
        }
        true
    }

    /// Instant stamped 2 m weighted A* with LOS/walkable string-pulling (ready on tick 0).
    pub fn next_waypoint(
        &mut self,
        tick: u64,
        x: f64,
        y: f64,
        goal: [f64; 2],
        big_mech: bool,
    ) -> [f64; 2] {
        let goal_shift = (goal[0] - self.cached_goal[0]).hypot(goal[1] - self.cached_goal[1]);
        let need_replan = self.cached_path.is_empty()
            || tick.saturating_sub(self.last_plan_tick) >= 6
            || goal_shift > 10.0;

        if need_replan {
            self.cached_path = self.find_path(x, y, goal[0], goal[1], big_mech);
            self.cached_goal = goal;
            self.last_plan_tick = tick;
        }

        while self.cached_path.len() > 1
            && (self.cached_path[0][0] - x).hypot(self.cached_path[0][1] - y) < 3.5
        {
            self.cached_path.remove(0);
        }

        if self.cached_path.is_empty() {
            return goal;
        }

        // String-pull up to 12 waypoints ahead along walkable line-of-sight for smooth movement
        let max_look = self.cached_path.len().min(12);
        if self.walk[self.cell_of(x, y)] != 0 {
            for idx in (1..max_look).rev() {
                let wp = self.cached_path[idx];
                if self.walk_segment(x, y, wp[0], wp[1]) {
                    return wp;
                }
            }
        }
        self.cached_path[0]
    }

    fn find_path(
        &mut self,
        sx_w: f64,
        sy_w: f64,
        gx_w: f64,
        gy_w: f64,
        big_mech: bool,
    ) -> Vec<[f64; 2]> {
        let Some(start) = self.snap_walkable(self.cell_of(sx_w, sy_w), 6) else {
            return vec![[gx_w, gy_w]];
        };
        let Some(goal) = self.snap_walkable(self.cell_of(gx_w, gy_w), 24) else {
            return vec![[gx_w, gy_w]];
        };
        if start == goal {
            return vec![[gx_w, gy_w]];
        }

        let w = self.nnx;
        let h = self.nny;
        let tx = (goal % w) as f32;
        let ty = (goal / w) as f32;
        let stamp = self.next_stamp();

        self.stamp[start] = stamp;
        self.g_score[start] = 0.0;
        self.parent[start] = u32::MAX;

        let heuristic = |idx: usize| -> f32 {
            let dx = (idx % w) as f32 - tx;
            let dy = (idx / w) as f32 - ty;
            (dx * dx + dy * dy).sqrt() * 2.0
        };

        let mut open = BinaryHeap::with_capacity(512);
        let h0 = heuristic(start);
        open.push(HeapNode {
            f: h0 * 1.25,
            idx: start,
        });

        let mut best_idx = start;
        let mut best_h = h0;
        let mut expansions = 0_usize;
        const MAX_EXPANSIONS: usize = 3600;
        const STEP_CARD: f32 = 2.0;
        const STEP_DIAG: f32 = 2.828_427;
        let tight_pen = if big_mech { 0.85_f32 } else { 0.0 };

        while let Some(HeapNode { idx, .. }) = open.pop() {
            if idx == goal {
                best_idx = idx;
                break;
            }
            expansions += 1;
            if expansions >= MAX_EXPANSIONS {
                break;
            }

            let cur_g = self.g_score[idx];
            let hc = heuristic(idx);
            if hc < best_h {
                best_h = hc;
                best_idx = idx;
            }

            let ci = (idx % w) as isize;
            let cj = (idx / w) as isize;
            for dj in -1_isize..=1 {
                let nj = cj + dj;
                if nj < 0 || nj >= h as isize {
                    continue;
                }
                let nju = nj as usize;
                let cju = cj as usize;
                for di in -1_isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ni = ci + di;
                    if ni < 0 || ni >= w as isize {
                        continue;
                    }
                    let niu = ni as usize;
                    let nidx = nju * w + niu;
                    if self.walk[nidx] == 0 {
                        continue;
                    }
                    if di != 0 && dj != 0 {
                        let ciu = ci as usize;
                        if self.walk[cju * w + niu] == 0 || self.walk[nju * w + ciu] == 0 {
                            continue;
                        }
                    }
                    let step = if di != 0 && dj != 0 {
                        STEP_DIAG
                    } else {
                        STEP_CARD
                    } + if self.tight[nidx] != 0 { tight_pen } else { 0.0 };
                    let ng = cur_g + step;
                    let visited = self.stamp[nidx] == stamp;
                    if !visited || ng < self.g_score[nidx] {
                        self.stamp[nidx] = stamp;
                        self.g_score[nidx] = ng;
                        self.parent[nidx] = idx as u32;
                        open.push(HeapNode {
                            f: ng + heuristic(nidx) * 1.25,
                            idx: nidx,
                        });
                    }
                }
            }
        }

        let mut rev = Vec::with_capacity(64);
        let mut cur = best_idx;
        let mut guard = 0_usize;
        while cur != u32::MAX as usize && guard < 512 {
            rev.push(self.cell_world(cur));
            let p = self.parent[cur];
            if p == u32::MAX {
                break;
            }
            cur = p as usize;
            guard += 1;
        }
        rev.reverse();
        if rev.len() > 1 {
            rev.remove(0);
        }
        if best_idx == goal {
            if let Some(last) = rev.last_mut() {
                *last = [gx_w, gy_w];
            }
        }
        rev
    }

    /// Bounded 8-way Dijkstra over the 2 m walkable grid (up to 56 m walk distance).
    /// Buckets reachable cover & shadow spots into 12 angular sectors x 2 distance rings,
    /// validates 3D solid LOS against all active threats, and returns `(goal, path, los_count)`.
    pub fn find_bfs_cover(
        &mut self,
        x: f64,
        y: f64,
        my_eye_off: f64,
        threat_eyes: &[[f64; 3]],
        escape_dir: [f64; 2],
        zone_center: [f64; 2],
        zone_radius: f64,
        next_center: [f64; 2],
        next_radius: f64,
        world: &World,
    ) -> Option<([f64; 2], Vec<[f64; 2]>, usize)> {
        if threat_eyes.is_empty() {
            return None;
        }
        let start = self.snap_walkable(self.cell_of(x, y), 6)?;
        let w = self.nnx;
        let h = self.nny;
        let stamp = self.next_stamp();

        self.stamp[start] = stamp;
        self.g_score[start] = 0.0;
        self.parent[start] = u32::MAX;

        let mut open = BinaryHeap::with_capacity(256);
        open.push(HeapNode { f: 0.0, idx: start });

        let safe_r = (zone_radius - 5.0).max(5.0);
        let safe_next_r = (next_radius - 8.0).max(4.0);
        const MAX_WALK_M: f32 = 56.0;
        const MAX_EXPANSIONS: usize = 680;
        const STEP_CARD: f32 = 2.0;
        const STEP_DIAG: f32 = 2.828_427;

        let mut buckets: [Option<(f64, f64, usize, [f64; 2], bool)>; 24] = [None; 24];
        let mut expansions = 0_usize;

        while let Some(HeapNode { f: cur_g, idx }) = open.pop() {
            if cur_g > self.g_score[idx] + 1e-3 {
                continue;
            }
            if cur_g > MAX_WALK_M {
                break;
            }
            expansions += 1;
            if expansions > MAX_EXPANSIONS {
                break;
            }

            let ci = (idx % w) as isize;
            let cj = (idx / w) as isize;
            let is_edge = self.near_cover[idx] != 0;
            let is_sample = is_edge || ((ci & 1) == 0 && (cj & 1) == 0);

            if cur_g >= 6.0 && is_sample {
                let pt = self.cell_world(idx);
                if (pt[0] - zone_center[0]).hypot(pt[1] - zone_center[1]) <= safe_r {
                    let dist_me = (pt[0] - x).hypot(pt[1] - y).max(1.0);
                    let dir_dot = ((pt[0] - x) * escape_dir[0] + (pt[1] - y) * escape_dir[1])
                        / dist_me;
                    let pre_score =
                        dir_dot * 32.0 - (cur_g as f64) * 1.2 + if is_edge { 14.0 } else { 0.0 };

                    let ang = (pt[1] - y).atan2(pt[0] - x);
                    let sec = (((ang + PI) / (2.0 * PI) * 12.0).floor() as isize)
                        .clamp(0, 11) as usize;
                    let ring = if cur_g < 22.0 { 0 } else { 1 };
                    let bidx = sec * 2 + ring;

                    if buckets[bidx].map_or(true, |b| pre_score > b.0) {
                        buckets[bidx] = Some((pre_score, cur_g as f64, idx, pt, is_edge));
                    }
                }
            }

            for dj in -1_isize..=1 {
                let nj = cj + dj;
                if nj < 0 || nj >= h as isize {
                    continue;
                }
                let nju = nj as usize;
                let cju = cj as usize;
                for di in -1_isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ni = ci + di;
                    if ni < 0 || ni >= w as isize {
                        continue;
                    }
                    let niu = ni as usize;
                    let nidx = nju * w + niu;
                    if self.walk[nidx] == 0 {
                        continue;
                    }
                    if di != 0 && dj != 0 {
                        let ciu = ci as usize;
                        if self.walk[cju * w + niu] == 0 || self.walk[nju * w + ciu] == 0 {
                            continue;
                        }
                    }
                    let ng = cur_g + if di != 0 && dj != 0 { STEP_DIAG } else { STEP_CARD };
                    if ng > MAX_WALK_M {
                        continue;
                    }
                    let visited = self.stamp[nidx] == stamp;
                    if !visited || ng < self.g_score[nidx] {
                        self.stamp[nidx] = stamp;
                        self.g_score[nidx] = ng;
                        self.parent[nidx] = idx as u32;
                        open.push(HeapNode { f: ng, idx: nidx });
                    }
                }
            }
        }

        let mut best: Option<(f64, usize, [f64; 2], usize)> = None;
        for slot in buckets.iter().flatten() {
            let &(_, walk_dist, cell_idx, pt, is_edge) = slot;
            let gz = world.terrain_height(pt[0], pt[1]);
            let cand_eye = [pt[0], pt[1], gz + my_eye_off];

            let mut los_count = 0_usize;
            let mut min_threat_d = f64::INFINITY;
            for &te in threat_eyes {
                let td = (pt[0] - te[0]).hypot(pt[1] - te[1]);
                min_threat_d = min_threat_d.min(td);
                if world.raycast_solid(te, cand_eye).is_none() {
                    los_count += 1;
                }
            }

            let mut score = if los_count == 0 {
                260.0
            } else if los_count == 1 {
                -25.0
            } else {
                -(los_count as f64) * 130.0
            };

            let dist_me = (pt[0] - x).hypot(pt[1] - y).max(1.0);
            let dir_dot = ((pt[0] - x) * escape_dir[0] + (pt[1] - y) * escape_dir[1]) / dist_me;
            score += dir_dot * 42.0;
            score += min_threat_d.min(90.0) * 1.15;
            score -= walk_dist * 1.85;
            if is_edge {
                score += 15.0;
            }
            if (pt[0] - next_center[0]).hypot(pt[1] - next_center[1]) > safe_next_r {
                score -= 22.0;
            }

            if best.map_or(true, |(bs, _, _, _)| score > bs) {
                best = Some((score, cell_idx, pt, los_count));
            }
        }

        let (_, best_idx, best_pt, best_los) = best?;
        let mut rev = Vec::with_capacity(32);
        let mut cur = best_idx;
        let mut guard = 0_usize;
        while cur != u32::MAX as usize && guard < 128 {
            rev.push(self.cell_world(cur));
            let p = self.parent[cur];
            if p == u32::MAX {
                break;
            }
            cur = p as usize;
            guard += 1;
        }
        rev.reverse();
        if rev.len() > 1 {
            rev.remove(0);
        }
        Some((best_pt, rev, best_los))
    }

    /// Nearest cover-edge walkable cell inside `next_zone` for safe idle positioning.
    pub fn find_cover_anchor(
        &self,
        x: f64,
        y: f64,
        next_center: [f64; 2],
        next_radius: f64,
        max_dist: f64,
    ) -> Option<[f64; 2]> {
        let c = self.cell_of(x, y);
        let ci = (c % self.nnx) as isize;
        let cj = (c / self.nnx) as isize;
        let r_cells = (max_dist * 0.5).ceil() as isize;
        let safe_r = (next_radius - 10.0).max(5.0);

        let mut best: Option<(f64, [f64; 2])> = None;
        for dj in (-r_cells..=r_cells).step_by(3) {
            let nj = cj + dj;
            if nj <= 3 || nj >= (self.nny as isize - 3) {
                continue;
            }
            for di in (-r_cells..=r_cells).step_by(3) {
                let ni = ci + di;
                if ni <= 3 || ni >= (self.nnx as isize - 3) {
                    continue;
                }
                let k = nj as usize * self.nnx + ni as usize;
                if self.walk[k] == 0 || self.near_cover[k] == 0 {
                    continue;
                }
                let pt = self.cell_world(k);
                let d_next = (pt[0] - next_center[0]).hypot(pt[1] - next_center[1]);
                if d_next > safe_r {
                    continue;
                }
                let d_me = (pt[0] - x).hypot(pt[1] - y);
                if d_me > max_dist {
                    continue;
                }
                let score = -d_me * 1.0 - d_next * 0.25;
                if best.map_or(true, |(bs, _)| score > bs) {
                    best = Some((score, pt));
                }
            }
        }
        best.map(|(_, pt)| pt)
    }
}