use crate::world::World;
use std::f64::consts::PI;
use std::time::Instant;

const INF: f32 = 1e9;

pub struct NavEngine {
    min_x: f64,
    min_y: f64,
    nnx: usize,
    nny: usize,
    walk: Vec<u8>,
    dist: Vec<f32>,
    work: Vec<f32>,
    hk: Vec<f32>,
    hv: Vec<u32>,
    goal_key: Option<(i32, i32)>,
    pending_key: Option<(i32, i32)>,
    pending_goal: [f64; 2],
    ready_goal: Option<[f64; 2]>,
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

        Self {
            min_x,
            min_y,
            nnx,
            nny,
            walk,
            dist: vec![INF; total],
            work: vec![INF; total],
            hk: Vec::with_capacity(total),
            hv: Vec::with_capacity(total),
            goal_key: None,
            pending_key: None,
            pending_goal: [0.0, 0.0],
            ready_goal: None,
        }
    }

    #[inline]
    pub fn cell_of(&self, x: f64, y: f64) -> usize {
        let i = (((x - self.min_x) * 0.5).round() as isize)
            .clamp(0, self.nnx as isize - 1) as usize;
        let j = (((y - self.min_y) * 0.5).round() as isize)
            .clamp(0, self.nny as isize - 1) as usize;
        j * self.nnx + i
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

    #[inline]
    pub fn is_pending(&self) -> bool {
        self.pending_key.is_some()
    }

    pub fn field(&mut self, gx: f64, gy: f64) {
        let key = ((gx / 5.0).round() as i32, (gy / 5.0).round() as i32);
        if self.pending_key == Some(key) {
            return;
        }
        if self.pending_key.is_none() && self.goal_key == Some(key) {
            return;
        }
        if self.pending_key.is_some()
            && (gx - self.pending_goal[0]).hypot(gy - self.pending_goal[1]) < 25.0
        {
            return;
        }

        self.work.fill(INF);
        self.hk.clear();
        self.hv.clear();

        let mut g = self.cell_of(gx, gy);
        if self.walk[g] == 0 {
            let w = self.nnx as isize;
            let h = self.nny as isize;
            let gi = (g % self.nnx) as isize;
            let gj = (g / self.nnx) as isize;
            let mut best: Option<usize> = None;
            let mut best_d2 = isize::MAX;
            for r in 1..30_isize {
                for dj in -r..=r {
                    for di in -r..=r {
                        let i = gi + di;
                        let j = gj + dj;
                        if i < 0 || j < 0 || i >= w || j >= h {
                            continue;
                        }
                        let k = j as usize * self.nnx + i as usize;
                        let d2 = di * di + dj * dj;
                        if self.walk[k] != 0 && d2 < best_d2 {
                            best_d2 = d2;
                            best = Some(k);
                        }
                    }
                }
                if best.is_some() {
                    break;
                }
            }
            if let Some(bk) = best {
                g = bk;
            }
        }

        self.work[g] = 0.0;
        self.push(g as u32, 0.0);
        self.pending_key = Some(key);
        self.pending_goal = [gx, gy];
    }

    #[inline]
    fn push(&mut self, k: u32, v: f32) {
        let mut i = self.hk.len();
        self.hk.push(v);
        self.hv.push(k);
        while i > 0 {
            let p = (i - 1) >> 1;
            if self.hk[p] <= v {
                break;
            }
            self.hk[i] = self.hk[p];
            self.hv[i] = self.hv[p];
            i = p;
        }
        self.hk[i] = v;
        self.hv[i] = k;
    }

    #[inline]
    fn pop(&mut self) -> (f32, u32) {
        let top_v = self.hk[0];
        let top_k = self.hv[0];
        let lv = self.hk.pop().unwrap();
        let lk = self.hv.pop().unwrap();
        let n = self.hk.len();
        if n > 0 {
            let mut i = 0;
            loop {
                let mut c = 2 * i + 1;
                if c >= n {
                    break;
                }
                if c + 1 < n && self.hk[c + 1] < self.hk[c] {
                    c += 1;
                }
                if self.hk[c] >= lv {
                    break;
                }
                self.hk[i] = self.hk[c];
                self.hv[i] = self.hv[c];
                i = c;
            }
            self.hk[i] = lv;
            self.hv[i] = lk;
        }
        (top_v, top_k)
    }

    pub fn pump(&mut self, budget_ms: f64, max_iter: usize) {
        if self.pending_key.is_none() || budget_ms <= 0.0 {
            return;
        }
        let w = self.nnx;
        let h = self.nny;
        let sqrt2 = std::f32::consts::SQRT_2;
        let t0 = Instant::now();
        let mut iter = 0_usize;

        while !self.hk.is_empty() {
            iter += 1;
            if iter > max_iter {
                return;
            }
            if (iter & 1023) == 0 && t0.elapsed().as_secs_f64() * 1000.0 > budget_ms {
                return;
            }
            let (v0, k_u32) = self.pop();
            let k = k_u32 as usize;
            if v0 > self.work[k] {
                continue;
            }
            let i = (k % w) as isize;
            let j = (k / w) as isize;

            for dj in -1_isize..=1 {
                let jj = j + dj;
                if jj < 0 || jj >= h as isize {
                    continue;
                }
                let jju = jj as usize;
                let ju = j as usize;
                for di in -1_isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ii = i + di;
                    if ii < 0 || ii >= w as isize {
                        continue;
                    }
                    let iiu = ii as usize;
                    let kk = jju * w + iiu;
                    if self.walk[kk] == 0 {
                        continue;
                    }
                    if di != 0 && dj != 0 {
                        let iu = i as usize;
                        if self.walk[ju * w + iiu] == 0 || self.walk[jju * w + iu] == 0 {
                            continue;
                        }
                    }
                    let nv = v0 + if di != 0 && dj != 0 { sqrt2 } else { 1.0 };
                    if nv < self.work[kk] {
                        self.work[kk] = nv;
                        self.push(kk as u32, nv);
                    }
                }
            }
        }

        std::mem::swap(&mut self.dist, &mut self.work);
        self.goal_key = self.pending_key.take();
        self.ready_goal = Some(self.pending_goal);
    }

    pub fn field_for(&self, gx: f64, gy: f64, tol: f64) -> bool {
        self.ready_goal
            .map_or(false, |rg| (gx - rg[0]).hypot(gy - rg[1]) < tol)
    }

    pub fn steer(&self, x: f64, y: f64) -> Option<[f64; 2]> {
        let w = self.nnx;
        let h = self.nny;
        let c = self.cell_of(x, y);
        let ci = (c % w) as isize;
        let cj = (c / w) as isize;
        let start_blocked = self.walk[c] == 0;
        let mut best = self.dist[c];
        let mut bi = 0_isize;
        let mut bj = 0_isize;

        for a in 0..24 {
            let ang = (a as f64 / 24.0) * 2.0 * PI;
            let ca = ang.cos();
            let sa = ang.sin();
            for r in 1..=5 {
                let i = ci + (ca * r as f64).round() as isize;
                let j = cj + (sa * r as f64).round() as isize;
                if i < 0 || j < 0 || i >= w as isize || j >= h as isize {
                    break;
                }
                let k = j as usize * w + i as usize;
                if self.walk[k] == 0 {
                    if start_blocked && r <= 2 {
                        continue;
                    }
                    break;
                }
                let v = self.dist[k] + r as f32 * 0.02;
                if v < best {
                    best = v;
                    bi = i - ci;
                    bj = j - cj;
                }
            }
        }

        if bi == 0 && bj == 0 {
            return None;
        }
        let l = ((bi * bi + bj * bj) as f64).sqrt();
        Some([bi as f64 / l, bj as f64 / l])
    }

    /// Nearby walkable in-zone point that breaks 3D solid LOS from `primary_eye`
    /// (and prefers breaking LOS from all active `threat_eyes`).
    pub fn cover_from(
        &self,
        x: f64,
        y: f64,
        my_eye_off: f64,
        threat_eyes: &[[f64; 3]],
        primary_eye: [f64; 3],
        max_r: f64,
        zone_center: [f64; 2],
        zone_radius: f64,
        world: &World,
    ) -> Option<[f64; 2]> {
        let safe_r = (zone_radius - 5.0).max(4.0);
        let mut best: Option<(f64, [f64; 2])> = None;
        let mut r = 6.0;
        while r <= max_r {
            for a in 0..16 {
                let ang = (a as f64 / 16.0) * 2.0 * PI;
                let px = x + ang.cos() * r;
                let py = y + ang.sin() * r;
                if self.walk[self.cell_of(px, py)] == 0 {
                    continue;
                }
                if (px - zone_center[0]).hypot(py - zone_center[1]) > safe_r {
                    continue;
                }
                let pz = world.terrain_height(px, py);
                let cand_eye = [px, py, pz + my_eye_off.min(7.0)];
                if world.raycast_solid(primary_eye, cand_eye).is_none() {
                    continue;
                }
                let extra_los = threat_eyes
                    .iter()
                    .filter(|&&te| world.raycast_solid(te, cand_eye).is_none())
                    .count();
                let away = (px - primary_eye[0]).hypot(py - primary_eye[1])
                    - (x - primary_eye[0]).hypot(y - primary_eye[1]);
                let s = r - away * 0.3 + extra_los as f64 * 25.0;
                if best.map_or(true, |(bs, _)| s < bs) {
                    best = Some((s, [px, py]));
                }
            }
            r += 6.0;
        }
        best.map(|(_, pt)| pt)
    }
}