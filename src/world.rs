use serde::Deserialize;
use std::fs;
use std::path::Path;

const CELL_SIZE: f64 = 16.0;

#[derive(Debug, Deserialize)]
struct WorldJson {
    terrain: TerrainDesc,
    #[serde(default)]
    prisms: Vec<RawPrism>,
    #[serde(default)]
    trees: Option<RawTrees>,
}

#[derive(Debug, Deserialize)]
struct TerrainDesc {
    nx: usize,
    ny: usize,
    #[serde(rename = "minX", alias = "min_x")]
    min_x: f64,
    #[serde(rename = "minY", alias = "min_y")]
    min_y: f64,
    step: f64,
}

#[derive(Debug, Deserialize)]
struct RawPrism {
    kind: String,
    base: f64,
    top: f64,
    ring: Vec<f64>,
    #[serde(default)]
    holes: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct RawTrees {
    fields: Vec<String>,
    data: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Prism {
    pub kind: String,
    pub base: f64,
    pub top: f64,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub ring: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Tree {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub height: f64,
    pub crown_r: f64,
    pub crown_base: f64,
    pub trunk_r: f64,
}

pub struct World {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub step: f64,
    pub nx: usize,
    pub ny: usize,
    heights: Vec<f32>,
    max_terrain_z: f64,
    pub prisms: Vec<Prism>,
    pub trees: Vec<Tree>,
    grid_nx: usize,
    grid_ny: usize,
    grid: Vec<Vec<u32>>,
}

impl World {
    pub fn load(map_dir: &str, files: &[String]) -> Result<Self, String> {
        let base = Path::new(map_dir);
        let world_rel = files
            .iter()
            .find(|f| f.ends_with("world.json"))
            .map(String::as_str)
            .unwrap_or("v2/world.json");
        let terrain_rel = files
            .iter()
            .find(|f| f.ends_with("terrain.i16"))
            .map(String::as_str)
            .unwrap_or("v2/terrain.i16");

        let world_bytes = fs::read(base.join(world_rel))
            .map_err(|e| format!("failed reading {world_rel}: {e}"))?;
        let wj: WorldJson = serde_json::from_slice(&world_bytes)
            .map_err(|e| format!("failed parsing {world_rel}: {e}"))?;

        let terrain_bytes = fs::read(base.join(terrain_rel))
            .map_err(|e| format!("failed reading {terrain_rel}: {e}"))?;
        let expected = wj.terrain.nx * wj.terrain.ny;
        if terrain_bytes.len() < expected * 2 {
            return Err("terrain.i16 shorter than expected".into());
        }
        let mut heights = Vec::with_capacity(expected);
        let mut max_terrain_z = f64::NEG_INFINITY;
        for chunk in terrain_bytes[..expected * 2].chunks_exact(2) {
            let cm = i16::from_le_bytes([chunk[0], chunk[1]]);
            let h = cm as f32 * 0.01;
            if (h as f64) > max_terrain_z {
                max_terrain_z = h as f64;
            }
            heights.push(h);
        }

        let min_x = wj.terrain.min_x;
        let min_y = wj.terrain.min_y;
        let max_x = min_x + (wj.terrain.nx.saturating_sub(1)) as f64 * wj.terrain.step;
        let max_y = min_y + (wj.terrain.ny.saturating_sub(1)) as f64 * wj.terrain.step;

        let grid_nx = (((max_x - min_x) / CELL_SIZE).ceil() as usize).max(1);
        let grid_ny = (((max_y - min_y) / CELL_SIZE).ceil() as usize).max(1);
        let mut grid = vec![Vec::new(); grid_nx * grid_ny];

        let mut prisms = Vec::with_capacity(wj.prisms.len());
        for rp in wj.prisms {
            let ring = flat_to_pts(&rp.ring);
            if ring.len() < 3 {
                continue;
            }
            let mut p_min_x = f64::INFINITY;
            let mut p_max_x = f64::NEG_INFINITY;
            let mut p_min_y = f64::INFINITY;
            let mut p_max_y = f64::NEG_INFINITY;
            for &[x, y] in &ring {
                p_min_x = p_min_x.min(x);
                p_max_x = p_max_x.max(x);
                p_min_y = p_min_y.min(y);
                p_max_y = p_max_y.max(y);
            }
            let holes = rp.holes.iter().map(|h| flat_to_pts(h)).collect();
            let idx = prisms.len() as u32;
            let gx0 = (((p_min_x - min_x) / CELL_SIZE).floor() as isize)
                .clamp(0, grid_nx as isize - 1) as usize;
            let gx1 = (((p_max_x - min_x) / CELL_SIZE).floor() as isize)
                .clamp(0, grid_nx as isize - 1) as usize;
            let gy0 = (((p_min_y - min_y) / CELL_SIZE).floor() as isize)
                .clamp(0, grid_ny as isize - 1) as usize;
            let gy1 = (((p_max_y - min_y) / CELL_SIZE).floor() as isize)
                .clamp(0, grid_ny as isize - 1) as usize;
            for gy in gy0..=gy1 {
                for gx in gx0..=gx1 {
                    grid[gy * grid_nx + gx].push(idx);
                }
            }
            prisms.push(Prism {
                kind: rp.kind,
                base: rp.base,
                top: rp.top,
                min_x: p_min_x,
                max_x: p_max_x,
                min_y: p_min_y,
                max_y: p_max_y,
                ring,
                holes,
            });
        }

        let trees = parse_trees(wj.trees);

        Ok(Self {
            min_x,
            min_y,
            max_x,
            max_y,
            step: wj.terrain.step,
            nx: wj.terrain.nx,
            ny: wj.terrain.ny,
            heights,
            max_terrain_z,
            prisms,
            trees,
            grid_nx,
            grid_ny,
            grid,
        })
    }

    #[inline]
    pub fn terrain_height(&self, x: f64, y: f64) -> f64 {
        let fx = ((x - self.min_x) / self.step).clamp(0.0, (self.nx - 1) as f64);
        let fy = ((y - self.min_y) / self.step).clamp(0.0, (self.ny - 1) as f64);
        let i0 = (fx.floor() as usize).min(self.nx - 1);
        let j0 = (fy.floor() as usize).min(self.ny - 1);
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let tx = fx - i0 as f64;
        let ty = fy - j0 as f64;

        let h00 = self.heights[j0 * self.nx + i0] as f64;
        let h10 = self.heights[j0 * self.nx + i1] as f64;
        let h01 = self.heights[j1 * self.nx + i0] as f64;
        let h11 = self.heights[j1 * self.nx + i1] as f64;

        let hx0 = h00 + (h10 - h00) * tx;
        let hx1 = h01 + (h11 - h01) * tx;
        hx0 + (hx1 - hx0) * ty
    }

    pub fn is_walkable_step(
        &self,
        pos: [f64; 3],
        nx: f64,
        ny: f64,
        radius: f64,
        height: f64,
        step_up: f64,
        max_slope_deg: f64,
    ) -> bool {
        if nx < self.min_x + radius
            || nx > self.max_x - radius
            || ny < self.min_y + radius
            || ny > self.max_y - radius
        {
            return false;
        }
        let gz = self.terrain_height(nx, ny);
        let mut support_z = gz;
        let mut on_prism = false;

        let gx0 = (((nx - radius - self.min_x) / CELL_SIZE).floor() as isize)
            .clamp(0, self.grid_nx as isize - 1) as usize;
        let gx1 = (((nx + radius - self.min_x) / CELL_SIZE).floor() as isize)
            .clamp(0, self.grid_nx as isize - 1) as usize;
        let gy0 = (((ny - radius - self.min_y) / CELL_SIZE).floor() as isize)
            .clamp(0, self.grid_ny as isize - 1) as usize;
        let gy1 = (((ny + radius - self.min_y) / CELL_SIZE).floor() as isize)
            .clamp(0, self.grid_ny as isize - 1) as usize;

        for gy in gy0..=gy1 {
            for gx in gx0..=gx1 {
                for &pid in &self.grid[gy * self.grid_nx + gx] {
                    let p = &self.prisms[pid as usize];
                    if nx + radius < p.min_x
                        || nx - radius > p.max_x
                        || ny + radius < p.min_y
                        || ny - radius > p.max_y
                    {
                        continue;
                    }
                    let overlaps = point_in_prism_2d(nx, ny, p)
                        || point_in_prism_2d(nx + radius, ny, p)
                        || point_in_prism_2d(nx - radius, ny, p)
                        || point_in_prism_2d(nx, ny + radius, p)
                        || point_in_prism_2d(nx, ny - radius, p);
                    if !overlaps {
                        continue;
                    }
                    if p.top <= pos[2] + step_up && p.top > support_z {
                        support_z = p.top;
                        on_prism = true;
                        continue;
                    }
                    if p.base >= pos[2] + height {
                        continue;
                    }
                    return false;
                }
            }
        }

        if on_prism {
            return (support_z - pos[2]) <= step_up;
        }

        let dx = nx - pos[0];
        let dy = ny - pos[1];
        let horiz = (dx * dx + dy * dy).sqrt();
        if horiz < 1e-4 {
            return (support_z - pos[2]) <= step_up;
        }

        let max_tan = max_slope_deg.to_radians().tan();
        if (gz - pos[2]) > (horiz * max_tan).max(step_up) {
            return false;
        }

        let ux = dx / horiz;
        let uy = dy / horiz;
        let steps = (horiz / 1.0).ceil().clamp(1.0, 8.0) as usize;
        let ds = horiz / steps as f64;
        let eps = 0.5 * self.step.max(0.5);
        let mut prev_z = pos[2].max(self.terrain_height(pos[0], pos[1]));

        for s in 1..=steps {
            let f = s as f64 / steps as f64;
            let px = pos[0] + dx * f;
            let py = pos[1] + dy * f;
            let cur_z = self.terrain_height(px, py);
            let dz = cur_z - prev_z;

            if dz > 0.0 && (dz / ds) > max_tan {
                return false;
            }

            let dhdx = (self.terrain_height(px + eps, py) - self.terrain_height(px - eps, py))
                / (2.0 * eps);
            let dhdy = (self.terrain_height(px, py + eps) - self.terrain_height(px, py - eps))
                / (2.0 * eps);
            let grad = (dhdx * dhdx + dhdy * dhdy).sqrt();
            let dir_grad = dhdx * ux + dhdy * uy;
            if dir_grad > max_tan || (grad > max_tan && cur_z > prev_z - 0.05) {
                return false;
            }

            prev_z = cur_z;
        }

        true
    }

    /// Fast 3D raycast using 2D Amanatides–Woo DDA grid traversal + prism deduplication.
    /// Returns `Some(t)` in `[0.0, 1.0]` for the closest hit, or `None` if unobstructed.
    pub fn raycast_solid(&self, origin: [f64; 3], target: [f64; 3]) -> Option<f64> {
        let dir = [
            target[0] - origin[0],
            target[1] - origin[1],
            target[2] - origin[2],
        ];
        let dist_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if dist_sq < 1e-12 {
            return None;
        }
        let dist = dist_sq.sqrt();

        let mut best_t = 1.000_001_f64;
        let mut hit = false;

        // Stack-allocated dedup buffer so multi-cell prisms are tested only once
        let mut seen = [u32::MAX; 32];
        let mut seen_len = 0_usize;

        // 2D DDA over the 16 m spatial hash
        let fx = (origin[0] - self.min_x) / CELL_SIZE;
        let fy = (origin[1] - self.min_y) / CELL_SIZE;
        let mut cx = (fx.floor() as isize).clamp(0, self.grid_nx as isize - 1);
        let mut cy = (fy.floor() as isize).clamp(0, self.grid_ny as isize - 1);

        let end_cx = (((target[0] - self.min_x) / CELL_SIZE).floor() as isize)
            .clamp(0, self.grid_nx as isize - 1);
        let end_cy = (((target[1] - self.min_y) / CELL_SIZE).floor() as isize)
            .clamp(0, self.grid_ny as isize - 1);

        let step_x: isize = if dir[0] > 1e-9 { 1 } else if dir[0] < -1e-9 { -1 } else { 0 };
        let step_y: isize = if dir[1] > 1e-9 { 1 } else if dir[1] < -1e-9 { -1 } else { 0 };

        let t_delta_x = if step_x != 0 {
            (CELL_SIZE / dir[0]).abs()
        } else {
            f64::INFINITY
        };
        let t_delta_y = if step_y != 0 {
            (CELL_SIZE / dir[1]).abs()
        } else {
            f64::INFINITY
        };

        let mut t_max_x = if step_x > 0 {
            ((cx + 1) as f64 * CELL_SIZE + self.min_x - origin[0]) / dir[0]
        } else if step_x < 0 {
            (cx as f64 * CELL_SIZE + self.min_x - origin[0]) / dir[0]
        } else {
            f64::INFINITY
        };
        let mut t_max_y = if step_y > 0 {
            ((cy + 1) as f64 * CELL_SIZE + self.min_y - origin[1]) / dir[1]
        } else if step_y < 0 {
            (cy as f64 * CELL_SIZE + self.min_y - origin[1]) / dir[1]
        } else {
            f64::INFINITY
        };
        if t_max_x < 0.0 {
            t_max_x = 0.0;
        }
        if t_max_y < 0.0 {
            t_max_y = 0.0;
        }

        let max_steps = (end_cx - cx).unsigned_abs() + (end_cy - cy).unsigned_abs() + 2;
        let mut t_enter = 0.0_f64;

        for _ in 0..max_steps {
            if t_enter >= best_t {
                break;
            }
            let cell = &self.grid[cy as usize * self.grid_nx + cx as usize];
            for &pid in cell {
                if seen[..seen_len].contains(&pid) {
                    continue;
                }
                if seen_len < seen.len() {
                    seen[seen_len] = pid;
                    seen_len += 1;
                }
                if let Some(t) = ray_hits_prism(origin, dir, best_t, &self.prisms[pid as usize]) {
                    best_t = t;
                    hit = true;
                }
            }
            if cx == end_cx && cy == end_cy {
                break;
            }
            if t_max_x < t_max_y {
                t_enter = t_max_x;
                t_max_x += t_delta_x;
                cx += step_x;
            } else {
                t_enter = t_max_y;
                t_max_y += t_delta_y;
                cy += step_y;
            }
            if cx < 0 || cy < 0 || cx >= self.grid_nx as isize || cy >= self.grid_ny as isize {
                break;
            }
        }

        // Fast terrain check (skip completely if both ends are above max campus terrain height)
        if origin[2].min(target[2]) <= self.max_terrain_z + 0.1 {
            let max_seg = dist * best_t.min(1.0);
            let steps = (max_seg / 4.0).ceil().max(1.0) as usize;
            for s in 1..=steps {
                let t = (s as f64 / steps as f64) * best_t.min(1.0);
                let px = origin[0] + dir[0] * t;
                let py = origin[1] + dir[1] * t;
                let pz = origin[2] + dir[2] * t;
                if pz <= self.terrain_height(px, py) - 0.05 {
                    best_t = t;
                    hit = true;
                    break;
                }
            }
        }

        if hit { Some(best_t) } else { None }
    }
}

fn flat_to_pts(flat: &[f64]) -> Vec<[f64; 2]> {
    flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}

fn parse_trees(raw: Option<RawTrees>) -> Vec<Tree> {
    let Some(rt) = raw else { return Vec::new() };
    let stride = rt.fields.len();
    if stride == 0 {
        return Vec::new();
    }
    let idx = |name: &str| rt.fields.iter().position(|f| f == name);
    let (Some(ix), Some(iy), Some(iz)) = (idx("x"), idx("y"), idx("z")) else {
        return Vec::new();
    };
    let ih = idx("height");
    let icr = idx("crownR");
    let icb = idx("crownBase");
    let itr = idx("trunkR");

    let mut out = Vec::with_capacity(rt.data.len() / stride);
    for row in rt.data.chunks_exact(stride) {
        out.push(Tree {
            x: row[ix],
            y: row[iy],
            z: row[iz],
            height: ih.map(|i| row[i]).unwrap_or(10.0),
            crown_r: icr.map(|i| row[i]).unwrap_or(4.0),
            crown_base: icb.map(|i| row[i]).unwrap_or(3.0),
            trunk_r: itr.map(|i| row[i]).unwrap_or(0.4),
        });
    }
    out
}

pub fn point_in_ring(x: f64, y: f64, ring: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = ring.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (ring[i][0], ring[i][1]);
        let (xj, yj) = (ring[j][0], ring[j][1]);
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub fn point_in_prism_2d(x: f64, y: f64, p: &Prism) -> bool {
    if x < p.min_x || x > p.max_x || y < p.min_y || y > p.max_y {
        return false;
    }
    if !point_in_ring(x, y, &p.ring) {
        return false;
    }
    for hole in &p.holes {
        if hole.len() >= 3 && point_in_ring(x, y, hole) {
            return false;
        }
    }
    true
}

fn ray_hits_aabb_2d(
    ox: f64,
    oy: f64,
    dx: f64,
    dy: f64,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    t_max: f64,
) -> bool {
    let mut t0 = 0.0_f64;
    let mut t1 = t_max;
    for &(o, d, lo, hi) in &[(ox, dx, min_x, max_x), (oy, dy, min_y, max_y)] {
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return false;
            }
        } else {
            let inv = 1.0 / d;
            let mut a = (lo - o) * inv;
            let mut b = (hi - o) * inv;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t0 = t0.max(a);
            t1 = t1.min(b);
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

fn ray_hits_prism(o: [f64; 3], d: [f64; 3], max_t: f64, p: &Prism) -> Option<f64> {
    if !ray_hits_aabb_2d(o[0], o[1], d[0], d[1], p.min_x, p.max_x, p.min_y, p.max_y, max_t) {
        return None;
    }
    let z_min = o[2].min(o[2] + d[2] * max_t);
    let z_max = o[2].max(o[2] + d[2] * max_t);
    if z_max < p.base || z_min > p.top {
        return None;
    }
    if o[2] >= p.base && o[2] <= p.top && point_in_prism_2d(o[0], o[1], p) {
        return Some(0.0);
    }

    let mut best: Option<f64> = None;
    let mut update = |t: f64| {
        if t >= 0.0 && t < max_t && best.map_or(true, |b| t < b) {
            best = Some(t);
        }
    };

    if d[2].abs() > 1e-9 {
        for &z_plane in &[p.top, p.base] {
            let t = (z_plane - o[2]) / d[2];
            if t >= 0.0 && t < max_t {
                let hx = o[0] + d[0] * t;
                let hy = o[1] + d[1] * t;
                if point_in_prism_2d(hx, hy, p) {
                    update(t);
                }
            }
        }
    }

    let check_edges = |ring: &[[f64; 2]], update: &mut dyn FnMut(f64)| {
        let n = ring.len();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            let ex = b[0] - a[0];
            let ey = b[1] - a[1];
            let denom = d[0] * ey - d[1] * ex;
            if denom.abs() < 1e-12 {
                continue;
            }
            let rx = a[0] - o[0];
            let ry = a[1] - o[1];
            let t = (rx * ey - ry * ex) / denom;
            let u = (rx * d[1] - ry * d[0]) / denom;
            if t >= 0.0 && t < max_t && u >= 0.0 && u <= 1.0 {
                let hz = o[2] + d[2] * t;
                if hz >= p.base && hz <= p.top {
                    update(t);
                }
            }
        }
    };
    check_edges(&p.ring, &mut update);
    for hole in &p.holes {
        if hole.len() >= 3 {
            check_edges(hole, &mut update);
        }
    }
    best
}