//! Nearest-neighbour distances. Every distance is computed exactly as R does
//! (FNN brute force / `stats::dist` / GEOS): `sqrt(sum_k (x_ik - x_jk)^2)`,
//! accumulated left to right from 0. Since `sqrt` is correctly rounded and
//! monotone, `min(sqrt(.)) == sqrt(min(.))`, so values match bit for bit.

use crate::kdtree::KdTree;
use crate::par::par_map;

#[inline]
pub fn sqdist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for k in 0..a.len() {
        let d = a[k] - b[k];
        s += d * d;
    }
    s
}

/// Distance from each query row to its nearest data row (`FNN::knnx.dist`, k = 1).
pub fn knnx_dist(query: &[f64], data: &[f64], p: usize) -> Vec<f64> {
    par_map(query.len() / p, |i| {
        let q = &query[i * p..(i + 1) * p];
        let mut best = f64::INFINITY;
        for x in data.chunks_exact(p) {
            let d = sqdist(x, q);
            if d < best {
                best = d;
            }
        }
        best.sqrt()
    })
}

/// Sorted neighbour lists (excluding self) truncated to `kk` entries.
/// Ties are broken by index so results are deterministic.
pub struct NeighbourLists {
    pub n: usize,
    pub p: usize,
    pub kk: usize,
    pub idx: Vec<u32>,
    pub d2: Vec<f64>,
    pub x: Vec<f64>,
}

impl NeighbourLists {
    pub fn build(x: Vec<f64>, n: usize, p: usize, kk: usize) -> Self {
        let kk = kk.min(n.saturating_sub(1)).max(1);
        let rows: Vec<(Vec<u32>, Vec<f64>)> = par_map(n, |i| {
                let xi = &x[i * p..(i + 1) * p];
                // bounded max-heap via sorted insertion (kk is small)
                let mut bi: Vec<u32> = Vec::with_capacity(kk + 1);
                let mut bd: Vec<f64> = Vec::with_capacity(kk + 1);
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let d = sqdist(&x[j * p..(j + 1) * p], xi);
                    if bd.len() == kk && d >= bd[kk - 1] {
                        continue;
                    }
                    let pos = bd.partition_point(|&v| v <= d);
                    bd.insert(pos, d);
                    bi.insert(pos, j as u32);
                    if bd.len() > kk {
                        bd.pop();
                        bi.pop();
                    }
                }
                (bi, bd)
            });
        let mut idx = Vec::with_capacity(n * kk);
        let mut d2 = Vec::with_capacity(n * kk);
        for (bi, bd) in rows {
            idx.extend_from_slice(&bi);
            d2.extend_from_slice(&bd);
        }
        NeighbourLists { n, p, kk, idx, d2, x }
    }

    /// `FNN::knn.dist(x, k = 1)`: distance to the nearest other point.
    pub fn nn_dist(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.d2[i * self.kk].sqrt()).collect()
    }

    /// For each point, the distance to the nearest point in a *different* fold
    /// (`CAST:::cv_distances` / `CAST:::distclust_distmat`).
    ///
    /// Most points find an other-fold neighbour in their cached list. The rest
    /// (fold interiors) query per-fold k-d trees of all other folds.
    pub fn cv_dist(&self, folds: &[i64]) -> Vec<f64> {
        let (n, p, kk) = (self.n, self.p, self.kk);
        let mut out = par_map(n, |i| {
            let fi = folds[i];
            for t in 0..kk {
                let j = self.idx[i * kk + t] as usize;
                if folds[j] != fi {
                    return self.d2[i * kk + t].sqrt();
                }
            }
            f64::NAN // resolved below
        });
        let misses: Vec<usize> = (0..n).filter(|&i| out[i].is_nan()).collect();
        if misses.is_empty() {
            return out;
        }
        // group points by fold label
        let mut labels: Vec<i64> = folds.to_vec();
        labels.sort_unstable();
        labels.dedup();
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); labels.len()];
        for (i, f) in folds.iter().enumerate() {
            let g = labels.binary_search(f).unwrap();
            members[g].push(i as u32);
        }
        let trees: Vec<KdTree> = members.into_iter().map(|m| KdTree::build(&self.x, p, m)).collect();
        let res = par_map(misses.len(), |t| {
            let i = misses[t];
            let gi = labels.binary_search(&folds[i]).unwrap();
            let q = &self.x[i * p..(i + 1) * p];
            // the cached list gives a valid upper bound: every listed point is
            // same-fold, so start from +inf and let the trees do the work
            let mut best = f64::INFINITY;
            for (g, tree) in trees.iter().enumerate() {
                if g != gi {
                    tree.search(q, &mut best);
                }
            }
            best.sqrt()
        });
        for (t, &i) in misses.iter().enumerate() {
            out[i] = res[t];
        }
        out
    }
}
