//! Rust kernels for the `knndm` Python package, exposed through a plain C ABI
//! (loaded with `ctypes`). Every function reproduces the corresponding R
//! computation bit for bit; see the module docs for the R sources.
//!
//! Conventions: matrices are row-major `f64`, labels are `i64`, return value
//! `0` means success. All pointers must be valid for the given lengths.

pub mod f80;
pub mod hclust;
pub mod kdtree;
pub mod kmeans;
pub mod nn;
pub mod par;

use f80::F80;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

/// Run `f`, converting a panic into the error code -1 (never unwind into C).
fn guard<F: FnOnce() -> i32>(f: F) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(-1)
}

fn guard_f64<F: FnOnce() -> f64>(f: F) -> f64 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(f64::NAN)
}

/// `twosamples::wass_stat(a, b)` with power = 1.
pub fn wass_stat(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64;
    let mut joint: Vec<(f64, bool)> = a.iter().map(|&v| (v, true)).chain(b.iter().map(|&v| (v, false))).collect();
    // R's order(): stable ascending (sort_by is stable)
    joint.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = joint.len() as f64;
    let nb = n - na;
    let diva = 1.0 / na;
    let divb = 1.0 / nb;
    let (mut ecur, mut fcur, mut out) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..joint.len().saturating_sub(1) {
        if joint[i].1 {
            ecur += diva;
        } else {
            fcur += divb;
        }
        let mut height = ecur - fcur;
        if height < 0.0 {
            height *= -1.0;
        }
        out += height * (joint[i + 1].0 - joint[i].0);
    }
    out
}

/// For each cluster level: `predict(prcomp, t(colMeans(x[labels == level, ])))`
/// with `colMeans` in 80-bit long double and the BLAS dgemv dot product.
pub fn cluster_pc1(x: &[f64], n: usize, p: usize, lab: &[i64], lev: &[i64], rot: &[f64], ctr: &[f64]) -> Vec<f64> {
    let maxlab = lev.iter().copied().max().unwrap_or(0).max(0) as usize;
    let mut pos = vec![usize::MAX; maxlab + 1];
    for (i, &l) in lev.iter().enumerate() {
        pos[l as usize] = i;
    }
    let mut sums = vec![F80::ZERO; lev.len() * p];
    let mut cnt = vec![0u64; lev.len()];
    for i in 0..n {
        let g = pos[lab[i] as usize];
        cnt[g] += 1;
        for j in 0..p {
            sums[g * p + j] = sums[g * p + j].add(F80::from_f64(x[i * p + j]));
        }
    }
    (0..lev.len())
        .map(|g| {
            let mut s = 0.0f64;
            for j in 0..p {
                let mean = sums[g * p + j].div_u64(cnt[g]).to_f64();
                s += (mean - ctr[j]) * rot[j];
            }
            s
        })
        .collect()
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        slice::from_raw_parts(p, n)
    }
}

unsafe fn sl_mut<'a, T>(p: *mut T, n: usize) -> &'a mut [T] {
    if n == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(p, n)
    }
}

/// Number of worker threads used by the parallel kernels.
#[no_mangle]
pub extern "C" fn knndm_threads() -> usize {
    par::threads()
}

/// # Safety
/// `a`/`b` must point to `na`/`nb` doubles.
#[no_mangle]
pub unsafe extern "C" fn knndm_wass_stat(a: *const f64, na: usize, b: *const f64, nb: usize) -> f64 {
    guard_f64(|| {
        wass_stat(sl(a, na), sl(b, nb))
    })
}

/// # Safety
/// `query` is m x p, `data` n x p, `out` has m slots.
#[no_mangle]
pub unsafe extern "C" fn knndm_knnx_dist(query: *const f64, m: usize, data: *const f64, n: usize, p: usize, out: *mut f64) -> i32 {
    guard(|| {
        let res = nn::knnx_dist(sl(query, m * p), sl(data, n * p), p);
        sl_mut(out, m).copy_from_slice(&res);
        0
    })
}

/// Build sorted neighbour lists for `x` (n x p). Free with `knndm_nnindex_free`.
///
/// # Safety
/// `x` must point to n * p doubles.
#[no_mangle]
pub unsafe extern "C" fn knndm_nnindex_new(x: *const f64, n: usize, p: usize, kk: usize) -> *mut nn::NeighbourLists {
    let v = sl(x, n * p).to_vec();
    match catch_unwind(AssertUnwindSafe(|| nn::NeighbourLists::build(v, n, p, kk))) {
        Ok(ix) => Box::into_raw(Box::new(ix)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `idx` must come from `knndm_nnindex_new` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn knndm_nnindex_free(idx: *mut nn::NeighbourLists) {
    if !idx.is_null() {
        drop(Box::from_raw(idx));
    }
}

/// `FNN::knn.dist(x, k = 1)`. # Safety: `out` has n slots.
#[no_mangle]
pub unsafe extern "C" fn knndm_nnindex_nn_dist(idx: *const nn::NeighbourLists, out: *mut f64) -> i32 {
    guard(|| {
        let ix = &*idx;
        sl_mut(out, ix.n).copy_from_slice(&ix.nn_dist());
        0
    })
}

/// Distance to the nearest point in another fold. # Safety: `folds`/`out` have n slots.
#[no_mangle]
pub unsafe extern "C" fn knndm_nnindex_cv_dist(idx: *const nn::NeighbourLists, folds: *const i64, out: *mut f64) -> i32 {
    guard(|| {
        let ix = &*idx;
        let res = ix.cv_dist(sl(folds, ix.n));
        sl_mut(out, ix.n).copy_from_slice(&res);
        0
    })
}

/// `hclust(dist(x), method)$merge` from coordinates `x` (n x p); `merge_out`
/// receives (n-1) rows of 2 (row-major).
///
/// # Safety
/// `x` has n * p doubles, `merge_out` 2 * (n - 1) slots.
#[no_mangle]
pub unsafe extern "C" fn knndm_hclust_merge(x: *const f64, n: usize, p: usize, iopt: i32, merge_out: *mut i64) -> i32 {
    guard(|| {
        if n < 2 {
            return 1;
        }
        let d = hclust::condensed_dist(sl(x, n * p), n, p);
        let merge = hclust::hclust(d, n, iopt);
        let out = sl_mut(merge_out, 2 * (n - 1));
        for (i, m) in merge.iter().enumerate() {
            out[2 * i] = m[0];
            out[2 * i + 1] = m[1];
        }
        0
    })
}

/// `cutree(tree, k)`. # Safety: `merge` has 2 * nm1 entries, `out` nm1 + 1.
#[no_mangle]
pub unsafe extern "C" fn knndm_cutree(merge: *const i64, nm1: usize, k: usize, out: *mut i64) -> i32 {
    guard(|| {
        if k < 1 || k > nm1 + 1 {
            return 1;
        }
        let m = sl(merge, 2 * nm1);
        let pairs: Vec<[i64; 2]> = (0..nm1).map(|i| [m[2 * i], m[2 * i + 1]]).collect();
        sl_mut(out, nm1 + 1).copy_from_slice(&hclust::cutree(&pairs, k));
        0
    })
}

/// Hartigan-Wong k-means from initial `centers` (k x p, overwritten).
/// Writes 1-based clusters to `out` and returns R's `ifault`.
///
/// # Safety
/// `x` has m * p doubles, `centers` k * p, `out` m slots.
#[no_mangle]
pub unsafe extern "C" fn knndm_kmeans_hw(
    x: *const f64,
    m: usize,
    p: usize,
    centers: *mut f64,
    k: usize,
    iter_max: usize,
    max_qtr: i64,
    out: *mut i64,
) -> i32 {
    guard(|| {
        let res = kmeans::kmns(sl(x, m * p), m, p, sl_mut(centers, k * p), k, iter_max, max_qtr);
        if !res.cluster.is_empty() {
            sl_mut(out, m).copy_from_slice(&res.cluster);
        }
        res.ifault
    })
}

/// See [`cluster_pc1`].
///
/// # Safety
/// Lengths: x n*p, labels n, levels nlev, rot/center p, out nlev.
#[no_mangle]
pub unsafe extern "C" fn knndm_cluster_pc1(
    x: *const f64,
    n: usize,
    p: usize,
    labels: *const i64,
    levels: *const i64,
    nlev: usize,
    rot: *const f64,
    center: *const f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        let res = cluster_pc1(sl(x, n * p), n, p, sl(labels, n), sl(levels, nlev), sl(rot, p), sl(center, p));
        sl_mut(out, nlev).copy_from_slice(&res);
        0
    })
}

/// `mean(x)` style long-double sum / n (exposed for testing the F80 emulation).
///
/// # Safety
/// `x` has n doubles.
#[no_mangle]
pub unsafe extern "C" fn knndm_ld_mean(x: *const f64, n: usize) -> f64 {
    guard_f64(|| {
        let mut acc = F80::ZERO;
        for &v in sl(x, n) {
            acc = acc.add(F80::from_f64(v));
        }
        acc.div_u64(n as u64).to_f64()
    })
}

/// Long-double sum (exposed for testing).
///
/// # Safety
/// `x` has n doubles.
#[no_mangle]
pub unsafe extern "C" fn knndm_ld_sum(x: *const f64, n: usize) -> f64 {
    guard_f64(|| {
        let mut acc = F80::ZERO;
        for &v in sl(x, n) {
            acc = acc.add(F80::from_f64(v));
        }
        acc.to_f64()
    })
}

/// `cumsum(x)` with R's long-double accumulator.
///
/// # Safety
/// `x` and `out` have n doubles.
#[no_mangle]
pub unsafe extern "C" fn knndm_ld_cumsum(x: *const f64, n: usize, out: *mut f64) -> i32 {
    guard(|| {
        let o = sl_mut(out, n);
        let mut acc = F80::ZERO;
        for (i, &v) in sl(x, n).iter().enumerate() {
            acc = acc.add(F80::from_f64(v));
            o[i] = acc.to_f64();
        }
        0
    })
}
