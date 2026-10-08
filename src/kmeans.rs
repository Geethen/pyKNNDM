//! Port of R's Hartigan-Wong `kmns.f` (KMNS / OPTRA / QTRAN), R 4.5.0.
//! `a` is row-major m x n, `c` row-major k x n (initial centres).

// Fortran `DATA BIG /1.E30/` is a single-precision literal.
const BIG: f64 = 1.0e30f32 as f64;

pub struct KmOut {
    pub cluster: Vec<i64>,
    pub ifault: i32,
}

pub fn kmns(a: &[f64], m: usize, n: usize, c: &mut [f64], k: usize, iter: usize, max_qtr: i64) -> KmOut {
    if k <= 1 || k >= m {
        return KmOut { cluster: vec![], ifault: 3 };
    }
    let mut ic1 = vec![0usize; m];
    let mut ic2 = vec![0usize; m];
    for i in 0..m {
        ic1[i] = 0;
        ic2[i] = 1;
        let mut dt = [0.0f64; 2];
        for il in 0..2 {
            let mut s = 0.0;
            for j in 0..n {
                let da = a[i * n + j] - c[il * n + j];
                s += da * da;
            }
            dt[il] = s;
        }
        if dt[0] > dt[1] {
            ic1[i] = 1;
            ic2[i] = 0;
            dt.swap(0, 1);
        }
        'l: for l in 2..k {
            let mut db = 0.0;
            for j in 0..n {
                let dc = a[i * n + j] - c[l * n + j];
                db += dc * dc;
                if db >= dt[1] {
                    continue 'l;
                }
            }
            if db >= dt[0] {
                dt[1] = db;
                ic2[i] = l;
            } else {
                dt[1] = dt[0];
                ic2[i] = ic1[i];
                dt[0] = db;
                ic1[i] = l;
            }
        }
    }
    let mut nc = vec![0i64; k];
    c.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..m {
        let l = ic1[i];
        nc[l] += 1;
        for j in 0..n {
            c[l * n + j] += a[i * n + j];
        }
    }
    let mut an1 = vec![0.0; k];
    let mut an2 = vec![0.0; k];
    let mut itran = vec![0i32; k];
    let mut ncp = vec![0i64; k];
    for l in 0..k {
        if nc[l] == 0 {
            return KmOut { cluster: vec![], ifault: 1 };
        }
        let aa = nc[l] as f64;
        for j in 0..n {
            c[l * n + j] /= aa;
        }
        an2[l] = aa / (aa + 1.0);
        an1[l] = BIG;
        if aa > 1.0 {
            an1[l] = aa / (aa - 1.0);
        }
        itran[l] = 1;
        ncp[l] = -1;
    }
    let mut d = vec![0.0f64; m];
    let mut live = vec![0i64; k];
    let mut indx: usize = 0;
    let mut max_qtr = max_qtr;
    let mut ifault = 2;
    let mi = m as i64;
    for _ in 0..iter {
        // ---------------- OPTRA ----------------
        for l in 0..k {
            if itran[l] == 1 {
                live[l] = mi + 1;
            }
        }
        let mut early = false;
        for i0 in 0..m {
            let ii = i0 as i64 + 1;
            indx += 1;
            let l1 = ic1[i0];
            let mut l2 = ic2[i0];
            let ll = l2;
            if nc[l1] != 1 {
                if ncp[l1] != 0 {
                    let mut de = 0.0;
                    for j in 0..n {
                        let df = a[i0 * n + j] - c[l1 * n + j];
                        de += df * df;
                    }
                    d[i0] = de * an1[l1];
                }
                let mut da = 0.0;
                for j in 0..n {
                    let db = a[i0 * n + j] - c[l2 * n + j];
                    da += db * db;
                }
                let mut r2 = da * an2[l2];
                'l: for l in 0..k {
                    if (ii >= live[l1] && ii >= live[l]) || l == l1 || l == ll {
                        continue;
                    }
                    let rr = r2 / an2[l];
                    let mut dc = 0.0;
                    for j in 0..n {
                        let dd = a[i0 * n + j] - c[l * n + j];
                        dc += dd * dd;
                        if dc >= rr {
                            continue 'l;
                        }
                    }
                    r2 = dc * an2[l];
                    l2 = l;
                }
                if r2 >= d[i0] {
                    ic2[i0] = l2;
                } else {
                    indx = 0;
                    live[l1] = mi + ii;
                    live[l2] = mi + ii;
                    ncp[l1] = ii;
                    ncp[l2] = ii;
                    transfer(a, c, n, i0, l1, l2, &mut nc, &mut an1, &mut an2);
                    ic1[i0] = l2;
                    ic2[i0] = l1;
                }
            }
            if indx == m {
                early = true;
                break;
            }
        }
        if early {
            ifault = 0;
            break;
        }
        for l in 0..k {
            itran[l] = 0;
            live[l] -= mi;
        }
        // ---------------- QTRAN ----------------
        let mut icoun: usize = 0;
        let mut istep: i64 = 0;
        let mut qdone = false;
        while !qdone {
            for i0 in 0..m {
                icoun += 1;
                istep += 1;
                if istep >= max_qtr {
                    max_qtr = -1;
                    qdone = true;
                    break;
                }
                let l1 = ic1[i0];
                let l2 = ic2[i0];
                if nc[l1] != 1 {
                    if istep <= ncp[l1] {
                        let mut da = 0.0;
                        for j in 0..n {
                            let db = a[i0 * n + j] - c[l1 * n + j];
                            da += db * db;
                        }
                        d[i0] = da * an1[l1];
                    }
                    if istep < ncp[l1] || istep < ncp[l2] {
                        let r2 = d[i0] / an2[l2];
                        let mut dd = 0.0;
                        let mut moved = true;
                        for j in 0..n {
                            let de = a[i0 * n + j] - c[l2 * n + j];
                            dd += de * de;
                            if dd >= r2 {
                                moved = false;
                                break;
                            }
                        }
                        if moved {
                            icoun = 0;
                            indx = 0;
                            itran[l1] = 1;
                            itran[l2] = 1;
                            ncp[l1] = istep + mi;
                            ncp[l2] = istep + mi;
                            transfer(a, c, n, i0, l1, l2, &mut nc, &mut an1, &mut an2);
                            ic1[i0] = l2;
                            ic2[i0] = l1;
                        }
                    }
                }
                if icoun == m {
                    qdone = true;
                    break;
                }
            }
        }
        if max_qtr < 0 {
            ifault = 4;
            break;
        }
        if k == 2 {
            ifault = 0;
            break;
        }
        ncp.iter_mut().for_each(|v| *v = 0);
    }
    KmOut { cluster: ic1.iter().map(|&v| v as i64 + 1).collect(), ifault }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn transfer(a: &[f64], c: &mut [f64], n: usize, i0: usize, l1: usize, l2: usize, nc: &mut [i64], an1: &mut [f64], an2: &mut [f64]) {
    let al1 = nc[l1] as f64;
    let alw = al1 - 1.0;
    let al2 = nc[l2] as f64;
    let alt = al2 + 1.0;
    for j in 0..n {
        c[l1 * n + j] = (c[l1 * n + j] * al1 - a[i0 * n + j]) / alw;
        c[l2 * n + j] = (c[l2 * n + j] * al2 + a[i0 * n + j]) / alt;
    }
    nc[l1] -= 1;
    nc[l2] += 1;
    an2[l1] = alw / al1;
    an1[l1] = BIG;
    if alw > 1.0 {
        an1[l1] = alw / (alw - 1.0);
    }
    an1[l2] = alt / al2;
    an2[l2] = alt / (alt + 1.0);
}
