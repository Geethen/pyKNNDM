//! Ports of R's Fortran `HCLUST`/`HCASS2` and C `cutree` (R 4.5.0).

const INF: f64 = 1.0e300;

#[inline]
fn ioffst(n: usize, i: usize, j: usize) -> usize {
    // 1-based (i < j) -> 0-based index into the condensed vector
    j + (i - 1) * n - (i * (i + 1)) / 2 - 1
}

/// Condensed euclidean distances (`stats::dist` / GEOS order of operations).
pub fn condensed_dist(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * (n - 1) / 2];
    // Row i (0-based) holds the pairs (i, j > i) contiguously; give each
    // thread whole rows.
    let mut rows: Vec<&mut [f64]> = Vec::with_capacity(n);
    let mut rest: &mut [f64] = &mut d;
    for i in 0..n.saturating_sub(1) {
        let (a, b) = rest.split_at_mut(n - 1 - i);
        rows.push(a);
        rest = b;
    }
    let rows: Vec<std::sync::Mutex<&mut [f64]>> = rows.into_iter().map(std::sync::Mutex::new).collect();
    crate::par::par_map(rows.len(), |i0| {
        let mut row = rows[i0].lock().unwrap();
        let xi = &x[i0 * p..(i0 + 1) * p];
        for (t, out) in row.iter_mut().enumerate() {
            let j0 = i0 + 1 + t;
            let mut s = 0.0f64;
            for k in 0..p {
                let dv = xi[k] - x[j0 * p + k];
                s += dv * dv;
            }
            *out = s.sqrt();
        }
    });
    d
}

/// Returns R's `merge` matrix as (n-1) pairs.
pub fn hclust(mut diss: Vec<f64>, n: usize, iopt: i32) -> Vec<[i64; 2]> {
    let mut ia = vec![0usize; n + 1];
    let mut ib = vec![0usize; n + 1];
    let mut membr = vec![1.0f64; n + 1];
    let mut nn = vec![0usize; n + 1];
    let mut disnn = vec![0.0f64; n + 1];
    let mut flag = vec![true; n + 1];
    let (mut im, mut jj, mut jm) = (0usize, 0usize, 0usize);
    let mut ncl = n;
    if iopt == 8 {
        for v in diss.iter_mut() {
            *v = *v * *v;
        }
    }
    for i in 1..n {
        let mut dmin = INF;
        for j in (i + 1)..=n {
            let v = diss[ioffst(n, i, j)];
            if dmin > v {
                dmin = v;
                jm = j;
            }
        }
        nn[i] = jm;
        disnn[i] = dmin;
    }
    let is_ward = iopt == 1 || iopt == 8;
    loop {
        let mut dmin = INF;
        for i in 1..n {
            if flag[i] && disnn[i] < dmin {
                dmin = disnn[i];
                im = i;
                jm = nn[i];
            }
        }
        ncl -= 1;
        let i2 = im.min(jm);
        let j2 = im.max(jm);
        ia[n - ncl] = i2;
        ib[n - ncl] = j2;
        flag[j2] = false;
        dmin = INF;
        let d12 = diss[ioffst(n, i2, j2)];
        for k in 1..=n {
            if flag[k] && k != i2 {
                let ind1 = if i2 < k { ioffst(n, i2, k) } else { ioffst(n, k, i2) };
                let ind2 = if j2 < k { ioffst(n, j2, k) } else { ioffst(n, k, j2) };
                let (d1, d2) = (diss[ind1], diss[ind2]);
                let v = if is_ward {
                    let t = (membr[i2] + membr[k]) * d1 + (membr[j2] + membr[k]) * d2 - membr[k] * d12;
                    t / (membr[i2] + membr[j2] + membr[k])
                } else {
                    match iopt {
                        2 => if d2 < d1 { d2 } else { d1 }, // Fortran MIN
                        3 => if d2 > d1 { d2 } else { d1 }, // Fortran MAX
                        4 => (membr[i2] * d1 + membr[j2] * d2) / (membr[i2] + membr[j2]),
                        5 => (d1 + d2) / 2.0,
                        6 => ((d1 + d2) - d12 / 2.0) / 2.0,
                        _ => {
                            (membr[i2] * d1 + membr[j2] * d2 - membr[i2] * membr[j2] * d12 / (membr[i2] + membr[j2]))
                                / (membr[i2] + membr[j2])
                        }
                    }
                };
                diss[ind1] = v;
                if i2 < k {
                    if v < dmin {
                        dmin = v;
                        jj = k;
                    }
                } else if v < disnn[k] {
                    disnn[k] = v;
                    nn[k] = i2;
                }
            }
        }
        membr[i2] += membr[j2];
        disnn[i2] = dmin;
        nn[i2] = jj;
        for i in 1..n {
            if flag[i] && (nn[i] == i2 || nn[i] == j2) {
                let mut dm = INF;
                let base = ioffst(n, i, i + 1) - (i + 1);
                for j in (i + 1)..=n {
                    if flag[j] {
                        let v = diss[base + j];
                        if v < dm {
                            dm = v;
                            jj = j;
                        }
                    }
                }
                nn[i] = jj;
                disnn[i] = dm;
            }
        }
        if ncl <= 1 {
            break;
        }
    }
    hcass2(n, &ia, &ib)
}

fn hcass2(n: usize, ia: &[usize], ib: &[usize]) -> Vec<[i64; 2]> {
    let mut iia: Vec<i64> = ia.iter().map(|&v| v as i64).collect();
    let mut iib: Vec<i64> = ib.iter().map(|&v| v as i64).collect();
    // Map each observation to the most recent merge step that absorbed it
    // (equivalent to the O(n^2) relabelling loop of HCASS2).
    let mut last = vec![0i64; n + 1];
    for i in 1..n {
        let (a, b) = (ia[i], ib[i]);
        iia[i] = if last[a] > 0 { last[a] } else { -(a as i64) };
        iib[i] = if last[b] > 0 { last[b] } else { -(b as i64) };
        // after merge i the cluster is represented by min(a, b) = a (ia < ib)
        last[a.min(b)] = i as i64;
    }
    let mut out = Vec::with_capacity(n - 1);
    for i in 1..n {
        let (mut a, mut b) = (iia[i], iib[i]);
        if a > 0 && b < 0 {
            std::mem::swap(&mut a, &mut b);
        }
        if a > 0 && b > 0 {
            let (k1, k2) = (a.min(b), a.max(b));
            a = k1;
            b = k2;
        }
        out.push([a, b]);
    }
    out
}

/// `cutree(tree, k)`: clusters numbered by first appearance of each group.
pub fn cutree(merge: &[[i64; 2]], k: usize) -> Vec<i64> {
    let n = merge.len() + 1;
    if k >= n {
        return (1..=n as i64).collect();
    }
    // union-find over the first n-k merges
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut rep = vec![0usize; n]; // merge step -> an observation in it
    for (step, m) in merge.iter().take(n - k).enumerate() {
        let a = if m[0] < 0 { (-m[0] - 1) as usize } else { rep[(m[0] - 1) as usize] };
        let b = if m[1] < 0 { (-m[1] - 1) as usize } else { rep[(m[1] - 1) as usize] };
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        parent[rb] = ra;
        rep[step] = a;
    }
    let mut label = vec![0i64; n];
    let mut ans = vec![0i64; n];
    let mut next = 0i64;
    for l in 0..n {
        let r = find(&mut parent, l);
        if label[r] == 0 {
            next += 1;
            label[r] = next;
        }
        ans[l] = label[r];
    }
    ans
}
