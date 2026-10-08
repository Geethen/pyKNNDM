"""Verbatim port of ``stats::kmeans`` with the default Hartigan-Wong algorithm.

Source: R 4.5.0 ``src/library/stats/R/kmeans.R`` and ``src/kmns.f``.
"""

from __future__ import annotations

import numpy as np

from ._rrng import RRNG

# Fortran ``DATA BIG /1.E30/`` is a REAL (single precision) literal.
_BIG = float(np.float32(1.0e30))


class KMeansError(RuntimeError):
    pass


def _unique_rows_idx(x: np.ndarray) -> list[int]:
    seen = set()
    keep = []
    for i in range(x.shape[0]):
        key = tuple(float(v) + 0.0 for v in x[i])  # identical(): -0 == 0
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep


def kmeans(x: np.ndarray, centers: int, rng: RRNG, iter_max: int = 10, kmns=None) -> np.ndarray:
    """``stats::kmeans(x, centers)$cluster`` (nstart = 1, Hartigan-Wong).

    ``kmns(x, c, k, iter_max, max_qtr) -> (cluster, ifault)`` may be swapped
    for a compiled implementation; centre sampling always happens here.
    """
    kmns = kmns or _kmns
    x = np.ascontiguousarray(x, dtype=float)
    m, p = x.shape
    k = int(centers)
    if k > m:
        raise KMeansError("cannot take a sample larger than the population when 'replace = FALSE'")
    idx = rng.sample_int(m, k)
    c = x[[i - 1 for i in idx], :].copy()
    if len(_unique_rows_idx(c)) < k:  # any(duplicated(centers))
        uniq = _unique_rows_idx(x)
        if len(uniq) < k:
            raise KMeansError("more cluster centers than distinct data points.")
        idx = rng.sample_int(len(uniq), k)
        c = x[[uniq[i - 1] for i in idx], :].copy()
    if k == 1:
        raise NotImplementedError("k = 1 uses MacQueen in R; not needed by knndm")
    ic1, ifault = kmns(x, c, k, iter_max, min(2147483647, 50 * m))
    if ifault == 1:
        raise KMeansError("empty cluster: try a better set of initial centers")
    if ifault == 3:
        raise KMeansError("number of cluster centres must lie between 1 and nrow(x)")
    return ic1


def _kmns(a: np.ndarray, c: np.ndarray, k: int, iter_max: int, max_qtr: int):
    m, n = a.shape
    A = a.tolist()
    C = c.tolist()
    if k <= 1 or k >= m:
        return None, 3
    ic1 = [0] * m
    ic2 = [0] * m
    for i in range(m):
        ic1[i] = 0
        ic2[i] = 1
        dt = [0.0, 0.0]
        for il in range(2):
            s = 0.0
            for j in range(n):
                da = A[i][j] - C[il][j]
                s = s + da * da
            dt[il] = s
        if dt[0] > dt[1]:
            ic1[i] = 1
            ic2[i] = 0
            dt[0], dt[1] = dt[1], dt[0]
        for l in range(2, k):
            db = 0.0
            skip = False
            for j in range(n):
                dc = A[i][j] - C[l][j]
                db = db + dc * dc
                if db >= dt[1]:
                    skip = True
                    break
            if skip:
                continue
            if db >= dt[0]:
                dt[1] = db
                ic2[i] = l
            else:
                dt[1] = dt[0]
                ic2[i] = ic1[i]
                dt[0] = db
                ic1[i] = l
    nc = [0] * k
    for l in range(k):
        for j in range(n):
            C[l][j] = 0.0
    for i in range(m):
        l = ic1[i]
        nc[l] += 1
        for j in range(n):
            C[l][j] = C[l][j] + A[i][j]
    an1 = [0.0] * k
    an2 = [0.0] * k
    itran = [0] * k
    ncp = [0] * k
    for l in range(k):
        if nc[l] == 0:
            return None, 1
        aa = float(nc[l])
        for j in range(n):
            C[l][j] = C[l][j] / aa
        an2[l] = aa / (aa + 1.0)
        an1[l] = _BIG
        if aa > 1.0:
            an1[l] = aa / (aa - 1.0)
        itran[l] = 1
        ncp[l] = -1
    d = [0.0] * m
    live = [0] * k
    state = {"indx": 0, "max_qtr": max_qtr}
    ifault = 2
    for _ij in range(iter_max):
        _optra(A, m, n, C, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, state)
        if state["indx"] == m:
            ifault = 0
            break
        _qtran(A, m, n, C, k, ic1, ic2, nc, an1, an2, ncp, d, itran, state)
        if state["max_qtr"] < 0:
            ifault = 4
            break
        if k == 2:
            ifault = 0
            break
        for l in range(k):
            ncp[l] = 0
    # ifault 2 (no convergence) and 4 (Quick-TRANSfer limit) only warn in R.
    return np.array([v + 1 for v in ic1], dtype=np.int64), ifault


def _optra(A, m, n, C, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, st):
    # Indices: points i are 0-based, but the Fortran compares I with LIVE and
    # stores M+I / I in NCP, so we keep a 1-based ``I`` for those quantities.
    for l in range(k):
        if itran[l] == 1:
            live[l] = m + 1
    for i0 in range(m):
        I = i0 + 1
        st["indx"] += 1
        l1 = ic1[i0]
        l2 = ic2[i0]
        ll = l2
        if nc[l1] != 1:
            if ncp[l1] != 0:
                de = 0.0
                for j in range(n):
                    df = A[i0][j] - C[l1][j]
                    de = de + df * df
                d[i0] = de * an1[l1]
            da = 0.0
            for j in range(n):
                db = A[i0][j] - C[l2][j]
                da = da + db * db
            r2 = da * an2[l2]
            for l in range(k):
                if (I >= live[l1] and I >= live[l]) or l == l1 or l == ll:
                    continue
                rr = r2 / an2[l]
                dc = 0.0
                skip = False
                for j in range(n):
                    dd = A[i0][j] - C[l][j]
                    dc = dc + dd * dd
                    if dc >= rr:
                        skip = True
                        break
                if skip:
                    continue
                r2 = dc * an2[l]
                l2 = l
            if r2 >= d[i0]:
                ic2[i0] = l2
            else:
                st["indx"] = 0
                live[l1] = m + I
                live[l2] = m + I
                ncp[l1] = I
                ncp[l2] = I
                al1 = float(nc[l1])
                alw = al1 - 1.0
                al2 = float(nc[l2])
                alt = al2 + 1.0
                for j in range(n):
                    C[l1][j] = (C[l1][j] * al1 - A[i0][j]) / alw
                    C[l2][j] = (C[l2][j] * al2 + A[i0][j]) / alt
                nc[l1] -= 1
                nc[l2] += 1
                an2[l1] = alw / al1
                an1[l1] = _BIG
                if alw > 1.0:
                    an1[l1] = alw / (alw - 1.0)
                an1[l2] = alt / al2
                an2[l2] = alt / (alt + 1.0)
                ic1[i0] = l2
                ic2[i0] = l1
        if st["indx"] == m:
            return
    for l in range(k):
        itran[l] = 0
        live[l] = live[l] - m


def _qtran(A, m, n, C, k, ic1, ic2, nc, an1, an2, ncp, d, itran, st):
    icoun = 0
    istep = 0
    while True:
        for i0 in range(m):
            icoun += 1
            istep += 1
            if istep >= st["max_qtr"]:
                st["max_qtr"] = -1
                return
            l1 = ic1[i0]
            l2 = ic2[i0]
            if nc[l1] != 1:
                if istep <= ncp[l1]:
                    da = 0.0
                    for j in range(n):
                        db = A[i0][j] - C[l1][j]
                        da = da + db * db
                    d[i0] = da * an1[l1]
                if istep < ncp[l1] or istep < ncp[l2]:
                    r2 = d[i0] / an2[l2]
                    dd = 0.0
                    moved = True
                    for j in range(n):
                        de = A[i0][j] - C[l2][j]
                        dd = dd + de * de
                        if dd >= r2:
                            moved = False
                            break
                    if moved:
                        icoun = 0
                        st["indx"] = 0
                        itran[l1] = 1
                        itran[l2] = 1
                        ncp[l1] = istep + m
                        ncp[l2] = istep + m
                        al1 = float(nc[l1])
                        alw = al1 - 1.0
                        al2 = float(nc[l2])
                        alt = al2 + 1.0
                        for j in range(n):
                            C[l1][j] = (C[l1][j] * al1 - A[i0][j]) / alw
                            C[l2][j] = (C[l2][j] * al2 + A[i0][j]) / alt
                        nc[l1] -= 1
                        nc[l2] += 1
                        an2[l1] = alw / al1
                        an1[l1] = _BIG
                        if alw > 1.0:
                            an1[l1] = alw / (alw - 1.0)
                        an1[l2] = alt / al2
                        an2[l2] = alt / (alt + 1.0)
                        ic1[i0] = l2
                        ic2[i0] = l1
            if icoun == m:
                return
