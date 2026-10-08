"""Distance and test-statistic helpers, ported for bit-exact agreement with R.

* ``ks_test_greater_pvalue``: ``stats::ks.test(x, y, alternative = "greater")$p.value``
  (R 4.5.0 ``ks.test.R`` + ``ks.c``).
* ``wass_stat``: ``twosamples::wass_stat`` 2.0.1 (``stat_functions.cpp``).
* ``knn_dist_self`` / ``knnx_dist``: ``FNN::knn.dist`` / ``FNN::knnx.dist`` with
  ``k = 1, algorithm = "brute"`` (ANN brute force: sqrt of the smallest
  squared distance, accumulated dimension by dimension).
"""

from __future__ import annotations

import math

import numpy as np

from ._rnum import r_cumsum, r_order


# --------------------------------------------------------------------------
# nearest-neighbour distances
# --------------------------------------------------------------------------
def _sqdist(query: np.ndarray, data: np.ndarray) -> np.ndarray:
    """ANN ``annDist``: sum over dims of (p - q)^2, accumulated left to right."""
    out = np.zeros((query.shape[0], data.shape[0]))
    for d in range(query.shape[1]):
        diff = data[None, :, d] - query[:, None, d]
        out = out + diff * diff
    return out


def knnx_dist(query: np.ndarray, data: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance from each query row to its nearest data row."""
    res = np.empty(query.shape[0])
    for s in range(0, query.shape[0], chunk):
        res[s : s + chunk] = np.sqrt(_sqdist(query[s : s + chunk], data).min(axis=1))
    return res


def knn_dist_self(data: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """``knn.dist(data, k = 1)``: second-smallest squared distance incl. self."""
    n = data.shape[0]
    res = np.empty(n)
    for s in range(0, n, chunk):
        sq = _sqdist(data[s : s + chunk], data)
        part = np.partition(sq, 1, axis=1)[:, 1]
        res[s : s + chunk] = np.sqrt(part)
    return res


def euclid_dist_matrix(x: np.ndarray) -> np.ndarray:
    """``as.matrix(stats::dist(x))`` (also equals GEOS point distances in 2-D):
    sqrt(sum over columns of (x_i - x_j)^2), accumulated left to right."""
    out = np.zeros((x.shape[0], x.shape[0]))
    for d in range(x.shape[1]):
        diff = x[:, None, d] - x[None, :, d]
        out = out + diff * diff
    return np.sqrt(out)


def cv_distances(x: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """CAST:::cv_distances with euclidean distance and FNN brute force."""
    out = np.full(x.shape[0], np.nan)
    for f in _unique_in_order(folds):
        test = folds == f
        train = ~test
        out[test] = knnx_dist(x[test], x[train])
    return out


def distclust_distmat(distm: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """CAST:::distclust_distmat: min distance from each point to other folds."""
    out = np.full(len(folds), np.nan)
    for f in _unique_in_order(folds):
        test = folds == f
        out[test] = distm[np.ix_(test, ~test)].min(axis=1)
    return out


def _unique_in_order(v) -> list:
    seen = set()
    out = []
    for x in np.asarray(v).tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# --------------------------------------------------------------------------
# Wasserstein statistic (twosamples::wass_stat, power = 1)
# --------------------------------------------------------------------------
def wass_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    joint = np.concatenate([a, b])
    na = float(len(a))
    labs = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
    o = r_order(joint)
    joint = joint[o].tolist()
    labs = labs[o].tolist()
    n = float(len(joint))
    nb = n - na
    diva = 1 / na
    divb = 1 / nb
    ecur = fcur = out = 0.0
    for i in range(len(joint) - 1):
        if labs[i]:
            ecur += diva
        else:
            fcur += divb
        height = ecur - fcur
        if height < 0.0:
            height *= -1.0
        out += height * (joint[i + 1] - joint[i])
    return out


# --------------------------------------------------------------------------
# two-sample Kolmogorov-Smirnov test, alternative = "greater"
# --------------------------------------------------------------------------
def ks_test_greater_pvalue(x: np.ndarray, y: np.ndarray, cumsum=r_cumsum) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n_x = float(len(x))
    n_y = len(y)
    exact = n_x * n_y < 10000
    w = np.concatenate([x, y])
    o = r_order(w) + 1  # 1-based like R
    steps = np.where(o <= n_x, 1 / n_x, -1 / n_y)
    z = cumsum(steps)
    ties = len(np.unique(w)) < (n_x + n_y)
    if ties:
        sw = np.sort(w, kind="stable")
        keep = np.concatenate([np.nonzero(np.diff(sw) != 0)[0], [int(n_x + n_y) - 1]])
        z = z[keep]
    stat = float(np.max(z))
    p = _psmirnov_upper(stat, n_x, float(n_y), w if ties else None, exact)
    return min(1.0, max(0.0, p))


def _psmirnov_upper(q: float, n_x: float, n_y: float, z, exact: bool) -> float:
    """psmirnov(q, sizes, z, alternative = "greater", exact, lower.tail = FALSE)."""
    if math.isnan(q):
        return math.nan
    if q <= 0:
        return 1.0  # 1 - lower.tail
    if q > 1:
        return 0.0
    n_x = math.floor(n_x)
    n_y = math.floor(n_y)
    if not exact:
        n = (n_x * n_y) / (n_x + n_y)  # prod(sizes)/sum(sizes)
        ret = -math.expm1((-2 * n) * (q * q))
        return 1 - ret
    zz = None
    if z is not None:
        dz = np.diff(np.sort(z, kind="stable")) != 0
        if dz.any():
            zz = [0] + dz.astype(int).tolist() + [1]
    m = int(n_x)
    n = int(n_y)
    md = float(m)
    nd = float(n)
    q = (0.5 + math.floor(q * md * nd - 1e-7)) / (md * nd)

    def test(r: float, s: float) -> bool:
        return (r - s) >= q

    u = [0.0] * (n + 1)
    for j in range(1, n + 1):
        if test(0.0, j / nd) and (zz is None or zz[j]):
            u[j] = 1.0
        else:
            u[j] = u[j - 1]
    for i in range(1, m + 1):
        if test(i / md, 0.0) and (zz is None or zz[i]):
            u[0] = 1.0
        for j in range(1, n + 1):
            if test(i / md, j / nd) and (zz is None or zz[i + j]):
                u[j] = 1.0
            else:
                v = float(i) / float(i + j)
                w = float(j) / float(i + j)
                u[j] = v * u[j] + w * u[j - 1]
    return u[n]
