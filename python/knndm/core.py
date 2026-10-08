"""Python port of ``CAST::knndm`` (CAST 1.1.2), bit-exact with R 4.5.

Supported: geographical (projected coordinates) and feature space, euclidean
distances, ``clustering = "hierarchical"`` (any ``hclust`` linkage) or
``"kmeans"``, ``test_prop`` train/test splitting, and the random-CV fallback.
Prediction locations must be supplied (``predpoints``); sampling them from a
``modeldomain`` polygon is not ported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import _backend
from ._kmeans import KMeansError
from ._rnum import LongDouble, nk_grid, predict_pc1, prcomp_rank1, r_colmeans, r_order, r_scale
from ._rrng import RRNG


@dataclass
class KNNDMResult:
    """Mirror of the ``knndm`` object returned by CAST.

    ``indx_train``/``indx_test`` are **0-based** index arrays (R's are 1-based).
    ``clusters`` holds integer fold labels, or ``"train"``/``"test"`` strings
    when ``test_prop`` is used.
    """

    clusters: np.ndarray
    indx_train: Any
    indx_test: Any
    Gij: np.ndarray
    Gj: np.ndarray
    Gjstar: np.ndarray
    W: float
    method: str
    q: Any
    space: str
    messages: list[str] = field(default_factory=list)


def knndm(
    tpoints,
    predpoints,
    space: str = "geographical",
    k: int = 10,
    maxp: float = 0.5,
    clustering: str = "hierarchical",
    linkf: str = "ward.D2",
    dist_fun: str = "euclidean",
    scale_vars: bool = True,
    test_prop: float | None = None,
    test_tolerance: float | None = None,
    nk_len: int = 100,
    seed: int | None = None,
    rng: RRNG | None = None,
    backend: str = "auto",
) -> KNNDMResult:
    """K-fold Nearest Neighbour Distance Matching (Linnenbrink et al. 2024).

    Parameters mirror ``CAST::knndm``. ``tpoints``/``predpoints`` are arrays of
    projected coordinates ``(n, 2)`` for ``space="geographical"`` or predictor
    tables ``(n, p)`` (array or DataFrame) for ``space="feature"``.

    ``seed`` reproduces ``set.seed(seed); knndm(...)`` in R. Alternatively pass
    an :class:`RRNG` as ``rng``; it is advanced exactly as R's global RNG is.
    ``backend`` is ``"python"``, ``"rust"`` or ``"auto"`` (Rust if available).
    """
    if rng is None:
        rng = RRNG(seed if seed is not None else int.from_bytes(np.random.bytes(4), "little"))
    elif seed is not None:
        rng.set_seed(seed)
    be = _backend.get(backend)
    msgs: list[str] = []

    if space == "geo":
        space = "geographical"
    if space not in ("geographical", "feature"):
        raise ValueError("dist_space must be one of 'geographical' or 'feature'")
    if dist_fun not in ("euclidean", "mahalanobis", "gower", "great_circle"):
        raise ValueError("dist_fun must be one of 'euclidean', 'mahalanobis', 'gower' or 'great_circle'")
    if dist_fun != "euclidean":
        raise NotImplementedError(f"dist_fun = '{dist_fun}' is not ported (only 'euclidean')")

    minp = None
    if test_prop is not None:
        if test_prop >= 1 or test_prop <= 0:
            raise ValueError("test_prop must be greater than 0 and smaller than 1")
        if test_tolerance is None:
            test_tolerance = 0.1
        k = 2
        maxp = test_prop + test_tolerance
        minp = test_prop - test_tolerance
        if maxp <= 0 or minp >= 1 or minp > maxp:
            raise ValueError("Misspecified tolerance. Resulted in infeasible minp/maxp values")
        if minp <= 0:
            msgs.append("warning: minp was set to 0.1")
            minp = 0.1
        elif maxp >= 1:
            msgs.append("warning: maxp was set to 0.9")
            maxp = 0.9
        if maxp == 1 / k:
            maxp = maxp + math.sqrt(np.finfo(float).eps)
    if nk_len % 1 != 0:
        raise ValueError("nk_len must be an integer")
    nk_len = int(nk_len)

    if space == "geographical":
        tp = _as_matrix(tpoints)
        pp = _as_matrix(predpoints)
        if tp.shape[1] < 2 or pp.shape[1] < 2:
            raise ValueError("geographical tpoints/predpoints need (n, 2) coordinates")
        tp, pp = tp[:, :2], pp[:, :2]
    else:
        tp = _as_matrix(tpoints, numeric_only=True)
        pp = _as_matrix(predpoints, numeric_only=True)
        if tp.shape[1] != pp.shape[1]:
            raise ValueError("tpoints and predpoints need to contain the predictor data and have the same colnames.")
        if np.isnan(pp).any():
            msgs.append("some prediction points contain NAs, which will be removed")
            pp = pp[~np.isnan(pp).any(axis=1)]
        if np.isnan(tp).any():
            msgs.append("some training points contain NAs, which will be removed")
            tp = tp[~np.isnan(tp).any(axis=1)]

    if clustering not in ("kmeans", "hierarchical"):
        raise ValueError("clustering must be one of `kmeans` or `hierarchical`")
    if test_prop is None and not (maxp < 1 and maxp > 1 / k):
        raise ValueError("maxp must be strictly between 1/k and 1")

    if space == "feature" and scale_vars:
        tp, center, sc = r_scale(tp)
        pp = (pp - center[None, :]) / sc[None, :]

    return _knndm_impl(tp, pp, space, k, maxp, minp, test_prop, clustering, linkf, nk_len, rng, be, msgs)


def _as_matrix(x, numeric_only: bool = False) -> np.ndarray:
    if hasattr(x, "to_numpy") and hasattr(x, "dtypes"):
        if numeric_only:
            bad = [c for c, dt in zip(x.columns, x.dtypes) if not np.issubdtype(dt, np.number)]
            if bad:
                raise NotImplementedError(f"categorical variables {bad} require gower distance (not ported)")
        x = x.to_numpy(dtype=float)
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return np.ascontiguousarray(a)


def _knndm_impl(tp, pp, space, k, maxp, minp, test_prop, clustering, linkf, nk_len, rng, be, msgs):
    n = tp.shape[0]
    is_geo = space == "geographical"

    Gj = be.knn_dist_self(tp)
    Gij = be.knnx_dist(pp, tp)

    pval = be.ks_greater_pvalue(Gj, Gij)
    if pval >= 0.05:
        if test_prop is not None:
            ntest = math.floor(test_prop * n)
            ntrain = n - ntest
            clusters = [1] * ntrain + [2] * ntest
            clust = np.array(rng.sample(clusters, n), dtype=np.int64)
        else:
            pool = list(range(1, k + 1)) * math.ceil(n / k)
            clust = np.array(rng.sample(pool, n), dtype=np.int64)
        Gjstar = be.cv_distances(tp, clust)
        k_final: Any = "random CV"
        W_final = be.wass_stat(Gjstar, Gij)
        msgs.append("Gij <= Gj; a random CV assignment is returned")
    else:
        distmat = None
        hc_merge = None
        if clustering == "hierarchical":
            distmat = be.dist_matrix(tp)
            hc_merge = be.hclust_merge(distmat, linkf)
        nks = nk_grid(k, n, nk_len)
        W = {nk: math.nan for nk in nks}
        groups: dict[int, np.ndarray] = {}
        rot, pc_center = prcomp_rank1(tp)

        for nk in nks:
            if clustering == "hierarchical":
                clust_nk = be.cutree(hc_merge, nk)
            else:
                try:
                    clust_nk = be.kmeans(tp, nk, rng)
                except KMeansError:
                    if is_geo:  # knndm_geo has no tryCatch around kmeans
                        raise
                    continue
            clust_k = _assign_folds(tp, clust_nk, n, k, test_prop, rot, pc_center, be)

            counts = np.bincount(clust_k, minlength=k + 1)[1:]
            counts = counts[counts > 0]
            max_ok = not np.any(counts / n > maxp)
            if test_prop is None:
                valid = max_ok
            else:
                props = [_ld_mean_count(int(np.sum(clust_k == 1)), n), _ld_mean_count(int(np.sum(clust_k == 2)), n)]
                valid = any(minp <= p <= maxp for p in props)
            # knndm_geo evaluates the W statistic twice: once for prop_valid and
            # once more under the plain maxp check (which matters for test_prop).
            if valid or (is_geo and max_ok):
                if clustering == "kmeans" or is_geo:
                    Gjstar_i = be.cv_distances(tp, clust_k)
                else:
                    Gjstar_i = be.distclust_distmat(distmat, clust_k)
                W[nk] = be.wass_stat(Gjstar_i, Gij)
                groups[nk] = clust_k

        wvals = np.array([W[nk] for nk in nks])
        if np.all(np.isnan(wvals)):
            if test_prop is not None:
                raise RuntimeError(
                    "No valid train/test configurations found in the range test_prop +/- tolerance. Increase tolerance."
                )
            raise RuntimeError("no fold configuration satisfied maxp; try a larger maxp")
        best = int(np.nanargmin(wvals))
        k_final = nks[best]
        W_final = float(np.nanmin(wvals))
        clust = groups[k_final]
        if clustering == "kmeans" or is_geo:
            Gjstar = be.cv_distances(tp, clust)
        else:
            Gjstar = be.distclust_distmat(distmat, clust)

    if test_prop is None:
        indx_train, indx_test = _create_spacetime_folds(clust, k, rng, msgs)
        clusters_out: np.ndarray = clust
    else:
        labels, counts = np.unique(clust, return_counts=True)
        if len(labels) < 2:
            raise RuntimeError("subscript out of bounds")
        dev1 = abs(counts[0] / len(clust) - test_prop)
        dev2 = abs(counts[1] / len(clust) - test_prop)
        test_class = 2 if dev1 > dev2 else 1
        is_test = clust == test_class
        clusters_out = np.where(is_test, "test", "train")
        indx_test = np.nonzero(is_test)[0]
        indx_train = np.nonzero(~is_test)[0]

    return KNNDMResult(
        clusters=clusters_out,
        indx_train=indx_train,
        indx_test=indx_test,
        Gij=np.asarray(Gij),
        Gj=np.asarray(Gj),
        Gjstar=np.asarray(Gjstar),
        W=W_final,
        method=clustering,
        q=k_final,
        space=space,
        messages=msgs,
    )


def _ld_mean_count(count: int, n: int) -> float:
    """``mean(logical)``: long double count / n, rounded to double."""
    acc = LongDouble()
    acc.add(float(count))
    return acc.div_int_to_double(n)


def _assign_folds(tp, clust_nk, n, k, test_prop, rot, pc_center, be) -> np.ndarray:
    """Map ``nk`` clusters to ``k`` folds as in knndm_geo/knndm_feature."""
    # table() levels (ascending), their counts, and each point's level
    levels, inv, freq = np.unique(clust_nk, return_inverse=True, return_counts=True)
    centr = be.cluster_pc1(tp, clust_nk, levels, rot, pc_center)
    o = r_order(centr)
    fr_sorted = freq[o]
    clust_k_of = np.zeros(len(levels), dtype=np.int64)  # 0 = NA
    if test_prop is None:
        ci = 1
        thresh = n / k
        for i in range(len(fr_sorted)):
            if fr_sorted[i] >= thresh:
                clust_k_of[i] = ci
                ci += 1
    used = set(int(v) for v in clust_k_of if v != 0)
    remaining = [v for v in range(1, k + 1) if v not in used]
    na_pos = np.nonzero(clust_k_of == 0)[0]
    if len(na_pos):
        if not remaining:
            raise RuntimeError("invalid 'times' argument")
        reps = math.ceil(len(levels) / len(remaining))
        fill = (remaining * reps)[: len(na_pos)]
        if len(fill) < len(na_pos):  # R pads with NA
            fill += [0] * (len(na_pos) - len(fill))
        clust_k_of[na_pos] = fill
    fold_of_level = np.empty(len(levels), dtype=np.int64)
    fold_of_level[o] = clust_k_of
    return fold_of_level[inv.ravel()]


def _create_spacetime_folds(clust: np.ndarray, k: int, rng: RRNG, msgs: list[str]):
    """CAST::CreateSpacetimeFolds(data.frame(clust), spacevar = "clust", k).

    ``createFolds`` always receives as many units as folds here, so folds are
    simply the unique cluster labels in order of first appearance. The RNG is
    advanced exactly as in R: ``seed = sample(1:1000, 1)`` then ``set.seed(seed)``.
    """
    uniq: list[int] = []
    seen = set()
    for v in clust.tolist():
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    if k > len(uniq):
        k = len(uniq)
        msgs.append(f"warning: k is higher than number of unique locations. k is set to {k}")
    if len(uniq) < 2:
        raise RuntimeError("invalid number of intervals")
    seed = rng.sample_int(1000, 1)[0]
    rng.set_seed(seed)
    indx_test = [np.nonzero(clust == u)[0] for u in uniq[:k]]
    indx_train = [np.nonzero(clust != u)[0] for u in uniq[:k]]
    return indx_train, indx_test
