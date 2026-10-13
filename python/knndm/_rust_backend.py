"""Backend backed by the compiled Rust kernels (``knndm_rs.dll``/``.so``), via ctypes.

The Rust crate exposes a plain C ABI and has no dependencies. Set
``KNNDM_RS_LIB`` to load a library from a custom path.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np

from . import _kmeans, _stats
from ._hclust import _method_index
from ._rrng import RRNG

_LIBNAMES = {"win32": "knndm_rs.dll", "darwin": "libknndm_rs.dylib"}

_dp = C.POINTER(C.c_double)
_ip = C.POINTER(C.c_int64)
_sz = C.c_size_t


def _load() -> C.CDLL:
    name = _LIBNAMES.get(sys.platform, "libknndm_rs.so")
    path = Path(os.environ.get("KNNDM_RS_LIB", Path(__file__).with_name(name)))
    if not path.exists():
        raise ImportError(f"Rust kernels not built: {path} missing (run `pixi run build-rust`)")
    lib = C.CDLL(str(path))
    sig = {
        "knndm_threads": (_sz, []),
        "knndm_wass_stat": (C.c_double, [_dp, _sz, _dp, _sz]),
        "knndm_knnx_dist": (C.c_int32, [_dp, _sz, _dp, _sz, _sz, _dp]),
        "knndm_nnindex_new": (C.c_void_p, [_dp, _sz, _sz, _sz]),
        "knndm_nnindex_free": (None, [C.c_void_p]),
        "knndm_nnindex_nn_dist": (C.c_int32, [C.c_void_p, _dp]),
        "knndm_nnindex_cv_dist": (C.c_int32, [C.c_void_p, _ip, _dp]),
        "knndm_hclust_merge": (C.c_int32, [_dp, _sz, _sz, C.c_int32, _ip]),
        "knndm_cutree": (C.c_int32, [_ip, _sz, _sz, _ip]),
        "knndm_kmeans_hw": (C.c_int32, [_dp, _sz, _sz, _dp, _sz, _sz, C.c_int64, _ip]),
        "knndm_kmeans_batch": (C.c_int32, [_dp, _sz, _sz, _dp, _ip, _sz, _sz, C.c_int64, _ip, _ip]),
        "knndm_cluster_pc1": (C.c_int32, [_dp, _sz, _sz, _ip, _ip, _sz, _dp, _dp, _dp]),
        "knndm_ld_mean": (C.c_double, [_dp, _sz]),
        "knndm_ld_sum": (C.c_double, [_dp, _sz]),
        "knndm_ld_cumsum": (C.c_int32, [_dp, _sz, _dp]),
    }
    for fname, (res, args) in sig.items():
        f = getattr(lib, fname)
        f.restype = res
        f.argtypes = args
    return lib


_lib = _load()


def _f(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _i(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.int64)


def _ok(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"Rust kernel {what} failed (code {code})")


def _check2d(a: np.ndarray, name: str, p: int | None = None) -> None:
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-d array")
    if p is not None and a.shape[1] != p:
        raise ValueError(f"{name} must have {p} columns, got {a.shape[1]}")


def _check_len(a: np.ndarray, n: int, name: str) -> None:
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{name} must be a 1-d array of length {n}, got shape {a.shape}")


def _pd(a: np.ndarray):
    return a.ctypes.data_as(_dp)


def _pi(a: np.ndarray):
    return a.ctypes.data_as(_ip)


class _NeighbourIndex:
    def __init__(self, x: np.ndarray, kk: int):
        x = _f(x)
        _check2d(x, "x")
        if x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("need at least 2 points and 1 column")
        self.n = x.shape[0]
        self._h = _lib.knndm_nnindex_new(_pd(x), self.n, x.shape[1], kk)
        if not self._h:
            raise RuntimeError("Rust kernel knndm_nnindex_new failed")

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _lib.knndm_nnindex_free(h)

    def nn_dist(self) -> np.ndarray:
        out = np.empty(self.n)
        _ok(_lib.knndm_nnindex_nn_dist(self._h, _pd(out)), "nn_dist")
        return out

    def cv_dist(self, folds) -> np.ndarray:
        folds = _i(folds)
        _check_len(folds, self.n, "folds")
        out = np.empty(self.n)
        _ok(_lib.knndm_nnindex_cv_dist(self._h, _pi(folds), _pd(out)), "cv_dist")
        return out


class RustBackend:
    name = "rust"

    def __init__(self, neighbours: int = 32):
        self._neighbours = neighbours
        self._index = None
        self._index_src = None

    def _idx(self, x: np.ndarray) -> _NeighbourIndex:
        if self._index is None or self._index_src is not x:
            self._index = _NeighbourIndex(x, self._neighbours)
            self._index_src = x
        return self._index

    def knn_dist_self(self, x):
        return self._idx(x).nn_dist()

    def knnx_dist(self, query, data):
        q, d = _f(query), _f(data)
        _check2d(q, "query")
        _check2d(d, "data", q.shape[1])
        if q.shape[1] < 1:
            raise ValueError("need at least 1 column")
        out = np.empty(q.shape[0])
        _ok(_lib.knndm_knnx_dist(_pd(q), q.shape[0], _pd(d), d.shape[0], q.shape[1], _pd(out)), "knnx_dist")
        return out

    def cv_distances(self, x, folds):
        return self._idx(x).cv_dist(folds)

    # In R these come from an explicit distance matrix; the values are the
    # same euclidean distances, so the Rust backend keeps the coordinates and
    # answers from the neighbour index instead of materialising n x n.
    def dist_matrix(self, x):
        return x

    def distclust_distmat(self, x, folds):
        return self._idx(x).cv_dist(folds)

    def hclust_merge(self, x, method: str):
        x = _f(x)
        _check2d(x, "x")
        n = x.shape[0]
        if n < 2:
            raise ValueError("must have n >= 2 objects to cluster")
        if n > 65536:
            raise ValueError("size cannot be NA nor exceed 65536")  # stats::hclust limit
        merge = np.empty((n - 1, 2), dtype=np.int64)
        _ok(_lib.knndm_hclust_merge(_pd(x), n, x.shape[1], _method_index(method), _pi(merge)), "hclust")
        return merge

    def cutree(self, merge, k: int):
        merge = _i(merge)
        if merge.ndim != 2 or merge.shape[1] != 2:
            raise ValueError("merge must be an (n-1, 2) array")
        out = np.empty(merge.shape[0] + 1, dtype=np.int64)
        if _lib.knndm_cutree(_pi(merge), merge.shape[0], int(k), _pi(out)) != 0:
            raise ValueError(f"elements of 'k' must be between 1 and {merge.shape[0] + 1}")
        return out

    def kmeans(self, x, nk: int, rng: RRNG):
        def kmns(a, c, k, iter_max, max_qtr):
            a, c = _f(a), _f(c).copy()
            _check2d(c, "centers", a.shape[1])
            if c.shape[0] != k:
                raise ValueError("centers must have k rows")
            out = np.empty(a.shape[0], dtype=np.int64)
            ifault = _lib.knndm_kmeans_hw(_pd(a), a.shape[0], a.shape[1], _pd(c), k, int(iter_max), int(max_qtr), _pi(out))
            if ifault < 0:
                raise RuntimeError("Rust kernel kmeans failed")
            return (out if ifault not in (1, 3) else None), ifault

        return _kmeans.kmeans(x, nk, rng, kmns=kmns)

    def kmeans_many(self, x, nks, rng: RRNG, stop_on_error: bool):
        def batch(a, centers, iter_max, max_qtr):
            if not centers:
                return []
            a = _f(a)
            m, p = a.shape
            for c in centers:
                _check2d(c, "centers", p)
            ks = np.array([c.shape[0] for c in centers], dtype=np.int64)
            cc = _f(np.concatenate(centers, axis=0))
            out = np.empty((len(centers), m), dtype=np.int64)
            ifault = np.empty(len(centers), dtype=np.int64)
            code = _lib.knndm_kmeans_batch(
                _pd(a), m, p, _pd(cc), _pi(ks), len(centers), int(iter_max), int(max_qtr), _pi(out), _pi(ifault)
            )
            _ok(code, "kmeans_batch")
            return [(out[j], int(ifault[j])) for j in range(len(centers))]

        return _kmeans.kmeans_many(x, nks, rng, stop_on_error, kmns_batch=batch)

    def wass_stat(self, a, b):
        a, b = _f(a).ravel(), _f(b).ravel()
        return _lib.knndm_wass_stat(_pd(a), len(a), _pd(b), len(b))

    def ks_greater_pvalue(self, x, y):
        return _stats.ks_test_greater_pvalue(x, y, cumsum=ld_cumsum)

    def cluster_pc1(self, x, labels, levels, rot, center):
        x, labels, levels = _f(x), _i(labels), _i(levels)
        rot, center = _f(rot), _f(center)
        _check2d(x, "x")
        n, p = x.shape
        _check_len(labels, n, "labels")
        _check_len(rot, p, "rot")
        _check_len(center, p, "center")
        if levels.ndim != 1 or not np.isin(labels, levels).all() or (levels < 0).any():
            raise ValueError("every label must appear in levels (non-negative ints)")
        out = np.empty(len(levels))
        _ok(
            _lib.knndm_cluster_pc1(_pd(x), n, p, _pi(labels), _pi(levels), len(levels), _pd(rot), _pd(center), _pd(out)),
            "cluster_pc1",
        )
        return out


def ld_mean(x) -> float:
    x = _f(x)
    return _lib.knndm_ld_mean(_pd(x), len(x))


def ld_sum(x) -> float:
    x = _f(x)
    return _lib.knndm_ld_sum(_pd(x), len(x))


def ld_cumsum(x) -> np.ndarray:
    x = _f(x)
    out = np.empty(len(x))
    _ok(_lib.knndm_ld_cumsum(_pd(x), len(x), _pd(out)), "ld_cumsum")
    return out
