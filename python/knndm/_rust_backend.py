"""Backend backed by the compiled Rust kernels (``knndm_rs.dll``/``.so``), via ctypes.

The Rust crate exposes a plain C ABI and has no dependencies, so it builds
without running any build-script executables (see ``.cargo/config.toml``).
"""

from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np

from . import _kmeans
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
        "knndm_cluster_pc1": (C.c_int32, [_dp, _sz, _sz, _ip, _ip, _sz, _dp, _dp, _dp]),
        "knndm_ld_mean": (C.c_double, [_dp, _sz]),
        "knndm_ld_sum": (C.c_double, [_dp, _sz]),
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


def _pd(a: np.ndarray):
    return a.ctypes.data_as(_dp)


def _pi(a: np.ndarray):
    return a.ctypes.data_as(_ip)


class _NeighbourIndex:
    def __init__(self, x: np.ndarray, kk: int):
        x = _f(x)
        self.n = x.shape[0]
        self._h = _lib.knndm_nnindex_new(_pd(x), self.n, x.shape[1], kk)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _lib.knndm_nnindex_free(h)

    def nn_dist(self) -> np.ndarray:
        out = np.empty(self.n)
        _lib.knndm_nnindex_nn_dist(self._h, _pd(out))
        return out

    def cv_dist(self, folds) -> np.ndarray:
        folds = _i(folds)
        out = np.empty(self.n)
        _lib.knndm_nnindex_cv_dist(self._h, _pi(folds), _pd(out))
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
        out = np.empty(q.shape[0])
        _lib.knndm_knnx_dist(_pd(q), q.shape[0], _pd(d), d.shape[0], q.shape[1], _pd(out))
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
        n = x.shape[0]
        merge = np.empty((n - 1, 2), dtype=np.int64)
        if _lib.knndm_hclust_merge(_pd(x), n, x.shape[1], _method_index(method), _pi(merge)) != 0:
            raise ValueError("must have n >= 2 objects to cluster")
        return merge

    def cutree(self, merge, k: int):
        merge = _i(merge)
        out = np.empty(merge.shape[0] + 1, dtype=np.int64)
        if _lib.knndm_cutree(_pi(merge), merge.shape[0], int(k), _pi(out)) != 0:
            raise ValueError(f"elements of 'k' must be between 1 and {merge.shape[0] + 1}")
        return out

    def kmeans(self, x, nk: int, rng: RRNG):
        def kmns(a, c, k, iter_max, max_qtr):
            a, c = _f(a), _f(c).copy()
            out = np.empty(a.shape[0], dtype=np.int64)
            ifault = _lib.knndm_kmeans_hw(_pd(a), a.shape[0], a.shape[1], _pd(c), k, int(iter_max), int(max_qtr), _pi(out))
            return (out if ifault not in (1, 3) else None), ifault

        return _kmeans.kmeans(x, nk, rng, kmns=kmns)

    def wass_stat(self, a, b):
        a, b = _f(a), _f(b)
        return _lib.knndm_wass_stat(_pd(a), len(a), _pd(b), len(b))

    def cluster_pc1(self, x, labels, levels, rot, center):
        x, labels, levels = _f(x), _i(labels), _i(levels)
        rot, center = _f(rot), _f(center)
        out = np.empty(len(levels))
        _lib.knndm_cluster_pc1(
            _pd(x), x.shape[0], x.shape[1], _pi(labels), _pi(levels), len(levels), _pd(rot), _pd(center), _pd(out)
        )
        return out


def ld_mean(x) -> float:
    x = _f(x)
    return _lib.knndm_ld_mean(_pd(x), len(x))


def ld_sum(x) -> float:
    x = _f(x)
    return _lib.knndm_ld_sum(_pd(x), len(x))
