"""Compute backends. Both must produce bit-identical results.

``PythonBackend`` is the readable reference port; ``RustBackend`` calls the
``knndm_rs`` shared library (loaded with ctypes) for speed.
"""

from __future__ import annotations

import numpy as np

from . import _hclust, _kmeans, _stats
from ._rnum import predict_pc1, r_colmeans
from ._rrng import RRNG


class PythonBackend:
    name = "python"

    knn_dist_self = staticmethod(_stats.knn_dist_self)
    knnx_dist = staticmethod(_stats.knnx_dist)
    cv_distances = staticmethod(_stats.cv_distances)
    distclust_distmat = staticmethod(_stats.distclust_distmat)
    wass_stat = staticmethod(_stats.wass_stat)
    dist_matrix = staticmethod(_stats.euclid_dist_matrix)
    cutree = staticmethod(_hclust.cutree)
    ks_greater_pvalue = staticmethod(_stats.ks_test_greater_pvalue)

    @staticmethod
    def hclust_merge(distmat: np.ndarray, method: str) -> np.ndarray:
        n = distmat.shape[0]
        merge, _height, _order = _hclust.hclust(_hclust.condensed_from_square(distmat), n, method)
        return merge

    @staticmethod
    def kmeans(x: np.ndarray, nk: int, rng: RRNG) -> np.ndarray:
        return _kmeans.kmeans(x, nk, rng)

    @staticmethod
    def kmeans_many(x: np.ndarray, nks, rng: RRNG, stop_on_error: bool):
        return _kmeans.kmeans_many(x, nks, rng, stop_on_error)

    @staticmethod
    def cluster_pc1(x, labels, levels, rot, center) -> np.ndarray:
        return np.array([predict_pc1(r_colmeans(x[labels == lv]), center, rot) for lv in levels])


def _rust_backend():
    try:
        from ._rust_backend import RustBackend
    except (ImportError, OSError):
        return None
    return RustBackend()


def get(name: str):
    if name == "python":
        return PythonBackend()
    if name in ("rust", "auto"):
        be = _rust_backend()
        if be is not None:
            return be
        if name == "rust":
            raise ImportError("Rust kernels are not built (run `pixi run build-rust`)")
        return PythonBackend()
    raise ValueError("backend must be 'python', 'rust' or 'auto'")
