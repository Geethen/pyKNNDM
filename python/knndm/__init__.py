"""Bit-exact Python port of ``CAST::knndm`` (k-fold nearest neighbour distance matching)."""

from ._rrng import RRNG
from .core import KNNDMResult, knndm
from .twcv import TWCVWeights, dwcv_weights, rake_weights, twcv_weights, weight_diagnostics, weighted_cv_error

__all__ = [
    "knndm",
    "KNNDMResult",
    "RRNG",
    "twcv_weights",
    "dwcv_weights",
    "TWCVWeights",
    "rake_weights",
    "weighted_cv_error",
    "weight_diagnostics",
]
__version__ = "0.1.0"
