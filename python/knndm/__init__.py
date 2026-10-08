"""Bit-exact Python port of ``CAST::knndm`` (k-fold nearest neighbour distance matching)."""

from ._rrng import RRNG
from .core import KNNDMResult, knndm

__all__ = ["knndm", "KNNDMResult", "RRNG"]
__version__ = "0.1.0"
