"""Bit-exact emulation of the R numeric primitives knndm relies on.

R on x86-64 accumulates ``sum``, ``cumsum`` and ``colMeans`` in x87 80-bit
``long double`` (64-bit significand, verified at runtime: in R on Windows,
``sum(c(1, 2^-60, -1)) == 2^-60``). Python has no such type, so it is
emulated exactly with integer arithmetic and round-half-to-even.
"""

from __future__ import annotations

import math

import numpy as np

_LD_BITS = 64  # x87 extended significand
_DBL_BITS = 53


def _round_bits(m: int, e: int, bits: int, sticky: bool = False) -> tuple[int, int]:
    """Round m * 2^e to ``bits`` significant bits (round half to even).

    ``sticky`` flags extra non-zero bits strictly below ``m``'s LSB.
    """
    if m == 0:
        return 0, 0
    neg = m < 0
    a = -m if neg else m
    bl = a.bit_length()
    if bl > bits:
        shift = bl - bits
        q = a >> shift
        rem = a & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (sticky or (q & 1))):
            q += 1
            if q.bit_length() > bits:
                q >>= 1
                shift += 1
        a, e = q, e + shift
    elif sticky:
        # Non-zero bits below an exactly representable value: only matters
        # when they reach the half-ulp, which callers guarantee they don't.
        pass
    return (-a if neg else a), e


def _from_double(x: float) -> tuple[int, int]:
    if x == 0.0:
        return 0, 0
    fr, ex = math.frexp(x)
    return int(fr * (1 << _DBL_BITS)), ex - _DBL_BITS


def _to_double(m: int, e: int) -> float:
    """Round m * 2^e to binary64 (subnormals keep fewer bits; overflow -> inf)."""
    if m == 0:
        return 0.0
    top = e + abs(m).bit_length() - 1  # exponent of the leading bit
    bits = _DBL_BITS if top >= -1022 else _DBL_BITS - (-1022 - top)
    if bits <= 0:
        # below the smallest subnormal: rounds to 0 or to 2^-1074 (half-even)
        half = top == -1075 and abs(m) != 1 << (abs(m).bit_length() - 1)
        v = math.ldexp(1.0, -1074) if half else 0.0
        return -v if m < 0 else v
    m, e = _round_bits(m, e, bits)
    if e + abs(m).bit_length() - 1 > 1023:
        return math.inf if m > 0 else -math.inf
    return math.ldexp(m, e)


class LongDouble:
    """An x87 extended-precision accumulator."""

    __slots__ = ("m", "e")

    def __init__(self) -> None:
        self.m = 0
        self.e = 0

    def add(self, x: float) -> None:
        xm, xe = _from_double(x)
        if xm == 0:
            return
        if self.m == 0:
            self.m, self.e = _round_bits(xm, xe, _LD_BITS)
            return
        if xe >= self.e:
            m = (xm << (xe - self.e)) + self.m
            e = self.e
        else:
            m = xm + (self.m << (self.e - xe))
            e = xe
        self.m, self.e = _round_bits(m, e, _LD_BITS)

    def to_double(self) -> float:
        return _to_double(self.m, self.e)

    def div_int_to_double(self, n: int) -> float:
        """``(double)(sum / n)`` with the division done in long double."""
        if self.m == 0:
            return 0.0
        neg = self.m < 0
        a = -self.m if neg else self.m
        s = max(0, _LD_BITS + 3 + n.bit_length() - a.bit_length())
        q, r = divmod(a << s, n)
        q = (q << 1) | (1 if r else 0)  # sticky bit
        qm, qe = _round_bits(q, self.e - s - 1, _LD_BITS)
        return _to_double(-qm if neg else qm, qe)


def r_sum(x) -> float:
    """``sum(x)`` for a double vector (rsum: long double accumulation)."""
    acc = LongDouble()
    for v in x:
        acc.add(float(v))
    return acc.to_double()


def r_cumsum(x) -> np.ndarray:
    """``cumsum(x)`` for a double vector."""
    acc = LongDouble()
    out = np.empty(len(x))
    for i, v in enumerate(x):
        acc.add(float(v))
        out[i] = acc.to_double()
    return out


def r_mean_ld(x) -> float:
    """One column of ``colMeans``: long double sum divided by the count."""
    acc = LongDouble()
    n = 0
    for v in x:
        acc.add(float(v))
        n += 1
    return acc.div_int_to_double(n)


def r_colmeans(mat: np.ndarray) -> np.ndarray:
    return np.array([r_mean_ld(mat[:, j]) for j in range(mat.shape[1])])


def r_order(x) -> np.ndarray:
    """``order(x)`` for doubles without NA: stable ascending (radix)."""
    return np.argsort(np.asarray(x, dtype=float), kind="stable")


def r_seq_len_out(start: float, end: float, length_out: int) -> np.ndarray:
    """``seq(from, to, length.out=)`` for doubles (seq.default)."""
    if length_out == 1:
        return np.array([start])
    if length_out == 2:
        return np.array([start, end])
    if start == end:
        return np.full(length_out, start)
    n1 = length_out - 1
    by = (end - start) / n1
    mid = [start + i * by for i in range(1, length_out - 1)]
    return np.array([start] + mid + [end])


def r_round_int(x: float) -> int:
    """``round(x)`` for digits = 0 (IEEE nearbyint, half to even)."""
    return int(round(x))


def nk_grid(k: int, n: int, nk_len: int) -> list[int]:
    """``unique(as.integer(round(exp(seq(log(k), log(n - 2), length.out = nk_len)))))``."""
    s = r_seq_len_out(math.log(k), math.log(n - 2), nk_len)
    out: list[int] = []
    seen = set()
    for v in s:
        r = r_round_int(math.exp(v))
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def r_scale_center(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``scale(x, center = TRUE, scale = FALSE)``."""
    center = r_colmeans(x)
    return x - center[None, :], center


def r_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``scale(x)`` (centre and divide by root-mean-square, R semantics)."""
    xc, center = r_scale_center(x)
    n = x.shape[0]
    denom = max(1, n - 1)
    sc = np.array([math.sqrt(r_sum(xc[:, j] * xc[:, j]) / denom) for j in range(x.shape[1])])
    with np.errstate(divide="ignore", invalid="ignore"):
        return xc / sc[None, :], center, sc


def prcomp_rank1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First rotation vector and centre of ``prcomp(x, center=TRUE, rank.=1)``.

    R calls its bundled reference LAPACK ``dgesdd`` (jobz = 'S'); ``_lapack``
    is a bit-exact port of that code path for up to 25 columns. Wider inputs
    fall back to SciPy's ``gesdd`` (same algorithm and sign convention, but
    optimised BLAS kernels may differ from R in the last bits).
    """
    from ._lapack import svd_vt

    xc, center = r_scale_center(np.asarray(x, dtype=float))
    if xc.shape[1] <= 25 and xc.shape[0] >= xc.shape[1]:
        _, vt = svd_vt(xc)
    else:
        from scipy.linalg import svd

        _, _, vt = svd(xc, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    return np.array(vt[0], dtype=float), center


def predict_pc1(centr: np.ndarray, center: np.ndarray, rot: np.ndarray) -> float:
    """``predict(prcomp, newdata)`` for one row: dgemv sequential dot product."""
    s = 0.0
    for j in range(len(rot)):
        s = s + (centr[j] - center[j]) * rot[j]
    return s
