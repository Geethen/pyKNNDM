"""Exact emulation of R's default random number generator.

Matches ``RNGkind("Mersenne-Twister", "Inversion", "Rejection")`` (R >= 3.6):
``set.seed`` scrambling, ``unif_rand`` (MT19937 + fixup), ``R_unif_index``
(rejection sampling via ``rbits``) and ``sample.int`` without replacement.
Ported from R 4.5.0 ``src/main/RNG.c`` and ``src/main/random.c``.
"""

from __future__ import annotations

import math

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_TEMPERING_MASK_B = 0x9D2C5680
_TEMPERING_MASK_C = 0xEFC60000
_I2_32M1 = 2.328306437080797e-10  # 1/(2^32 - 1)
_U32 = 0xFFFFFFFF


class RRNG:
    """R's Mersenne-Twister RNG with ``sample.kind = "Rejection"``."""

    def __init__(self, seed: int | None = None):
        self.mt = [0] * _N
        self.mti = _N + 1
        if seed is not None:
            self.set_seed(seed)

    # -- seeding ---------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Equivalent of ``set.seed(seed)``."""
        s = int(seed) & _U32
        for _ in range(50):  # initial scrambling
            s = (69069 * s + 1) & _U32
        dummy = [0] * (_N + 1)
        for j in range(_N + 1):
            s = (69069 * s + 1) & _U32
            dummy[j] = s
        # FixupSeeds(MERSENNE_TWISTER, initial = 1): dummy[0] = mti = N
        self.mti = _N
        self.mt = dummy[1:]

    def get_state(self) -> tuple:
        return (self.mti, tuple(self.mt))

    def set_state(self, state: tuple) -> None:
        self.mti = state[0]
        self.mt = list(state[1])

    # -- core generator -------------------------------------------------
    def _sgenrand(self, seed: int) -> None:
        mt = self.mt
        for i in range(_N):
            mt[i] = seed & 0xFFFF0000
            seed = (69069 * seed + 1) & _U32
            mt[i] |= (seed & 0xFFFF0000) >> 16
            seed = (69069 * seed + 1) & _U32
        self.mti = _N

    def _genrand(self) -> float:
        mt = self.mt
        if self.mti >= _N:
            if self.mti == _N + 1:
                self._sgenrand(4357)
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & _TEMPERING_MASK_B
        y ^= (y << 15) & _TEMPERING_MASK_C
        y ^= y >> 18
        return float(y) * 2.3283064365386963e-10

    def unif_rand(self) -> float:
        """``unif_rand()``: a draw in (0, 1)."""
        x = self._genrand()
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    def runif(self, n: int) -> list[float]:
        return [self.unif_rand() for _ in range(n)]

    def _rbits(self, bits: int) -> int:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: float) -> int:
        """``R_unif_index(dn)``: uniform integer in [0, dn)."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    # -- sampling ---------------------------------------------------------
    def sample_int(self, n: int, size: int | None = None) -> list[int]:
        """``sample.int(n, size)`` without replacement; 1-based results."""
        if size is None:
            size = n
        if size > n:
            raise ValueError(
                "cannot take a sample larger than the population when 'replace = FALSE'"
            )
        if n > 1e7 and size <= n / 2:
            raise NotImplementedError("R's hashed sampling (n > 1e7) is not emulated")
        if size < 2:
            return [self.unif_index(n) + 1 for _ in range(size)]
        x = list(range(n))
        out = [0] * size
        for i in range(size):
            j = self.unif_index(n)
            out[i] = x[j] + 1
            n -= 1
            x[j] = x[n]
        return out

    def sample(self, x: list, size: int | None = None) -> list:
        """``sample(x, size)`` without replacement for a vector of length != 1."""
        if len(x) == 1:
            raise ValueError("sample() on a length-1 vector has special R semantics")
        idx = self.sample_int(len(x), len(x) if size is None else size)
        return [x[i - 1] for i in idx]
