//! Exact software emulation of x87 80-bit `long double` accumulation.
//!
//! R on x86-64 sums (`sum`, `cumsum`, `colMeans`, `mean`) in `long double`,
//! i.e. a 64-bit significand with round-half-to-even. Values here are
//! `(-1)^neg * mant * 2^exp` with `mant` normalised to bit 63 (or zero).

#[derive(Clone, Copy, Debug)]
pub struct F80 {
    neg: bool,
    mant: u64,
    exp: i32,
}

/// Round `r * 2^e` (+ a sticky fraction below the LSB) to `bits` bits, RNE.
fn round_u128(mut r: u128, mut e: i32, sticky: bool, bits: u32) -> (u128, i32) {
    if r == 0 {
        return (0, 0);
    }
    let bl = 128 - r.leading_zeros();
    if bl > bits {
        let shift = bl - bits;
        let q = r >> shift;
        let rem = r & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        let mut q2 = q;
        if rem > half || (rem == half && (sticky || (q & 1) == 1)) {
            q2 += 1;
        }
        let mut sh = shift as i32;
        if (128 - q2.leading_zeros()) > bits {
            q2 >>= 1;
            sh += 1;
        }
        r = q2;
        e += sh;
    } else {
        // Exactly representable; a sticky fraction alone is < half an ulp.
        let sh = bits - bl;
        r <<= sh;
        e -= sh as i32;
    }
    (r, e)
}

fn ldexp(m: f64, e: i32) -> f64 {
    let mut x = m;
    let mut e = e;
    while e > 1000 {
        x *= 2f64.powi(1000);
        e -= 1000;
    }
    while e < -1000 {
        x *= 2f64.powi(-1000);
        e += 1000;
    }
    x * 2f64.powi(e)
}

impl F80 {
    pub const ZERO: F80 = F80 { neg: false, mant: 0, exp: 0 };

    pub fn from_f64(x: f64) -> F80 {
        if x == 0.0 {
            return F80::ZERO;
        }
        let bits = x.to_bits();
        let neg = (bits >> 63) == 1;
        let bexp = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        let (m, e) = if bexp == 0 { (frac, -1074) } else { (frac | (1u64 << 52), bexp - 1075) };
        let lz = m.leading_zeros();
        F80 { neg, mant: m << lz, exp: e - lz as i32 }
    }

    pub fn from_u64(x: u64) -> F80 {
        if x == 0 {
            return F80::ZERO;
        }
        let lz = x.leading_zeros();
        F80 { neg: false, mant: x << lz, exp: -(lz as i32) }
    }

    fn mag_ge(&self, o: &F80) -> bool {
        // both normalised and non-zero
        (self.exp, self.mant) >= (o.exp, o.mant)
    }

    pub fn add(self, o: F80) -> F80 {
        if o.mant == 0 {
            return self;
        }
        if self.mant == 0 {
            return o;
        }
        let (big, small) = if self.mag_ge(&o) { (self, o) } else { (o, self) };
        let shift = (big.exp - small.exp) as u32;
        let b = (big.mant as u128) << 62;
        let s_full = (small.mant as u128) << 62;
        let (s, sticky) = if shift >= 126 {
            (0u128, true)
        } else {
            let s = s_full >> shift;
            (s, (s << shift) != s_full)
        };
        let e = big.exp - 62;
        let (r, sticky) = if big.neg == small.neg {
            (b + s, sticky)
        } else if sticky {
            (b - s - 1, true)
        } else {
            (b - s, false)
        };
        if r == 0 {
            return F80::ZERO;
        }
        let (m, e2) = round_u128(r, e, sticky, 64);
        F80 { neg: big.neg, mant: m as u64, exp: e2 }
    }

    pub fn div_u64(self, n: u64) -> F80 {
        if self.mant == 0 {
            return self;
        }
        let num = (self.mant as u128) << 64;
        let q = num / n as u128;
        let rem = num % n as u128;
        let (m, e) = round_u128(q, self.exp - 64, rem != 0, 64);
        F80 { neg: self.neg, mant: m as u64, exp: e }
    }

    pub fn to_f64(self) -> f64 {
        if self.mant == 0 {
            return 0.0;
        }
        // value = mant * 2^exp with mant in [2^63, 2^64): the leading bit has
        // weight 2^(exp + 63). Binary64 keeps 53 bits for normal numbers and
        // fewer below 2^-1022, never bits below 2^-1074.
        let top = self.exp + 63;
        let bits: i32 = if top >= -1022 { 53 } else { (53 - (-1022 - top)).max(0) };
        let v = if bits == 0 {
            // below half the smallest subnormal or exactly a tie to zero
            let half = top == -1075 && self.mant > (1u64 << 63);
            if half { 2f64.powi(-1074) } else { 0.0 }
        } else {
            let (m, e) = round_u128(self.mant as u128, self.exp, false, bits as u32);
            if e + (128 - (m.leading_zeros() as i32)) - 1 > 1023 {
                f64::INFINITY
            } else {
                ldexp(m as f64, e)
            }
        };
        if self.neg {
            -v
        } else {
            v
        }
    }
}

