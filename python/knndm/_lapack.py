"""Bit-exact port of the part of reference LAPACK ``DGESDD`` that R 4.5.0 runs
for ``svd(x, nu = 0, nv = k)`` -> ``La.svd`` -> ``dgesdd(JOBZ='S')`` on a TALL
matrix (m >= n), returning only the singular values and the right singular
vectors (``VT``).

Sources ported: R-4-5-0 ``src/modules/lapack/{dlapack.f,dlamch.f,dlartg.f90}``
and ``src/extra/blas/{blas.f,blas2.f90}`` (LAPACK 3.12.x flavour: DLARF1F,
Blue's DNRM2, the 3.10 DLARTG).  Copies live in ``reference/native/lapack``.

Paths ported (Fortran DGESDD, M >= N):
  * Path 3  (M >= MNTHR = INT(N*11/6)):  DGEQRF(DGEQR2) -> R -> DGEBRD(DGEBD2)
            -> DBDSDC('U','I') -> DORMBR('P','R','T').
  * Path 5  (M <  MNTHR):                DGEBRD(DGEBD2) on A -> DBDSDC
            -> DORMBR('P','R','T').
The left-vector work (DORGQR, DORMBR('Q'), DGEMM, U rotations) never feeds
back into S or VT and is therefore skipped.

Exactness notes: no FMA is used (plain IEEE double ops).  Reductions are kept
sequential: ``np.cumsum`` is a strictly left-to-right accumulation, so
``0.0 + np.cumsum(products)[-1]`` equals the Fortran running ``TEMP`` loop.
Elementwise numpy operations are IEEE-identical to the scalar ones.

Limitations: n <= 25 (DBDSDC divide-and-conquer, SMLSIZ=25, not ported);
n <= 33 would be needed for DORMLQ to stay unblocked, implied by n <= 25.
DGEQRF/DGEBRD stay unblocked for n < 32 (NB=32), also implied.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# DLAMCH constants (R's dlamch.f, rounding mode assumed: RND = ONE)
# ---------------------------------------------------------------------------
_EPS = 2.0 ** -53                      # DLAMCH('E') = EPSILON(0)*0.5
_PREC = _EPS * 2.0                     # DLAMCH('P') = EPS*RADIX
_SFMIN = 2.2250738585072014e-308       # DLAMCH('S') = TINY (1/HUGE < TINY)
_HUGE = 1.7976931348623157e308         # DLAMCH('O')

# DLARTG (dlartg.f90) constants
_SAFMIN = 2.0 ** -1022
_SAFMAX = 1.0 / _SAFMIN
_RTMIN = math.sqrt(_SAFMIN)
_RTMAX = math.sqrt(_SAFMAX / 2)

# DNRM2 (blas2.f90, Blue's algorithm) constants
_TSML = 2.0 ** -511
_TBIG = 2.0 ** 486
_SSML = 2.0 ** 537
_SBIG = 2.0 ** -538

_SMLSIZ = 25                           # ILAENV(9, 'DBDSDC')

_sign = math.copysign                  # Fortran SIGN(a, b) = copysign(abs(a), b)


def _fsign(a, b):
    return math.copysign(abs(a), b)


def _fdiv(a, b):
    """IEEE division (Python raises on /0)."""
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0.0:
            return math.nan
        neg = (math.copysign(1.0, a) < 0) != (math.copysign(1.0, b) < 0)
        return -math.inf if neg else math.inf


# ---------------------------------------------------------------------------
# BLAS level 1: DNRM2 (blas2.f90, reference BLAS 3.9.1+ / Blue)
# ---------------------------------------------------------------------------
def _dnrm2(x: np.ndarray) -> float:
    n = x.shape[0]
    if n <= 0:
        return 0.0
    ax = np.abs(x)
    amax = float(ax.max())
    if amax <= _TBIG and not bool(np.any((ax > 0.0) & (ax < _TSML))):
        # every element is zero or mid-range: only 'amed' accumulates
        amed = 0.0 + float(np.cumsum(ax * ax)[-1])
        return 1.0 * math.sqrt(amed)
    # generic (rare) path, element by element in Fortran order
    notbig = True
    asml = amed = abig = 0.0
    for v in ax.tolist():
        if v > _TBIG:
            t = v * _SBIG
            abig = abig + t * t
            notbig = False
        elif v < _TSML:
            if notbig:
                t = v * _SSML
                asml = asml + t * t
        else:
            amed = amed + v * v
    if abig > 0.0:
        if amed > 0.0 or amed > _HUGE or amed != amed:
            abig = abig + (amed * _SBIG) * _SBIG
        scl = 1.0 / _SBIG
        sumsq = abig
    elif asml > 0.0:
        if amed > 0.0 or amed > _HUGE or amed != amed:
            amed = math.sqrt(amed)
            asml = math.sqrt(asml) / _SSML
            if asml > amed:
                ymin, ymax = amed, asml
            else:
                ymin, ymax = asml, amed
            scl = 1.0
            q = ymin / ymax
            sumsq = ymax * ymax * (1.0 + q * q)
        else:
            scl = 1.0 / _SSML
            sumsq = asml
    else:
        scl = 1.0
        sumsq = amed
    return scl * math.sqrt(sumsq)


# ---------------------------------------------------------------------------
# DLAPY2
# ---------------------------------------------------------------------------
def _dlapy2(x: float, y: float) -> float:
    if x != x:
        return x
    if y != y:
        return y
    xabs = abs(x)
    yabs = abs(y)
    w = max(xabs, yabs)
    z = min(xabs, yabs)
    if z == 0.0 or w > _HUGE:
        return w
    q = z / w
    return w * math.sqrt(1.0 + q * q)


# ---------------------------------------------------------------------------
# DLARTG (dlartg.f90, LAPACK 3.10+)
# ---------------------------------------------------------------------------
def _dlartg(f: float, g: float):
    """Returns (c, s, r)."""
    f1 = abs(f)
    g1 = abs(g)
    if g == 0.0:
        return 1.0, 0.0, f
    if f == 0.0:
        return 0.0, _fsign(1.0, g), g1
    if _RTMIN < f1 < _RTMAX and _RTMIN < g1 < _RTMAX:
        d = math.sqrt(f * f + g * g)
        c = f1 / d
        r = _fsign(d, f)
        s = g / r
        return c, s, r
    u = min(_SAFMAX, max(_SAFMIN, f1, g1))
    fs = f / u
    gs = g / u
    d = math.sqrt(fs * fs + gs * gs)
    c = abs(fs) / d
    r = _fsign(d, f)
    s = gs / r
    r = r * u
    return c, s, r


# ---------------------------------------------------------------------------
# DLARFG: generate elementary reflector.  x is a (writable) numpy view of the
# N-1 trailing elements; returns (new alpha, tau); x is scaled in place.
# ---------------------------------------------------------------------------
def _dlarfg(alpha: float, x: np.ndarray):
    if x.shape[0] == 0:                 # N <= 1
        return alpha, 0.0
    xnorm = _dnrm2(x)
    if xnorm == 0.0:
        return alpha, 0.0
    beta = -_fsign(_dlapy2(alpha, xnorm), alpha)
    safmin = _SFMIN / _EPS
    knt = 0
    if abs(beta) < safmin:
        rsafmn = 1.0 / safmin
        while True:
            knt += 1
            x *= rsafmn                 # DSCAL
            beta = beta * rsafmn
            alpha = alpha * rsafmn
            if not (abs(beta) < safmin and knt < 20):
                break
        xnorm = _dnrm2(x)
        beta = -_fsign(_dlapy2(alpha, xnorm), alpha)
    tau = (beta - alpha) / beta
    sc = 1.0 / (alpha - beta)
    if sc != 1.0:                       # DSCAL returns early if DA == ONE
        x *= sc
    for _ in range(knt):
        beta = beta * safmin
    return beta, tau


# ---------------------------------------------------------------------------
# ILADLC / ILADLR
# ---------------------------------------------------------------------------
def _iladlc(m: int, n: int, c: np.ndarray) -> int:
    """Last non-zero column of C[:m, :n] (1-based count, 0 if none)."""
    if n == 0:
        return 0
    if m == 0:
        return 0
    if c[0, n - 1] != 0.0 or c[m - 1, n - 1] != 0.0:
        return n
    nz = np.flatnonzero((c[:m, :n] != 0.0).any(axis=0))
    return int(nz[-1]) + 1 if nz.size else 0


def _iladlr(m: int, n: int, c: np.ndarray) -> int:
    """Last non-zero row of C[:m, :n] (1-based count, 0 if none)."""
    if m == 0:
        return 0
    if n == 0:
        return 0
    if c[m - 1, 0] != 0.0 or c[m - 1, n - 1] != 0.0:
        return m
    nz = np.flatnonzero((c[:m, :n] != 0.0).any(axis=1))
    return int(nz[-1]) + 1 if nz.size else 0


# ---------------------------------------------------------------------------
# DLARF1F: apply H = I - tau * v v**T with v(1) = 1 (implicit, not stored).
# ``vt`` holds v(2:), C is a writable 2-D view.  Uses DGEMV/DAXPY/DGER/DSCAL
# in exactly the reference order.
# ---------------------------------------------------------------------------
def _dlarf1f_left(vt: np.ndarray, tau: float, c: np.ndarray) -> None:
    mm, nn = c.shape
    lastv = 1
    lastc = 0
    if tau != 0.0:
        lastv = mm
        while lastv > 1 and vt[lastv - 2] == 0.0:
            lastv -= 1
        lastc = _iladlc(lastv, nn, c)
    if lastc == 0:
        return
    if lastv == 1:
        c[0, :lastc] *= (1.0 - tau)     # DSCAL(lastc, ONE-TAU, C, LDC)
        return
    vv = vt[:lastv - 1]
    # DGEMV('Transpose'): w(j) = 0 + 1 * (sequential sum_i C(i+1,j) * v(i+1))
    prod = c[1:lastv, :lastc] * vv[:, None]
    w = 0.0 + np.cumsum(prod, axis=0)[-1]
    w = 0.0 + w                         # y = y + alpha*temp with y = 0
    # DAXPY(lastc, 1, C(1,:), w)
    w = w + c[0, :lastc]
    # DAXPY(lastc, -tau, w, C(1,:))
    mt = -tau
    c[0, :lastc] = c[0, :lastc] + mt * w
    # DGER(lastv-1, lastc, -tau, v, w, C(2:,:)): skips columns with w(j) == 0
    t = mt * w
    cols = np.flatnonzero(w != 0.0)
    if cols.size:
        sub = c[1:lastv, :lastc]
        sub[:, cols] = sub[:, cols] + vv[:, None] * t[cols][None, :]


def _dlarf1f_right(vt: np.ndarray, tau: float, c: np.ndarray) -> None:
    mm, nn = c.shape
    lastv = 1
    lastc = 0
    if tau != 0.0:
        lastv = nn
        while lastv > 1 and vt[lastv - 2] == 0.0:
            lastv -= 1
        lastc = _iladlr(mm, lastv, c)
    if lastc == 0:
        return
    if lastv == 1:
        c[:lastc, 0] *= (1.0 - tau)     # DSCAL(lastc, ONE-TAU, C, 1)
        return
    # DGEMV('No transpose'): y = 0; y += (1*x(j)) * C(:,j) for j = 1..lastv-1
    w = np.zeros(lastc)
    for j in range(lastv - 1):
        temp = 1.0 * float(vt[j])
        w = w + temp * c[:lastc, j + 1]
    # DAXPY(lastc, 1, C(:,1), w)
    w = w + c[:lastc, 0]
    # DAXPY(lastc, -tau, w, C(:,1))
    mt = -tau
    c[:lastc, 0] = c[:lastc, 0] + mt * w
    # DGER(lastc, lastv-1, -tau, w, v, C(:,2:)): skips columns with v(j) == 0
    for j in range(lastv - 1):
        vj = float(vt[j])
        if vj != 0.0:
            temp = mt * vj
            c[:lastc, j + 1] = c[:lastc, j + 1] + w * temp


# ---------------------------------------------------------------------------
# DGEQR2 (unblocked; DGEQRF uses it directly because NB=32 >= K for n < 32)
# ---------------------------------------------------------------------------
def _dgeqr2(a: np.ndarray) -> None:
    m, n = a.shape
    for i in range(min(m, n)):
        alpha, tau = _dlarfg(float(a[i, i]), a[i + 1:, i])
        a[i, i] = alpha
        if i < n - 1:
            _dlarf1f_left(a[i + 1:, i], tau, a[i:, i + 1:])


# ---------------------------------------------------------------------------
# DGEBD2 for M >= N (upper bidiagonal).  Returns d, e, taup (tauq unused).
# ---------------------------------------------------------------------------
def _dgebd2(a: np.ndarray):
    m, n = a.shape
    d = [0.0] * n
    e = [0.0] * max(n - 1, 0)
    taup = [0.0] * n
    for i in range(n):
        alpha, tauq = _dlarfg(float(a[i, i]), a[i + 1:, i])
        a[i, i] = alpha
        d[i] = alpha
        if i < n - 1:
            _dlarf1f_left(a[i + 1:, i], tauq, a[i:, i + 1:])
            alpha, tp = _dlarfg(float(a[i, i + 1]), a[i, i + 2:])
            a[i, i + 1] = alpha
            taup[i] = tp
            e[i] = alpha
            _dlarf1f_right(a[i, i + 2:], tp, a[i + 1:, i + 1:])
        else:
            taup[i] = 0.0
    return d, e, taup


# ---------------------------------------------------------------------------
# DLASV2 / DLAS2
# ---------------------------------------------------------------------------
def _dlasv2(f: float, g: float, h: float):
    """Returns (ssmin, ssmax, snr, csr, snl, csl)."""
    ft = f
    fa = abs(ft)
    ht = h
    ha = abs(h)
    pmax = 1
    swap = ha > fa
    if swap:
        pmax = 3
        ft, ht = ht, ft
        fa, ha = ha, fa
    gt = g
    ga = abs(gt)
    if ga == 0.0:
        ssmin = ha
        ssmax = fa
        clt = 1.0
        crt = 1.0
        slt = 0.0
        srt = 0.0
    else:
        gasmal = True
        if ga > fa:
            pmax = 2
            if (fa / ga) < _EPS:
                gasmal = False
                ssmax = ga
                if ha > 1.0:
                    ssmin = fa / (ga / ha)
                else:
                    ssmin = (fa / ga) * ha
                clt = 1.0
                slt = ht / gt
                srt = 1.0
                crt = ft / gt
        if gasmal:
            dd = fa - ha
            if dd == fa:
                l = 1.0
            else:
                l = dd / fa
            m = gt / ft
            t = 2.0 - l
            mm = m * m
            tt = t * t
            s = math.sqrt(tt + mm)
            if l == 0.0:
                r = abs(m)
            else:
                r = math.sqrt(l * l + mm)
            a = 0.5 * (s + r)
            ssmin = ha / a
            ssmax = fa * a
            if mm == 0.0:
                if l == 0.0:
                    t = _fsign(2.0, ft) * _fsign(1.0, gt)
                else:
                    t = gt / _fsign(dd, ft) + m / t
            else:
                t = (m / (s + t) + m / (r + l)) * (1.0 + a)
            l = math.sqrt(t * t + 4.0)
            crt = 2.0 / l
            srt = t / l
            clt = (crt + srt * m) / a
            slt = (ht / ft) * srt / a
    if swap:
        csl = srt
        snl = crt
        csr = slt
        snr = clt
    else:
        csl = clt
        snl = slt
        csr = crt
        snr = srt
    if pmax == 1:
        tsign = _fsign(1.0, csr) * _fsign(1.0, csl) * _fsign(1.0, f)
    elif pmax == 2:
        tsign = _fsign(1.0, snr) * _fsign(1.0, csl) * _fsign(1.0, g)
    else:
        tsign = _fsign(1.0, snr) * _fsign(1.0, snl) * _fsign(1.0, h)
    ssmax = _fsign(ssmax, tsign)
    ssmin = _fsign(ssmin, tsign * _fsign(1.0, f) * _fsign(1.0, h))
    return ssmin, ssmax, snr, csr, snl, csl


def _dlas2(f: float, g: float, h: float):
    """Returns (ssmin, ssmax)."""
    fa = abs(f)
    ga = abs(g)
    ha = abs(h)
    fhmn = min(fa, ha)
    fhmx = max(fa, ha)
    if fhmn == 0.0:
        ssmin = 0.0
        if fhmx == 0.0:
            ssmax = ga
        else:
            q = min(fhmx, ga) / max(fhmx, ga)
            ssmax = max(fhmx, ga) * math.sqrt(1.0 + q * q)
    else:
        if ga < fhmx:
            as_ = 1.0 + fhmn / fhmx
            at = (fhmx - fhmn) / fhmx
            q = ga / fhmx
            au = q * q
            c = 2.0 / (math.sqrt(as_ * as_ + au) + math.sqrt(at * at + au))
            ssmin = fhmn * c
            ssmax = fhmx / c
        else:
            au = fhmx / ga
            if au == 0.0:
                ssmin = (fhmn * fhmx) / ga
                ssmax = ga
            else:
                as_ = 1.0 + fhmn / fhmx
                at = (fhmx - fhmn) / fhmx
                p1 = as_ * au
                p2 = at * au
                c = 1.0 / (math.sqrt(1.0 + p1 * p1) + math.sqrt(1.0 + p2 * p2))
                ssmin = (fhmn * c) * au
                ssmin = ssmin + ssmin
                ssmax = ga / (c + c)
    return ssmin, ssmax


# ---------------------------------------------------------------------------
# DLASR(SIDE='L', PIVOT='V', DIRECT=F/B) applied to rows ll0 .. ll0+mm-1 of VT
# (only the VT / left-side variant is needed).  cs, ss are 1-based lists.
# ---------------------------------------------------------------------------
def _dlasr_lv(forward: bool, mm: int, cs, ss, vt: np.ndarray, ll0: int) -> None:
    js = range(1, mm) if forward else range(mm - 1, 0, -1)
    for j in js:
        ct = cs[j]
        st = ss[j]
        if ct != 1.0 or st != 0.0:
            r0 = vt[ll0 + j - 1]
            r1 = vt[ll0 + j]
            temp = r1.copy()
            new1 = ct * temp - st * r0
            new0 = st * temp + ct * r0
            r1[:] = new1
            r0[:] = new0


def _drot_rows(vt: np.ndarray, i0: int, i1: int, c: float, s: float) -> None:
    dx = vt[i0]
    dy = vt[i1]
    dtemp = c * dx + s * dy
    ny = c * dy - s * dx
    dy[:] = ny
    dx[:] = dtemp


# ---------------------------------------------------------------------------
# DBDSQR('U', N, NCVT=N, NRU=N, NCC=0): only d, e, VT are tracked.
# d, e are 1-based lists (index 0 unused); vt is an (n, n) ndarray.
# Returns info.
# ---------------------------------------------------------------------------
def _dbdsqr(n: int, d, e, vt: np.ndarray) -> int:
    if n == 0:
        return 0
    eps = _EPS
    unfl = _SFMIN
    nm1 = n - 1
    idir = 0
    tolmul = max(10.0, min(100.0, math.pow(eps, -0.125)))
    tol = tolmul * eps
    maxitr = 6
    smax = 0.0
    for i in range(1, n + 1):
        smax = max(smax, abs(d[i]))
    for i in range(1, n):
        smax = max(smax, abs(e[i]))
    smin = 0.0
    # tol >= 0
    sminoa = abs(d[1])
    if sminoa != 0.0:
        mu = sminoa
        for i in range(2, n + 1):
            mu = abs(d[i]) * _fdiv(mu, mu + abs(e[i - 1]))
            sminoa = min(sminoa, mu)
            if sminoa == 0.0:
                break
    sminoa = sminoa / math.sqrt(float(n))
    thresh = max(tol * sminoa, maxitr * (n * (n * unfl)))
    maxitdivn = maxitr * n
    iterdivn = 0
    it = -1
    oldll = -1
    oldm = -1
    m = n
    info = 0
    cr = [0.0] * (n + 1)   # WORK(1..)      (cs / cosr)
    sr = [0.0] * (n + 1)   # WORK(NM1+1..)  (sn / sinr)
    cl = [0.0] * (n + 1)   # WORK(NM12+1..) (oldcs / cosl)
    sl = [0.0] * (n + 1)   # WORK(NM13+1..) (oldsn / sinl)
    shift = 0.0
    while True:                                   # label 60
        if m <= 1:
            break
        if it >= n:
            it -= n
            iterdivn += 1
            if iterdivn >= maxitdivn:             # label 200
                info = 0
                for i in range(1, n):
                    if e[i] != 0.0:
                        info += 1
                return info
        smax = abs(d[m])
        found = False
        ll = 0
        for lll in range(1, m):
            ll = m - lll
            abss = abs(d[ll])
            abse = abs(e[ll])
            if abse <= thresh:
                found = True
                break
            smax = max(smax, abss, abse)
        if found:                                 # label 80
            e[ll] = 0.0
            if ll == m - 1:
                m -= 1
                continue
        else:
            ll = 0
        ll += 1                                   # label 90
        if ll == m - 1:
            sigmn, sigmx, sinr, cosr, sinl, cosl = _dlasv2(d[m - 1], e[m - 1], d[m])
            d[m - 1] = sigmx
            e[m - 1] = 0.0
            d[m] = sigmn
            _drot_rows(vt, m - 2, m - 1, cosr, sinr)
            m -= 2
            continue
        if ll > oldm or m < oldll:
            if abs(d[ll]) >= abs(d[m]):
                idir = 1
            else:
                idir = 2
        goto60 = False
        if idir == 1:
            if abs(e[m - 1]) <= abs(tol) * abs(d[m]):
                e[m - 1] = 0.0
                continue
            mu = abs(d[ll])
            smin = mu
            for lll in range(ll, m):
                if abs(e[lll]) <= tol * mu:
                    e[lll] = 0.0
                    goto60 = True
                    break
                mu = abs(d[lll + 1]) * _fdiv(mu, mu + abs(e[lll]))
                smin = min(smin, mu)
        else:
            if abs(e[ll]) <= abs(tol) * abs(d[ll]):
                e[ll] = 0.0
                continue
            mu = abs(d[m])
            smin = mu
            for lll in range(m - 1, ll - 1, -1):
                if abs(e[lll]) <= tol * mu:
                    e[lll] = 0.0
                    goto60 = True
                    break
                mu = abs(d[lll]) * _fdiv(mu, mu + abs(e[lll]))
                smin = min(smin, mu)
        if goto60:
            continue
        oldll = ll
        oldm = m
        if (n * tol * _fdiv(smin, smax)) <= max(eps, 0.01 * tol):
            shift = 0.0
        else:
            if idir == 1:
                sll = abs(d[ll])
                shift, r = _dlas2(d[m - 1], e[m - 1], d[m])
            else:
                sll = abs(d[m])
                shift, r = _dlas2(d[ll], e[ll], d[ll + 1])
            if sll > 0.0:
                q = shift / sll
                if q * q < eps:
                    shift = 0.0
        it = it + m - ll
        cnt = m - ll + 1
        if shift == 0.0:
            if idir == 1:
                cs = 1.0
                oldcs = 1.0
                oldsn = 0.0
                for i in range(ll, m):
                    cs, sn, r = _dlartg(d[i] * cs, e[i])
                    if i > ll:
                        e[i - 1] = oldsn * r
                    oldcs, oldsn, d[i] = _dlartg(oldcs * r, d[i + 1] * sn)
                    k = i - ll + 1
                    cr[k] = cs
                    sr[k] = sn
                    cl[k] = oldcs
                    sl[k] = oldsn
                h = d[m] * cs
                d[m] = h * oldcs
                e[m - 1] = h * oldsn
                _dlasr_lv(True, cnt, cr, sr, vt, ll - 1)
                if abs(e[m - 1]) <= thresh:
                    e[m - 1] = 0.0
            else:
                cs = 1.0
                oldcs = 1.0
                oldsn = 0.0
                for i in range(m, ll, -1):
                    cs, sn, r = _dlartg(d[i] * cs, e[i - 1])
                    if i < m:
                        e[i] = oldsn * r
                    oldcs, oldsn, d[i] = _dlartg(oldcs * r, d[i - 1] * sn)
                    k = i - ll
                    cr[k] = cs
                    sr[k] = -sn
                    cl[k] = oldcs
                    sl[k] = -oldsn
                h = d[ll] * cs
                d[ll] = h * oldcs
                e[ll] = h * oldsn
                _dlasr_lv(False, cnt, cl, sl, vt, ll - 1)
                if abs(e[ll]) <= thresh:
                    e[ll] = 0.0
        else:
            if idir == 1:
                f = (abs(d[ll]) - shift) * (_fsign(1.0, d[ll]) + shift / d[ll])
                g = e[ll]
                for i in range(ll, m):
                    cosr, sinr, r = _dlartg(f, g)
                    if i > ll:
                        e[i - 1] = r
                    f = cosr * d[i] + sinr * e[i]
                    e[i] = cosr * e[i] - sinr * d[i]
                    g = sinr * d[i + 1]
                    d[i + 1] = cosr * d[i + 1]
                    cosl, sinl, r = _dlartg(f, g)
                    d[i] = r
                    f = cosl * e[i] + sinl * d[i + 1]
                    d[i + 1] = cosl * d[i + 1] - sinl * e[i]
                    if i < m - 1:
                        g = sinl * e[i + 1]
                        e[i + 1] = cosl * e[i + 1]
                    k = i - ll + 1
                    cr[k] = cosr
                    sr[k] = sinr
                    cl[k] = cosl
                    sl[k] = sinl
                e[m - 1] = f
                _dlasr_lv(True, cnt, cr, sr, vt, ll - 1)
                if abs(e[m - 1]) <= thresh:
                    e[m - 1] = 0.0
            else:
                f = (abs(d[m]) - shift) * (_fsign(1.0, d[m]) + shift / d[m])
                g = e[m - 1]
                for i in range(m, ll, -1):
                    cosr, sinr, r = _dlartg(f, g)
                    if i < m:
                        e[i] = r
                    f = cosr * d[i] + sinr * e[i - 1]
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i]
                    g = sinr * d[i - 1]
                    d[i - 1] = cosr * d[i - 1]
                    cosl, sinl, r = _dlartg(f, g)
                    d[i] = r
                    f = cosl * e[i - 1] + sinl * d[i - 1]
                    d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1]
                    if i > ll + 1:
                        g = sinl * e[i - 2]
                        e[i - 2] = cosl * e[i - 2]
                    k = i - ll
                    cr[k] = cosr
                    sr[k] = -sinr
                    cl[k] = cosl
                    sl[k] = -sinl
                e[ll] = f
                if abs(e[ll]) <= thresh:
                    e[ll] = 0.0
                _dlasr_lv(False, cnt, cl, sl, vt, ll - 1)
    # label 160: make singular values positive
    for i in range(1, n + 1):
        if d[i] == 0.0:
            d[i] = 0.0
        if d[i] < 0.0:
            d[i] = -d[i]
            vt[i - 1] *= -1.0                     # DSCAL(NCVT, NEGONE, VT(I,1))
    # sort into decreasing order (insertion sort, one swap per vector)
    for i in range(1, n):
        isub = 1
        smin = d[1]
        for j in range(2, n + 2 - i):
            if d[j] <= smin:
                isub = j
                smin = d[j]
        last = n + 1 - i
        if isub != last:
            d[isub] = d[last]
            d[last] = smin
            tmp = vt[isub - 1].copy()
            vt[isub - 1] = vt[last - 1]
            vt[last - 1] = tmp
    return 0


# ---------------------------------------------------------------------------
# DLASDQ('U', SQRE=0, N, NCVT=N, NRU=N, NCC=0) then DBDSDC's final sort.
# d, e: 1-based lists.  vt starts as the identity.
# ---------------------------------------------------------------------------
def _dbdsdc_small(n: int, d, e, vt: np.ndarray) -> int:
    """DBDSDC('U','I') for 2 <= N <= SMLSIZ (DLASDQ -> DBDSQR path)."""
    vt[:, :] = np.eye(n)                          # DLASET('A', N, N, 0, 1, VT)
    info = _dbdsqr(n, d, e, vt)
    # DLASDQ: selection sort (ascending!) of D with row swaps of VT
    for i in range(1, n + 1):
        isub = i
        smin = d[i]
        for j in range(i + 1, n + 1):
            if d[j] < smin:
                isub = j
                smin = d[j]
        if isub != i:
            d[isub] = d[i]
            d[i] = smin
            tmp = vt[isub - 1].copy()
            vt[isub - 1] = vt[i - 1]
            vt[i - 1] = tmp
    if info != 0:
        return info
    # DBDSDC label 40: sort into decreasing order
    for ii in range(2, n + 1):
        i = ii - 1
        kk = i
        p = d[i]
        for j in range(ii, n + 1):
            if d[j] > p:
                kk = j
                p = d[j]
        if kk != i:
            d[kk] = d[i]
            d[i] = p
            tmp = vt[i - 1].copy()
            vt[i - 1] = vt[kk - 1]
            vt[kk - 1] = tmp
    return 0


# ---------------------------------------------------------------------------
# DORMBR('P','R','T', N, N, N, A, TAUP, VT) -> DORMLQ -> DORML2 -> DLARF1F
# a holds the DGEBD2 output (row i holds the P(i) reflector at columns i+2..)
# ---------------------------------------------------------------------------
def _dormbr_prt(a: np.ndarray, taup, vt: np.ndarray) -> None:
    n = vt.shape[0]
    if n <= 1:
        return
    k = n - 1
    # DORML2('R','N', M=n, N=n-1, K=n-1, A(1,2), TAU, C=VT(1,2)): I = K..1
    for i in range(k, 0, -1):
        # A_sub(I, I+1:) == a[I-1, I+1:n]; C = VT_sub(:, I:) == vt[:, I:n]
        _dlarf1f_right(a[i - 1, i + 1:n], taup[i - 1], vt[:, i:n])


# ---------------------------------------------------------------------------
# DLASCL('G', 0, 0, CFROM, CTO, M, N, A, ...) on a numpy array
# ---------------------------------------------------------------------------
def _dlascl(cfrom: float, cto: float, a: np.ndarray) -> None:
    smlnum = _SFMIN
    bignum = 1.0 / smlnum
    cfromc = cfrom
    ctoc = cto
    while True:
        cfrom1 = cfromc * smlnum
        if cfrom1 == cfromc:
            mul = ctoc / cfromc
            done = True
        else:
            cto1 = ctoc / bignum
            if cto1 == ctoc:
                mul = ctoc
                done = True
                cfromc = 1.0
            elif abs(cfrom1) > abs(ctoc) and ctoc != 0.0:
                mul = smlnum
                done = False
                cfromc = cfrom1
            elif abs(cto1) > abs(cfromc):
                mul = bignum
                done = False
                ctoc = cto1
            else:
                mul = ctoc / cfromc
                done = True
                if mul == 1.0:
                    return
        a *= mul
        if done:
            return


# ---------------------------------------------------------------------------
# DGESDD driver (JOBZ='S', M >= N)
# ---------------------------------------------------------------------------
def svd_vt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact ``La.svd(x, nu=0, nv=ncol(x))`` of R 4.5.0 (reference LAPACK).

    Returns ``(d, vt)``: singular values (length n) and ``vt`` (n x n, rows are
    right singular vectors), for a tall (m >= n) finite matrix with n <= 25.
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 2:
        raise ValueError("x must be a 2-D array")
    m, n = xa.shape
    if m < n:
        raise NotImplementedError("only tall matrices (m >= n) are supported")
    if n > _SMLSIZ:
        raise NotImplementedError(
            "n > 25 needs DBDSDC divide-and-conquer (DLASD0), not ported")
    if m == 0 or n == 0:
        return np.zeros(0), np.zeros((n, n))
    if not np.all(np.isfinite(xa)):
        raise ValueError("infinite or missing values in 'x'")

    a = np.array(xa, dtype=np.float64, order="F", copy=True)
    minmn = n
    mnthr = int(minmn * 11.0 / 6.0)

    # --- scaling (DLANGE 'M', DLASCL) ---
    eps = _PREC
    smlnum = math.sqrt(_SFMIN) / eps
    bignum = 1.0 / smlnum
    anrm = float(np.abs(a).max())
    iscl = 0
    if anrm > 0.0 and anrm < smlnum:
        iscl = 1
        _dlascl(anrm, smlnum, a)
    elif anrm > bignum:
        iscl = 1
        _dlascl(anrm, bignum, a)

    if m >= mnthr:
        # Path 3: QR, then bidiagonalise R
        _dgeqr2(a)
        r = np.zeros((n, n), order="F")
        for j in range(n):                       # DLACPY 'U' + DLASET 'L' zero
            r[:j + 1, j] = a[:j + 1, j]
        work = r
    else:
        # Path 5: bidiagonalise A directly
        work = a
    d0, e0, taup = _dgebd2(work)

    d = [0.0] + d0
    e = [0.0] + e0 + [0.0]
    vt = np.zeros((n, n))
    if n == 1:
        # DBDSDC N == 1: VT(1,1) = 1, D = ABS(D)
        vt[0, 0] = 1.0
        d[1] = abs(d[1])
        info = 0
    else:
        info = _dbdsdc_small(n, d, e, vt)
    if info != 0:
        raise RuntimeError(f"DBDSDC did not converge (info={info})")
    _dormbr_prt(work, taup, vt)

    s = np.array(d[1:], dtype=np.float64)
    if iscl == 1:
        if anrm > bignum:
            _dlascl(bignum, anrm, s)
        if anrm < smlnum:
            _dlascl(smlnum, anrm, s)
    return s, vt
