"""Target-weighted cross-validation (TWCV) for kNNDM and other CV designs.

Port of the weighting estimators in Brenning & Suesse (2026), *Aligning
Validation with Deployment in Spatial Prediction: Target-Weighted
Cross-Validation* (arXiv:2603.29981), reference code
https://github.com/alexanderbrenning/TWCVpaper (``R/weights.R``,
``R/weighting_twcv.R``, ``R/cv_designs_simstudy.R``, ``R/cv_generic.R``,
``R/task_descriptors.R``, ``R/loss_metrics.R``).

Each held-out CV prediction is a *validation task* described by its realised
prediction distance ``d`` (distance to the nearest training point of the other
folds) and, optionally, covariates. Each prediction location is a *deployment
task* with ``d`` = distance to the nearest sample. The descriptors are
discretised at quantiles of the deployment distribution and the validation
tasks are raked (iterative proportional fitting) so that their weighted
margins match the deployment margins. Weighted CV losses then estimate the
error of the map rather than of the CV design.

Numerics follow R 4.5 on x86-64 Windows (``quantile`` type 7, ``cut``, ``sum``
and ``mean`` in 80-bit long double, and R-level ``sqrt`` rounded through the x87
80-bit format), so results agree with the R code bit for bit. On Linux, R's
``sqrt`` is correctly rounded, so ``d_cv`` and RMSE can differ by 1 ulp.
One deliberate difference: when a deployment quantile bin is empty (with tied
descriptors, e.g. distances on a regular raster, or fewer distinct prediction
values than bins), R's ``tabulate_proportions`` silently shifts the target
proportions onto the wrong bins; here the empty bin gets a target share of 0
and a warning is issued. Importance-weighted CV (IWCV) is not ported.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping

import numpy as np

from ._rnum import LongDouble, _round_bits, _to_double, r_sqrt, r_sum
from ._stats import _sqdist, knnx_dist

__all__ = [
    "TWCVWeights",
    "twcv_weights",
    "dwcv_weights",
    "rake_weights",
    "weighted_cv_error",
    "weight_diagnostics",
]


@dataclass
class TWCVWeights:
    """Calibration weights for the validation tasks (one per training point).

    ``weights`` (shrunk, mean 1) are what to use; ``weights_raw`` are the raked
    weights before shrinkage. ``d_cv`` and ``d_pred`` are the realised CV
    prediction distances and the deployment (prediction-to-sample) distances.
    ``n_bins`` gives the number of levels each balancing variable was cut into.
    """

    weights: np.ndarray
    weights_raw: np.ndarray
    converged: bool
    iterations: int
    balancing_vars: list[str]
    shrink_lambda: float
    balance_by: float
    d_cv: np.ndarray
    d_pred: np.ndarray
    n_bins: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# R numeric primitives
# --------------------------------------------------------------------------
def _ld(fr: Fraction) -> Fraction:
    """Round an exact rational to x87 long double (64-bit significand)."""
    if fr == 0:
        return Fraction(0)
    neg = fr < 0
    num, den = abs(fr.numerator), fr.denominator
    s = max(0, 67 + den.bit_length() - num.bit_length())
    q, r = divmod(num << s, den)
    q = (q << 1) | (1 if r else 0)
    m, e = _round_bits(q, -s - 1, 64)
    v = Fraction(m) * (Fraction(2) ** e)
    return -v if neg else v


def _fr_to_double(fr: Fraction) -> float:
    if fr == 0:
        return 0.0
    neg = fr < 0
    num, den = abs(fr.numerator), fr.denominator
    s = max(0, 67 + den.bit_length() - num.bit_length())
    q, r = divmod(num << s, den)
    q = (q << 1) | (1 if r else 0)
    v = _to_double(q, -s - 1)
    return -v if neg else v


def r_mean(x) -> float:
    """``mean(x)`` for finite doubles (summary.c real_mean, long double)."""
    x = [float(v) for v in x]
    n = len(x)
    if n == 0:
        return math.nan
    if not all(math.isfinite(v) for v in x):
        return float(np.mean(x))  # NaN / Inf propagate as in R
    acc = LongDouble()
    for v in x:
        acc.add(v)
    if math.isfinite(acc.to_double()):
        s = _ld(Fraction(acc.m) * Fraction(2) ** acc.e / n)
        t = Fraction(0)
        for v in x:
            t = _ld(t + _ld(Fraction(v) - s))
        s = _ld(s + _ld(t / n))
    else:  # the sum overflowed: R retries with x[k] / n terms
        s = Fraction(0)
        for v in x:
            s = _ld(s + Fraction(v / n))  # dx[k] / n is a double division
        t = Fraction(0)
        for v in x:
            t = _ld(t + _ld(_ld(Fraction(v) - s) / n))
        s = _ld(s + t)
    return _fr_to_double(s)


def _r_seq_by(by: float) -> np.ndarray:
    """``seq(0, 1, by = by)``."""
    if not (0 < by <= 1):
        raise ValueError("balance_by must be in (0, 1]")
    n = int(1.0 / by + 1e-10)
    return np.minimum(np.array([0.0 + i * by for i in range(n + 1)]), 1.0)


def r_quantile7(x, probs) -> np.ndarray:
    """``stats::quantile(x, probs, na.rm = TRUE, type = 7, names = FALSE)``."""
    x = np.asarray(x, dtype=float)
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    out = np.empty(len(probs))
    for j, p in enumerate(np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)):
        if n == 0:
            out[j] = math.nan
            continue
        index = 1 + max(n - 1, 0) * p
        lo = math.floor(index)  # R's default fuzz is 0 for type 7
        hi = math.ceil(index)
        qs = x[lo - 1]
        if index > lo and x[hi - 1] != qs:
            h = index - lo
            qs = (1 - h) * qs + h * x[hi - 1]
        out[j] = qs
    return out


# --------------------------------------------------------------------------
# discretisation (prepare_for_balancing)
# --------------------------------------------------------------------------
def _is_numeric(a: np.ndarray) -> bool:
    return a.dtype.kind in "fiub"


def _is_na(v) -> bool:
    return v is None or type(v).__name__ in ("NAType", "NaTType") or (
        isinstance(v, (float, np.floating)) and math.isnan(v)
    )


def _r_key(v):
    """Category key as R's ``as.character`` gives it (None = NA)."""
    if _is_na(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.15g" % (float(v) + 0.0)  # as.character(): 15 digits, -0 -> "0"
    return str(v)


def _discretise(x, x_ref, by: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Codes 1..L for ``x`` and ``x_ref`` (0 = NA) and the number of levels L.

    Numeric descriptors with more than two distinct values are cut at the
    type-7 quantiles of ``x_ref`` (outer breaks widened to -Inf/Inf); anything
    else is treated as categorical with the levels found in ``x_ref``.
    """
    x = np.asarray(x)
    x_ref = np.asarray(x_ref)
    if _is_numeric(x_ref):
        ref = x_ref.astype(float)
        ok_ref = ref[~np.isnan(ref)]
        numeric = len(np.unique(ok_ref)) > 2
    else:
        numeric = False

    if numeric:
        qt = r_quantile7(ref, _r_seq_by(by))
        qt = np.array(list(dict.fromkeys(qt.tolist())))  # unique(), keeps order

        def code(v):
            v = np.asarray(v, dtype=float)
            if len(qt) < 2:  # degenerate: one level
                return np.where(np.isnan(v), 0, 1).astype(np.int64), 1
            br = qt.copy()
            br[0], br[-1] = -math.inf, math.inf
            br = np.sort(br)
            # cut(include.lowest = TRUE, right = TRUE): (b[i-1], b[i]], first closed
            c = np.searchsorted(br, v, side="left")
            c = np.where(c == 0, 1, c)
            return np.where(np.isnan(v), 0, c).astype(np.int64), len(br) - 1

        cx, L = code(x)
        cr, _ = code(ref)
        return cx, cr, L

    def keys(a):
        return [_r_key(v) for v in a.tolist()]

    kref = keys(x_ref)
    levels = sorted({k for k in kref if k is not None})
    pos = {k: i + 1 for i, k in enumerate(levels)}
    cr = np.array([pos.get(k, 0) if k is not None else 0 for k in kref], dtype=np.int64)
    cx = np.array([pos.get(k, 0) if k is not None else 0 for k in keys(x)], dtype=np.int64)
    return cx, cr, len(levels)


def _target_margin(codes: np.ndarray, n_levels: int) -> np.ndarray:
    """``tabulate_proportions``: share of each level among non-NA codes."""
    counts = np.bincount(codes[codes > 0], minlength=n_levels + 1)[1:].astype(float)
    total = counts.sum()
    if total == 0:
        raise ValueError("deployment descriptor has no non-missing values")
    if (counts == 0).any():
        warnings.warn(
            f"{int((counts == 0).sum())} of {n_levels} quantile bins hold no prediction "
            "locations (tied values?); they get a target share of 0, unlike the R code",
            stacklevel=3,
        )
    return counts / total


# --------------------------------------------------------------------------
# raking, normalisation and shrinkage
# --------------------------------------------------------------------------
def rake_weights(
    balance: Mapping[str, np.ndarray],
    target_margins: Mapping[str, np.ndarray],
    base_weights=None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, bool, int]:
    """Iterative proportional fitting (``rake_weights`` in TWCVpaper).

    ``balance[v]`` holds level codes 1..L (0 = missing) for each validation
    task and ``target_margins[v]`` the L target proportions. Rows with a
    missing level keep their weight for that margin, and target levels with
    no validation tasks are left unmatched.
    Returns ``(weights, converged, iterations)``.
    """
    names = list(target_margins)
    if not names:
        raise ValueError("at least one balancing variable is required")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1")
    codes = {m: np.asarray(balance[m], dtype=np.int64) for m in names}
    n = len(codes[names[0]])
    if n == 0:
        raise ValueError("no validation tasks to weight")
    for m in names:
        L = len(target_margins[m])
        if codes[m].shape != (n,) or codes[m].min() < 0 or codes[m].max() > L:
            raise ValueError(f"balance[{m!r}] must hold {n} codes in 0..{L}")
    w = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float).copy()
    if w.shape != (n,):
        raise ValueError("base_weights must have one entry per validation task")
    rows = {
        m: [np.flatnonzero(codes[m] == lev) for lev in range(1, len(target_margins[m]) + 1)]
        for m in names
    }

    rel_change = math.inf
    it = 0
    for it in range(1, max_iter + 1):
        w_old = w.copy()
        for m in names:
            p = np.asarray(target_margins[m], dtype=float)
            total = r_sum(w)
            for lev, idx in enumerate(rows[m]):
                if len(idx) == 0:
                    continue
                current = r_sum(w[idx])
                if current > 0:
                    w[idx] = w[idx] * ((total * p[lev]) / current)
        rel_change = float(np.max(np.abs(w - w_old) / np.maximum(np.abs(w_old), 1e-12)))
        if rel_change < tol:
            break
    converged = it < max_iter or rel_change < tol
    return w, bool(converged), it


def _normalise(w: np.ndarray) -> np.ndarray:
    return w / r_mean(w)


def _shrink(w: np.ndarray, lam: float) -> np.ndarray:
    if not (0 <= lam <= 1):
        raise ValueError("shrink_lambda must be in [0, 1]")
    w = _normalise(w)
    return (1 - lam) * w + lam


# --------------------------------------------------------------------------
# task descriptors
# --------------------------------------------------------------------------
def _xy(a, what: str) -> np.ndarray:
    a = np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"{what} must be an (n, 2) array of projected coordinates")
    return a


def _folds(folds, n: int) -> np.ndarray:
    if hasattr(folds, "clusters") and hasattr(folds, "indx_test"):
        if np.asarray(folds.clusters).dtype.kind in "OUS":  # "train"/"test" labels
            raise ValueError("a test_prop train/test split is not a CV design; pass k-fold labels")
        folds = folds.clusters
    f = np.asarray(folds)
    if f.ndim != 1 or len(f) != n:
        raise ValueError(f"folds must have one label per training point ({n})")
    if any(_is_na(v) for v in f.tolist()):
        raise ValueError("folds must not contain missing labels")
    if len(set(f.tolist())) < 2:
        raise ValueError("at least two folds are needed for CV prediction distances")
    return f


def _cv_prediction_distances(xy: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """``compute_cv_prediction_distances``: held-out point to nearest other-fold point."""
    d = np.full(len(folds), np.nan)
    for f in dict.fromkeys(folds.tolist()):
        test = folds == f
        if not (~test).any():
            continue
        # R: sqrt(outer(dx)^2 + outer(dy)^2) with R-level sqrt, then min;
        # sqrt is monotone, so min first (in chunks to bound memory)
        q, ref = xy[test], xy[~test]
        sq = np.concatenate(
            [_sqdist(q[s : s + 2048], ref).min(axis=1) for s in range(0, len(q), 2048)]
        )
        d[test] = r_sqrt(sq)
    return d


def _columns(x, n: int, what: str, names=None) -> dict[str, np.ndarray]:
    if x is None:
        return {}
    if hasattr(x, "columns"):
        if len(x) != n:
            raise ValueError(f"{what} must have {n} rows")
        cols = [str(c) for c in x.columns]
        if len(set(cols)) != len(cols):
            raise ValueError(f"{what} has duplicate column names")
        out = {}
        for j, c in enumerate(cols):
            col = x.iloc[:, j]
            if getattr(col.dtype, "kind", "O") in "iuf":  # incl. nullable Int64/Float64
                out[c] = col.to_numpy(dtype=float, na_value=np.nan)
            else:
                out[c] = np.asarray(col.to_numpy(), dtype=object)
        return out
    a = np.asarray(x)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != n:
        raise ValueError(f"{what} must have {n} rows")
    if names is not None and len(names) != a.shape[1]:
        raise ValueError(f"{what} has {a.shape[1]} columns, expected {len(names)}")
    names = names or [f"x{j + 1}" for j in range(a.shape[1])]
    return {names[j]: _infer_column(a[:, j]) for j in range(a.shape[1])}


def _infer_column(col: np.ndarray) -> np.ndarray:
    """Per-column type for mixed object arrays (e.g. ``df.values``)."""
    if col.dtype.kind != "O":
        return col
    vals = [v for v in col.tolist() if not _is_na(v)]
    if vals and all(
        isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
        for v in vals
    ):
        return np.array([math.nan if _is_na(v) else float(v) for v in col.tolist()])
    return col


# --------------------------------------------------------------------------
# public estimators
# --------------------------------------------------------------------------
def twcv_weights(
    train_xy,
    pred_xy,
    folds,
    train_covariates=None,
    pred_covariates=None,
    *,
    include_distance: bool = True,
    balance_by: float = 0.2,
    shrink_lambda: float = 0.2,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> TWCVWeights:
    """Target-weighted CV weights (``compute_twcv_weights`` in TWCVpaper).

    Parameters
    ----------
    train_xy, pred_xy
        ``(n, 2)`` / ``(m, 2)`` projected coordinates of the training samples
        and of the prediction (deployment) locations.
    folds
        Fold label per training point, e.g. ``res.clusters``, or a
        ``KNNDMResult`` itself.
    train_covariates, pred_covariates
        Optional covariates to balance on, one row per training / prediction
        location (arrays or DataFrames). DataFrame columns are matched by name;
        non-numeric columns, and numeric ones with at most two distinct values
        at the prediction locations, are balanced as categories.
    include_distance
        Also balance the prediction distance ``d`` (the paper's default).
    balance_by
        Quantile step for discretising numeric descriptors (0.2 = quintiles).
    shrink_lambda
        Shrinkage towards uniform weights, ``(1 - lambda) * w + lambda``.
    max_iter, tol
        Raking stops after ``max_iter`` sweeps or when the largest relative
        weight change falls below ``tol``.

    Returns
    -------
    TWCVWeights
        Use ``.weights`` with :func:`weighted_cv_error`.
    """
    txy = _xy(train_xy, "train_xy")
    pxy = _xy(pred_xy, "pred_xy")
    f = _folds(folds, txy.shape[0])

    tcov = _columns(train_covariates, txy.shape[0], "train_covariates")
    if (train_covariates is None) != (pred_covariates is None):
        raise ValueError("pass both train_covariates and pred_covariates, or neither")
    pcov = _columns(pred_covariates, pxy.shape[0], "pred_covariates", names=list(tcov) or None)
    if set(tcov) != set(pcov):
        raise ValueError(f"covariate names differ: {sorted(tcov)} vs {sorted(pcov)}")
    if "d" in tcov:
        raise ValueError("covariate name 'd' is reserved for the prediction distance")

    d_cv = _cv_prediction_distances(txy, f)
    d_pred = knnx_dist(pxy, txy)

    sample = dict(tcov)
    target = {k: pcov[k] for k in tcov}
    if include_distance:
        sample["d"], target["d"] = d_cv, d_pred
    if not sample:
        raise ValueError("nothing to balance: give covariates or include_distance=True")

    balance, margins, n_bins = {}, {}, {}
    for v in sample:
        cx, cr, L = _discretise(sample[v], target[v], balance_by)
        balance[v] = cx
        margins[v] = _target_margin(cr, L)
        n_bins[v] = L

    w, conv, it = rake_weights(balance, margins, max_iter=max_iter, tol=tol)
    if not r_sum(w) > 0:
        raise ValueError("all calibration weights are zero: the validation tasks only "
                         "cover descriptor bins that hold no prediction locations")
    w_raw = _normalise(w)
    return TWCVWeights(
        weights=_shrink(w_raw, shrink_lambda),
        weights_raw=w_raw,
        converged=conv,
        iterations=it,
        balancing_vars=list(sample),
        shrink_lambda=float(shrink_lambda),
        balance_by=float(balance_by),
        d_cv=d_cv,
        d_pred=d_pred,
        n_bins=n_bins,
    )


def dwcv_weights(
    train_xy,
    pred_xy,
    folds,
    *,
    balance_by: float = 0.2,
    shrink_lambda: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> TWCVWeights:
    """Distance-weighted CV (``compute_dwcv_weights``): balance ``d`` only."""
    return twcv_weights(
        train_xy, pred_xy, folds, include_distance=True, balance_by=balance_by,
        shrink_lambda=shrink_lambda, max_iter=max_iter, tol=tol,
    )


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(w)
    if not ok.any():
        return math.nan
    ws = r_sum(w[ok])
    if not math.isfinite(ws) or ws <= 0:
        return math.nan
    return r_sum(w[ok] * x[ok]) / ws


def weighted_cv_error(obs, pred, weights=None) -> dict[str, float]:
    """Weighted CV performance (``summarize_losses``): bias, mse, rmse, mae.

    ``obs`` and ``pred`` are the observed values and out-of-fold predictions
    for the training points, in the same order as the weights. Bias is
    ``mean(obs - pred)``.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if obs.shape != pred.shape or obs.ndim != 1:
        raise ValueError("obs and pred must be 1-D arrays of equal length")
    w = np.ones(len(obs)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != obs.shape:
        raise ValueError("weights must be 1-D with one entry per observation")
    if np.isfinite(w).all():  # in R a non-finite sum(weights) skips the rescaling
        ws = r_sum(w)
        if ws > 0 and math.isfinite(ws):
            w = w / ws
    err = obs - pred
    se = err * err
    mse = _wmean(se, w)
    rmse = float(r_sqrt(mse)) if not mse < 0 else math.nan  # R: NaN for negative
    return {"bias": _wmean(err, w), "mse": mse, "rmse": rmse, "mae": _wmean(np.abs(err), w)}


def weight_diagnostics(weights) -> dict[str, float]:
    """``summarize_weight_vector``: effective sample size, extremes and Gini."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    n = len(w)
    if n == 0:
        return {"n_cases": 0, "effective_sample_size": math.nan, "ess_fraction": math.nan,
                "max_weight": math.nan, "q95_weight": math.nan, "q99_weight": math.nan,
                "gini": math.nan}
    nn = w[w >= 0]
    ess = math.nan
    gini = math.nan
    q95 = math.nan
    if len(nn):
        s = r_sum(nn)
        if s > 0:
            p = nn / s
            with np.errstate(divide="ignore"):  # Inf if the sum overflowed, as in R
                ess = float(np.float64(1.0) / r_sum(p * p))
            g = np.sort(nn)
            m = len(g)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                gini = float(2 * np.float64(r_sum(np.arange(1, m + 1) * g)) / (m * r_sum(g)) - (m + 1) / m)
        q95 = float(r_quantile7(nn, [0.95])[0])
    return {
        "n_cases": n,
        "effective_sample_size": ess,
        "ess_fraction": ess / n,
        "max_weight": float(w.max()),
        "q95_weight": q95,
        "q99_weight": float(r_quantile7(w, [0.99])[0]),
        "gini": gini,
    }
