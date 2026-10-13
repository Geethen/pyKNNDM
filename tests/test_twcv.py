"""Bit-exact comparison of the TWCV port against the TWCVpaper R code.

Fixtures: ``tests/fixtures/twcv/*.json`` from ``reference/make_twcv_fixtures.R``
(doubles stored as C99 hex floats; ``null`` marks NA).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import knndm

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "twcv").glob("twcv_*.json"))


def _hex(v) -> np.ndarray:
    if isinstance(v, str):
        v = [v]
    return np.array([np.nan if s is None else float.fromhex(s) for s in v], dtype=float)


def _bits(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _same(got, exp, what: str) -> None:
    got = np.atleast_1d(np.asarray(got, dtype=float))
    exp = np.atleast_1d(np.asarray(exp, dtype=float))
    assert got.shape == exp.shape, f"{what}: shape {got.shape} != {exp.shape}"
    bad = np.flatnonzero(_bits(got) != _bits(exp))
    assert bad.size == 0, f"{what}: first mismatch at {bad[0]}: {got[bad[0]]!r} != {exp[bad[0]]!r}"


def _column(v):
    if v and isinstance(v[0], str) and not _is_hexfloat(v[0]):
        return np.array(v, dtype=object)
    if any(s is None for s in v) and not any(isinstance(s, str) and not _is_hexfloat(s) for s in v):
        return _hex(v)
    if all(isinstance(s, str) and not _is_hexfloat(s) or s is None for s in v):
        return np.array(v, dtype=object)
    return _hex(v)


def _is_hexfloat(s: str) -> bool:
    try:
        float.fromhex(s)
        return "0x" in s.lower() or s.lower() in ("nan", "inf", "-inf")
    except ValueError:
        return False


def _covariates(spec: dict, names):
    """None, a 2-D float array (numeric only) or a DataFrame (any categorical)."""
    if not names:
        return None
    cols = {n: _column(spec[n]) for n in names}
    if all(c.dtype.kind == "f" for c in cols.values()):
        return np.column_stack([cols[n] for n in names])
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({n: cols[n] for n in names})


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_matches_r(path: Path):
    fx = json.loads(path.read_text())
    args, exp = fx["args"], fx["expected"]
    names = [v for v in args["balancing_vars"] if v != "d"]
    train, pred = fx["train"], fx["pred"]
    txy = np.column_stack([_hex(train["x"]), _hex(train["y"])])
    pxy = np.column_stack([_hex(pred["x"]), _hex(pred["y"])])

    tw = knndm.twcv_weights(
        txy, pxy, np.array(fx["folds"]),
        train_covariates=_covariates(train, names),
        pred_covariates=_covariates(pred, names),
        include_distance="d" in args["balancing_vars"],
        balance_by=args["by"],
        shrink_lambda=args["shrink_lambda"],
    )

    _same(tw.d_cv, _hex(exp["d_cv"]), "d_cv")
    _same(tw.d_pred, _hex(exp["d_pred"]), "d_pred")
    _same(tw.weights_raw, _hex(exp["weights_raw"]), "weights_raw")
    _same(tw.weights, _hex(exp["weights"]), "weights")
    assert tw.converged == exp["converged"]
    assert tw.iterations == exp["iterations"]

    obs, pv = _hex(fx["obs"]), _hex(fx["pred_values"])
    expected_w, expected_u = exp["weighted"], exp["unweighted"]
    for label, got, e in (
        ("weighted", knndm.weighted_cv_error(obs, pv, tw.weights), expected_w),
        ("unweighted", knndm.weighted_cv_error(obs, pv), expected_u),
    ):
        for key in ("bias", "mse", "rmse", "mae"):
            _same(got[key], _hex(e[key]), f"{label}.{key}")

    diag = knndm.weight_diagnostics(tw.weights)
    dexp = exp["diagnostics"]
    assert diag["n_cases"] == dexp["n_cases"]
    for key in ("effective_sample_size", "ess_fraction", "max_weight", "q95_weight", "q99_weight", "gini"):
        _same(diag[key], _hex(dexp[key]), f"diagnostics.{key}")


# --------------------------------------------------------------------------
# non-fixture unit tests
# --------------------------------------------------------------------------
def _toy(n=150, m=400, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1000, (4, 2))
    txy = centers[rng.integers(0, 4, n)] + rng.normal(0, 40, (n, 2))
    pxy = rng.uniform(0, 1000, (m, 2))
    folds = rng.permutation(np.arange(n) % 5)
    return txy, pxy, folds


def test_weights_have_unit_mean():
    txy, pxy, folds = _toy()
    tw = knndm.twcv_weights(txy, pxy, folds)
    assert tw.weights.mean() == pytest.approx(1.0, abs=1e-12)
    assert tw.weights_raw.mean() == pytest.approx(1.0, abs=1e-12)
    assert np.all(tw.weights > 0)


def test_full_shrinkage_gives_ones():
    txy, pxy, folds = _toy()
    tw = knndm.twcv_weights(txy, pxy, folds, shrink_lambda=1.0)
    assert np.array_equal(tw.weights, np.ones(len(txy)))


def test_knndm_result_as_folds():
    txy, pxy, _ = _toy()
    res = knndm.knndm(txy, pxy, k=5, rng=knndm.RRNG(1))
    a = knndm.twcv_weights(txy, pxy, res)
    b = knndm.twcv_weights(txy, pxy, res.clusters)
    assert np.array_equal(_bits(a.weights), _bits(b.weights))


def test_covariate_argument_errors():
    txy, pxy, folds = _toy()
    cov_t = np.random.default_rng(1).normal(size=(len(txy), 1))
    cov_p = np.random.default_rng(2).normal(size=(len(pxy), 1))
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, folds, train_covariates=cov_t)
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, folds, pred_covariates=cov_p)
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, folds, train_covariates=cov_t[:-1], pred_covariates=cov_p)
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, folds, train_covariates=np.hstack([cov_t, cov_t]), pred_covariates=cov_p)
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, folds[:-1])


def test_matched_distributions_give_near_uniform_weights():
    # Training tasks drawn from the same covariate distribution as the
    # deployment tasks, d not balanced: raking has nothing to correct.
    rng = np.random.default_rng(3)
    n = 1000
    txy = rng.uniform(0, 1000, (n, 2))
    pxy = rng.uniform(0, 1000, (n, 2))
    folds = np.arange(n) % 5
    cov_p = rng.normal(size=(n, 1))
    cov_t = np.sort(cov_p[:, 0])[rng.permutation(n)][:, None]  # same empirical distribution
    tw = knndm.twcv_weights(
        txy, pxy, folds, train_covariates=cov_t, pred_covariates=cov_p,
        include_distance=False, shrink_lambda=0.0,
    )
    assert np.allclose(tw.weights, 1.0, atol=1e-12)
    assert tw.converged


# --------------------------------------------------------------------------
# R primitives vs R (fixtures: r_sqrt.json, r_quantile.json, r_mean.json)
# --------------------------------------------------------------------------
def _fx(name: str) -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / "twcv" / name).read_text())


def test_r_sqrt_matches_r():
    from knndm._rnum import r_sqrt

    fx = _fx("r_sqrt.json")
    x, exp = _hex(fx["x"]), _hex(fx["sqrt"])
    assert float.fromhex("0x1.f6bcffb28882fp+1") in x
    _same(r_sqrt(x), exp, "r_sqrt")


def test_r_quantile7_matches_r():
    from knndm.twcv import r_quantile7

    for i, cs in enumerate(_fx("r_quantile.json")["cases"]):
        x = _hex(cs["x"])  # null -> NaN
        _same(r_quantile7(x, _hex(cs["p"])), _hex(cs["q"]), f"quantile case {i}")


def test_r_mean_matches_r():
    from knndm.twcv import r_mean

    for i, cs in enumerate(_fx("r_mean.json")["cases"]):
        _same(r_mean(_hex(cs["x"])), _hex(cs["mean"]), f"mean case {i}")


# --------------------------------------------------------------------------
# further unit tests
# --------------------------------------------------------------------------
def test_binary_numeric_matches_int_codes():
    txy, pxy, folds = _toy()
    rng = np.random.default_rng(5)
    ct = rng.integers(0, 2, len(txy))
    cp = rng.integers(0, 2, len(pxy))
    a = knndm.twcv_weights(txy, pxy, folds, train_covariates=ct.astype(float)[:, None],
                           pred_covariates=cp[:, None])
    b = knndm.twcv_weights(txy, pxy, folds, train_covariates=ct[:, None], pred_covariates=cp[:, None])
    assert np.array_equal(_bits(a.weights), _bits(b.weights))


def test_pandas_nullable_int_with_na():
    pd = pytest.importorskip("pandas")
    txy, pxy, folds = _toy()
    rng = np.random.default_rng(6)
    ct = rng.integers(0, 3, len(txy)).astype(float)
    cp = rng.integers(0, 3, len(pxy))
    ct[:5] = np.nan
    dt = pd.DataFrame({"c": pd.array([pd.NA if np.isnan(v) else int(v) for v in ct], dtype="Int64")})
    dp = pd.DataFrame({"c": pd.array(cp, dtype="Int64")})
    a = knndm.twcv_weights(txy, pxy, folds, train_covariates=dt, pred_covariates=dp)
    b = knndm.twcv_weights(txy, pxy, folds, train_covariates=pd.DataFrame({"c": ct}),
                           pred_covariates=pd.DataFrame({"c": cp.astype(float)}))
    assert np.all(np.isfinite(a.weights))
    assert np.array_equal(_bits(a.weights), _bits(b.weights))


def test_bad_folds_raise():
    txy, pxy, folds = _toy()
    f = folds.astype(float)
    f[3] = np.nan
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, f)
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, np.ones(len(txy), dtype=int))


def test_dwcv_rejects_covariates():
    txy, pxy, folds = _toy()
    cov = np.zeros((len(txy), 1))
    with pytest.raises(TypeError):
        knndm.dwcv_weights(txy, pxy, folds, train_covariates=cov)


def test_weighted_cv_error_edge_cases():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = obs + np.array([0.1, -0.2, 0.3, -0.1])
    w = np.array([1.0, np.nan, 1.0, 1.0])
    out = knndm.weighted_cv_error(obs, pred, w)  # must not raise
    assert set(out) == {"bias", "mse", "rmse", "mae"}
    out = knndm.weighted_cv_error(obs, pred, np.array([1.0, 1.0, -1.0, 1.0]))
    assert out["mse"] < 0
    assert np.isnan(out["rmse"])


# --------------------------------------------------------------------------
# empty bins, object arrays, test_prop results
# --------------------------------------------------------------------------
def test_empty_deployment_bin_warns():
    txy, pxy, folds = _toy(n=150, m=58)
    # heavily tied deployment covariate: with balance_by=0.1 one quantile bin is empty
    base = [3, 1, 1, 0, 0, 0, 1, 4, 3, 5, 3, 3, 5, 4, 3, 3, 3, 5, 1, 4, 4, 0, 2, 5, 3, 0, 4, 4, 5]
    cp = np.tile(np.array(base, dtype=float), 2)[:, None]
    ct = np.random.default_rng(8).uniform(0, 5, len(txy))[:, None]
    with pytest.warns(UserWarning):
        tw = knndm.twcv_weights(txy, pxy, folds, train_covariates=ct, pred_covariates=cp,
                                include_distance=False, balance_by=0.1)
    assert np.all(np.isfinite(tw.weights))


def test_object_ndarray_matches_dataframe():
    pd = pytest.importorskip("pandas")
    txy, pxy, folds = _toy()
    rng = np.random.default_rng(9)
    nt, npred = len(txy), len(pxy)
    ft, fp = rng.normal(size=nt), rng.normal(size=npred)
    st = rng.choice(["a", "b", "c"], nt, p=[0.6, 0.3, 0.1])
    sp = rng.choice(["a", "b", "c"], npred, p=[0.2, 0.3, 0.5])
    arr_t = np.empty((nt, 2), dtype=object)
    arr_t[:, 0], arr_t[:, 1] = ft, st
    arr_p = np.empty((npred, 2), dtype=object)
    arr_p[:, 0], arr_p[:, 1] = fp, sp
    df_t = pd.DataFrame({"x1": ft, "x2": st})
    df_p = pd.DataFrame({"x1": fp, "x2": sp})
    a = knndm.twcv_weights(txy, pxy, folds, train_covariates=arr_t, pred_covariates=arr_p)
    b = knndm.twcv_weights(txy, pxy, folds, train_covariates=df_t, pred_covariates=df_p)
    assert np.array_equal(_bits(a.weights), _bits(b.weights))


def test_test_prop_result_raises():
    txy, pxy, _ = _toy()
    res = knndm.knndm(txy, pxy, test_prop=0.25, rng=knndm.RRNG(1))
    with pytest.raises(ValueError):
        knndm.twcv_weights(txy, pxy, res)


# --------------------------------------------------------------------------
# negative zero, all-zero weights, non-finite diagnostics
# --------------------------------------------------------------------------
def test_negative_zero_is_zero():
    txy, pxy, folds = _toy()
    rng = np.random.default_rng(11)
    ct = rng.integers(0, 2, len(txy)).astype(float)
    cp = rng.integers(0, 2, len(pxy))
    neg = np.where(ct == 0.0, -0.0, ct)
    assert np.signbit(neg[ct == 0.0]).all()
    a = knndm.twcv_weights(txy, pxy, folds, train_covariates=neg[:, None], pred_covariates=cp[:, None])
    b = knndm.twcv_weights(txy, pxy, folds, train_covariates=ct[:, None], pred_covariates=cp[:, None])
    assert np.array_equal(_bits(a.weights), _bits(b.weights))


def test_all_zero_weights_raise():
    # every training task lies in a deployment bin with target share 0
    txy, _, folds = _toy(n=150, m=58)
    pxy = np.random.default_rng(12).uniform(0, 1000, (58, 2))
    base = [3, 1, 1, 0, 0, 0, 1, 4, 3, 5, 3, 3, 5, 4, 3, 3, 3, 5, 1, 4, 4, 0, 2, 5, 3, 0, 4, 4, 5]
    cp = np.tile(np.array(base, dtype=float), 2)[:, None]
    ct = np.full((len(txy), 1), 1.05)  # inside the empty bin (1, 1.1]
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="zero"):
        knndm.twcv_weights(txy, pxy, folds, train_covariates=ct, pred_covariates=cp,
                           include_distance=False, balance_by=0.1)


@pytest.mark.filterwarnings("error")
def test_weight_diagnostics_overflow_like_r():
    d = knndm.weight_diagnostics([1e308, 1e308])
    assert d["n_cases"] == 2
    assert d["effective_sample_size"] == np.inf
    assert d["ess_fraction"] == np.inf
    assert np.isnan(d["gini"])
    assert d["max_weight"] == 1e308
