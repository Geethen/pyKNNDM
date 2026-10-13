"""Unit checks for the exact numeric emulation (values verified in R 4.5)."""

import numpy as np
import pytest

from knndm import RRNG
from knndm._backend import PythonBackend, _rust_backend
from knndm._rnum import nk_grid, r_colmeans, r_cumsum, r_sum


def test_rng_runif_matches_r():
    r = RRNG(42)
    assert [f"{v:.17g}" for v in r.runif(3)] == ["0.91480604349635541", "0.93707541329786181", "0.28613953478634357"]
    assert [f"{v:.17g}" for v in RRNG(-7).runif(2)] == ["0.32649485557340086", "0.53986111702397466"]


def test_rng_sampling_matches_r():
    r = RRNG(42)
    assert r.sample_int(1000, 1) == [561]
    assert r.sample_int(10, 10) == [5, 1, 9, 2, 4, 7, 10, 8, 6, 3]
    assert r.sample_int(100000, 5) == [95833, 64110, 96666, 64809, 5403]
    assert r.sample([1, 2, 3] * 4, 10) == [1, 2, 2, 3, 2, 3, 3, 2, 1, 1]


def test_long_double_accumulation():
    # R: sum(c(1, 2^-60, -1)) == 2^-60 (x87 64-bit significand)
    assert r_sum([1.0, 2.0**-60, -1.0]) == 2.0**-60
    x = [0.1, 0.2, 0.3, 1e16, -1e16, 0.7]
    assert r_sum(x) == 1.299609375
    assert r_colmeans(np.array(x)[:, None])[0] == 0.21660156249999998
    assert r_cumsum([1.0, 2.0**-60, -1.0])[2] == 2.0**-60


@pytest.mark.skipif(_rust_backend() is None, reason="Rust kernels not built")
def test_rust_long_double():
    from knndm._rust_backend import ld_cumsum, ld_mean, ld_sum

    assert ld_sum([1.0, 2.0**-60, -1.0]) == 2.0**-60
    assert ld_mean([0.1, 0.2, 0.3, 1e16, -1e16, 0.7]) == 0.21660156249999998
    rs = np.random.default_rng(0)
    for _ in range(200):
        v = rs.normal(0, 10.0 ** rs.integers(-5, 12), rs.integers(1, 50))
        assert ld_sum(v) == r_sum(v)
        assert ld_mean(v) == r_colmeans(v[:, None])[0]
        assert np.array_equal(ld_cumsum(v), r_cumsum(v))


def test_nk_grid():
    assert nk_grid(10, 100, 100)[0] == 10 and nk_grid(10, 100, 100)[-1] == 98


@pytest.mark.skipif(_rust_backend() is None, reason="Rust kernels not built")
@pytest.mark.parametrize("method", ["ward.D2", "ward.D", "single", "complete", "average", "mcquitty", "median", "centroid"])
def test_rust_hclust_matches_python(method):
    rs = np.random.default_rng(3)
    x = np.round(rs.normal(0, 50, (90, 2)), 1)  # rounding creates ties
    py, ru = PythonBackend(), _rust_backend()
    m_py = py.hclust_merge(py.dist_matrix(x), method)
    m_ru = ru.hclust_merge(x, method)
    assert np.array_equal(m_py, m_ru)
    for k in (2, 5, 17, 60, 89, 90):
        assert np.array_equal(py.cutree(m_py, k), ru.cutree(m_ru, k))


@pytest.mark.skipif(_rust_backend() is None, reason="Rust kernels not built")
def test_rust_kmeans_and_distances_match_python():
    rs = np.random.default_rng(5)
    x = rs.normal(0, 1, (300, 3))
    py, ru = PythonBackend(), _rust_backend()
    for nk in (3, 7, 20):
        a = py.kmeans(x, nk, RRNG(nk))
        b = ru.kmeans(x, nk, RRNG(nk))
        assert np.array_equal(a, b)
        folds = (a % 4) + 1
        assert np.array_equal(py.cv_distances(x, folds), ru.cv_distances(x, folds))
    q = rs.normal(0, 1, (50, 3))
    assert np.array_equal(py.knnx_dist(q, x), ru.knnx_dist(q, x))
    assert np.array_equal(py.knn_dist_self(x), ru.knn_dist_self(x))
    g, h = rs.random(100), rs.random(250)
    assert py.wass_stat(g, h) == ru.wass_stat(g, h)


# --- regression tests for review findings (expected values computed in R 4.5) ---
EDGE_MEANS = [
    ([2.0**-1022, 2.0**-1022, 0.0], "0x1.5555555555556p-1023"),  # subnormal result
    ([2.0**-1074, 0.0, 0.0], "0x0p+0"),
    ([3 * 2.0**-1074, 0.0], "0x1p-1073"),
    ([2.0**-1074, 0.0], "0x0p+0"),  # exact tie rounds to even (zero)
]


@pytest.mark.parametrize("vals,expected", EDGE_MEANS)
def test_long_double_subnormal_means(vals, expected):
    assert r_colmeans(np.array(vals)[:, None])[0] == float.fromhex(expected)
    be = _rust_backend()
    if be is not None:
        from knndm._rust_backend import ld_mean

        assert ld_mean(vals) == float.fromhex(expected)


def test_long_double_overflow_headroom():
    # R: sum(c(1e308, 1e308)) == Inf, but sum(c(1e308, 1e308, -1e308)) is finite
    assert r_sum([1e308, 1e308]) == np.inf
    assert r_sum([1e308, 1e308, -1e308]) == float.fromhex("0x1.1ccf385ebc8ap+1023")
    if _rust_backend() is not None:
        from knndm._rust_backend import ld_sum

        assert ld_sum([1e308, 1e308]) == np.inf
        assert ld_sum([1e308, 1e308, -1e308]) == float.fromhex("0x1.1ccf385ebc8ap+1023")


def test_hclust_three_points_complete():
    # R: hclust(dist(c(0, 10, 11)), "complete")$merge == rbind(c(-2, -3), c(-1, 1))
    x = np.array([[0.0], [10.0], [11.0]])
    for be in [PythonBackend()] + ([_rust_backend()] if _rust_backend() is not None else []):
        m = be.hclust_merge(be.dist_matrix(x), "complete")
        assert m.tolist() == [[-2, -3], [-1, 1]]
        assert be.cutree(m, 2).tolist() == [1, 2, 2]


@pytest.mark.skipif(_rust_backend() is None, reason="Rust kernels not built")
def test_rust_rejects_bad_shapes_instead_of_crashing():
    be = _rust_backend()
    with pytest.raises(ValueError):
        be.knnx_dist(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        be.cv_distances(np.zeros((5, 2)), np.array([1, 2, 1]))
    assert isinstance(be.wass_stat(np.array([np.nan, 1.0]), np.array([0.0, 2.0])), float)  # must not abort


def test_knndm_input_validation():
    from knndm import knndm

    tp = np.random.default_rng(0).uniform(0, 100, (20, 2))
    with pytest.raises(ValueError):
        knndm(tp)  # predpoints required
    with pytest.raises(ValueError):
        knndm(np.vstack([tp, [[np.nan, 1.0]]]), tp, seed=1)
    with pytest.raises(ValueError):
        knndm(tp, tp, nk_len=0, seed=1)


def test_knndm_k_validation():
    from knndm import knndm

    tp = np.random.default_rng(0).uniform(0, 100, (20, 2))
    for k in (1, 0, 2.5, 21):
        with pytest.raises(ValueError):
            knndm(tp, tp, k=k, seed=1)


def test_knndm_zero_variance_predictor():
    from knndm import knndm

    rs = np.random.default_rng(0)
    tp = np.column_stack([rs.normal(size=20), np.ones(20)])
    with pytest.raises(ValueError, match="zero variance"):
        knndm(tp, tp, space="feature", k=3, seed=1)


def test_as_matrix_rejects_bad_shapes():
    from knndm.core import _as_matrix

    with pytest.raises(ValueError):
        _as_matrix(np.zeros((4, 2, 2)))
    with pytest.raises(ValueError):
        _as_matrix(np.zeros((4, 0)))


def test_feature_dataframe_columns_matched_by_name():
    pd = pytest.importorskip("pandas")
    from knndm import knndm

    rs = np.random.default_rng(1)
    t = pd.DataFrame({"a": rs.normal(size=30), "b": rs.normal(size=30) * 3})
    p = pd.DataFrame({"a": rs.normal(size=40), "b": rs.normal(size=40) * 3})
    r1 = knndm(t, p, space="feature", k=3, seed=1)
    r2 = knndm(t, p[["b", "a"]], space="feature", k=3, seed=1)
    r3 = knndm(t, p.assign(extra=1.0)[["extra", "b", "a"]], space="feature", k=3, seed=1)
    for r in (r2, r3):
        assert r.W == r1.W and r.q == r1.q
        assert np.array_equal(r.clusters, r1.clusters)
