"""Bit-exact comparison against CAST::knndm fixtures generated in R.

Fixtures: ``tests/fixtures/*.json`` from ``reference/make_fixtures.R``
(doubles stored as C99 hex floats so they round-trip exactly).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from knndm import RRNG, knndm
from knndm._backend import _rust_backend

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
BACKENDS = ["python"] + (["rust"] if _rust_backend() is not None else [])


def _hex(v) -> np.ndarray:
    if isinstance(v, str):
        v = [v]
    return np.array([float.fromhex(s) for s in v], dtype=float)


def _points(spec, space):
    if space == "geographical":
        return np.column_stack([_hex(spec["x"]), _hex(spec["y"])])
    cols = spec["colnames"]
    return np.column_stack([_hex(spec[c]) for c in cols])


def _bits(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_matches_r(path: Path, backend: str):
    fx = _load(path)
    space = fx["space"]
    args = {k: v for k, v in (fx.get("args") or {}).items() if v is not None}
    kw = {}
    for name in ("k", "maxp", "clustering", "linkf", "nk_len", "test_prop", "test_tolerance", "scale_vars"):
        if name in args:
            kw[name] = args[name]
    if "k" in kw:
        kw["k"] = int(kw["k"])
    if "nk_len" in kw:
        kw["nk_len"] = int(kw["nk_len"])
    tp = _points(fx["tpoints"], space)
    pp = _points(fx["predpoints"], space)
    rng = RRNG(fx["knndm_seed"])

    if "error" in fx:
        with pytest.raises(Exception):
            knndm(tp, pp, space=space, rng=rng, backend=backend, **kw)
        return

    res = knndm(tp, pp, space=space, rng=rng, backend=backend, **kw)

    assert np.array_equal(_bits(res.Gj), _bits(_hex(fx["Gj"]))), "Gj"
    assert np.array_equal(_bits(res.Gij), _bits(_hex(fx["Gij"]))), "Gij"
    expected_q = fx["q"]
    assert (res.q == expected_q) or (str(res.q) == str(expected_q)), f"q {res.q} != {expected_q}"
    assert _bits([res.W])[0] == _bits(_hex(fx["W"]))[0], f"W {res.W!r} != {_hex(fx['W'])[0]!r}"
    assert np.array_equal(_bits(res.Gjstar), _bits(_hex(fx["Gjstar"]))), "Gjstar"

    clusters = fx["clusters"]
    if isinstance(clusters, list) and clusters and isinstance(clusters[0], str):
        assert list(res.clusters) == clusters, "clusters"
    else:
        assert np.array_equal(res.clusters, np.array(clusters)), "clusters"

    train = fx["indx_train"]
    test = fx["indx_test"]
    if isinstance(res.indx_test, list):
        assert len(res.indx_test) == len(test)
        for got, exp in zip(res.indx_test, test):
            assert np.array_equal(got + 1, np.atleast_1d(exp))
        for got, exp in zip(res.indx_train, train):
            assert np.array_equal(got + 1, np.atleast_1d(exp))
    else:
        assert np.array_equal(res.indx_test + 1, np.atleast_1d(test[0]))
        assert np.array_equal(res.indx_train + 1, np.atleast_1d(train[0]))

    # the RNG must be left exactly where R leaves .Random.seed
    rs = np.array(fx["rng_after"], dtype=np.int64) & 0xFFFFFFFF
    assert rs[1] == rng.mti, "RNG position"
    assert np.array_equal(rs[2:], np.array(rng.mt, dtype=np.int64)), "RNG state"
