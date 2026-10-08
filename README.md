# knndm (Python)

A bit-exact Python port of [`CAST::knndm`](https://rdrr.io/cran/CAST/man/knndm.html)
(k-fold Nearest Neighbour Distance Matching, Linnenbrink et al. 2024), with
optional Rust kernels for speed.

*Bit-exact* means that for the same inputs and seed, every output is the
same IEEE double or integer as in R 4.5.0 + CAST 1.1.2: `Gj`, `Gij`,
`Gjstar`, `W`, `q`, the clusters, the fold indices, and the state of R's RNG
(`.Random.seed`) afterwards. The test suite checks this against fixtures
generated in R.

```python
import numpy as np
from knndm import knndm

res = knndm(train_xy, pred_xy, k=10, seed=1)            # = set.seed(1); knndm(...)
res.clusters, res.indx_train, res.indx_test, res.W, res.q
res = knndm(train_df, pred_df, space="feature", clustering="kmeans", seed=1)
```

`indx_train`/`indx_test` are **0-based** (R's are 1-based). Use `seed=` to
reproduce `set.seed(seed); knndm(...)`, or pass `rng=RRNG(...)` to carry R's
RNG state across several calls.

## Scope

| CAST feature | Status |
|---|---|
| `dist_space = "geographical"` (projected coordinates) | ported |
| `dist_space = "feature"` (numeric predictors, `scale_vars`) | ported |
| `dist_fun = "euclidean"` | ported |
| `clustering = "hierarchical"` (every `hclust` linkage) / `"kmeans"` | ported |
| `test_prop` / `test_tolerance` train/test split | ported |
| random-CV fallback (when `Gij <= Gj`) | ported |
| `predpoints` supplied by the user | required |
| `modeldomain` sampling, `mahalanobis`, `gower`/categorical, `great_circle` | not ported |

Pass geographical inputs as `(n, 2)` arrays of *projected* coordinates (what
`sf::st_coordinates` returns).

## How bit-exactness is achieved

Every numeric step mirrors the code R actually runs (sources are in `reference/native/`):

- **RNG:** R's Mersenne-Twister seeding, `unif_rand`, rejection-sampling
  `R_unif_index` and `sample.int` (`_rrng.py`).
- **80-bit long double:** on x86-64, R's `sum`, `cumsum`, `colMeans` and `mean`
  accumulate in x87 80-bit long double. This is emulated exactly, in Python with
  integers and in Rust with a soft-float (`_rnum.py`, `src/f80.rs`).
- **Clustering:** `hclust` (Fortran `HCLUST`/`HCASS2`), `cutree`, and
  Hartigan-Wong `kmeans` (`kmns.f`, including its single-precision `1.E30`
  constant) are ported line by line.
- **PCA:** `prcomp` uses a port of R's reference-LAPACK `dgesdd` code path
  (`_lapack.py`), verified bit-exact on ~1,500 matrices.
- **Tests and distances:** `ks.test` (exact/asymptotic, with ties),
  `twosamples::wass_stat`, and FNN brute-force / `stats::dist` / GEOS distances
  are all reproduced in R's operation order.
- **Fold creation:** `CreateSpacetimeFolds` / `caret::createFolds`, including the
  RNG draws they make.

Caveat: with more than 25 predictors, PCA falls back to SciPy's LAPACK. Fold
labels could then differ from R in near-tie cases, and a warning is added to
`res.messages`.

## Rust backend

`backend="auto"` (default) uses the Rust kernels when built, else pure Python;
both give identical results. The Rust crate has **no dependencies** and a
plain C ABI loaded with `ctypes`, so building it never executes build-script
binaries (this machine's endpoint policy blocks newly built `.exe`s; DLLs are fine).

The Rust kernels speed things up in four ways:

- **Parallel neighbour search:** neighbour lists are built with threads, and
  CV distances use cached lists plus a per-fold k-d tree fallback. Pruning is
  provably exact, because IEEE arithmetic is monotone.
- **Faster `hclust`:** R's algorithm, but it iterates only the clusters still active.
- **`cutree` and `hcass2`:** rewritten with union-find / O(n) relabelling, with
  identical output.
- **Parallel kmeans:** all starting centres are drawn first in R's order, then
  the solves run in parallel.

Build (user-level, no admin):

```bash
pixi run build-rust      # conda-forge rust + MinGW linker -> python/knndm/knndm_rs.dll
```

## Development

```bash
uv venv .venv && uv pip install --python .venv numpy scipy pytest
pixi run build-rust
.venv/Scripts/python -m pytest              # 40 R fixtures x 2 backends + unit tests
Rscript reference/make_fixtures.R tests/fixtures   # regenerate fixtures (needs CAST)
python bench/bench.py                       # timings vs R (see bench/RESULTS.md)
```

## Performance

See `bench/RESULTS.md`. On a 14-thread laptop the Rust backend is roughly
10–90× faster than R, with `W`/`q` identical to R on every benchmark dataset.
