# reference

Material used to port `CAST::knndm`; none of it is part of the installed package.

- `knndm_source.R`, `helpers_source.R`, `helpers2_source.R`: printed R function bodies
  from CAST 1.1.2 (dumped with `dump.R` to `dump4.R`), kept as the porting reference.
- `ldtest.R`: probes how R's `sum`/`mean`/`colMeans`/`cumsum` behave with 80-bit long double.
- `make_fixtures.R`: generates the fixtures in `tests/fixtures/` (needs R, CAST, sf, jsonlite).
- `lapack_check/`: `gen_cases.R` writes random matrices with R's `La.svd` results
  (ndjson); `compare.py` checks the `dgesdd` port in `knndm._lapack` against them bit for bit.
- `native/`: upstream C, Fortran and C++ sources from R, FNN/ANN, twosamples and
  reference LAPACK that the port follows. Each file keeps its own upstream license
  (see file headers). Included for reference only and not compiled into the package.
