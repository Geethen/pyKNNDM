# Contributing

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The suite runs every R fixture on the pure-Python backend, and on the Rust backend
too if `python/knndm/knndm_rs.{dll,so,dylib}` is built (see README, Installation).

## Regenerating fixtures

Fixtures in `tests/fixtures/` come from R. This needs R 4.5.0 with CAST 1.1.2, sf and jsonlite:

```bash
Rscript reference/make_fixtures.R tests/fixtures
```

## Bit-exactness rule

The package promises results identical to R. Any change to numerical code must keep
both backends (Python and Rust) identical to the fixtures and to each other. Do not
loosen fixture comparisons or "fix" a fixture to make a test pass. New behaviour
should come with a fixture generated from R where R defines it.
