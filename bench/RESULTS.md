# knndm benchmark: R CAST::knndm vs Python port

- CPU: Intel(R) Core(TM) Ultra 7 165U | cores=12 logical=14
- R: 4.5.0, CAST 1.1.2
- Python: Python 3.13.16
- Repeats: median of 3 for n<2000, 1 run for n>=2000. Python backend skipped for n>=5000.
- W identical column compares float.hex vs sprintf("%a") strings exactly (py = python backend, rs = rust backend).

| dataset | clustering | R s | python s | rust s | speedup rust vs R | W identical to R | q identical |
|---|---|---|---|---|---|---|---|
| geo_500 | hierarchical | 2.000 | 0.875 | 0.180 | 11.1x | py:yes/rs:yes | py:yes/rs:yes |
| geo_500 | kmeans | 2.350 | 4.687 | 0.298 | 7.9x | py:yes/rs:yes | py:yes/rs:yes |
| geo_1000 | hierarchical | 4.560 | 4.857 | 0.239 | 19.1x | py:yes/rs:yes | py:yes/rs:yes |
| geo_1000 | kmeans | 5.130 | 16.930 | 0.562 | 9.1x | py:yes/rs:yes | py:yes/rs:yes |
| geo_2000 | hierarchical | 15.590 | 20.860 | 0.403 | 38.7x | py:yes/rs:yes | py:yes/rs:yes |
| geo_2000 | kmeans | 17.700 | 63.986 | 1.212 | 14.6x | py:yes/rs:yes | py:yes/rs:yes |
| geo_5000 | hierarchical | 102.080 | n/a | 1.196 | 85.4x | rs:yes | rs:yes |
| geo_5000 | kmeans | 111.830 | n/a | 4.560 | 24.5x | rs:yes | rs:yes |
| feat_500 | hierarchical | 4.120 | 1.673 | 0.222 | 18.6x | py:yes/rs:yes | py:yes/rs:yes |
| feat_500 | kmeans | 4.510 | 7.890 | 0.489 | 9.2x | py:yes/rs:yes | py:yes/rs:yes |
| feat_2000 | hierarchical | 29.860 | 18.378 | 0.451 | 66.1x | py:yes/rs:yes | py:yes/rs:yes |
| feat_2000 | kmeans | 25.970 | 105.333 | 3.140 | 8.3x | py:yes/rs:yes | py:yes/rs:yes |
