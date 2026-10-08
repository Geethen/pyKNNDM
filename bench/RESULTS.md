# knndm benchmark: R CAST::knndm vs Python port

- CPU: Intel(R) Core(TM) Ultra 7 165U | cores=12 logical=14
- R: 4.5.0, CAST 1.1.2
- Python: Python 3.13.16
- Repeats: median of 3 for n<2000, 1 run for n>=2000. Python backend skipped for n>=5000.
- W identical column compares float.hex vs sprintf("%a") strings exactly (py = python backend, rs = rust backend).

| dataset | clustering | R s | python s | rust s | speedup rust vs R | W identical to R | q identical |
|---|---|---|---|---|---|---|---|
| geo_500 | hierarchical | 2.000 | 0.866 | 0.200 | 10.0x | py:yes/rs:yes | py:yes/rs:yes |
| geo_500 | kmeans | 2.350 | 5.147 | 0.255 | 9.2x | py:yes/rs:yes | py:yes/rs:yes |
| geo_1000 | hierarchical | 4.560 | 5.027 | 0.252 | 18.1x | py:yes/rs:yes | py:yes/rs:yes |
| geo_1000 | kmeans | 5.130 | 17.923 | 0.365 | 14.0x | py:yes/rs:yes | py:yes/rs:yes |
| geo_2000 | hierarchical | 15.590 | 25.111 | 0.426 | 36.6x | py:yes/rs:yes | py:yes/rs:yes |
| geo_2000 | kmeans | 17.700 | 66.301 | 0.534 | 33.2x | py:yes/rs:yes | py:yes/rs:yes |
| geo_5000 | hierarchical | 102.080 | n/a | 1.223 | 83.5x | rs:yes | rs:yes |
| geo_5000 | kmeans | 111.830 | n/a | 1.378 | 81.1x | rs:yes | rs:yes |
| feat_500 | hierarchical | 4.120 | 1.685 | 0.206 | 20.0x | py:yes/rs:yes | py:yes/rs:yes |
| feat_500 | kmeans | 4.510 | 7.875 | 0.279 | 16.2x | py:yes/rs:yes | py:yes/rs:yes |
| feat_2000 | hierarchical | 29.860 | 19.402 | 0.468 | 63.8x | py:yes/rs:yes | py:yes/rs:yes |
| feat_2000 | kmeans | 25.970 | 105.521 | 0.840 | 30.9x | py:yes/rs:yes | py:yes/rs:yes |
