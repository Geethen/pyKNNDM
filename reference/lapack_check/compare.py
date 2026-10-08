"""Compare knndm._lapack.svd_vt with R's La.svd output bit for bit.
usage: python -I compare.py cases.ndjson [--max-m N]
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from knndm._lapack import svd_vt  # noqa: E402


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def main():
    path = sys.argv[1]
    max_m = int(sys.argv[sys.argv.index("--max-m") + 1]) if "--max-m" in sys.argv else 10**9
    tot = okd = okv = 0
    bad = []
    t0 = time.time()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m, n = r["m"], r["n"]
            for k in ("x", "d", "vt"):
                if isinstance(r[k], str):
                    r[k] = [r[k]]
            if m > max_m:
                continue
            x = np.array([float.fromhex(h) for h in r["x"]]).reshape((n, m)).T
            d_ref = np.array([float.fromhex(h) for h in r["d"]])
            vt_ref = np.array([float.fromhex(h) for h in r["vt"]]).reshape((n, n)).T
            d, vt = svd_vt(x)
            tot += 1
            ed = bool(np.array_equal(bits(d), bits(d_ref)))
            ev = bool(np.array_equal(bits(vt), bits(vt_ref)))
            okd += ed
            okv += ev
            if not (ed and ev):
                bad.append((r["id"], r["kind"], m, n, ed, ev,
                            float(np.max(np.abs(vt - vt_ref)))))
    print(f"cases={tot} d_exact={okd} vt_exact={okv} time={time.time()-t0:.1f}s")
    for b in bad[:40]:
        print("MISMATCH id=%d kind=%s m=%d n=%d d_ok=%s vt_ok=%s maxdiff=%.3g" % b)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
