import sys, os, time, csv, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import numpy as np
from knndm import knndm

CFGS = [("geo_500","geographical"),("geo_1000","geographical"),("geo_2000","geographical"),
        ("geo_5000","geographical"),("feat_500","feature"),("feat_2000","feature")]

def rd(f):
    with open(f) as fh:
        r = csv.reader(fh); next(r)
        return np.array([[float.fromhex(v) for v in row] for row in r])

def main():
    out = []; py_est = {}
    for nm, sp in CFGS:
        t = rd(os.path.join(HERE, "data", nm + "_t.csv"))
        p = rd(os.path.join(HERE, "data", nm + "_p.csv"))
        n = len(t)
        for cl in ("hierarchical", "kmeans"):
            for be in ("python", "rust"):
                if be == "python" and n >= 5000:
                    print(f"SKIP python {nm} {cl}: n>=5000", flush=True); continue
                reps = 1 if n >= 2000 else 3
                ts = []; o = None
                for _ in range(reps):
                    t0 = time.perf_counter()
                    o = knndm(t, p, space=sp, k=10, clustering=cl, seed=1, backend=be)
                    ts.append(time.perf_counter() - t0)
                sec = statistics.median(ts)
                print(nm, cl, be, round(sec, 3), flush=True)
                out.append(dict(name=nm, space=sp, n=n, clustering=cl, backend=be,
                                seconds=sec, q=o.q, W_hex=float(o.W).hex()))
                with open(os.path.join(HERE, "results_py.csv"), "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
                    w.writeheader(); w.writerows(out)
    report(out)

def report(out=None):
    if out is None:
        with open(os.path.join(HERE, "results_py.csv")) as fh:
            out = list(csv.DictReader(fh))
    with open(os.path.join(HERE, "results_r.csv")) as fh:
        R = {(r["name"], r["clustering"]): r for r in csv.DictReader(fh)}
    py = {(r["name"], r["clustering"], r["backend"]): r for r in out}
    lines = ["| dataset | clustering | R s | python s | rust s | speedup rust vs R | W identical to R | q identical |",
             "|---|---|---|---|---|---|---|---|"]
    for nm, _ in CFGS:
        for cl in ("hierarchical", "kmeans"):
            r = R.get((nm, cl)); a = py.get((nm, cl, "python")); b = py.get((nm, cl, "rust"))
            f = lambda x: f"{float(x['seconds']):.3f}" if x else "n/a"
            sp = f"{float(r['seconds'])/float(b['seconds']):.1f}x" if r and b else "n/a"
            def same(x):
                if not (x and r): return "n/a"
                w = x["W_hex"] == r["W_hex"]; return "yes" if w else "NO"
            def sameq(x):
                if not (x and r): return "n/a"
                return "yes" if float(x["q"]) == float(r["q"]) else "NO"
            wi = "/".join(f"{k}:{same(x)}" for k, x in (("py", a), ("rs", b)) if x)
            qi = "/".join(f"{k}:{sameq(x)}" for k, x in (("py", a), ("rs", b)) if x)
            lines.append(f"| {nm} | {cl} | {f(r)} | {f(a)} | {f(b)} | {sp} | {wi} | {qi} |")
    tbl = "\n".join(lines); print(tbl)
    return tbl

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report": report()
    else: main()
