"""Generate the data behind docs/knndm_explainer.html.

Runs knndm on a small clustered example and records the intermediate steps
CAST does not return (W for every candidate nk, folds per nk, the PC1 ordering
of clusters), re-using the package internals so the numbers are the real ones.

    python docs/make_explainer_data.py > docs/explainer_data.json
    python docs/make_explainer_data.py --build   # also rebuild docs/knndm_explainer.html
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from knndm import RRNG, knndm  # noqa: E402
from knndm import _backend  # noqa: E402
from knndm._rnum import nk_grid, prcomp_rank1, r_order  # noqa: E402
from knndm.core import _assign_folds  # noqa: E402

K = 5
SEED = 7


def make_data():
    rs = np.random.default_rng(42)
    centers = np.array([[150, 780], [260, 640], [420, 860], [700, 720], [820, 300], [560, 240], [300, 260]], float)
    sizes = [28, 18, 22, 26, 24, 16, 16]
    pts = [c + rs.normal(0, 38, (s, 2)) for c, s in zip(centers, sizes)]
    tp = np.round(np.vstack(pts), 2)
    pp = np.round(rs.uniform(0, 1000, (600, 2)), 2)
    return tp, pp


def ecdf_points(v):
    v = np.sort(np.asarray(v))
    return [round(float(x), 3) for x in v]


def main():
    tp, pp = make_data()
    n = tp.shape[0]
    res = knndm(tp, pp, k=K, seed=SEED, backend="python")
    be = _backend.get("python")

    # --- replay the nk loop to expose W(nk) and folds(nk)
    distmat = be.dist_matrix(tp)
    merge = be.hclust_merge(distmat, "ward.D2")
    nks = nk_grid(K, n, 100)
    rot, ctr = prcomp_rank1(tp)
    grid = []
    for nk in nks:
        clust_nk = be.cutree(merge, nk)
        folds = _assign_folds(tp, clust_nk, n, K, None, rot, ctr, be)
        counts = np.bincount(folds, minlength=K + 1)[1:]
        valid = not np.any(counts / n > 0.5)
        entry = {"nk": nk, "valid": bool(valid), "folds": folds.tolist(), "sizes": counts.tolist()}
        if valid:
            gjs = be.cv_distances(tp, folds)
            entry["W"] = be.wass_stat(gjs, res.Gij)
            entry["Gjstar"] = ecdf_points(gjs)
        grid.append(entry)
    assert res.q == min((g for g in grid if g["valid"]), key=lambda g: g["W"])["nk"], "replay must match knndm"

    # --- the cluster -> fold merge at the chosen q
    clust_q = be.cutree(merge, res.q)
    levels = np.unique(clust_q)
    pc1 = be.cluster_pc1(tp, clust_q, levels, rot, ctr)
    o = r_order(pc1)
    folds_q = np.array(res.clusters)
    merge_rows = []
    for rank, idx in enumerate(o):
        lv = int(levels[idx])
        members = clust_q == lv
        merge_rows.append({
            "cluster": lv,
            "size": int(members.sum()),
            "pc1": round(float(pc1[idx]), 2),
            "fold": int(folds_q[members][0]),
            "big": bool(members.sum() >= n / K),
            "centroid": [round(float(v), 1) for v in tp[members].mean(axis=0)],
        })

    # --- random CV for comparison (R's sample(rep(1:k, ...)))
    rng = RRNG(SEED)
    pool = list(range(1, K + 1)) * math.ceil(n / K)
    rand_folds = np.array(rng.sample(pool, n))
    gj_rand = be.cv_distances(tp, rand_folds)

    out = {
        "k": K,
        "n": n,
        "train": tp.tolist(),
        "pred": pp.tolist(),
        "Gij": ecdf_points(res.Gij),
        "Gj": ecdf_points(res.Gj),
        "random": {"folds": rand_folds.tolist(), "Gjstar": ecdf_points(gj_rand), "W": be.wass_stat(gj_rand, res.Gij)},
        "q": int(res.q),
        "W": res.W,
        "grid": grid,
        "merge_q": merge_rows,
        "pc1_dir": [round(float(v), 4) for v in rot],
        "pc1_center": [round(float(v), 2) for v in ctr],
    }
    if "--build" in sys.argv[1:]:
        here = Path(__file__).resolve().parent
        data = json.dumps(out, separators=(",", ":"))
        (here / "explainer_data.json").write_text(data, encoding="utf-8", newline="")
        page = (here / "explainer_template.html").read_text(encoding="utf-8").replace("__DATA__", data)
        (here / "knndm_explainer.html").write_text(page, encoding="utf-8", newline="")
    else:
        json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
