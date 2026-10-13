"""Draw the README overview figure (light and dark SVGs) from a real knndm run.

    python docs/make_readme_figure.py   # writes docs/img/knndm-overview-{light,dark}.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math  # noqa: E402

from knndm import RRNG, knndm  # noqa: E402
from knndm import _backend  # noqa: E402
from make_explainer_data import K, SEED, make_data  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "pred": "#c3c2b7",
        "folds": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "series": {"Gij": "#0b0b0b", "Gj": "#eb6834", "Gjstar": "#2a78d6"},
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "pred": "#4a4a46",
        "folds": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "series": {"Gij": "#f0efec", "Gj": "#d95926", "Gjstar": "#3987e5"},
    },
}
W, H = 980, 470
FONT = 'font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"'


def ecdf_path(v, sx, sy):
    v = np.sort(v)
    n = len(v)
    pts = [(sx(v[0]), sy(0))]
    for i, x in enumerate(v):
        pts += [(sx(x), sy(i / n)), (sx(x), sy((i + 1) / n))]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def draw(res, gj_rand, tp, pp, t):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT}>',
           f'<rect width="{W}" height="{H}" rx="12" fill="{t["surface"]}"/>']
    txt = lambda x, y, s, size=13, fill=t["ink2"], anchor="start", weight=400: out.append(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')

    # --- left: map of folds
    x0, y0, side = 36, 78, 360
    txt(x0, 36, "Folds in geographic space", 16, t["ink"], weight=600)
    txt(x0, 58, f"{len(tp)} clustered training samples, {len(pp)} prediction locations, k = {K}", 12.5)
    out.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" rx="6" fill="none" stroke="{t["grid"]}"/>')
    m = lambda p: (x0 + p[0] / 1000 * side, y0 + side - p[1] / 1000 * side)
    for p in pp:
        x, y = m(p)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6" fill="{t["pred"]}"/>')
    folds = np.asarray(res.clusters)
    for p, f in zip(tp, folds):
        x, y = m(p)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{t["folds"][f - 1]}" stroke="{t["surface"]}" stroke-width="1.5"/>')
    # fold legend (identity is never colour alone: number + swatch)
    lx = x0 + side + 22
    txt(lx, y0 + 12, "CV fold", 12.5, t["ink"], weight=600)
    for i in range(K):
        yy = y0 + 36 + i * 24
        out.append(f'<circle cx="{lx + 6}" cy="{yy - 4}" r="5" fill="{t["folds"][i]}"/>')
        txt(lx + 18, yy, f"Fold {i + 1}  <tspan fill=\"{t['muted']}\">n={int((folds == i + 1).sum())}</tspan>", 12.5)
    yy = y0 + 36 + K * 24 + 8
    out.append(f'<circle cx="{lx + 6}" cy="{yy - 4}" r="2.2" fill="{t["pred"]}"/>')
    txt(lx + 18, yy, "prediction", 12.5)

    # --- right: ECDFs of nearest-neighbour distances
    px0, py0, pw, ph = 580, 78, 368, 330
    txt(px0 - 40, 36, "Nearest-neighbour distances", 16, t["ink"], weight=600)
    txt(px0 - 40, 58, f"kNNDM folds match the prediction task (W = {res.W:.1f})", 12.5)
    xmax = float(np.ceil(max(res.Gij.max(), res.Gjstar.max(), gj_rand.max()) / 50) * 50)
    sx = lambda v: px0 + v / xmax * pw
    sy = lambda f: py0 + ph - f * ph
    for f in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{px0}" x2="{px0 + pw}" y1="{sy(f):.1f}" y2="{sy(f):.1f}" stroke="{t["grid"] if f else t["axis"]}"/>')
        txt(px0 - 8, sy(f) + 4, f"{f:g}", 11, t["muted"], "end")
    step = 50 if xmax <= 300 else 100
    for v in np.arange(0, xmax + 1, step):
        txt(sx(v), py0 + ph + 18, f"{v:g}", 11, t["muted"], "middle")
    txt(px0 + pw / 2, py0 + ph + 40, "distance to nearest training sample (m)", 12, t["ink2"], "middle")
    out.append(f'<text transform="translate({px0 - 34},{py0 + ph / 2}) rotate(-90)" font-size="12" fill="{t["ink2"]}" text-anchor="middle">ECDF</text>')
    series = [("Gj", gj_rand, f"CV distances (random {K}-fold)", "6 4"),
              ("Gij", res.Gij, "prediction-to-sample (target)", None),
              ("Gjstar", res.Gjstar, "CV distances (kNNDM)", None)]
    for key, v, _, dash in series:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path d="{ecdf_path(v, sx, sy)}" fill="none" stroke="{t["series"][key]}" stroke-width="2.2"{d} stroke-linejoin="round"/>')
    # legend, bottom-right inside the plot
    lx, ly = px0 + pw - 210, py0 + ph - 78
    for i, (key, _, label, dash) in enumerate(series):
        yy = ly + i * 22
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{yy - 4}" y2="{yy - 4}" stroke="{t["series"][key]}" stroke-width="2.5"{d}/>')
        txt(lx + 30, yy, label, 12, t["ink2"])
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    tp, pp = make_data()
    res = knndm(tp, pp, k=K, seed=SEED, backend="python")
    pool = list(range(1, K + 1)) * math.ceil(len(tp) / K)
    gj_rand = _backend.get("python").cv_distances(tp, np.array(RRNG(SEED).sample(pool, len(tp))))
    outdir = Path(__file__).resolve().parent / "img"
    outdir.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        (outdir / f"knndm-overview-{name}.svg").write_text(draw(res, gj_rand, tp, pp, theme), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
