# -*- coding: utf-8 -*-
"""Fig. 1 - accuracy vs measured board speed, (a) H = 96, (b) H = 720.

    python tools/paper/fig_pareto_board.py
    -> analysis/figures/fig_pareto_board.png (600 dpi) + .svg

Up and to the right is better (user, 2026-09-30):
  x: forecasts per second = 1000 / latency of one forecast for all 7 channels of the
     MetroPT3 cell, measured on the NUCLEO-F103RB (results/board/board_latency_20260929.csv;
     ports without a streaming API are measured at D = 1 and scaled by 7 at H = 720 -
     linear, checked at H = 96).  Energy per forecast is 3.3 V x ~30 mA x latency, so
     the ordering is the same in energy.
  y: test MSE averaged over the six manufacturing datasets (results/_archive/pre_raw_20261007/mfg_raw.csv),
     drawn with the axis inverted so that better (lower) is up; the unit stays MSE so
     the values can be read against Tables 3-4.
Only models with a C port are placed (CycleNet ported 2026-09-30); at H = 720 DLinear
and CycleNet do not fit the flash - DLinear, the more accurate of the two, is drawn as a
dashed reference line instead of a point.

Size and fonts follow the CF-JEPA KBS R2 figures (figsize 11 x 4.2, labels 13 pt,
ticks 12 pt, bold "(a) ..." titles, tab blue / tab red).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.patheffects as pe   # noqa: E402
import pandas as pd               # noqa: E402

plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.edgecolor": "#444444", "axes.linewidth": 0.8})

OURS, OTHER, REF, FRONT = "#1f77b4", "#8c8c8c", "#d62728", "#1f77b4"
OUT = "analysis/figures/fig_pareto_board.png"

BOARD = {"SparseTSF": "stsf", "MixLinear": "mixlinear", "DiPE_Linear": "dipe",
         "SparseTSF_MLP": "stsf_mlp", "FITS": "fits", "DLinear": "dlinear",
         "CycleNet": "cyclenet", "FEATHer_CI": "feather_ci", "FEATHer_mix": "feather_mix"}
LABEL = {"SparseTSF": "SparseTSF", "MixLinear": "MixLinear", "DiPE_Linear": "DiPE-Linear",
         "SparseTSF_MLP": "SparseTSF-MLP", "FITS": "FITS", "DLinear": "DLinear",
         "CycleNet": "CycleNet",
         "FEATHer_CI": "FEATHer-CI (ours)", "FEATHer_mix": "FEATHer-mix (ours)"}
DS = ["Steel", "MetroPT3", "WindSCADA", "N2O", "EWELD", "BlastFurnace"]

# label placement in points from the marker: (dx, dy, ha, va); default = to the left
LABEL_AT = {
    (96, "SparseTSF"): (0, -11, "center", "top"),
    (96, "MixLinear"): (-9, 0, "right", "center"),
    (96, "FEATHer_CI"): (0, -13, "center", "top"),
    (96, "SparseTSF_MLP"): (-9, 0, "right", "center"),
    (96, "DiPE_Linear"): (0, -11, "center", "top"),
    (96, "FITS"): (0, 10, "center", "bottom"),
    (96, "CycleNet"): (9, -3, "left", "top"),
    (96, "FEATHer_mix"): (12, 5, "left", "bottom"),
    (96, "DLinear"): (-9, 0, "right", "center"),
    (720, "SparseTSF"): (0, -11, "center", "top"),
    (720, "MixLinear"): (0, -11, "center", "top"),
    (720, "FEATHer_CI"): (-13, 0, "right", "center"),
    (720, "SparseTSF_MLP"): (0, 10, "center", "bottom"),
    (720, "FITS"): (0, -11, "center", "top"),
    (720, "DiPE_Linear"): (0, 10, "center", "bottom"),
    (720, "FEATHer_mix"): (0, 14, "center", "bottom"),
}
XLIM = {96: (0.8, 40), 720: (0.045, 30)}
# 'better' arrow (tail, head) in axes fraction, placed in an empty corner of each panel
ARROW = {96: ((0.80, 0.80), (0.95, 0.96)), 720: ((0.05, 0.56), (0.20, 0.74))}
HALO = [pe.withStroke(linewidth=3, foreground="white")]


def latency(lat, bm, H):
    r = lat[(lat.model == bm) & (lat.H == H) & (lat.direct_dft == 0)]
    if (r.D == 7).any():
        return float(r[r.D == 7].latency_ms.iloc[0])
    if (r.D == 1).any():
        return 7 * float(r[r.D == 1].latency_ms.iloc[0])
    return None


def frontier(points):
    """non-dominated points (faster and lower MSE are better), fastest first"""
    out, best = [], float("inf")
    for p in sorted(points, key=lambda p: -p[1]):
        if p[2] < best:
            out.append(p)
            best = p[2]
    return out


# F4-SSM ICICIC style (user, 2026-09-30): one colour per model (tab10), black-edged
# circles, ours = red star; then (same day) direct labels at every point, no legend.
COLOR = {"SparseTSF": "#8c564b", "MixLinear": "#1f77b4", "DiPE_Linear": "#9467bd",
         "SparseTSF_MLP": "#ff7f0e", "FITS": "#17becf", "CycleNet": "#e377c2",
         "DLinear": "#2ca02c", "FEATHer_CI": "#d62728", "FEATHer_mix": "#d62728"}
ORDER = ["SparseTSF", "MixLinear", "DiPE_Linear", "SparseTSF_MLP", "FITS", "CycleNet",
         "DLinear", "FEATHer_CI", "FEATHer_mix"]


def style(m):
    """scatter kwargs per model"""
    if m == "FEATHer_mix":
        return dict(marker="*", s=420, color=COLOR[m], edgecolor="black", linewidth=0.9, zorder=5)
    if m == "FEATHer_CI":
        return dict(marker="*", s=300, facecolor="white", edgecolor=COLOR[m], linewidth=1.6,
                    zorder=4)
    return dict(marker="o", s=110, color=COLOR[m], edgecolor="black", linewidth=0.8, zorder=3)


def panel(ax, tag, H, acc, lat, fits):
    ax.grid(True, ls=":", color="#cfcfcf", lw=0.8, zorder=0)
    for m in ORDER:
        if not fits(m, H):
            continue
        ms = latency(lat, BOARD[m], H)
        if ms is None:
            continue
        x, y = 1000.0 / ms, acc[(m, H)]
        ax.scatter([x], [y], **style(m))
        ours = m.startswith("FEATHer")
        ox, oy, ha, va = LABEL_AT.get((H, m), (-9, 0, "right", "center"))
        ax.annotate(LABEL[m], (x, y), xytext=(ox, oy), textcoords="offset points",
                    fontsize=10.5 if ours else 10, ha=ha, va=va,
                    color=COLOR[m] if ours else "#333333",
                    fontweight="bold" if m == "FEATHer_mix" else "normal",
                    path_effects=HALO, zorder=6)
    if not fits("DLinear", H):
        y = acc[("DLinear", H)]
        ax.axhline(y, color=COLOR["DLinear"], lw=1.3, ls=(0, (5, 3)), zorder=1)
        ax.annotate("DLinear: more accurate, but does not fit (546 KB of weights)",
                    (0.985, y), xycoords=("axes fraction", "data"), xytext=(0, 5),
                    textcoords="offset points", fontsize=9.5, color=COLOR["DLinear"],
                    ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(*XLIM[H])
    ax.invert_yaxis()
    ax.set_xlabel("Forecasts per second (log scale)")
    ax.set_title(f"({tag}) Horizon H = {H}", fontweight="bold")


def main():
    raw = pd.read_csv("results/_archive/pre_raw_20261007/mfg_raw.csv")
    per = raw.groupby(["model", "data", "pred_len"]).MSE.mean()
    acc = {(m, H): sum(per[(m, d, H)] for d in DS) / len(DS)
           for m in BOARD for H in (96, 720)}
    lat = pd.read_csv("results/board/board_latency_20260929.csv")
    dep = pd.read_csv("analysis/deployment/deploy_cells.csv")
    dep = dep[dep.data == "MetroPT3"]
    state = {(r.model, int(r.H)): r.state for r in dep.itertuples()}

    def fits(m, H):
        return state.get((m, H)) == "FITS"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, tag, H in zip(axes, "ab", (96, 720)):
        panel(ax, tag, H, acc, lat, fits)
    axes[0].set_ylabel("Test MSE (lower is better)")
    axes[0].set_ylim(0.642, 0.555)
    axes[1].set_ylim(0.882, 0.745)

    fig.tight_layout(w_pad=4.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=600)
    fig.savefig(OUT.replace(".png", ".svg"))
    print(f"Saved -> {OUT}")
    print(f"Saved -> {OUT.replace('.png', '.svg')}")


if __name__ == "__main__":
    main()
