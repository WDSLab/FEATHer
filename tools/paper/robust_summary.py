# -*- coding: utf-8 -*-
"""
Robustness sweep summary — degradation metrics, heatmaps, and the
FEATHer-vs-SparseTSF story that justifies FEATHer's 453 params over
SparseTSF's 41.

Reads results/robust_results.csv (schema: exp_tag, train_exp_tag, model,
data, pred_len, seq_len, seed, fault_type, severity, MSE, MAE, RMSE,
CORR, R2, MSE_clean, Dw, timestamp). The worker writes
Dw = MSE_corrupt / MSE_clean (>= 1, higher = worse). Produces:

  (1) Per-(fault_type, model) MSE x severity heatmap (one PNG per data,
      horizon) — saved under tools/paper/figures/.
  (2) D_w table: retention ratio MSE_corrupt / MSE_clean aggregated over
      (data, pred_len, seed); FEATHer vs best baseline per fault axis.
  (3) mu_rel = MSE_clean / MSE_corrupt (in (0,1], 1 = perfect retention).
  (4) AUC robustness score: severity-integral of the degradation curve on
      a normalized [0,1] severity axis (quant's bit axis is reversed so
      that "more severe" always increases). Lower = more robust.
  (5) mCE normalized to a reference model (default SparseTSF, the 41-param
      linear rival): Hendrycks & Dietterich (ICLR'19) style corruption
      error, per (data, pred_len, fault) ratio of summed error, averaged.
      < 1 = more robust than the reference.
  (6) FEATHer-vs-SparseTSF degradation curve: mean Dw vs severity for both,
      the fitted slope, and a paired Wilcoxon test at the top severity of
      each fault axis (reuses tools/paper/wilcoxon.py).

Framing guard (see manuscript/notes/robustness_justification_evidence.md):
the claim is NOT "more params => more robust" (monotone, counter-attackable)
but "a 41-param linear head lacks representational headroom to absorb sensor
faults; FEATHer's multi-band + FFT-gate inductive bias buys graceful
degradation at still-sub-1K." Hence every comparison is anchored to SparseTSF.

Directly addresses:
  R6 #7a  — "no noise robustness experiments"  (edge-PdM selling point)
  R8 #7   — partial; energy/RAM/bit-width still need on-device measurement

Usage:
    python tools/paper/robust_summary.py
    python tools/paper/robust_summary.py --metric MSE --no_heatmap
    python tools/paper/robust_summary.py --reference SparseTSF --focus FEATHer
    python tools/paper/robust_summary.py --output_dir tools/paper/figures
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_DIR)))
sys.path.insert(0, _THIS_DIR)  # so `import wilcoxon` (sibling) resolves

from wilcoxon import wilcoxon_pair  # noqa: E402  reuse the paired signed-rank test
# AUDIT-FIX (2026-07-14, P3): import the severity tables from the torch-free
# single source instead of a hand-copied mirror — a silent drift here would
# make the top-severity Wilcoxon match 0 rows (NaN p) with no error.
from utils.severity import SEVERITY_SWEEPS as _SEVERITY_SWEEPS  # noqa: E402
from utils.severity import CLEAN_SENTINEL  # noqa: E402

LOWER_IS_BETTER = {"MSE", "MAE", "RMSE"}

_QUANT_CLEAN_BITS = float(CLEAN_SENTINEL["quant"])


def normalized_severity(fault_type, severity):
    """Map a raw severity onto [0, 1] where 0 == clean and 1 == worst,
    monotonically increasing in corruption strength.

    gauss/miss/impulse: linear in the raw value (max sweep value -> 1).
    quant: distortion ~ 1/bits, so map (1/bits - 1/clean) / (1/min_bits -
    1/clean). Fewer bits -> larger value.
    """
    if fault_type == "quant":
        inv = 1.0 / max(float(severity), 1e-9)
        inv_clean = 1.0 / _QUANT_CLEAN_BITS
        inv_worst = 1.0 / min(_SEVERITY_SWEEPS["quant"])
        denom = (inv_worst - inv_clean) or 1.0
        return (inv - inv_clean) / denom
    sweep = _SEVERITY_SWEEPS.get(fault_type)
    worst = max(sweep) if sweep else 1.0
    return float(severity) / (worst or 1.0)


# ---------------------------------------------------------------------------
# (2) D_w retention-ratio table
# ---------------------------------------------------------------------------
def _common_cells(df, cell_cols=("data", "pred_len", "seed")):
    """AUDIT-FIX (2026-07-19, #6): restrict to the (data, pred_len, seed)
    cells present for EVERY model, so Dw / mu_rel / AUC all pool over the
    same denominator (mCE already does this). Without it, a model whose
    sweep is only partially done gets its degradation averaged over a
    DIFFERENT, easier/harder cell set and its rank is silently distorted.
    A no-op once the sweep is 100% complete; warns on any drop."""
    models = df["model"].unique()
    if len(models) <= 1:
        return df
    cols = list(cell_cols)
    per_cell = df.groupby(cols)["model"].nunique()
    common = per_cell[per_cell == len(models)].reset_index()[cols]
    if len(common) < len(per_cell):
        print(f"[warn] robust_summary: {len(per_cell) - len(common)} of "
              f"{len(per_cell)} ({', '.join(cols)}) cells are not covered by "
              f"all {len(models)} models — pooling over the {len(common)} "
              f"common cells only (pass a completed sweep to avoid this).",
              file=sys.stderr)
    return df.merge(common, on=cols, how="inner")


def dw_table(df, metric):
    """For each (model, fault_type), aggregate the retention RATIO
    metric_corrupt / metric_clean across all (data, pred_len, seed) cells
    and severities. 1.0 = no degradation, higher = worse.

    For MSE we reuse the worker-written `Dw` column (already a ratio); for
    other metrics we recompute the SAME ratio form so units never mix.
    """
    df = _common_cells(df)
    work = df[df["fault_type"] != "clean"].copy()
    if metric == "MSE" and "Dw" in work.columns:
        work["dw_row"] = work["Dw"]
    else:
        clean = (df[df["fault_type"] == "clean"]
                 .groupby(["model", "data", "pred_len", "seed"])[metric]
                 .first()
                 .rename("clean_val")
                 .reset_index())
        work = work.merge(clean, on=["model", "data", "pred_len", "seed"], how="left")
        if metric in LOWER_IS_BETTER:
            work["dw_row"] = work[metric] / work["clean_val"].replace(0, np.nan)
        else:
            # bigger-is-better: invert so >1 still means "degraded"
            work["dw_row"] = work["clean_val"].replace(0, np.nan) / work[metric]

    agg = (work.groupby(["model", "fault_type"])["dw_row"]
                 .agg(["mean", "std", "count"])
                 .reset_index())
    agg.columns = ["model", "fault_type", "Dw_mean", "Dw_std", "n"]
    return agg


def best_vs_feather(dw, focus="FEATHer"):
    """For each fault_type, pick the model with lowest Dw_mean and show how
    `focus` compares (lower Dw = more robust)."""
    rows = []
    for ft, sub in dw.groupby("fault_type"):
        sub = sub.sort_values("Dw_mean")
        best_row = sub.iloc[0]
        foc = sub[sub["model"] == focus]
        if foc.empty:
            continue
        f = foc.iloc[0]
        rows.append({
            "fault_type": ft,
            "best_model": best_row["model"],
            "best_Dw": best_row["Dw_mean"],
            f"{focus}_Dw": f["Dw_mean"],
            f"{focus}_rank": int(sub["Dw_mean"].rank().loc[f.name]),
            "n_models": len(sub),
        })
    return pd.DataFrame(rows).sort_values("fault_type")


# ---------------------------------------------------------------------------
# (3) mu_rel = MSE_clean / MSE_corrupt (retention fraction, higher = better)
# ---------------------------------------------------------------------------
def mu_rel_table(df, metric="MSE"):
    """Relative retention mu_rel = metric_clean / metric_corrupt in (0, 1].
    1.0 = perfectly retained accuracy under corruption. This is the
    reciprocal of D_w and is the form cited in the evidence note."""
    df = _common_cells(df)   # AUDIT-FIX (#6): shared denominator across models
    corrupt = df[df["fault_type"] != "clean"].copy()
    clean = (df[df["fault_type"] == "clean"]
             .groupby(["model", "data", "pred_len", "seed"])[metric]
             .first().rename("clean_val").reset_index())
    corrupt = corrupt.merge(clean, on=["model", "data", "pred_len", "seed"], how="left")
    if metric in LOWER_IS_BETTER:
        corrupt["mu_rel"] = corrupt["clean_val"] / corrupt[metric].replace(0, np.nan)
    else:
        corrupt["mu_rel"] = corrupt[metric] / corrupt["clean_val"].replace(0, np.nan)
    agg = (corrupt.groupby(["model", "fault_type"])["mu_rel"]
                  .agg(["mean", "std", "count"]).reset_index())
    agg.columns = ["model", "fault_type", "mu_rel_mean", "mu_rel_std", "n"]
    return agg


# ---------------------------------------------------------------------------
# (4) AUC robustness score — integral of the degradation curve
# ---------------------------------------------------------------------------
def auc_table(df, metric="MSE"):
    """Per (model, fault_type): area under the degradation curve on a
    normalized [0,1] severity axis. Curve y = mean D_w (metric_corrupt /
    metric_clean) at each severity, anchored by the clean point (x=0, y=1).
    Trapezoidal integral; lower AUC = more robust. Because x is normalized
    to [0,1], the AUC of a perfectly robust model (flat y=1) is 1.0, so
    (AUC - 1) is the excess degradation area.
    """
    dw = _per_severity_dw(df, metric)
    rows = []
    for (model, ft), sub in dw.groupby(["model", "fault_type"]):
        xs = [0.0]
        ys = [1.0]  # clean anchor: no degradation
        for _, r in sub.sort_values("sev_norm").iterrows():
            xs.append(r["sev_norm"])
            ys.append(r["dw"])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        auc = float(np.trapz(ys, xs))
        rows.append({"model": model, "fault_type": ft,
                     "AUC": auc, "excess_AUC": auc - 1.0, "n_sev": len(sub)})
    return pd.DataFrame(rows).sort_values(["fault_type", "AUC"])


def _per_severity_dw(df, metric="MSE"):
    """Mean D_w (metric_corrupt / metric_clean) per (model, fault_type,
    severity) with a normalized severity column attached."""
    df = _common_cells(df)   # AUDIT-FIX (#6): shared denominator (feeds AUC)
    corrupt = df[df["fault_type"] != "clean"].copy()
    if metric == "MSE" and "Dw" in corrupt.columns:
        corrupt["dw_row"] = corrupt["Dw"]
    else:
        clean = (df[df["fault_type"] == "clean"]
                 .groupby(["model", "data", "pred_len", "seed"])[metric]
                 .first().rename("clean_val").reset_index())
        corrupt = corrupt.merge(clean, on=["model", "data", "pred_len", "seed"], how="left")
        if metric in LOWER_IS_BETTER:
            corrupt["dw_row"] = corrupt[metric] / corrupt["clean_val"].replace(0, np.nan)
        else:
            corrupt["dw_row"] = corrupt["clean_val"].replace(0, np.nan) / corrupt[metric]
    g = (corrupt.groupby(["model", "fault_type", "severity"])["dw_row"]
                .mean().rename("dw").reset_index())
    g["sev_norm"] = g.apply(
        lambda r: normalized_severity(r["fault_type"], r["severity"]), axis=1)
    return g


# ---------------------------------------------------------------------------
# (5) mCE normalized to a reference model (SparseTSF)
# ---------------------------------------------------------------------------
def mce_table(df, reference="SparseTSF", metric="MSE"):
    """Hendrycks & Dietterich (ICLR'19) mean Corruption Error, normalized to
    `reference`. For each (data, pred_len, fault_type):
        CE_model = sum_severity mean_seed(metric_model)
                 / sum_severity mean_seed(metric_reference)
    then average CE over (data, pred_len) per fault, and over faults for the
    overall mCE. Using the summed-error RATIO cancels the per-dataset scale,
    so different datasets/horizons are comparable. < 1 = more robust than
    the reference; = 1 is the reference itself.
    """
    corrupt = df[df["fault_type"] != "clean"].copy()
    # mean over seeds at each (model, data, pred_len, fault, severity)
    cell = (corrupt.groupby(["model", "data", "pred_len", "fault_type", "severity"])[metric]
                   .mean().reset_index())
    # sum over severities -> per (model, data, pred_len, fault)
    summed = (cell.groupby(["model", "data", "pred_len", "fault_type"])[metric]
                  .sum().rename("err_sum").reset_index())
    ref = (summed[summed["model"] == reference]
           .rename(columns={"err_sum": "ref_sum"})
           .drop(columns=["model"]))
    merged = summed.merge(ref, on=["data", "pred_len", "fault_type"], how="left")
    merged["CE"] = merged["err_sum"] / merged["ref_sum"].replace(0, np.nan)

    # AUDIT-FIX (2026-07-14, P2-6): a left-merge + skipna mean silently
    # averaged models over DIFFERENT cell sets whenever reference rows were
    # partially missing. Restrict every model to the cell set that (a) has a
    # reference value and (b) is present for ALL models, and say what was
    # dropped — an mCE from unequal cell sets is not comparable.
    cell_cols = ["data", "pred_len", "fault_type"]
    no_ref = merged.loc[merged["CE"].isna(), cell_cols].drop_duplicates()
    if not no_ref.empty:
        print(f"[mCE] WARNING: {len(no_ref)} (data, pred_len, fault) cell(s) "
              f"lack a {reference} reference — dropped for ALL models:",
              file=sys.stderr)
        for _, r in no_ref.iterrows():
            print(f"        {r['data']} | H={r['pred_len']} | {r['fault_type']}",
                  file=sys.stderr)
        merged = merged.dropna(subset=["CE"])

    n_models = merged["model"].nunique()
    per_cell = merged.groupby(cell_cols)["model"].nunique()
    common = per_cell[per_cell == n_models].reset_index()[cell_cols]
    n_dropped = int((per_cell < n_models).sum())
    if n_dropped:
        print(f"[mCE] WARNING: {n_dropped} cell(s) missing for some model(s) "
              f"— mCE computed over the {len(common)} cells common to all "
              f"{n_models} models.", file=sys.stderr)
        merged = merged.merge(common, on=cell_cols, how="inner")

    per_fault = (merged.groupby(["model", "fault_type"])["CE"]
                       .mean().reset_index())
    per_fault_wide = per_fault.pivot(index="model", columns="fault_type",
                                     values="CE")
    # AUDIT-FIX (2026-07-23): mCE must average a FIXED corruption set. A plain
    # mean(axis=1) uses pandas' default skipna=True, so a model missing an
    # ENTIRE fault axis (its sweep for that fault never ran, or all its cells
    # got dropped for lacking a reference) would be averaged over fewer faults
    # and could jump up the robustness ranking (sort_values("mCE")) silently —
    # the headline robustness result. Average with skipna=False so an absent
    # fault axis yields NaN mCE (sorts last, excluded from the ranking) + warn.
    fault_cols = list(per_fault_wide.columns)
    incomplete = per_fault_wide.index[
        per_fault_wide[fault_cols].isna().any(axis=1)].tolist()
    if incomplete:
        print(f"[mCE] WARNING: {len(incomplete)} model(s) lack a full fault axis "
              f"({','.join(map(str, fault_cols))}) — mCE set NaN (excluded from "
              f"ranking): {incomplete}", file=sys.stderr)
    per_fault_wide["mCE"] = per_fault_wide[fault_cols].mean(axis=1, skipna=False)
    per_fault_wide["n_cells"] = merged.groupby("model").size()
    return per_fault_wide.sort_values("mCE").reset_index()


# ---------------------------------------------------------------------------
# (6) FEATHer-vs-reference degradation curve + slope + top-severity Wilcoxon
# ---------------------------------------------------------------------------
def degradation_curve(df, focus="FEATHer", reference="SparseTSF", metric="MSE"):
    """Mean D_w vs normalized severity for `focus` and `reference` per fault
    axis, the fitted slope (per unit normalized severity; steeper = worse),
    and a paired one-sided Wilcoxon at the TOP severity of each fault axis
    (H1: focus better than reference). Returns (curve_df, slope_df)."""
    dw = _per_severity_dw(df, metric)
    dw = dw[dw["model"].isin([focus, reference])]

    # Slope per (model, fault): linear fit dw ~ sev_norm, anchored with clean.
    slope_rows = []
    for (model, ft), sub in dw.groupby(["model", "fault_type"]):
        xs = np.concatenate([[0.0], sub["sev_norm"].to_numpy()])
        ys = np.concatenate([[1.0], sub["dw"].to_numpy()])
        if len(xs) >= 2 and np.ptp(xs) > 0:
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = np.nan
        slope_rows.append({"model": model, "fault_type": ft, "slope": slope})
    slope_df = pd.DataFrame(slope_rows)

    # Top-severity paired Wilcoxon (focus vs reference) per fault.
    wil_rows = []
    for ft in sorted(df[df["fault_type"] != "clean"]["fault_type"].unique()):
        sweep = _SEVERITY_SWEEPS.get(ft, [])
        # "top severity" = worst corruption: max value, except quant = fewest bits
        top_sev = min(sweep) if ft == "quant" else max(sweep) if sweep else None
        if top_sev is None:
            continue
        sub = df[(df["fault_type"] == ft) & (df["severity"] == top_sev)]
        p, n, md = wilcoxon_pair(sub, metric, focus, reference)
        wil_rows.append({
            "fault_type": ft, "top_severity": top_sev, "n_pairs": n,
            "median_diff": md, "p_value": p,
            "focus_better_05": (p is not None and not np.isnan(p) and p < 0.05
                                and md < 0),
        })
    wil_df = pd.DataFrame(wil_rows)

    slope_wide = slope_df.pivot(index="fault_type", columns="model", values="slope")
    slope_wide = slope_wide.merge(
        wil_df.set_index("fault_type")[["top_severity", "n_pairs", "median_diff",
                                        "p_value", "focus_better_05"]],
        left_index=True, right_index=True, how="left")
    return dw, slope_wide.reset_index()


def plot_degradation_curve(dw, focus, reference, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available — install for the curve plot",
              file=sys.stderr)
        return
    faults = sorted(dw["fault_type"].unique())
    fig, axes = plt.subplots(1, len(faults), figsize=(4 * len(faults), 4),
                             squeeze=False)
    for ax, ft in zip(axes[0], faults):
        for model, style in [(focus, "-o"), (reference, "--s")]:
            sub = dw[(dw["model"] == model) & (dw["fault_type"] == ft)]
            sub = sub.sort_values("sev_norm")
            xs = np.concatenate([[0.0], sub["sev_norm"].to_numpy()])
            ys = np.concatenate([[1.0], sub["dw"].to_numpy()])
            ax.plot(xs, ys, style, label=model)
        ax.set_title(ft)
        ax.set_xlabel("normalized severity")
        ax.set_ylabel("D_w  (MSE_corrupt / MSE_clean)")
        ax.axhline(1.0, color="gray", lw=0.7, ls=":")
        ax.legend()
    fig.suptitle(f"Degradation curve: {focus} vs {reference}", y=1.03)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"degradation_{focus}_vs_{reference}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# (1) heatmap (unchanged)
# ---------------------------------------------------------------------------
def heatmap(df, metric, data, pred_len, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available — install for heatmaps", file=sys.stderr)
        return

    sub = df[(df["data"] == data) & (df["pred_len"] == pred_len) & (df["fault_type"] != "clean")]
    if sub.empty:
        return

    fault_types = sorted(sub["fault_type"].unique())
    fig, axes = plt.subplots(1, len(fault_types), figsize=(4 * len(fault_types), 6),
                              squeeze=False)
    for ax, ft in zip(axes[0], fault_types):
        sub_ft = sub[sub["fault_type"] == ft]
        pivot = sub_ft.groupby(["model", "severity"])[metric].mean().unstack(fill_value=np.nan)
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{ft} ({metric})")
        ax.set_xlabel("severity")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{data}  H={int(pred_len)}", y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"robust_{data}_H{int(pred_len)}_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Robustness summary tables + heatmaps")
    p.add_argument("--csv", default="results/robust_results.csv")
    p.add_argument("--exp_tag", default="robust")
    p.add_argument("--metric", default="MSE",
                   choices=["MSE", "MAE", "RMSE", "CORR", "R2"])
    p.add_argument("--reference", default="SparseTSF",
                   help="Reference model for mCE + degradation curve (the "
                        "41-param linear rival FEATHer must out-degrade).")
    p.add_argument("--focus", default="FEATHer",
                   help="Model whose robustness we are arguing for.")
    p.add_argument("--no_heatmap", action="store_true",
                   help="Skip PNG heatmap generation.")
    p.add_argument("--no_curve", action="store_true",
                   help="Skip the degradation-curve PNG.")
    p.add_argument("--output_dir", default="tools/paper/figures",
                   help="Where to save figures.")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[empty] {args.csv} does not exist yet", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.csv)
    if args.exp_tag:
        df = df[df["exp_tag"] == args.exp_tag]
    if df.empty:
        print(f"No rows for exp_tag={args.exp_tag}", file=sys.stderr)
        sys.exit(1)

    models = sorted(df["model"].unique())
    have_ref = args.reference in models
    have_focus = args.focus in models

    print(f"\n=== (2) D_w retention ratio per (model, fault_type), metric={args.metric} ===")
    dw = dw_table(df, args.metric)
    print(dw.round(4).to_string(index=False))

    print(f"\n=== {args.focus} vs best model per fault axis (D_w) ===")
    if have_focus:
        print(best_vs_feather(dw, focus=args.focus).round(4).to_string(index=False))
    else:
        print(f"  [skip] {args.focus} not in results")

    print(f"\n=== (3) mu_rel = {args.metric}_clean / {args.metric}_corrupt (higher=better) ===")
    print(mu_rel_table(df, args.metric).round(4).to_string(index=False))

    print(f"\n=== (4) AUC robustness score (normalized-severity integral of D_w; lower=better) ===")
    print(auc_table(df, args.metric).round(4).to_string(index=False))

    print(f"\n=== (5) mCE normalized to {args.reference} (<1 = more robust than {args.reference}) ===")
    if have_ref:
        print(mce_table(df, reference=args.reference, metric=args.metric)
              .round(4).to_string(index=False))
    else:
        print(f"  [skip] reference {args.reference} not in results")

    print(f"\n=== (6) Degradation curve slope + top-severity Wilcoxon: "
          f"{args.focus} vs {args.reference} ===")
    if have_ref and have_focus:
        curve, slope_tbl = degradation_curve(df, focus=args.focus,
                                              reference=args.reference,
                                              metric=args.metric)
        print(slope_tbl.round(4).to_string(index=False))
        print(f"  (slope = d D_w / d(norm severity); smaller = flatter = more robust)")
        if not args.no_curve:
            plot_degradation_curve(curve, args.focus, args.reference, args.output_dir)
    else:
        print(f"  [skip] need both {args.focus} and {args.reference} in results")

    if not args.no_heatmap:
        for (data, h), _ in df.groupby(["data", "pred_len"]):
            heatmap(df, args.metric, data, h, args.output_dir)


if __name__ == "__main__":
    main()
