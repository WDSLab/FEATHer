#!/usr/bin/env python
"""RevIN-fragility A/B judgment table.

Reads the two isolated A/B CSVs and prints, per dataset, the 3-arm comparison
that decides whether removing RevIN (no_revin) is a general win or whether a
robust-RevIN (median/MAD) is needed instead:

    clean MSE  : feather_mlp_ab.csv   (does removal hurt clean anywhere?)
    robust Dw  : robust_mlp_ab.csv    (does removal restore robustness?)

Arms (all MLP-on except feather_base):
    feather_base        canonical (no MLP, RevIN on, gate on)  = reference
    feather_mlp16       MLP on,  RevIN on,  gate on
    feather_mlp_gateoff MLP on,  RevIN on,  gate UNIFORM        = "gate ~0" control
    feather_mlp_norevin MLP on,  RevIN OFF, gate on             = treatment

Decision (per dataset, mlp16 -> norevin):
    clean improves AND Dw drops  -> removal wins here
    clean HURTS but Dw drops     -> need robust-RevIN (median/MAD), not removal
    Dw already ~1.0 at baseline  -> no fragility here (RevIN not the problem)

Usage (PYTHONUTF8=1 on Windows):
    python tools/audit/revin_ab_summary.py
    python tools/audit/revin_ab_summary.py --results_dir "<Dropbox>/results"
"""
import argparse, os
import pandas as pd

ARMS = ["feather_base", "feather_mlp16", "feather_mlp_gateoff",
        "feather_mlp_norevin", "feather_mlp_robustrevin"]
DEFAULT_DIR = r"C:\Users\lee\Dropbox\WDSLab\05. 실험\FEATHer\results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=DEFAULT_DIR)
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    acc = pd.read_csv(os.path.join(a.results_dir, "feather_mlp_ab.csv"))
    rob = pd.read_csv(os.path.join(a.results_dir, "robust_mlp_ab.csv"))

    print("=" * 66)
    print("CLEAN MSE / MAE  (feather_mlp_ab.csv; seeds = distinct seed count)")
    print("=" * 66)
    g = (acc.groupby(["data", "exp_tag"])
            .agg(seeds=("seed", "nunique"), H=("pred_len", "nunique"),
                 MSE=("MSE", "mean"), MAE=("MAE", "mean")).round(4))
    print(g)

    print("\n" + "=" * 66)
    print("ROBUST Dw  (robust_mlp_ab.csv; mean over faults x severities, drop clean)")
    print("=" * 66)
    rr = rob[rob.get("fault_type", "clean") != "clean"].copy()
    dw = (rr.groupby(["data", "train_exp_tag"])
            .agg(n=("Dw", "size"), Dw=("Dw", "mean")).round(4))
    print(dw)
    print("\n Dw by fault (norevin vs mlp16), per dataset:")
    for ds in sorted(rr.data.unique()):
        sub = rr[rr.data == ds]
        piv = sub.pivot_table(index="train_exp_tag", columns="fault_type",
                              values="Dw").reindex([x for x in ARMS
                              if x in sub.train_exp_tag.unique()]).round(3)
        print(f"\n [{ds}]"); print(piv)

    print("\n" + "=" * 66)
    print("DECISION per dataset  (mlp16 -> norevin)")
    print("=" * 66)
    acc_p = acc.groupby(["data", "exp_tag"]).MSE.mean().unstack("exp_tag")
    dw_p = dw["Dw"].unstack("train_exp_tag")
    for ds in sorted(set(acc_p.index) | set(dw_p.index)):
        m16 = acc_p.loc[ds, "feather_mlp16"] if ds in acc_p.index and "feather_mlp16" in acc_p.columns else float("nan")
        mnr = acc_p.loc[ds, "feather_mlp_norevin"] if ds in acc_p.index and "feather_mlp_norevin" in acc_p.columns else float("nan")
        d16 = dw_p.loc[ds, "feather_mlp16"] if ds in dw_p.index and "feather_mlp16" in dw_p.columns else float("nan")
        dnr = dw_p.loc[ds, "feather_mlp_norevin"] if ds in dw_p.index and "feather_mlp_norevin" in dw_p.columns else float("nan")
        verdict = "?"
        if pd.notna(d16) and d16 < 1.05:
            verdict = "no fragility (baseline Dw already ~1.0)"
        elif pd.notna(mnr) and pd.notna(dnr):
            clean_better = mnr <= m16 + 1e-9
            robust_better = dnr < d16 - 1e-3
            if clean_better and robust_better:
                verdict = "REMOVAL WINS (clean+robust both improve)"
            elif robust_better and not clean_better:
                verdict = "NEED ROBUST-RevIN (Dw drops but clean hurts)"
            elif not robust_better:
                verdict = "removal does NOT help robustness here"
        print(f"  {ds:>13}: clean mlp16={m16:.4f} -> norevin={mnr if pd.notna(mnr) else float('nan'):.4f}"
              f" | Dw mlp16={d16 if pd.notna(d16) else float('nan'):.3f} -> norevin={dnr if pd.notna(dnr) else float('nan'):.3f}"
              f"  => {verdict}")


if __name__ == "__main__":
    main()
