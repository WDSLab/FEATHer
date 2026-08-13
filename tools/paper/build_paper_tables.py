# -*- coding: utf-8 -*-
"""Publication mfg tables (per model: two stacked metric rows; 1st/2nd hi-lit;
FEATHer forced to the BOTTOM). FEATHer = residual-MLP rerun (feather_v2_mlp, 08-04).

  analysis/01_accuracy_mfg.xlsx    per model: MSE row / MAE row
    1_avg 2_per_horizon (all 14) ; 3_edge_avg 4_edge_per_horizon (edge-7)
  analysis/02_robustness_mfg.xlsx  per model: Dw row / corruptMSE row  (edge-7)
    1_avg (per-dataset, fault-mean) ; 2_per_fault (per dataset x fault)
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

BEST = PatternFill("solid", fgColor="C6EFCE")
SECOND = PatternFill("solid", fgColor="FFEB9C")
HDR = PatternFill("solid", fgColor="D9E1F2")
BOLD = Font(bold=True)
CT = Alignment(horizontal="center", vertical="center")
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)

R = "results"
MFG = ["Steel", "WindSCADA", "BlastFurnace", "EWELD", "MetroPT3", "N2O"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Exchange", "Traffic", "Weather"]
HS = [96, 192, 336, 720]
FAULTS = ["gauss", "impulse", "miss", "quant"]
REF_PARAMS = {
    # FEATHer = FINAL N cell (ds16_k7_B4) at D=7/H=96 (Steel 822); scales with
    # D and H (up to 4522 @ BlastFurnace D=28 / H720) — still << 128KB flash.
    "SparseTSF": 41, "MixLinear": 49, "DiPE_Linear": 267, "FEATHer": 822,
    "FITS": 1200, "CycleNet": 9536, "DLinear": 18624, "PatchTST": 35171,
    "TimeMixer": 75497, "TQNet": 86192, "LMS_AutoTSF": 181318,
    "TimesNet": 605479, "iTransformer": 841568, "MDMLP_EIA": 1632832,
}
# Edge/deployment comparison = MCU-deployable baselines only (compare like-for-like).
# DLinear excluded: not deployable -- at H720 it needs 139,680 params x 4B = 545KB
# >> 128KB flash (F103RB); it cannot run on the target device, so it is out of the
# deployable comparison set entirely.
EDGE7 = ["SparseTSF", "MixLinear", "DiPE_Linear", "FITS", "CycleNet", "FEATHer"]


def order_models(models):
    others = sorted([m for m in models if m != "FEATHer"], key=lambda m: REF_PARAMS.get(m, 1e12))
    return others + (["FEATHer"] if "FEATHer" in models else [])


# ---------- generic per-model two-metric-row writer ----------
def write_sheet(wb, name, models, specs, mats, metric_labels, two_row):
    """specs=[(group,sub)]; mats={label: DataFrame[model x MultiIndex(group,sub)]}."""
    ws = wb.create_sheet(name)
    hdr2 = 2 if two_row else 1
    for i, t in enumerate(("model", "params", "metric")):
        ws.cell(1, i + 1, t).font = BOLD
        if two_row:
            ws.merge_cells(start_row=1, start_column=i + 1, end_row=2, end_column=i + 1)
    col0 = 4
    colmap = []
    if two_row:
        i, cc = 0, col0
        while i < len(specs):
            g = specs[i][0]; j = i
            while j < len(specs) and specs[j][0] == g:
                j += 1
            ws.merge_cells(start_row=1, start_column=cc, end_row=1, end_column=cc + (j - i) - 1)
            hc = ws.cell(1, cc, g); hc.font = BOLD; hc.alignment = CT; hc.fill = HDR
            for k in range(j - i):
                sc = ws.cell(2, cc + k, specs[i + k][1]); sc.font = BOLD; sc.alignment = CT; sc.fill = HDR
                colmap.append((cc + k, specs[i + k]))
            cc += (j - i); i = j
    else:
        for k, spec in enumerate(specs):
            hc = ws.cell(1, col0 + k, spec[0]); hc.font = BOLD; hc.alignment = CT; hc.fill = HDR
            colmap.append((col0 + k, spec))

    top = hdr2 + 1
    row_of = {}
    for mi, m in enumerate(models):
        r0 = top + mi * len(metric_labels)
        ws.merge_cells(start_row=r0, start_column=1, end_row=r0 + len(metric_labels) - 1, end_column=1)
        ws.merge_cells(start_row=r0, start_column=2, end_row=r0 + len(metric_labels) - 1, end_column=2)
        nm = ws.cell(r0, 1, m); nm.alignment = CT
        if m == "FEATHer":
            nm.font = BOLD
        ws.cell(r0, 2, REF_PARAMS.get(m, "")).alignment = CT
        for li, lab in enumerate(metric_labels):
            rr = r0 + li
            ws.cell(rr, 3, lab)
            row_of[(m, lab)] = rr
            df = mats[lab]
            for (ec, spec) in colmap:
                if m in df.index and spec in df.columns and pd.notna(df.loc[m, spec]):
                    ws.cell(rr, ec, round(float(df.loc[m, spec]), 4))
    # highlight within each metric per column (lower better)
    for (ec, spec) in colmap:
        for lab in metric_labels:
            vals = [(ws.cell(row_of[(m, lab)], ec).value, row_of[(m, lab)]) for m in models
                    if isinstance(ws.cell(row_of[(m, lab)], ec).value, (int, float))]
            if len(vals) >= 2:
                vals.sort()
                ws.cell(vals[0][1], ec).fill = BEST; ws.cell(vals[0][1], ec).font = BOLD
                ws.cell(vals[1][1], ec).fill = SECOND
    last = top + len(models) * len(metric_labels) - 1
    for row in ws.iter_rows(min_row=1, max_row=last, min_col=1, max_col=col0 + len(specs) - 1):
        for cell in row:
            cell.border = THIN
    ws.column_dimensions["A"].width = 13
    ws.column_dimensions["B"].width = 9
    ws.column_dimensions["C"].width = 10


# ---------- accuracy ----------
def acc_matrix(d, datasets, metric, per_h):
    specs, series = [], {}
    for ds in datasets:
        piv = d[d.data == ds].pivot_table(index="model", columns="pred_len", values=metric)
        piv = piv.reindex(columns=[h for h in HS if h in piv.columns]); piv["Avg"] = piv.mean(axis=1)
        if per_h:
            for h in piv.columns:
                specs.append((ds, str(h))); series[(ds, str(h))] = piv[h]
        else:
            specs.append((ds, "Avg")); series[(ds, "Avg")] = piv["Avg"]
    df = pd.DataFrame(series); df.columns = pd.MultiIndex.from_tuples(df.columns)
    df[("OVERALL", "Avg")] = df[[c for c in df.columns if c[1] == "Avg"]].mean(axis=1)
    specs.append(("OVERALL", "Avg"))
    return specs, df


def load_acc():
    b = pd.read_csv(f"{R}/baselines/fcst_results_baselines.csv")
    b = b[b.exp_tag == "main"]
    # FEATHer = FINAL grid-tuned 3-seed run (2026-08-13): mfg = canonical N cell
    # (ds16_k7_B4), LTSF = per-dataset CI cell. exp_tag "final" in feather_final.csv.
    # LTSF rows fill in once the --group ltsf final sweep lands.
    f = pd.read_csv(f"{R}/feather/feather_final.csv")
    f = f[f.exp_tag == "final"]
    return pd.concat([b, f])


# ---------- robustness ----------
def rob_matrix(d, datasets, valuecol, per_fault):
    specs, series = [], {}
    for ds in datasets:
        sub = d[d.data == ds]
        if per_fault:
            piv = sub.pivot_table(index="model", columns="fault_type", values=valuecol)
            piv = piv.reindex(columns=[f for f in FAULTS if f in piv.columns]); piv["Avg"] = piv.mean(axis=1)
            for f in piv.columns:
                specs.append((ds, f)); series[(ds, f)] = piv[f]
        else:
            specs.append((ds, "Avg")); series[(ds, "Avg")] = sub.groupby("model")[valuecol].mean()
    df = pd.DataFrame(series); df.columns = pd.MultiIndex.from_tuples(df.columns)
    df[("OVERALL", "Avg")] = df[[c for c in df.columns if c[1] == "Avg"]].mean(axis=1)
    specs.append(("OVERALL", "Avg"))
    return specs, df


def load_rob():
    # FEATHer = residual-MLP robustness (robust_feather_v2_mlp)
    d = pd.concat([pd.read_csv(f"{R}/baselines/robust_results_baselines.csv"),
                   pd.read_csv(f"{R}/feather/robust_feather_v2_mlp.csv")])
    d = d[d.fault_type != "clean"].copy()
    for c in ("Dw", "MSE"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Dw"])


if __name__ == "__main__":
    import os
    os.makedirs("analysis", exist_ok=True)

    acc = load_acc()
    all_m = order_models([m for m in acc.model.unique() if m in REF_PARAMS])
    edge = order_models([m for m in EDGE7 if m in acc.model.unique()])
    wb = Workbook(); wb.remove(wb.active)
    for name, models, per_h in [("1_avg", all_m, False), ("2_per_horizon", all_m, True),
                                ("3_edge_avg", edge, False), ("4_edge_per_horizon", edge, True)]:
        sp, mse = acc_matrix(acc, MFG, "MSE", per_h)
        _, mae = acc_matrix(acc, MFG, "MAE", per_h)
        write_sheet(wb, name, models, sp, {"MSE": mse, "MAE": mae}, ["MSE", "MAE"], per_h)
    wb.save("analysis/01_accuracy_mfg.xlsx"); print("saved 01_accuracy_mfg.xlsx")

    # ---- LTSF accuracy (same 4-sheet layout over the 8 LTSF datasets) ----
    all_l = order_models([m for m in acc.model.unique() if m in REF_PARAMS])
    edge_l = order_models([m for m in EDGE7 if m in acc.model.unique()])
    wb = Workbook(); wb.remove(wb.active)
    for name, models, per_h in [("1_avg", all_l, False), ("2_per_horizon", all_l, True),
                                ("3_edge_avg", edge_l, False), ("4_edge_per_horizon", edge_l, True)]:
        sp, mse = acc_matrix(acc, LTSF, "MSE", per_h)
        _, mae = acc_matrix(acc, LTSF, "MAE", per_h)
        write_sheet(wb, name, models, sp, {"MSE": mse, "MAE": mae}, ["MSE", "MAE"], per_h)
    wb.save("analysis/03_accuracy_ltsf.xlsx"); print("saved 03_accuracy_ltsf.xlsx")

    rob = load_rob()
    redge = order_models([m for m in EDGE7 if m in rob.model.unique()])
    wb = Workbook(); wb.remove(wb.active)
    for name, per_f in [("1_avg", False), ("2_per_fault", True)]:
        sp, dw = rob_matrix(rob, MFG, "Dw", per_f)
        _, cm = rob_matrix(rob, MFG, "MSE", per_f)
        write_sheet(wb, name, redge, sp, {"Dw": dw, "corruptMSE": cm}, ["Dw", "corruptMSE"], per_f)
    wb.save("analysis/02_robustness_mfg.xlsx"); print("saved 02_robustness_mfg.xlsx | edge:", redge)
