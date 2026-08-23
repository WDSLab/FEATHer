# -*- coding: utf-8 -*-
"""Publication mfg tables (per model: two stacked metric rows; 1st/2nd hi-lit;
FEATHer forced to the BOTTOM). FEATHer = residual-MLP rerun (feather_v2_mlp, 08-04).

  analysis/01_accuracy_mfg.xlsx    per model: MSE row / MAE row
    1_avg 2_per_horizon (all 14) ; 3_edge_avg 4_edge_per_horizon (edge-7)
  analysis/02_robustness_mfg.xlsx  per model: Dw row / corruptMSE row  (edge-7)
    1_avg (per-dataset, fault-mean) ; 2_per_fault (per dataset x fault)
"""
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

BEST = PatternFill("solid", fgColor="C6EFCE")
SECOND = PatternFill("solid", fgColor="FFEB9C")
HDR = PatternFill("solid", fgColor="DDDDDD")
OURS = PatternFill("solid", fgColor="FFF2CC")
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
def write_sheet(wb, name, models, specs, mats, metric_labels, two_row, title=None):
    """specs=[(group,sub)]; mats={label: DataFrame[model x MultiIndex(group,sub)]}.

    Layout (CF-JEPA convention): merged caption on row 1, blank spacer row 2,
    header from row 3 (row 3-4 when `two_row`), body below.
    """
    ws = wb.create_sheet(name)
    ncols = 3 + len(specs)
    h1 = 1
    if title:
        ws.cell(1, 1, title).font = Font(size=10)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
        h1 = 3                      # 1 = caption, 2 = spacer, 3 = header
    h2 = h1 + 1 if two_row else h1
    for i, t in enumerate(("model", "params", "metric")):
        c = ws.cell(h1, i + 1, t); c.font = BOLD; c.alignment = CT; c.fill = HDR
        if two_row:
            ws.merge_cells(start_row=h1, start_column=i + 1, end_row=h2, end_column=i + 1)
    col0 = 4
    colmap = []
    if two_row:
        i, cc = 0, col0
        while i < len(specs):
            g = specs[i][0]; j = i
            while j < len(specs) and specs[j][0] == g:
                j += 1
            ws.merge_cells(start_row=h1, start_column=cc, end_row=h1, end_column=cc + (j - i) - 1)
            hc = ws.cell(h1, cc, g); hc.font = BOLD; hc.alignment = CT; hc.fill = HDR
            for k in range(j - i):
                sc = ws.cell(h2, cc + k, specs[i + k][1]); sc.font = BOLD; sc.alignment = CT; sc.fill = HDR
                colmap.append((cc + k, specs[i + k]))
            cc += (j - i); i = j
    else:
        for k, spec in enumerate(specs):
            hc = ws.cell(h1, col0 + k, spec[0]); hc.font = BOLD; hc.alignment = CT; hc.fill = HDR
            colmap.append((col0 + k, spec))

    top = h2 + 1
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
    for row in ws.iter_rows(min_row=h1, max_row=last, min_col=1, max_col=col0 + len(specs) - 1):
        for cell in row:
            cell.border = THIN
    # "ours" band on the identity columns (best/2nd fills own the value columns)
    if "FEATHer" in models:
        r = row_of[("FEATHer", metric_labels[0])]
        for rr in range(r, r + len(metric_labels)):
            for cc in (1, 2, 3):
                ws.cell(rr, cc).fill = OURS
    ws.column_dimensions["A"].width = 13
    ws.column_dimensions["B"].width = 9
    ws.column_dimensions["C"].width = 10
    ws.freeze_panes = ws.cell(top, col0)


# ---------- accuracy ----------
def acc_matrix(d, datasets, metric, per_h, common=None):
    """common = 데이터셋 목록을 주면 OVERALL을 그 위에서만 계산.

    모델마다 커버리지가 다르면 OVERALL이 서로 다른 데이터셋 집합의 평균이 되어
    비교가 성립하지 않는다 (덜 돈 모델이 유리/불리해짐). OVERALL은 반드시
    '모든 모델이 가진' 공통 집합에서만 낸다.
    """
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
    avg_cols = [c for c in df.columns if c[1] == "Avg"]
    if common is not None:
        avg_cols = [c for c in avg_cols if c[0] in common]
    lab = "OVERALL" if common is None or len(common) == len(datasets) else \
          f"OVERALL ({len(common)}ds)"
    df[(lab, "Avg")] = df[avg_cols].mean(axis=1)
    specs.append((lab, "Avg"))
    return specs, df


def common_datasets(d, datasets, models):
    """표시할 모든 모델이 4호라이즌 전부 가진 데이터셋만."""
    ok = []
    for ds in datasets:
        sub = d[d.data == ds]
        if all(sub[sub.model == m].pred_len.nunique() == len(HS) for m in models):
            ok.append(ds)
    return ok


def load_acc():
    """병합된 벤치마크별 raw 파일이 단일 소스 — `tools/paper/merge_raw.py`가 만든다.

    mfg  = 베이스라인 13 + FEATHer-N (mix + 잔차MLP, ds16_k7_B4)
    ltsf = 베이스라인 13 + FEATHer-CIN (ci_full + 잔차MLP, 데이터셋별 arch;
           레벨 앵커는 train-only 규칙 → Exchange만 last_std)
    """
    parts = []
    for name in ("mfg", "ltsf"):
        p = f"{R}/{name}_raw.csv"
        if not os.path.exists(p):
            raise SystemExit(f"[merge 필요] {p} 없음 — python tools/paper/merge_raw.py")
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True)


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
    d = pd.concat([pd.read_csv(f"{R}/_archive/robustness/robust_results_baselines.csv"),
                   pd.read_csv(f"{R}/_archive/robustness/robust_feather_v2_mlp.csv")])
    d = d[d.fault_type != "clean"].copy()
    for c in ("Dw", "MSE"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Dw"])


CAP = {  # sheet -> caption suffix (CF-JEPA convention: merged caption on row 1)
    "1_avg": "all models, horizon-averaged",
    "2_per_horizon": "all models, per horizon",
    "3_edge_avg": "MCU-deployable models, horizon-averaged",
    "4_edge_per_horizon": "MCU-deployable models, per horizon",
    "2_per_fault": "MCU-deployable models, per fault type",
}
PROTO = ("seq_len=96, 3 seeds (2025-2027), best-val-epoch selection, "
         "MSE training. Lower is better; best shaded green, 2nd yellow.")


def save(wb, path, label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb.save(path)
    print(f"saved {path}  [{label}]")


def accuracy_book(acc, datasets, table_no, what, out):
    all_m = order_models([m for m in acc.model.unique() if m in REF_PARAMS])
    edge = order_models([m for m in EDGE7 if m in acc.model.unique()])
    wb = Workbook(); wb.remove(wb.active)
    for name, models, per_h in [("1_avg", all_m, False), ("2_per_horizon", all_m, True),
                                ("3_edge_avg", edge, False), ("4_edge_per_horizon", edge, True)]:
        common = common_datasets(acc, datasets, models)
        sp, mse = acc_matrix(acc, datasets, "MSE", per_h, common)
        _, mae = acc_matrix(acc, datasets, "MAE", per_h, common)
        title = f"Table {table_no}. {what} — {CAP[name]}. {PROTO}"
        if len(common) < len(datasets):
            gap = [d for d in datasets if d not in common]
            title += (f" OVERALL is averaged over the {len(common)} datasets every model "
                      f"covers; {', '.join(gap)} not yet run for all models and are "
                      f"EXCLUDED from OVERALL (per-dataset columns still shown).")
        write_sheet(wb, name, models, sp, {"MSE": mse, "MAE": mae}, ["MSE", "MAE"],
                    per_h, title=title)
    save(wb, out, f"{len(all_m)} models x {len(datasets)} datasets")


if __name__ == "__main__":
    acc = load_acc()

    accuracy_book(acc, MFG, 2, "Forecasting accuracy on the six manufacturing datasets",
                  "analysis/mfg/main_table.xlsx")

    # FEATHer's LTSF cell is CIN (channel-INDEPENDENT), so its count differs from
    # the mfg N cell: quote it under the same D=7/H=96 convention (ETTh1 = 420).
    REF_PARAMS["FEATHer"] = 420
    accuracy_book(acc, LTSF, 3, "Generalization to the eight standard LTSF benchmarks",
                  "analysis/ltsf/main_table.xlsx")
    REF_PARAMS["FEATHer"] = 822

    rob = load_rob()
    redge = order_models([m for m in EDGE7 if m in rob.model.unique()])
    wb = Workbook(); wb.remove(wb.active)
    for name, per_f in [("1_avg", False), ("2_per_fault", True)]:
        sp, dw = rob_matrix(rob, MFG, "Dw", per_f)
        _, cm = rob_matrix(rob, MFG, "MSE", per_f)
        cap = CAP["2_per_fault"] if per_f else CAP["3_edge_avg"]
        write_sheet(wb, name, redge, sp, {"Dw": dw, "corruptMSE": cm}, ["Dw", "corruptMSE"],
                    per_f, title=(f"Table R. Sensor-fault robustness on the manufacturing "
                                  f"datasets — {cap}. Dw = corrupt/clean MSE ratio. "
                                  f"NOT REPORTED IN THE PAPER (FEATHer ranks last)."))
    save(wb, "analysis/diagnostics/robustness.xlsx", f"edge: {redge}")
