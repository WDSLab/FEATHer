"""Build analysis/04_complexity.xlsx from results/analysis/complexity.csv (paper Table-5 style).

Layout: for each representative dataset, a block of 4 horizon columns; each model
has a Parameters row and a MACs row. FEATHer = FINAL frozen cells
(mfg = N / LTSF = CIN, measured 2026-08-17).
"""
import csv
import os
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

CSV = "results/analysis/complexity.csv"
OUT = "analysis/deployment/complexity.xlsx"

# representative datasets: small / mid / high-channel
DATASETS = [("Steel", "Steel (D=6, mfg)"),
            ("ETTh1", "ETTh1 (D=7, LTSF)"),
            ("Weather", "Weather (D=21, LTSF)"),
            ("BlastFurnace", "BlastFurnace (D=28, mfg)")]
HORIZONS = [96, 192, 336, 720]
MODELS = ["SparseTSF", "MixLinear", "DiPE_Linear", "FITS", "CycleNet", "DLinear"]
MODEL_LABEL = {"DiPE_Linear": "DiPE-Linear"}

rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
def get(model, data, h, col):
    for r in rows:
        if r["model"] == model and r["data"] == data and int(r["pred_len"]) == h:
            return int(r[col])
    return None

def fmt(n):
    if n is None: return "-"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)

wb = Workbook()
ws = wb.active
ws.title = "Complexity"

# styles
bold = Font(bold=True)
center = Alignment(horizontal="center", vertical="center")
hdr_fill = PatternFill("solid", fgColor="D9E1F2")
mfill = PatternFill("solid", fgColor="F2F2F2")
tbd_fill = PatternFill("solid", fgColor="FFF2CC")
thin = Side(style="thin", color="BBBBBB")
border = Border(left=thin, right=thin, top=thin, bottom=thin)

# Title
ws.cell(1, 1, "Table 5. Model Complexity — Parameters and MACs across datasets and horizons").font = Font(bold=True, size=12)
ws.cell(2, 1, "MACs = matmul/conv (FlopCounterMode) + FFT (N·log2N per transform). Element-wise freq ops excluded. FEATHer = FINAL cells (mfg: N / LTSF: CIN).").font = Font(italic=True, size=9)

# Header rows (row 4: dataset groups, row 5: horizons)
r_grp, r_hz, r_start = 4, 5, 6
ws.cell(r_grp, 1, "Model").font = bold
ws.cell(r_grp, 2, "Metric").font = bold
col = 3
for _, dlabel in DATASETS:
    ws.merge_cells(start_row=r_grp, start_column=col, end_row=r_grp, end_column=col + 3)
    c = ws.cell(r_grp, col, dlabel); c.font = bold; c.alignment = center; c.fill = hdr_fill
    for j, h in enumerate(HORIZONS):
        hc = ws.cell(r_hz, col + j, f"H{h}"); hc.font = bold; hc.alignment = center; hc.fill = hdr_fill
    col += 4
ws.cell(r_hz, 1, "").fill = hdr_fill
ws.cell(r_hz, 2, "").fill = hdr_fill

# Pre-compute per-column minimums (best = lowest params / lowest MACs),
# excluding the TBD FEATHer row.
best = Font(bold=True, color="006100")
best_fill = PatternFill("solid", fgColor="C6EFCE")
min_params, min_macs = {}, {}
for dkey, _ in DATASETS:
    for h in HORIZONS:
        ps = [get(m, dkey, h, "params") for m in MODELS]
        ms = [get(m, dkey, h, "macs_total") for m in MODELS]
        ps = [x for x in ps if x is not None]
        ms = [x for x in ms if x is not None]
        min_params[(dkey, h)] = min(ps) if ps else None
        min_macs[(dkey, h)] = min(ms) if ms else None

# Body: 2 rows per model (Params, MACs)
row = r_start
def write_model(name, label, is_tbd=False):
    global row
    ws.merge_cells(start_row=row, start_column=1, end_row=row + 1, end_column=1)
    mc = ws.cell(row, 1, label); mc.font = bold; mc.alignment = center
    ws.cell(row, 2, "Params").alignment = center
    ws.cell(row + 1, 2, "MACs").alignment = center
    col = 3
    for dkey, _ in DATASETS:
        for j, h in enumerate(HORIZONS):
            if is_tbd:
                p, m = "TBD", "TBD"
            else:
                pv = get(name, dkey, h, "params")
                mv = get(name, dkey, h, "macs_total")
                p, m = fmt(pv), fmt(mv)
            pc = ws.cell(row, col + j, p); pc.alignment = center
            mc2 = ws.cell(row + 1, col + j, m); mc2.alignment = center
            if is_tbd:
                pc.fill = tbd_fill; mc2.fill = tbd_fill
            else:
                # highlight best (lowest) params / MACs in this column
                if pv is not None and pv == min_params.get((dkey, h)):
                    pc.font = best; pc.fill = best_fill
                if mv is not None and mv == min_macs.get((dkey, h)):
                    mc2.font = best; mc2.fill = best_fill
        col += 4
    # shading for readability
    for rr in (row, row + 1):
        for cc in range(1, col):
            ws.cell(rr, cc).border = border
    row += 2

for m in MODELS:
    write_model(m, MODEL_LABEL.get(m, m))
write_model("FEATHer", "FEATHer (ours)")

# widths
ws.column_dimensions["A"].width = 15
ws.column_dimensions["B"].width = 9
for i in range(3, 3 + len(DATASETS) * 4):
    ws.column_dimensions[chr(64 + i) if i <= 26 else "A" + chr(64 + i - 26)].width = 8

# second sheet: raw params @ each dataset (all 12) for reference
ws2 = wb.create_sheet("Params_all")
ws2.cell(1, 1, "Params @ H=96 (all datasets)").font = bold
ALL_DS = ["Steel","WindSCADA","BlastFurnace","N2O","MetroPT3","EWELD",
          "ETTh1","ETTh2","ETTm1","ETTm2","Weather","Exchange"]
ws2.cell(2, 1, "Model").font = bold
for j, d in enumerate(ALL_DS):
    ws2.cell(2, 2 + j, d).font = bold
for i, m in enumerate(MODELS):
    ws2.cell(3 + i, 1, MODEL_LABEL.get(m, m))
    for j, d in enumerate(ALL_DS):
        v = get(m, d, 96, "params")
        ws2.cell(3 + i, 2 + j, v if v is not None else "-")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
wb.save(OUT)
print(f"wrote -> {OUT}")
print(f"  Complexity sheet: {len(MODELS)+1} models x {len(DATASETS)} datasets x {len(HORIZONS)} horizons")
