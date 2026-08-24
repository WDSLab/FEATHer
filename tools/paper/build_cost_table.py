# -*- coding: utf-8 -*-
"""Complexity / deployment-cost table (paper Table 5 layout, MCU-adapted).

Model x {small, middle, large channel-count dataset} x horizon, with
Parameters / MACs / flash stacked per model — the layout LTSF complexity
tables use, so the numbers stay comparable to what reviewers expect.

Three adaptations, all forced by this paper being about deployment:

  * Latency is measured on the TARGET BOARD (NUCLEO-F103RB, Cortex-M3 72 MHz,
    no FPU), not on a GPU. GPU inference separates nothing here — every model
    in the lineup runs in under a few ms — whereas the board spreads them over
    47-1312 ms and reverses the parameter ordering (DiPE has 339 params and is
    the slowest of the five).
  * A flash column and a fits/does-not-fit verdict, since that is what splits
    the lineup into the deployable set.
  * Dataset tiers are channel counts (Steel 6 / WindSCADA 15 / BlastFurnace
    28), mirroring the small/middle/large tiering of the original table.

Board latency exists for five models at the pre-08 453-parameter FEATHer, so
those cells are labelled and the rest left blank rather than guessed.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

TIERS = [("Steel", "small (D=6)"), ("WindSCADA", "middle (D=15)"),
         ("BlastFurnace", "large (D=28)")]
HS = [96, 192, 336, 720]

FLASH_KB, FIRMWARE_KB, TABLES_KB = 128, 15, 1.5
MAX_PARAMS = int((FLASH_KB - FIRMWARE_KB - TABLES_KB) * 1024 / 4)

# deployment/EDGE_EXPERIMENT.md, 2026-07-27. FEATHer measured at the pre-08
# 453-param configuration — flagged in the sheet, pending re-measurement.
BOARD_MS = {"SparseTSF": 47, "FEATHer": 349, "FITS": 796, "DiPE_Linear": 1312}
BOARD_MJ = {"SparseTSF": 4.8, "FEATHer": 35, "FITS": 80, "DiPE_Linear": 132}
BOARD_NOTE = {"FEATHer": "453p"}

HDR = PatternFill("solid", fgColor="DDDDDD")
OURS = PatternFill("solid", fgColor="FFF2CC")
NOFIT = PatternFill("solid", fgColor="F2F2F2")
BOLD = Font(bold=True)
CT = Alignment(horizontal="center", vertical="center")
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)


def _fmt_params(n):
    n = int(n)
    return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}K" if n >= 1e4 else f"{n:,}")


def _fmt_macs(n):
    if n != n:
        return "—"
    n = float(n)
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n/div:.2f}{suf}"
    return f"{n:.0f}"


def _autosize(ws):
    for col in ws.columns:
        w = max((len(str(c.value).split("\n")[0]) for c in col if c.value is not None),
                default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(w + 2, 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/mfg_raw.csv")
    ap.add_argument("--complexity", default="results/complexity.csv")
    ap.add_argument("--out", default="analysis/mfg/T2_cost.xlsx")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cx = pd.read_csv(args.complexity).set_index(["model", "data", "pred_len"])

    # params actually constructed, per (model, data, horizon)
    P = df.groupby(["model", "data", "pred_len"]).num_params.max()
    worst = df.groupby("model").num_params.max()
    deployable = {m: bool(v <= MAX_PARAMS) for m, v in worst.items()}
    order = worst.sort_values().index.tolist()

    wb = Workbook()
    ws = wb.active
    ws.title = "cost"

    # ---- header: two rows (dataset tier, horizon) ----
    ws.cell(row=1, column=1, value="Model")
    ws.cell(row=1, column=2, value="Metric")
    col = 3
    for ds, label in TIERS:
        ws.cell(row=1, column=col, value=f"{ds}\n{label}")
        ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + 3)
        for h in HS:
            ws.cell(row=2, column=col, value=h)
            col += 1
    ws.cell(row=1, column=col, value="worst-case\nflash (KB)")
    ws.merge_cells(start_row=1, start_column=col, end_row=2, end_column=col)
    ws.cell(row=1, column=col + 1, value="fits 128KB\nMCU")
    ws.merge_cells(start_row=1, start_column=col + 1, end_row=2, end_column=col + 1)
    ws.cell(row=1, column=col + 2, value="board latency\n(Cortex-M3)")
    ws.merge_cells(start_row=1, start_column=col + 2, end_row=2, end_column=col + 2)
    ws.cell(row=1, column=col + 3, value="energy\n/inference")
    ws.merge_cells(start_row=1, start_column=col + 3, end_row=2, end_column=col + 3)
    ncols = col + 3
    for rr in (1, 2):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=rr, column=c)
            cell.fill, cell.font, cell.alignment, cell.border = HDR, BOLD, CT, THIN

    # ---- body: two metric rows per model ----
    r = 3
    for m in order:
        for metric in ("Parameters", "MACs"):
            ws.cell(row=r, column=2, value=metric).border = THIN
            c = 3
            for ds, _ in TIERS:
                for h in HS:
                    if metric == "Parameters":
                        v = _fmt_params(P.get((m, ds, h), float("nan"))) \
                            if (m, ds, h) in P.index else "—"
                    else:
                        v = _fmt_macs(cx.macs_total.get((m, ds, h), float("nan")))
                    cell = ws.cell(row=r, column=c, value=v)
                    cell.alignment, cell.border = CT, THIN
                    c += 1
            if metric == "Parameters":
                ws.cell(row=r, column=1, value=m).border = THIN
                w = int(worst[m])
                ws.cell(row=r, column=c, value=round(w * 4 / 1024, 1))
                ws.cell(row=r, column=c + 1, value="yes" if deployable[m] else "no")
                lat = BOARD_MS.get(m)
                note = f" ({BOARD_NOTE[m]})" if m in BOARD_NOTE else ""
                ws.cell(row=r, column=c + 2,
                        value=f"{lat} ms{note}" if lat else "not ported")
                mj = BOARD_MJ.get(m)
                ws.cell(row=r, column=c + 3, value=f"{mj} mJ" if mj else "—")
                for cc in range(c, c + 4):
                    ws.cell(row=r, column=cc).alignment = CT
                    ws.cell(row=r, column=cc).border = THIN
                    ws.merge_cells(start_row=r, start_column=cc, end_row=r + 1, end_column=cc)
                if m == "FEATHer":
                    for cc in range(1, ncols + 1):
                        ws.cell(row=r, column=cc).fill = OURS
                        ws.cell(row=r + 1, column=cc).fill = OURS
                elif not deployable[m]:
                    ws.cell(row=r, column=c + 1).fill = NOFIT
            r += 1
        ws.merge_cells(start_row=r - 2, start_column=1, end_row=r - 1, end_column=1)

    r += 1
    for line in [
        f"Deployable = worst-case parameters over all 6 datasets and 4 horizons fit "
        f"{FLASH_KB} KB flash minus ~{FIRMWARE_KB} KB firmware and ~{TABLES_KB} KB "
        f"tables at 4 B/param, i.e. <= {MAX_PARAMS:,} parameters.",
        "MACs are FFT-aware (matmul + N log2 N per transform); '—' = not measured "
        "(ceiling models are outside the deployable set).",
        "Board latency and energy: NUCLEO-F103RB, Cortex-M3 @72 MHz, no FPU, -O2 "
        "soft-float, measured 2026-07-27 (deployment/EDGE_EXPERIMENT.md). "
        "Parameter order does NOT predict latency: DiPE runs 339 parameters and is "
        "the slowest of the ported five.",
        "FEATHer's board figures are from the pre-08 453-parameter configuration and "
        "are pending re-measurement at the shipping 822-4,522 configuration.",
    ]:
        ws.cell(row=r, column=1, value=line)
        r += 1

    _autosize(ws)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wb.save(args.out)
    print(f"wrote {args.out}  ({len(order)} models)")
    print(f"deployable: {sorted(m for m in order if deployable[m])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
