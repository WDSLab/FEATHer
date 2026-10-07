# -*- coding: utf-8 -*-
"""제조 벤치마크 정확도 표 (논문 Table 1).

    python tools/paper/table_mfg.py               # FEATHer 포함 (기본)
    python tools/paper/table_mfg.py --no_feather  # 베이스라인만

입력은 2026-08-31 에 분리한 두 파일:
    results/_archive/pre_raw_20261007/mfg_baselines.csv   확정 — 재실험 없음
    results/_archive/pre_raw_20261007/feather_mfg.csv     모델이 확정되면 갱신됨

2026-09-04 에 모델이 확정(452p)되어 기본이 FEATHer 포함으로 바뀌었다.  그 전에는
옛 모델이 표에 박히는 것을 막으려고 `(pending)` 행으로 비워 두는 게 기본이었다.
feather_mfg.csv 의 provenance 는 merge_raw.py 가 num_params 로 검증한다.

출력:
    analysis/mfg/T1_accuracy.xlsx
      1_avg           모델 x 데이터셋 (4호라이즌 평균), MSE 행 / MAE 행
      2_per_horizon   모델 x (데이터셋 x 호라이즌), MSE
    + 콘솔에 markdown 요약
"""
import argparse
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

R = "results"
OUT_DIR = "analysis/mfg"
MFG = ["Steel", "WindSCADA", "BlastFurnace", "EWELD", "MetroPT3", "N2O"]
HS = [96, 192, 336, 720]

# 티어 분리 (2026-07-31 결정, 08-04 재확인):
#   메인 표 = 엣지 tier 만. 천장 모델은 표에서 빼고 부록/Pareto 로만 보여준다
#   ("왜 SOTA 가 아니냐" 방어 + accuracy-per-resource 서사).
#   ⚠ DLinear(18,624p)는 엣지 tier 최상단이자 H=96 에서 우리를 이기는 모델이라
#   반드시 남긴다 — 빼면 "숨겼다"가 되어 즉살.
#   ⚠ 천장 모델을 완전히 버리지도 않는다 — Related Work 에 "크기는 제외 기준이
#   아니다(41~1.6M 스팬이 요점)"라고 써두었으므로 부록 시트에 남겨야 자기모순이 없다.
EDGE = ["SparseTSF", "MixLinear", "TimeBase", "DiPE_Linear", "CMoS", "SparseTSF_MLP", "FITS",
        "PhaseFormer", "CycleNet", "DLinear", "SAMformer", "FEATHer_mix", "FEATHer_CI"]   # 10-06 +4
# 우리 모델은 두 변종을 모두 싣는다 (merge_raw.py 참조).  도메인별로 나은 쪽만
# 싣는 것은 선택 편향이다 - CI 가 mfg 에서 지는 것도 표에 그대로 보인다.
OURS_PREFIX = "FEATHer"

BEST = PatternFill("solid", fgColor="C6EFCE")
SECOND = PatternFill("solid", fgColor="FFEB9C")
HDR = PatternFill("solid", fgColor="DDDDDD")
OURS = PatternFill("solid", fgColor="FFF2CC")
PEND = PatternFill("solid", fgColor="F2F2F2")
BOLD = Font(bold=True)
CT = Alignment(horizontal="center", vertical="center")
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)

PENDING = "FEATHer (pending)"


def load(with_feather):
    b = pd.read_csv(f"{R}/_archive/pre_raw_20261007/mfg_baselines.csv")
    if with_feather:
        f = pd.read_csv(f"{R}/_archive/pre_raw_20261007/feather_mfg.csv")
        b = pd.concat([b, f], ignore_index=True)
    return b[b.data.isin(MFG)]


def split_tier(cells):
    edge = cells[cells.model.isin(EDGE)]
    ceil = cells[~cells.model.isin(EDGE)]
    return edge, ceil


def cell_stats(df):
    """(model, data, pred_len) -> 시드 평균/표준편차."""
    return df.groupby(["model", "data", "pred_len"]).agg(
        MSE=("MSE", "mean"), MSE_sd=("MSE", "std"),
        MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        params=("num_params", "max"), n=("seed", "size")).reset_index()


def order_models(cells):
    """H=96 파라미터 오름차순. FEATHer 는 맨 아래(우리 모델 관례)."""
    p = (cells[cells.pred_len == 96].groupby("model").params.max()
         .sort_values())
    ms = [m for m in p.index if not m.startswith(OURS_PREFIX)]
    return ms + [m for m in p.index if m.startswith(OURS_PREFIX)]


def _hi(ws, rows, col):
    """한 열에서 1위 초록 / 2위 노랑 (숫자 셀만)."""
    vals = [(r, ws.cell(r, col).value) for r in rows
            if isinstance(ws.cell(r, col).value, (int, float))]
    if len(vals) < 2:
        return
    vals.sort(key=lambda t: t[1])
    ws.cell(vals[0][0], col).fill = BEST
    ws.cell(vals[0][0], col).font = BOLD
    ws.cell(vals[1][0], col).fill = SECOND


def _head(ws, row, col, text, width=None):
    c = ws.cell(row, col, text)
    c.font = BOLD
    c.fill = HDR
    c.alignment = CT
    c.border = THIN
    if width:
        ws.column_dimensions[get_column_letter(col)].width = width
    return c


def sheet_avg(wb, cells, models, title):
    ws = wb.create_sheet("1_avg")
    ws.cell(1, 1, title).font = Font(size=10)

    avg = (cells.groupby(["model", "data"])[["MSE", "MAE"]].mean().reset_index())
    ov = cells.groupby("model")[["MSE", "MAE"]].mean()
    par = cells[cells.pred_len == 96].groupby("model").params.max()
    par720 = cells[cells.pred_len == 720].groupby("model").params.max()

    r0 = 3
    _head(ws, r0, 1, "Model", 18)
    _head(ws, r0, 2, "Metric", 8)
    _head(ws, r0, 3, "params\nH96", 10)
    _head(ws, r0, 4, "params\nH720", 10)
    for j, d in enumerate(MFG):
        _head(ws, r0, 5 + j, d, 14)
    _head(ws, r0, 5 + len(MFG), "OVERALL", 12)

    r = r0 + 1
    mse_rows, mae_rows = [], []
    for m in models:
        for k, metric in enumerate(("MSE", "MAE")):
            c = ws.cell(r, 1, m if k == 0 else "")
            if m == PENDING:
                c.fill = PEND
            elif m.startswith(OURS_PREFIX):
                c.fill = OURS
                c.font = BOLD
            ws.cell(r, 2, metric).alignment = CT
            if k == 0:
                ws.cell(r, 3, int(par[m]) if m in par else None).alignment = CT
                ws.cell(r, 4, int(par720[m]) if m in par720 else None).alignment = CT
            for j, d in enumerate(MFG):
                s = avg[(avg.model == m) & (avg.data == d)]
                v = round(float(s[metric].iloc[0]), 4) if len(s) else None
                ws.cell(r, 5 + j, v).alignment = CT
            v = round(float(ov.loc[m, metric]), 4) if m in ov.index else None
            cc = ws.cell(r, 5 + len(MFG), v)
            cc.alignment = CT
            cc.font = BOLD
            (mse_rows if k == 0 else mae_rows).append(r)
            r += 1

    for col in range(5, 6 + len(MFG)):
        _hi(ws, mse_rows, col)
        _hi(ws, mae_rows, col)
    ws.freeze_panes = "C4"
    return avg, ov, par


def sheet_horizon(wb, cells, models, title):
    ws = wb.create_sheet("2_per_horizon")
    ws.cell(1, 1, title).font = Font(size=10)

    _head(ws, 3, 1, "Model", 18)
    col = 2
    spans = []
    for d in MFG:
        for h in HS:
            _head(ws, 4, col, str(h), 9)
            spans.append((d, h, col))
            col += 1
        c = ws.cell(3, col - len(HS), d)
        c.font, c.fill, c.alignment, c.border = BOLD, HDR, CT, THIN
        ws.merge_cells(start_row=3, start_column=col - len(HS),
                       end_row=3, end_column=col - 1)
    _head(ws, 4, col, "OVERALL", 11)
    ws.cell(3, col, "").fill = HDR

    rows = []
    r = 5
    for m in models:
        c = ws.cell(r, 1, m)
        if m == PENDING:
            c.fill = PEND
        elif m.startswith(OURS_PREFIX):
            c.fill = OURS
            c.font = BOLD
        for d, h, cc in spans:
            s = cells[(cells.model == m) & (cells.data == d) & (cells.pred_len == h)]
            ws.cell(r, cc, round(float(s.MSE.iloc[0]), 4) if len(s) else None
                    ).alignment = CT
        s = cells[cells.model == m]
        ws.cell(r, col, round(float(s.MSE.mean()), 4) if len(s) else None
                ).font = BOLD
        rows.append(r)
        r += 1
    for _, _, cc in spans:
        _hi(ws, rows, cc)
    _hi(ws, rows, col)
    ws.freeze_panes = "B5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_feather", action="store_true",
                    help="베이스라인만 (FEATHer 행을 pending 으로 비운다)")
    a = ap.parse_args()
    a.with_feather = not a.no_feather

    df = load(a.with_feather)
    cells = cell_stats(df)
    seeds = sorted(df.seed.unique())

    note = ("FEATHer included." if a.with_feather
            else "FEATHer pending band-axis model freeze.")
    base = (f"Manufacturing benchmark — test MSE/MAE, seq_len=96, "
            f"{len(seeds)} seeds {seeds}, best-val epoch selection. ")
    t_edge = ("Table 1. " + base
              + "EDGE TIER (MCU-relevant parameter budgets). " + note)
    t_full = ("Appendix. " + base
              + "FULL LINEUP incl. ceiling models — shown so that model size is "
                "not the exclusion criterion; the main comparison is Table 1. " + note)

    edge_cells, _ = split_tier(cells)
    edge_models = order_models(edge_cells)
    all_models = order_models(cells)
    if not a.with_feather:
        edge_models = edge_models + [PENDING]
        all_models = all_models + [PENDING]

    os.makedirs(OUT_DIR, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    avg, ov, par = sheet_avg(wb, edge_cells, edge_models, t_edge)
    wb["1_avg"].title = "1_edge_avg"
    sheet_horizon(wb, edge_cells, edge_models, t_edge)
    wb["2_per_horizon"].title = "2_edge_per_horizon"
    sheet_avg(wb, cells, all_models, t_full)
    wb["1_avg"].title = "A1_full_lineup"
    out = f"{OUT_DIR}/T1_accuracy.xlsx"
    wb.save(out)

    # ---- 콘솔 요약 ----
    def show(label, models, sub):
        av = sub.groupby(["model", "data"])[["MSE", "MAE"]].mean().reset_index()
        o = sub.groupby("model")[["MSE", "MAE"]].mean()
        pr = sub[sub.pred_len == 96].groupby("model").params.max()
        print(label)
        print("| Model | params | " + " | ".join(MFG) + " | OVERALL |")
        print("|" + "---|" * (len(MFG) + 3))
        for m in models:
            if m == PENDING:
                print(f"| *{m}* | — | " + " | ".join(["—"] * len(MFG)) + " | — |")
                continue
            row = [f"{av[(av.model == m) & (av.data == d)].MSE.iloc[0]:.4f}"
                   if len(av[(av.model == m) & (av.data == d)]) else "—"
                   for d in MFG]
            print(f"| {m} | {int(pr[m]):,} | " + " | ".join(row)
                  + f" | **{o.loc[m, 'MSE']:.4f}** |")
        print()

    show("### Table 1 — EDGE TIER (main)", edge_models, edge_cells)
    show("### Appendix — full lineup", all_models, cells)
    print(f"→ {out}   시트: 1_edge_avg / 2_edge_per_horizon / A1_full_lineup")
    print(f"   {len(df)}행 / 엣지 {edge_cells.model.nunique()}모델 / "
          f"전체 {cells.model.nunique()}모델")
    bad = cells[cells.n != len(seeds)]
    if len(bad):
        print(f"   ⚠ 시드 미달 셀 {len(bad)}개")


if __name__ == "__main__":
    main()
