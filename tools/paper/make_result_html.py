# -*- coding: utf-8 -*-
"""One local HTML page with the paper's result tables, rebuilt from the raw results (2026-10-13).

    python tools/paper/make_result_html.py            # -> results/result.html

Sources: results/raw_mfg.csv · raw_ltsf.csv (merge_incoming.py first), docs/chosen_mfg.json, protocol.ltsf_config_id,
results/board/chip_verdict_20261012.csv (+ _candidates_ for settings changed by the chip-best rule),
board_latency_20260929.csv · board_latency_candidates_20261007.csv · board_stream_20261012.csv.
Rules: mfg ranks are computed only among the models that fit all 24 industrial cells; other lightweight models are
shown with their off-chip cells marked; LTSF uses the authors' official settings and the same pool. Test errors of
the redesign arms are never shown (pre-registration docs/prereg_redesign_1012.md).
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "results"))
import protocol as P  # noqa: E402

RES = os.path.join(ROOT, "results")
BOARD = os.path.join(RES, "board")
OUT = os.path.join(RES, "result.html")

POOL = ["FEATHer_mix", "FEATHer_CI", "DiPE_Linear", "CMoS", "SparseTSF_MLP", "SparseTSF", "MixLinear", "TimeBase"]
PART = ["DLinear", "SAMformer", "PhaseFormer", "CycleNet", "FITS"]
OURS = {"FEATHer_mix", "FEATHer_CI"}
NAME = {"FEATHer_mix": "FEATHer-mix (ours)", "FEATHer_CI": "FEATHer-CI (ours)", "DiPE_Linear": "DiPE",
        "SparseTSF_MLP": "SparseTSF-MLP", "LMS_AutoTSF": "LMS-AutoTSF", "MDMLP_EIA": "MDMLP-EIA"}
VENUE = {"FEATHer_mix": "ours", "FEATHer_CI": "ours", "DiPE_Linear": "DASFAA'26", "CMoS": "ICML'25",
         "SparseTSF_MLP": "ICML'24", "SparseTSF": "ICML'24", "MixLinear": "ICLR'26", "TimeBase": "ICML'25",
         "DLinear": "AAAI'23", "SAMformer": "ICML'24", "PhaseFormer": "ICLR'26", "CycleNet": "NeurIPS'24",
         "FITS": "ICLR'24", "PatchTST": "ICLR'23", "TimeMixer": "ICLR'24", "TQNet": "ICML'25",
         "LMS_AutoTSF": "EAAI'26", "TimesNet": "ICLR'23", "iTransformer": "ICLR'24", "MDMLP_EIA": "AAAI'26"}
B2R = {"feather_mix": "FEATHer_mix", "feather_ci": "FEATHer_CI", "stsf": "SparseTSF", "mixlinear": "MixLinear",
       "timebase": "TimeBase", "dipe": "DiPE_Linear", "stsf_mlp": "SparseTSF_MLP", "cmos": "CMoS", "fits": "FITS",
       "cyclenet": "CycleNet", "dlinear": "DLinear", "samformer": "SAMformer", "phaseformer": "PhaseFormer"}
R2B = {v: k for k, v in B2R.items()}
HS = P.HS
MFG = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
MFG_D = {"Steel": 6, "MetroPT3": 7, "WindSCADA": 14, "N2O": 14, "EWELD": 14, "BlastFurnace": 28}
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange", "Electricity", "Traffic"]


def nm(m):
    return NAME.get(m, m)


def esc(x):
    return html.escape(str(x))


# ------------------------------------------------------------------ data
def load():
    ch = json.load(open(os.path.join(ROOT, "docs", "chosen_mfg.json"), encoding="utf-8"))
    r = pd.read_csv(os.path.join(RES, "raw_mfg.csv"))
    main = r[r.role.isin(["main", "reference"]) & (r.seq_len == 96)]
    cfg = P.mfg_main_config_id          # frozen / ref-default / chosen (single-candidate models included)
    main = main[main.apply(lambda x: x.config_id == cfg(x.model), axis=1)]
    lt = pd.read_csv(os.path.join(RES, "raw_ltsf.csv"))
    lt = lt[lt.role.isin(["main", "reference"]) & (lt.seq_len == 96)]
    lt = lt[lt.apply(lambda x: x.config_id == P.ltsf_config_id(x.model, x.data), axis=1)]
    # chip verdict at the chosen setting
    v = pd.read_csv(os.path.join(BOARD, "chip_verdict_20261012.csv"))
    vc = pd.read_csv(os.path.join(BOARD, "chip_verdict_candidates_20261012.csv"))
    rows = []
    for b, reg in B2R.items():
        setting = (P.mfg_main_config_id(reg) or "").replace("cand:", "")
        x = vc[(vc.model == b) & (vc.config == setting)]
        if not len(x):
            x = v[v.model == b]
        rows.append(x.assign(model=reg))
    chip = pd.concat(rows)[["model", "data", "H", "verdict", "flash_B", "ram_B", "ram_stream_B"]]
    chip = chip.rename(columns={"H": "pred_len"})
    return ch, r, main, lt, chip


def cell_means(df):
    return df.groupby(["model", "data", "pred_len"]).agg(MSE=("MSE", "mean"), MAE=("MAE", "mean"),
                                                         params=("num_params", "max"),
                                                         seeds=("seed", "nunique")).reset_index()


def ranks(c, pool):
    x = c[c.model.isin(pool)].copy()
    for m in ("MSE", "MAE"):
        x["r" + m] = x.groupby(["data", "pred_len"])[m].rank(method="min")
    return x


# ------------------------------------------------------------------ html helpers
def table(head, body, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def f3(x, d=3):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def mrow(m):
    return f'<span class="{"ours" if m in OURS else ""}">{esc(nm(m))}</span>'


def sec_wide(c, chip, datasets, pool, part):
    """Paper layout: rows = dataset x H (+ dataset mean), columns = method x (MSE, MAE).  Bold = best in the pool,
    underline = second; red X = that cell exceeds the chip (value kept for reference, not ranked)."""
    models = [m for m in pool + part if m in set(c.model)]
    cc = c.merge(chip, on=["model", "data", "pred_len"], how="left") if chip is not None else c.assign(verdict="FIT")
    cc["verdict"] = cc.verdict.fillna("FIT")
    val = cc.set_index(["data", "pred_len", "model"])
    h1 = '<th rowspan="2">데이터</th><th rowspan="2">H</th>' + "".join(
        f'<th colspan="2" class="{"g ours" if m in OURS else ("g" if m in pool else "g part")}">{esc(nm(m))}'
        f'{"" if m in pool else " *"}</th>' for m in models)
    h2 = "".join('<th class="g">MSE</th><th>MAE</th>' for _ in models)
    rows = []

    def fmt(d, h, m, k, order):
        if (d, h, m) not in val.index:
            return "<td>–</td>"
        v = val.loc[(d, h, m), k]
        off = val.loc[(d, h, m), "verdict"] != "FIT"
        s = f3(v)
        if off:
            s = f'<span class="off">{s}✗</span>'
        elif m in pool and order and abs(v - order[0]) < 1e-12:
            s = f"<b>{s}</b>"
        elif m in pool and len(order) > 1 and abs(v - order[1]) < 1e-12:
            s = f"<u>{s}</u>"
        cls = ' class="g"' if k == "MSE" else ""
        return f"<td{cls}>{s}</td>"

    for d in datasets:
        title = f"{d} ({MFG_D[d]})" if d in MFG_D else d
        for i, h in enumerate(HS + ["평균"]):
            if h == "평균":
                sub = cc[cc.data == d].groupby("model")[["MSE", "MAE"]].mean()
                tds = ""
                for k_m in models:
                    for k in ("MSE", "MAE"):
                        okv = sorted(sub.loc[[m for m in pool if m in sub.index], k])
                        v = sub.loc[k_m, k] if k_m in sub.index else float("nan")
                        s = f3(v)
                        if k_m in pool and abs(v - okv[0]) < 1e-12:
                            s = f"<b>{s}</b>"
                        elif k_m in pool and abs(v - okv[1]) < 1e-12:
                            s = f"<u>{s}</u>"
                        tds += f'<td class="{"g " if k == "MSE" else ""}avg">{s}</td>'
                rows.append(f'<tr class="avgrow"><td class="avg">평균</td>{tds}</tr>')
                continue
            ok = cc[(cc.data == d) & (cc.pred_len == h) & cc.model.isin(pool) & (cc.verdict == "FIT")]
            order = {k: sorted(ok[k]) for k in ("MSE", "MAE")}
            tds = "".join(fmt(d, h, m, k, order[k]) for m in models for k in ("MSE", "MAE"))
            lead = f'<td rowspan="5" class="ds">{esc(title)}</td>' if i == 0 else ""
            rows.append(f"<tr>{lead}<td>{h}</td>{tds}</tr>")
    # overall mean + 1st-place counts (pool only)
    tot = cc.groupby("model")[["MSE", "MAE"]].mean()
    x = ranks(cc[cc.verdict == "FIT"], pool)
    tds = cnt = ""
    for m in models:
        for k in ("MSE", "MAE"):
            okv = sorted(tot.loc[[p for p in pool if p in tot.index], k])
            v = tot.loc[m, k]
            s = f3(v)
            if m in pool and abs(v - okv[0]) < 1e-12:
                s = f"<b>{s}</b>"
            elif m in pool and abs(v - okv[1]) < 1e-12:
                s = f"<u>{s}</u>"
            tds += f'<td class="{"g " if k == "MSE" else ""}avg">{s}</td>'
            n = int(((x.model == m) & (x["r" + k] == 1)).sum()) if m in pool else None
            cnt += f'<td class="{"g " if k == "MSE" else ""}avg">{"" if n is None else n}</td>'
    rows.append(f'<tr class="avgrow"><td colspan="2"><b>전체 평균</b></td>{tds}</tr>')
    rows.append(f'<tr class="avgrow"><td colspan="2"><b>1등 칸 수</b> (풀 안)</td>{cnt}</tr>')
    return (f'<div class="tw"><table class="wide"><thead><tr>{h1}</tr><tr>{h2}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def sec_tall(c, chip, datasets, pool, part):
    """One table per dataset: rows = method x (MSE, MAE), columns = H96..H720 + mean.  Bold = best in the pool,
    underline = second; red X = that cell exceeds the chip (kept for reference, not ranked)."""
    models = [m for m in pool + part if m in set(c.model)]
    cc = c.merge(chip, on=["model", "data", "pred_len"], how="left") if chip is not None else c.assign(verdict="FIT")
    cc["verdict"] = cc.verdict.fillna("FIT")
    out = []
    for d in datasets:
        sub = cc[cc.data == d]
        val = sub.set_index(["model", "pred_len"])
        mean = sub.groupby("model")[["MSE", "MAE"]].mean()
        cols = HS + ["avg"]

        def order(k, h):
            if h == "avg":
                return sorted(mean.loc[[m for m in pool if m in mean.index], k])
            ok = sub[(sub.pred_len == h) & sub.model.isin(pool) & (sub.verdict == "FIT")]
            return sorted(ok[k])

        ords = {(k, h): order(k, h) for k in ("MSE", "MAE") for h in cols}
        rows = []
        for i, m in enumerate(models):
            if m == part[0] if part else False:
                rows.append(f'<tr class="sep"><td colspan="{2 + len(cols)}"><i>칩에 일부만 들어가는 모델 (순위 제외)</i></td></tr>')
            for j, k in enumerate(("MSE", "MAE")):
                tds = ""
                for h in cols:
                    if h == "avg":
                        v, off = mean.loc[m, k], False
                    elif (m, h) in val.index:
                        v, off = val.loc[(m, h), k], val.loc[(m, h), "verdict"] != "FIT"
                    else:
                        tds += "<td>–</td>"; continue
                    s, o = f3(v), ords[(k, h)]
                    if off:
                        s = f'<span class="off">{s}✗</span>'
                    elif m in pool and o and abs(v - o[0]) < 1e-12:
                        s = f"<b>{s}</b>"
                    elif m in pool and len(o) > 1 and abs(v - o[1]) < 1e-12:
                        s = f"<u>{s}</u>"
                    tds += f'<td class="{"avg" if h == "avg" else ""}">{s}</td>'
                lead = f'<td rowspan="2" class="ds">{mrow(m)}</td>' if j == 0 else ""
                cls = ("m2" if j else "m1") + ("" if m in pool else " part")
                rows.append(f'<tr class="{cls}">{lead}<td class="met">{k}</td>{tds}</tr>')
        head = "<tr><th>모델</th><th>지표</th>" + "".join(f"<th>H{h}</th>" for h in HS) + '<th class="avg">평균</th></tr>'
        colg = '<colgroup><col class="c-model"><col class="c-met">' + "<col>" * (len(HS) + 1) + "</colgroup>"
        title = f"{d} (D = {MFG_D[d]})" if d in MFG_D else d
        out.append(f'<h4>{esc(title)}</h4><div class="tw tallw"><table class="tall">{colg}<thead>{head}</thead>'
                   f'<tbody>{"".join(rows)}</tbody></table></div>')
    return "".join(out)


def sec_grid(c, chip, datasets, pool, part):
    """One table: columns = dataset x (MSE, MAE); rows = H block (H96..H720, then the 4-H mean) x method.
    Bold = best in the pool for that dataset/H/metric, underline = second; red X = exceeds the chip there."""
    models = [m for m in pool + part if m in set(c.model)]
    cc = c.merge(chip, on=["model", "data", "pred_len"], how="left") if chip is not None else c.assign(verdict="FIT")
    cc["verdict"] = cc.verdict.fillna("FIT")
    val = cc.set_index(["model", "data", "pred_len"])
    mean = cc.groupby(["model", "data"])[["MSE", "MAE"]].mean()
    blocks = HS + ["avg"]
    ords = {}
    for d in datasets:
        for k in ("MSE", "MAE"):
            for h in HS:
                ok = cc[(cc.data == d) & (cc.pred_len == h) & cc.model.isin(pool) & (cc.verdict == "FIT")]
                ords[(d, h, k)] = sorted(ok[k])
            ords[(d, "avg", k)] = sorted(mean.loc[[(m, d) for m in pool if (m, d) in mean.index], k])
    lab = lambda d: f"{d}<br><span class='mut'>D = {MFG_D[d]}</span>" if d in MFG_D else d
    h1 = '<th rowspan="2" class="hcol">H</th><th rowspan="2">모델</th>' + "".join(
        f'<th colspan="2" class="dgrp">{lab(d)}</th>' for d in datasets)
    h2 = "".join('<th class="dgrp">MSE</th><th>MAE</th>' for _ in datasets)
    rows = []
    n_rows = len(models) + (1 if part else 0)
    for h in blocks:
        for i, m in enumerate(models):
            lead = (f'<td rowspan="{n_rows}" class="hcol">{"평균" if h == "avg" else h}</td>' if i == 0 else "")
            if part and m == part[0]:
                rows.append(f'<tr class="sep">{"" if i else lead}<td colspan="{1 + 2 * len(datasets)}">'
                            f'<i>칩에 일부만 들어가는 모델 (순위 제외)</i></td></tr>')
            tds = ""
            for d in datasets:
                for k in ("MSE", "MAE"):
                    if h == "avg":
                        if (m, d) not in mean.index:
                            tds += "<td>–</td>"; continue
                        v, off = mean.loc[(m, d), k], False
                    else:
                        if (m, d, h) not in val.index:
                            tds += "<td>–</td>"; continue
                        v, off = val.loc[(m, d, h), k], val.loc[(m, d, h), "verdict"] != "FIT"
                    s, o = f3(v), ords[(d, h, k)]
                    if off:
                        s = f'<span class="off">{s}✗</span>'
                    elif m in pool and o and abs(v - o[0]) < 1e-12:
                        s = f"<b>{s}</b>"
                    elif m in pool and len(o) > 1 and abs(v - o[1]) < 1e-12:
                        s = f"<u>{s}</u>"
                    tds += f'<td class="{"dgrp" if k == "MSE" else ""}">{s}</td>'
            cls = ("part " if m not in pool else "") + ("last" if i == len(models) - 1 else "")
            rows.append(f'<tr class="{cls}">{lead}<td class="mname">{mrow(m)}</td>{tds}</tr>')
    return (f'<div class="tw"><table class="grid"><thead><tr>{h1}</tr><tr>{h2}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def sec_pairs(c, chip, datasets, pool, part, per=2):
    """Tables of `per` datasets each: columns = dataset (group) x (H96..H720, mean); rows = method x (MSE, MAE).
    Bold = best in the pool for that dataset/H/metric, underline = second; red X = exceeds the chip there."""
    models = [m for m in pool + part if m in set(c.model)]
    cc = c.merge(chip, on=["model", "data", "pred_len"], how="left") if chip is not None else c.assign(verdict="FIT")
    cc["verdict"] = cc.verdict.fillna("FIT")
    val = cc.set_index(["model", "data", "pred_len"])
    mean = cc.groupby(["model", "data"])[["MSE", "MAE"]].mean()
    cols = HS + ["avg"]

    def order(d, h, k):
        if h == "avg":
            return sorted(mean.loc[[(m, d) for m in pool if (m, d) in mean.index], k])
        ok = cc[(cc.data == d) & (cc.pred_len == h) & cc.model.isin(pool) & (cc.verdict == "FIT")]
        return sorted(ok[k])

    out = []
    for g in range(0, len(datasets), per):
        ds = datasets[g:g + per]
        ords = {(d, h, k): order(d, h, k) for d in ds for h in cols for k in ("MSE", "MAE")}
        lab = lambda d: f"{d} <span class='mut'>(D = {MFG_D[d]})</span>" if d in MFG_D else d
        h1 = '<th rowspan="2" class="mh">모델</th><th rowspan="2" class="mh">지표</th>' + "".join(
            f'<th colspan="{len(cols)}" class="dgrp">{lab(d)}</th>' for d in ds)
        h2 = "".join("".join(f'<th class="{"dgrp" if i == 0 else ""}{" avg" if h == "avg" else ""}">'
                             f'{"평균" if h == "avg" else h}</th>' for i, h in enumerate(cols)) for d in ds)
        rows = []
        for m in models:
            if part and m == part[0]:
                rows.append(f'<tr class="sep"><td colspan="{2 + len(cols) * len(ds)}"><i>칩에 일부만 들어가는 모델 '
                            f'(순위 제외)</i></td></tr>')
            for j, k in enumerate(("MSE", "MAE")):
                tds = ""
                for d in ds:
                    for i, h in enumerate(cols):
                        cl = ("dgrp" if i == 0 else "") + (" avg" if h == "avg" else "")
                        if h == "avg":
                            if (m, d) not in mean.index:
                                tds += f'<td class="{cl}">–</td>'; continue
                            v, off = mean.loc[(m, d), k], False
                        else:
                            if (m, d, h) not in val.index:
                                tds += f'<td class="{cl}">–</td>'; continue
                            v, off = val.loc[(m, d, h), k], val.loc[(m, d, h), "verdict"] != "FIT"
                        s, o = f3(v), ords[(d, h, k)]
                        if off:
                            s = f'<span class="off">{s}✗</span>'
                        elif m in pool and o and abs(v - o[0]) < 1e-12:
                            s = f"<b>{s}</b>"
                        elif m in pool and len(o) > 1 and abs(v - o[1]) < 1e-12:
                            s = f"<u>{s}</u>"
                        tds += f'<td class="{cl}">{s}</td>'
                lead = f'<td rowspan="2" class="ds">{mrow(m)}</td>' if j == 0 else ""
                cls = ("m2" if j else "m1") + ("" if m in pool else " part")
                rows.append(f'<tr class="{cls}">{lead}<td class="met">{k}</td>{tds}</tr>')
        out.append(f'<div class="tw pairw"><table class="pairs"><thead><tr>{h1}</tr><tr>{h2}</tr></thead>'
                   f'<tbody>{"".join(rows)}</tbody></table></div>')
    return "".join(out)


# ------------------------------------------------------------------ sections
def sec_mfg_summary(ch, c, chip, lat):
    x = ranks(c, POOL)
    fit = chip.groupby("model").verdict.apply(lambda s: int((s == "FIT").sum()))
    body = []
    best = {k: c[c.model.isin(POOL)].groupby("model")[k].mean().min() for k in ("MSE", "MAE")}
    bestH = {h: c[c.model.isin(POOL) & (c.pred_len == h)].groupby("model").MSE.mean().min() for h in HS}
    for group, models in (("pool", POOL), ("part", PART)):
        for m in models:
            y = c[c.model == m]
            if not len(y):
                continue
            mse, mae = y.MSE.mean(), y.MAE.mean()
            hm = [y[y.pred_len == h].MSE.mean() for h in HS]
            par = y[y.data == "MetroPT3"].set_index("pred_len").params
            prm = f'{int(par.get(96, 0)):,} → {int(par.get(720, 0)):,}'
            setting = (P.mfg_main_config_id(m) or "").replace("cand:", "")
            la = lat.get(m, ("–", "–"))
            if group == "pool":
                z = x[x.model == m]
                rk = f'{z.rMSE.mean():.2f} / {z.rMAE.mean():.2f}'
                wn = f'{int((z.rMSE == 1).sum())} / {int((z.rMAE == 1).sum())}'
                b = lambda v, k: f'<b>{f3(v)}</b>' if abs(v - best[k]) < 1e-12 else f3(v)
                cells = [b(mse, "MSE"), b(mae, "MAE")] + [f'<b>{f3(v)}</b>' if abs(v - bestH[h]) < 1e-12 else f3(v)
                                                          for v, h in zip(hm, HS)]
            else:
                rk = wn = '<span class="mut">순위 제외</span>'
                cells = [f3(mse), f3(mae)] + [f3(v) for v in hm]
            body.append([mrow(m), VENUE.get(m, ""), esc(setting), prm, f'{fit.get(m, 0)}/24', *cells, rk, wn,
                         la[0], la[1]])
        if group == "pool":
            body.append(['<i>칩에 일부만 들어가는 모델 (칩 밖 칸 포함 평균, 순위 없음)</i>'] + [""] * 14)
    head = ["모델", "발표처", "설정", "파라미터 (MetroPT3 H96→H720)", "칩 안 칸", "평균 MSE", "평균 MAE",
            "H96", "H192", "H336", "H720", "평균 순위 MSE / MAE", "1등 칸 MSE / MAE",
            "보드 지연 H96 (ms)", "보드 지연 H720 (ms)"]
    return table(head, body)


def sec_detail(c, chip, datasets, models_pool, models_part, chip_marks=True):
    out = []
    cc = c.merge(chip, on=["model", "data", "pred_len"], how="left") if chip is not None else c.assign(verdict="FIT")
    for d in datasets:
        sub = cc[cc.data == d]
        ok = sub[sub.model.isin(models_pool)]
        bm = {(h, k): ok[ok.pred_len == h][k].min() for h in HS for k in ("MSE", "MAE")}
        body = []
        for m in models_pool + models_part:
            y = sub[sub.model == m].set_index("pred_len")
            if not len(y):
                continue
            row = [mrow(m)]
            for k in ("MSE", "MAE"):
                for h in HS:
                    if h not in y.index:
                        row.append("–"); continue
                    val = y.loc[h, k]
                    off = chip_marks and y.loc[h, "verdict"] != "FIT"
                    s = f3(val)
                    if off:
                        s = f'<span class="off">{s} ✗</span>'
                    elif m in models_pool and abs(val - bm[(h, k)]) < 1e-12:
                        s = f"<b>{s}</b>"
                    row.append(s)
            body.append(row)
        head = ["모델"] + [f"MSE H{h}" for h in HS] + [f"MAE H{h}" for h in HS]
        title = f"{d} (D = {MFG_D[d]})" if d in MFG_D else d
        out.append(f"<h4>{esc(title)}</h4>" + table(head, body, "det"))
    return "".join(out)


def sec_ltsf_summary(lc):
    x = ranks(lc, POOL)
    body = []
    best = {k: lc[lc.model.isin(POOL)].groupby("model")[k].mean().min() for k in ("MSE", "MAE")}
    for m in POOL:
        y = lc[lc.model == m]
        z = x[x.model == m]
        per = [y[y.data == d].MSE.mean() for d in LTSF]
        b = lambda v, k: f'<b>{f3(v)}</b>' if abs(v - best[k]) < 1e-12 else f3(v)
        body.append([mrow(m), b(y.MSE.mean(), "MSE"), b(y.MAE.mean(), "MAE"),
                     f'{z.rMSE.mean():.2f} / {z.rMAE.mean():.2f}',
                     f'{int((z.rMSE == 1).sum())} / {int((z.rMAE == 1).sum())}', f'{int(z.rMSE.max())}']
                    + [f3(v) for v in per])
    body.append(['<i>참고: 칩 풀 밖 경량 모델</i>'] + [""] * 13)
    for m in PART:
        y = lc[lc.model == m]
        body.append([mrow(m), f3(y.MSE.mean()), f3(y.MAE.mean()), "", "", ""]
                    + [f3(y[y.data == d].MSE.mean()) for d in LTSF])
    head = ["모델", "평균 MSE", "평균 MAE", "평균 순위 MSE / MAE", "1등 칸 MSE / MAE", "최악 순위"] + \
           [f"{d} MSE" for d in LTSF]
    return table(head, body)


# ------------------------------------------------------------------ experiment setup (10-13)
MIXES = {"FEATHer_mix", "SAMformer"}          # lightweight models whose forecast of one channel reads other channels
COMMIT = {"CMoS": "b696a0c", "CycleNet": "d807e51", "DLinear": "0c11366", "DiPE_Linear": "b26e250", "FITS": "d040bb0",
          "LMS_AutoTSF": "412884a", "MDMLP_EIA": "0d68442", "MixLinear": "71b5db6", "PatchTST": "204c21e",
          "PhaseFormer": "ed1db61", "SAMformer": "71f10ea", "SparseTSF": "b8c2740", "SparseTSF_MLP": "b8c2740",
          "TQNet": "15e19cb", "TimeBase": "369b330", "TimeMixer": "e246105", "TimesNet": "4e938a1",
          "iTransformer": "c2426e6"}                      # docs/ENV.md (setup_baselines.sh)


def sec_models(c, lc, chip):
    fit = chip.groupby("model").verdict.apply(lambda s: int((s == "FIT").sum()))
    groups = (("칩 풀 — mfg 24칸 전부 칩 안 (순위 매김)", POOL), ("칩에 일부 칸만 들어가는 경량 모델 (표에 싣고 순위 제외)", PART),
              ("서버급 참고 — 칩에 안 들어감 (정확도 천장)", P.HEAVY))
    body = []
    for title, models in groups:
        body.append([f"<i>{title}</i>"] + [""] * 6)
        for m in models:
            y = c[(c.model == m) & (c.data == "MetroPT3")].set_index("pred_len").params
            z = lc[(lc.model == m) & (lc.data == "ETTh1") & (lc.pred_len == 96)].params
            mp = f"{int(y.get(96)):,} → {int(y.get(720)):,}" if 96 in y and 720 in y else "–"
            heavy = m in P.HEAVY
            chn = "–" if heavy else ("<b>섞음</b>" if m in MIXES else "독립")
            body.append([mrow(m), VENUE.get(m, ""), chn, mp, f"{int(z.max()):,}" if len(z) else "–",
                         "✗" if heavy else f"{fit.get(m, 0)}/24", COMMIT.get(m, "이 저장소")])
    head = ["모델", "발표처", "채널 처리", "파라미터 mfg (MetroPT3 D 7, H96 → H720)", "파라미터 LTSF (ETTh1 H96)",
            "mfg 칩 안 칸", "공식 코드 커밋"]
    return table(head, body)


def sec_settings():
    sys.path.insert(0, os.path.join(ROOT, "tools", "results"))
    import select_verdict as S   # noqa: E402
    a = S.selection_rows("mfg")
    vc = pd.read_csv(os.path.join(BOARD, "chip_verdict_candidates_20261012.csv"))
    v0 = pd.read_csv(os.path.join(BOARD, "chip_verdict_20261012.csv"))
    fitc = vc.groupby(["model", "config"]).verdict.apply(lambda s: int((s == "FIT").sum()))
    fit0 = v0.groupby("model").verdict.apply(lambda s: int((s == "FIT").sum()))
    lt = json.load(open(P.CHOSEN_LTSF_FILE, encoding="utf-8"))
    body, full, lexs = [], [], []
    for m in [x for x in POOL + PART if x not in OURS]:
        cs = P.CANDIDATES[m]
        ids = ["cand:" + x for x in cs]
        s = a[(a.model == m) & a.config_id.isin(ids)]
        b = R2B[m]
        # chip cells per candidate; models whose published settings share one structure are measured once
        fits = {x: int(fitc[(b, x)]) if (b, x) in fitc.index else int(fit0.get(b, 0)) for x in cs}
        top = max(fits.values())
        acc, _, lines = S.choose(s, ids, 24, "data")                                    # accuracy only
        sub = ["cand:" + x for x in cs if fits[x] == top]
        win, _, _ = S.choose(s[s.config_id.isin(sub)], sub, 24, "data")                 # chip-best, then val
        bad = {ln.split()[2].rstrip(":") for ln in lines if "not eligible" in ln}
        rank = {ln.split()[0]: ln.split()[4] for ln in lines if "mean val rank" in ln}
        chosen = P.mfg_main_config_id(m)
        assert win == chosen, (m, win, chosen)
        cells = []
        for x in cs:
            k = "cand:" + x
            r = "제외" if k in bad else rank.get(k, "–")
            t = f"{esc(x)} {fits[x]}/24·{r}"
            cells.append(f"<b>{t}</b>" if k == chosen else (t if fits[x] == top else f'<span class="mut">{t}</span>'))
        full.append(f"<li>{mrow(m)}: " + " &nbsp;|&nbsp; ".join(cells) + "</li>")
        c0 = chosen.replace("cand:", "")
        acc_s = "같음" if acc == chosen else f'{esc(acc.replace("cand:", ""))} ({fits[acc.replace("cand:", "")]}/24)'
        lex = [f'{esc(k.split("|")[1])} → {esc(v.replace("cand:", ""))}' for k, v in sorted(lt.items())
               if k.split("|")[0] == m]
        if m == "DLinear":
            lex.append("ETTh2 lr 5e-3 (공식 0.05 발산)")
        lexs.append(f"<li>{mrow(m)}: " + ", ".join(lex) + "</li>") if lex else None
        body.append([mrow(m), str(len(cs)), f"<b>{esc(c0)}</b> ({fits[c0]}/24)", acc_s,
                     "공식" + (f" + 검증 {len(lex)}칸" if lex else "")])
    body.append([mrow("FEATHer_mix") + " / " + mrow("FEATHer_CI"), "–", "동결 (고르기 없음)", "–", "동결 · 주기만 데이터별"])
    body.append(["서버급 참고 7개", "–", "모델 기본 설정", "–", "공식"])
    head = ["모델", "후보 수", "mfg 고른 설정 (칩 안 칸)", "정확도만으로 골랐다면", "LTSF"]
    rule = """<div class="card"><ul>
<li><b>출처</b>: 저자가 공개한 설정만 (데이터별 최종 스크립트 + 기본 설정). <b>mfg</b>: ① 24칸 중 칩에 가장 많이 들어가는 후보 → ② 그중
6데이터 검증 평균 순위 1등 하나를 전부에 씀 (시드 2025, 테스트 안 봄; 건강 검사 = best epoch ≤ 2 칸이 절반 넘으면 제외).
①은 10-12 사후 변경(우리 순위를 낮춤 → 공개).</li>
<li><b>LTSF</b>: 데이터(·H)별 공식 설정. 공식이 없는 칸만 그 데이터 검증 손실로 후보 중 선택.
<b>주기·길이 제한 값</b>은 규칙으로 계산 (고르는 대상 아님).</li></ul></div>"""
    more = ('<details><summary>후보 전체 (출처 데이터 칩 안 칸·검증 평균 순위; 굵게 = 고름, 회색 = ①에서 빠짐)</summary><ul class="note">'
            + "".join(full) + "</ul></details>"
            + '<details><summary>LTSF 검증 선택 칸</summary><ul class="note">' + "".join(lexs) + "</ul></details>")
    return rule + table(head, body) + more


# process, source, what is forecast (utils/prep_manufacturing.py); period columns = protocol §5.8
MFG_INFO = {
    "Steel": ("철강 공장 전력 (광양)", "UCI #851", "전력·무효전력·역률·CO2", 24, 96),
    "WindSCADA": ("풍력 터빈 SCADA (Kelmarsh 1호기, 2017)", "Zenodo 5841834", "풍속·회전수·베어링 온도·피치·출력", 48, 144),
    "BlastFurnace": ("고로 공정 (MISIS)", "Mendeley 6d7jbc7tb5 (Data in Brief 2025)", "송풍·압력·가스 온도·성분 27개 + 용선 Si", 48, 24),
    "MetroPT3": ("지하철 공기압축기 (APU)", "UCI #791 (Sci. Data 2022)", "압력 5·모터 전류·오일 온도 (아날로그만)", 24, "298 (검출된 약 5시간 주기, 하루 아님)"),
    "N2O": ("하수처리장 생물 반응조 (Avedøre)", "Mendeley xmbxhscgpr", "송풍량·NH4·NO3·O2·SS·온도·PO4·유입량·N2O", 24, 288),
    "EWELD": ("금속 제조 공장 14곳 전력 부하", "Figshare 21893808 (Sci. Data 2023)", "공장별 15분 전력량 (kWh)", 24, 96),
}
# standard LTSF benchmarks (public files; split = data_loader.py)
LTSF_INFO = {"ETTh1": (7, "1시간", 17420, "12/4/4개월"), "ETTh2": (7, "1시간", 17420, "12/4/4개월"),
             "ETTm1": (7, "15분", 69680, "12/4/4개월"), "ETTm2": (7, "15분", 69680, "12/4/4개월"),
             "Weather": (21, "10분", 52696, "70/10/20 %"), "Exchange": (8, "1일", 7588, "70/10/20 %"),
             "Electricity": (321, "1시간", 26304, "70/10/20 %"), "Traffic": (862, "1시간", 17544, "70/10/20 %")}


def _span(minutes):
    if minutes < 60 * 48:
        return f"{minutes / 60:g}시간"
    return f"{minutes / 1440:.1f}일".replace(".0일", "일")


def sec_data():
    body = []
    for d in MFG:
        p = os.path.join(ROOT, "data", d)
        x = pd.read_csv(os.path.join(p, "data.csv"), usecols=[0], parse_dates=["date"]).date
        n, D = len(x), MFG_D[d]
        step = int(x.diff().median().total_seconds() // 60)
        mp = os.path.join(p, "mask.csv")
        miss = (1 - pd.read_csv(mp).iloc[:, 1:].to_numpy().mean()) * 100 if os.path.exists(mp) else 0.0
        proc, src, chans, fft_p, day = MFG_INFO[d]
        ntr, nte = int(n * 0.7), int(n * 0.2)
        step_s = f"{step}분" if step < 60 else f"{step // 60}시간"
        body.append([f"<b>{esc(d)}</b>", esc(proc), str(D), step_s, f"{n:,}",
                     f'{x.iloc[0]:%Y-%m} ~ {x.iloc[-1]:%Y-%m}', f"{ntr:,} / {n - ntr - nte:,} / {nte:,}",
                     f"{miss:.1f} %" if miss >= 0.05 else "0", f"{fft_p} / {day}",
                     f"{_span(96 * step)} ~ <b>{_span(720 * step)}</b>", esc(chans), esc(src)])
    head = ["데이터", "공정", "채널 D", "간격", "길이 (행)", "기간", "학습 / 검증 / 테스트 (행)", "결측 (채점 제외)",
            "주기 FFT≤48 / 하루 (CycleNet·FITS·TQNet)", "H96 ~ H720 실제 시간", "채널 내용", "출처"]
    mfg = table(head, body)
    body = []
    for d in LTSF:
        D, st, n, sp = LTSF_INFO[d]
        body.append([f"<b>{esc(d)}</b>", str(D), st, f"{n:,}", sp])
    lt = table(["데이터", "채널 D", "간격", "길이 (행)", "분할"], body)
    rules = """<div class="card"><ul>
<li><b>분할</b>: 시간순. mfg 와 Weather·Exchange·Electricity·Traffic = 70 / 10 / 20 %, ETT = 12 / 4 / 4개월 (관례). 검증·테스트 구간은 앞 L 스텝을 입력으로 겹쳐 씀.</li>
<li><b>표준화</b>: 채널별 학습 구간 평균·표준편차. 지표는 표준화된 값에서 계산.</li>
<li><b>결측</b>: 일정 시간 격자로 만든 뒤 빈 칸은 직전 값으로 채워 <b>입력</b>에 쓰고, <b>손실·지표에서는 제외</b>(mask.csv, 관측값만 채점).
보간·미래 값 사용 없음 (맨 앞 구간만 뒤 값).</li>
<li><b>전처리</b>: 연속 측정 채널만 (상태·범주·설정값 신호 제외). MetroPT3 1분 평균, N2O 5분 평균. BlastFurnace 는 학습 구간 기준 0.5~99.5 % 자르기 (센서 오류값).
EWELD 는 C24·C25 금속 제조 사용자 중 2018~2020 이 99 % 이상 있는 ID 순 14곳.</li>
<li><b>예측 방식</b>: 모든 채널을 동시에 예측 (M 모드). 마지막 열은 원 논문의 관심 변수(Usage_kWh · Power · Si · 오일 온도 · N2O)지만 채점은 전 채널 평균.</li>
</ul></div>"""
    return (rules + "<h4>산업 데이터 (mfg-6, 주 실험)</h4>" + mfg
            + "<h4>일반 벤치마크 (LTSF-8, 일반화 확인)</h4>" + lt)


def sec_training():
    # values = run.py (worker command) + scripts/train.py (optimizer, early stopping, metrics)
    rows = [
        ("입력 길이 L", "96", "입력 길이 실험만 48 · 192 · 336 (mfg). L 720 은 7채널 입력 창만으로 SRAM 거의 소진 → 안 함"),
        ("예측 길이 H", "96 · 192 · 336 · 720", "모든 데이터"),
        ("배치 · 에폭", "batch 32 · 최대 50 epoch", "학습 구간은 섞어서 뽑고 마지막 불완전 배치는 버림 (검증·테스트는 전부 사용)"),
        ("조기 종료", "patience 10 (검증 손실)", "검증 손실이 가장 낮은 epoch 의 가중치로 테스트 <b>1회</b>"),
        ("최적화", "Adam · CosineAnnealing (T_max 50) · gradient clip 1.0 · weight decay 0", "SAMformer 만 SAM (방법의 일부)"),
        ("학습률", "모델·데이터별 고른 설정의 값", "설정 B 참고 (저자 값; 탐색하지 않음)"),
        ("손실", "관측값 MSE", "결측 칸은 손실에서 제외. 공식이 Huber 인 모델(PhaseFormer)도 MSE 로 통일"),
        ("정밀도", "float32 (AMP 끔)", "FITS 복소 기울기 · TimesNet half cuFFT 문제 → 전 모델 동일"),
        ("시드", "2025 · 2026 · 2027", "설정 고르기는 2025 하나. cuDNN 결정적 모드"),
        ("지표", "MSE · MAE (표준화 값, 관측 칸만)", "관측 칸 전체에 대한 평균 (배치 평균의 평균 아님). 표의 값 = 3시드 평균"),
    ]
    common = table(["항목", "값", "비고"], [[f"<b>{esc(a)}</b>", b, c] for a, b, c in rows])
    own = [
        ("FEATHer", "스펙트럼 분리 항 λ 0.01 (LTSF Electricity 0)", "우리 방법의 일부"),
        ("TimeBase", "기저 직교 항", "공식 계수"),
        ("DiPE", "시간/주파수 손실 비율 · 전문가 라우터 온도 일정", "공식 설정의 값"),
        ("SAMformer", "SAM 최적화 (rho mfg 0.5 기본 · LTSF 0.7 공식)", "Adam 대신"),
    ]
    unify = [
        ("에폭 · patience · batch · 학습률 일정(lradj)", "공식 스크립트마다 다름 → 위 공통값"),
        ("PhaseFormer 손실", "Huber → MSE"),
        ("FITS 학습 방식", "공식 train_mode 선택 대신 예측 구간 손실 하나로 (다른 모델과 같은 방식)"),
        ("모델 구조·학습률·드롭아웃 등", "<b>통일하지 않음</b> — 고른 저자 설정 그대로"),
    ]
    return ('<div class="card">원칙: <b>학습 절차는 모든 모델에 똑같이</b>, <b>모델 자체(구조·lr)와 방법 고유의 손실 항은 저자 것 그대로</b>. '
            '테스트는 모델·칸마다 한 번만 봄.</div>' + common
            + "<h4>방법 고유 학습 항 (유지)</h4>" + table(["모델", "항", "값"], [[esc(a), b, c] for a, b, c in own])
            + "<h4>공식과 다르게 통일한 것</h4>" + table(["항목", "처리"], [[a, b] for a, b in unify]))


def sec_chip_rule():
    rows = [
        ("칩", "STM32F103RB (NUCLEO-F103RB) · Cortex-M3 72 MHz · <b>flash 128 KB · SRAM 20 KB</b> · FPU 없음",
         "가장 싼 급의 범용 MCU. 모든 모델 float32 (양자화는 범위 밖)"),
        ("판정 단위", "칸 = 모델 × 데이터 × H (mfg 24칸), <b>실제 채널 수 D</b>, 고른 설정",
         "chip_verdict.py: 칸마다 보드 펌웨어 전체를 빌드 (CubeIDE 플래그 · 실제 링커 스크립트 · HAL 포함)"),
        ("flash 에 넣는 것", "펌웨어 전체 − 골든 테스트 벡터", "골든 = 시험용 입력·정답 (현장 노드에는 없음)"),
        ("SRAM 에 넣는 것", ".data + .bss + 힙·스택 예약 0x600 B (입력 창 L × D 포함)",
         "출력을 채널 하나씩 내보내는 포트는 그 기준 (H × D 출력 버퍼를 다 들고 있지 않음)"),
        ("C 포트 규칙", "원래 계산은 그대로, 메모리는 정리만", "채널 독립 모델은 채널 하나씩 · 작업 공간 공유 · 재계산 금지 "
         "(FEATHer-mix 의 정규화+밴드 재계산만 명시적 예외) · 96점 FFT 는 전 모델 같은 커널"),
        ("보드 실측", "가장 빠듯한 칸을 실제 보드에 올려 출력 일치 · 스택 · 지연 확인 (10-12, 전부 PASS)",
         "CubeIDE GCC 14.3. 판정 빌드(WSL gcc 10.3)와 결론 일치. PhaseFormer BF H336 은 실제 링크도 실패"),
        ("지연 (표 1)", "7채널 1회 예측, 72 MHz", "H720 의 채널 독립 전체 버퍼 포트는 1채널 측정 × 7"),
        ("전류", "13개 모델 실측 29.2~30.0 mA", "에너지 = 3.3 V × 30 mA × 지연"),
        ("순위 풀", "24칸 전부 칩 안인 모델만 순위", "일부 칸만 들어가는 모델은 표에 싣고 칩 밖 칸 ✗ (값은 참고)"),
    ]
    return table(["항목", "기준", "비고"], [[f"<b>{esc(a)}</b>", b, c] for a, b, c in rows])


def sec_eval_rule():
    rows = [
        ("칸 값", "3시드 평균 MSE · MAE", "표준화 값, 관측 칸만"),
        ("칸별 순위", "칸마다 순위 풀 8개 안에서 MSE · MAE 각각 (동점 = 낮은 순위)", "1등 칸 수 = 그 칸 1위 횟수"),
        ("평균 순위", "24칸(LTSF 32칸) 칸별 순위의 평균", "평균 MSE 와 함께 봄 (데이터마다 오차 크기가 달라 평균만 보면 큰 데이터가 좌우)"),
        ("개선 %", "칸마다 (없음 − 있음) / 있음 × 100 의 평균", "평균끼리 나눈 값은 쓰지 않음"),
        ("검정 단위", "<b>데이터</b> (mfg 6쌍, H 평균) 단측 Wilcoxon", "칸 수(24)·시드 변동은 보조. 6쌍 단측 최소 p = 0.016"),
        ("보조 지표", "CORR · R2 는 참고만", "모양 지표로 주장하지 않음 (사전 규칙으로 기각)"),
        ("사전 등록", "섞기만의 효과 · 안정성 가설 · 재설계 판정은 결과 전에 규칙을 문서로 고정",
         "docs/prereg_*.md, 판정 도구도 결과 전에 커밋"),
    ]
    return table(["항목", "규칙", "비고"], [[f"<b>{esc(a)}</b>", b, c] for a, b, c in rows])


def sec_posthoc():
    rows = [
        ("10-12", "mfg 베이스라인 설정 = 칩에 가장 많이 들어가는 공개 설정 중 검증 1등 (프로토콜 5.5)",
         "main 결과를 본 뒤. <b>우리 순위를 낮춤</b>", "CMoS · FITS · PhaseFormer 설정 변경, 정확도 우선 결과는 표 주석"),
        ("10-11", "정식 3시드에서 best epoch ≤ 2 를 실패로 보지 않음, 발산만 실패 (프로토콜 5.6)",
         "결과를 본 뒤", "근거: 거의 모든 모델·데이터에 있고 일찍 멈춘 시드가 더 나쁘지 않음"),
        ("10-07", "공식 설정 없는 LTSF 칸 = 그 데이터 검증으로 선택 (처음엔 ETTh1 설정)", "실행 전 (Fable 검토)", ""),
        ("10-07", "칩 필터 후보 단위 → 칸 단위", "실행 전 (Fable 검토)", "CMoS · PhaseFormer 후보 복원"),
        ("예외", "DLinear ETTh2 lr 0.05 → 5e-3", "공식 값이 발산", "검증으로 재선택"),
        ("예외", "CycleNet 은 입력 길이별 공식 설정이 있으나 모든 L 에 L96 설정 하나", "", "다른 모델과 같은 규칙"),
        ("예외", "입력 길이를 넘는 값 = 허용 최대값 (MixLinear lpf, FITS cut)", "", ""),
        ("FEATHer", "채널 방식 mfg = mix · LTSF = CI 는 결과를 보고 정함", "", "<b>두 변종 모두 보고</b>"),
        ("FEATHer", "LTSF 주기·예외 lr 은 7월 LTSF 검증 탐색 값", "", "LTSF 는 모든 모델이 데이터별 설정"),
        ("숫자", "섞기 효과 원고 5.9 % → 섞기만 2.1 % (사전 등록 판정)", "", "원고 미반영"),
        ("정리 필요", "DLinear 후보 동점(Exchange = Electricity) → 목록 순서로 결정", "", "프로토콜 동점 규칙에 한 줄 추가"),
        ("정리 필요", "MetroPT3 '하루 주기' 298 = 검출된 약 5시간 주기", "", "프로토콜·원고 표현 수정"),
    ]
    return ('<div class="card">원고 원칙(10-07 결정): 베이스라인 쪽 사후 변경은 모두 밝힌다. "베이스라인 설정은 검증 손실로 골랐다"까지만 쓰고, '
            '"FEATHer 포함 모든 모델을 검증으로 골랐다"처럼 사실과 다른 문장은 쓰지 않는다.</div>'
            + table(["언제", "내용", "시점", "비고"], [[esc(a), b, c, d] for a, b, c, d in rows]))


def sec_heavy(c, lc):
    body = []
    for m in P.HEAVY:
        y, z = c[c.model == m], lc[lc.model == m]
        if not len(y) and not len(z):
            continue
        q = z[(z.data == "ETTh1") & (z.pred_len == 96)]
        p = int(q.params.max()) if len(q) else None
        body.append([esc(nm(m)), VENUE.get(m, ""), f"{p:,}" if p else "–", f3(y.MSE.mean()), f3(y.MAE.mean()),
                     f3(z.MSE.mean()), f3(z.MAE.mean())])
    for m in ("FEATHer_mix", "FEATHer_CI"):
        y, z = c[c.model == m], lc[lc.model == m]
        p = int(z[(z.data == "ETTh1") & (z.pred_len == 96)].params.max())
        body.append([mrow(m), "ours", f"{p:,}", f3(y.MSE.mean()), f3(y.MAE.mean()), f3(z.MSE.mean()), f3(z.MAE.mean())])
    head = ["모델", "발표처", "파라미터 (ETTh1 H96)", "mfg 평균 MSE", "mfg 평균 MAE", "LTSF 평균 MSE", "LTSF 평균 MAE"]
    return table(head, body)


def sec_chip(chip, stream):
    body = []
    for m in POOL + PART:
        x = chip[chip.model == m]
        fit = x[x.verdict == "FIT"]
        ram = fit.ram_stream_B.fillna(fit.ram_B)
        off = x[x.verdict != "FIT"]
        offs = ", ".join(f"{d} H{h}" for d, h in sorted(zip(off.data, off.pred_len))) if 0 < len(off) <= 4 else (
            f"{len(off)}칸" if len(off) else "없음")
        s = stream[stream.model == R2B.get(m, "")]
        st = "; ".join(f'{r.data} H{int(r.H)}: {r.status}' + (f' {float(r.latency_ms) / 1000:.1f} s'
                       if r.status == "PASS" else "") for r in s.itertuples())
        body.append([mrow(m), f"{len(fit)}/24", f'{fit.flash_B.max() / 1024:.1f}' if len(fit) else "–",
                     f'{ram.max() / 1024:.1f}' if len(fit) else "–", esc(offs), esc(st) or "–"])
    head = ["모델", "칩 안 칸", "최대 flash (KB, 한도 128)", "최대 SRAM (KB, 한도 20)", "칩 밖 칸",
            "보드 실측 (가장 빠듯한 칸, 실제 채널 수)"]
    return table(head, body)


def sec_mixing(r):
    k = ["data", "pred_len"]
    mm = lambda df: df.groupby(k).MSE.mean()
    main = r[(r.role == "main") & (r.seq_len == 96)]
    ctl = r[(r.role == "control") & (r.seq_len == 96)]
    mix = mm(main[(main.model == "FEATHer_mix") & (main.config_id == "frozen")])

    def gain(without, with_):
        g = ((without - with_) / with_ * 100).dropna()
        per = g.groupby("data").mean()
        return g.mean(), int((per > 0).sum()), len(per)

    rows = []
    ci = mm(main[(main.model == "FEATHer_CI") & (main.config_id == "frozen")])
    a = gain(ci, mix)
    rows.append(["FEATHer: CI → mix (섞기 + 채널별 용량)", f"{a[0]:+.1f} %", f"{a[1]}/{a[2]}", "원고의 5.9 % (칸별 규칙)"])
    sep = mm(ctl[ctl.config_id == "ctrl:FEATHer_mixsep@frozen"])
    a = gain(sep, mix)
    rows.append(["FEATHer: 채널별 MLP → mix (섞기만)", f"{a[0]:+.1f} %", f"{a[1]}/{a[2]}",
                 "사전 등록 판정: M = 2.1 %, p 0.047 → 5.9 % 대신 M"])
    host = {"SparseTSF": "cand:ETTh1", "SparseTSF_MLP": "cand:Traffic"}
    for h, cid in host.items():
        base = mm(main[(main.model == h) & (main.config_id == cid)])
        for arm in ("mix", "mixM"):
            cfg = f"ctrl:{h}_{arm}@{cid}"
            w = mm(ctl[ctl.config_id == cfg])
            if len(w):
                a = gain(base, w)
                rows.append([f"{nm(h)} + 섞기 통로 ({arm})", f"{a[0]:+.1f} %", f"{a[1]}/{a[2]}", "대조군"])
    fn = mm(ctl[ctl.config_id == "ctrl:FEATHer_frontnone@frozen"])
    a = gain(fn, mix)
    rows.append(["FEATHer: 밴드 + FFT 게이트 제거 → 있음", f"{a[0]:+.1f} %", f"{a[1]}/{a[2]}",
                 "보드 지연 161 → 83 ms (H96), 208 → 130 ms (H720)"])
    head = ["비교 (없음 → 있음)", "MSE 개선 (칸별 % 평균, 분모 = 있음)", "좋아진 데이터", "비고"]
    return table(head, rows)


def sec_ablation(r):
    k = ["data", "pred_len"]
    main = r[(r.role == "main") & (r.seq_len == 96) & (r.model == "FEATHer_mix") & (r.config_id == "frozen")]
    base = main.groupby(k).MSE.mean()
    ab = r[(r.role == "ablation") & (r.seq_len == 96)]
    label = {a: v[0] for a, v in P.ABLATION_ARMS.items()}
    body = []
    for cid, g in ab.groupby("config_id"):
        arm = cid[4:]
        x = g.groupby(k).MSE.mean()
        d = ((x - base) / base * 100).dropna()
        per = d.groupby("data").mean()
        hs = sorted(d.index.get_level_values(1).unique())
        body.append([esc(arm), esc(label.get(arm, "")), "H" + ",".join(map(str, hs)), f"{per.mean():+.2f} %",
                     f"{int((per > 0).sum())}/{len(per)}", f"{int(g.num_params.max()):,}"])
    body.sort(key=lambda r_: r_[0])
    head = ["arm", "내용", "H", "MSE 변화 (데이터별 % 평균, + = 나빠짐)", "나빠진 데이터", "최대 파라미터"]
    return table(head, body)


def sec_lookback(r, ch):
    lb = r[(r.role.isin(["lookback", "main"]))]
    cfg = P.mfg_main_config_id
    pool = P.FEATHER + P.ON_CHIP + ["CycleNet", "DLinear"]
    body = []
    for m in pool:
        x = lb[(lb.model == m) & (lb.config_id == cfg(m))]
        row = [mrow(m)]
        for L in (48, 96, 192, 336):
            y = x[x.seq_len == L]
            n = y.groupby(["data", "pred_len"]).ngroups
            row.append(f3(y.MSE.mean()) if n == 24 and y.seed.nunique() == 3 else f'<span class="mut">{n}/24</span>')
        body.append(row)
    head = ["모델", "L48", "L96", "L192", "L336"]
    return table(head, body)


def sec_redesign(r):
    rd = r[(r.role == "redesign") | ((r.role == "main") & (r.model == "FEATHer_mix") & (r.config_id == "frozen"))]
    rd = rd[(rd.seq_len == 96) & (rd.seed == 2025)]
    rd = rd.assign(cfg=rd.config_id.where(rd.role == "redesign", "A frozen"))
    v = rd.groupby(["data", "cfg"]).val_loss.mean().unstack()
    full = rd.groupby("cfg").size()
    v = v[[c for c in v.columns if full[c] == 24]]
    rk = v.rank(axis=1)
    order = rk.mean().sort_values().index
    body = [[esc(d)] + [f3(v.loc[d, c], 4) for c in order] for d in v.index]
    body.append(["<b>평균 순위</b>"] + [f"<b>{rk[c].mean():.2f}</b>" for c in order])
    return table(["데이터 (검증 손실, 4H 평균, 시드 2025)"] + [esc(c) for c in order], body)


# ------------------------------------------------------------------ page
CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e6;--head:#f1f1f3;--acc:#0b63c9;--ours:#0a7d4f;--off:#b3261e;--card:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121214;--fg:#ececee;--mut:#9a9aa2;--line:#2c2c31;--head:#1c1c20;--acc:#6fa8ff;--ours:#4cd39a;--off:#ff8a80;--card:#17171a}}
:root[data-theme="dark"]{--bg:#121214;--fg:#ececee;--mut:#9a9aa2;--line:#2c2c31;--head:#1c1c20;--acc:#6fa8ff;--ours:#4cd39a;--off:#ff8a80;--card:#17171a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI","Malgun Gothic",sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
h4{font-size:14px;margin:18px 0 6px}.meta{color:var(--mut);font-size:12.5px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:14px 0 6px;font-size:13px}nav a{color:var(--acc);text-decoration:none}
.note{color:var(--mut);font-size:12.5px;margin:4px 0 10px}.note.below{margin:6px 0 4px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}
.tallw{max-width:862px}.pairw{margin:0 0 14px}
table.pairs th,table.pairs td{padding:3px 8px;font-size:12.5px}table.pairs tbody tr:hover{background:transparent}
table.pairs thead th{text-align:center}table.pairs th.mh{text-align:left;vertical-align:bottom}
table.pairs .dgrp{border-left:1px solid var(--line)}table.pairs .avg{background:var(--head);font-weight:400}
table.pairs tr.m1 td{border-bottom:none;padding-top:5px}table.pairs tr.m2 td{border-bottom:1px solid var(--line);padding-bottom:5px}
table.pairs td.ds{text-align:left;vertical-align:middle;font-weight:500;border-bottom:1px solid var(--line);border-right:1px solid var(--line);white-space:nowrap}
table.pairs td.met{color:var(--mut);font-size:11.5px;text-align:left}
table.pairs tr.sep td{text-align:left;color:var(--mut);background:var(--head);font-size:12px;padding:4px 8px;border-bottom:1px solid var(--line)}
table.pairs tr.part td{color:var(--mut)}table.pairs tr.part td.ds span{color:var(--mut)}
table.grid th,table.grid td{padding:3px 8px;font-size:12.5px}table.grid tbody tr:hover{background:transparent}
table.grid th.dgrp,table.grid td.dgrp{border-left:1px solid var(--line)}table.grid thead th{text-align:center}
table.grid td.hcol{font-weight:700;text-align:center;vertical-align:middle;border-right:1px solid var(--line);background:var(--head);border-bottom:2px solid var(--line)}
table.grid th.hcol{width:56px}table.grid td.mname{text-align:left;white-space:nowrap}
table.grid tr td{border-bottom:none}table.grid tr.last td{border-bottom:2px solid var(--line)}
table.grid tr.sep td{text-align:left;color:var(--mut);background:var(--head);font-size:12px;padding:3px 8px;border-top:1px dashed var(--line)}
table.grid tr.part td{color:var(--mut)}table.grid tr.part td span:not(.off){color:var(--mut)}.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--head);font-weight:600;font-size:12.5px;position:sticky;top:0}td:first-child,th:first-child{text-align:left}
tbody tr:hover{background:var(--head)}.ours{color:var(--ours);font-weight:600}.off{color:var(--off)}.mut{color:var(--mut)}
table.det td{min-width:62px}
table.wide th,table.wide td{padding:4px 7px;font-size:12.5px}table.wide .g{border-left:1px solid var(--line)}
table.wide th.part{color:var(--mut)}table.wide td.ds{font-weight:600;vertical-align:top;background:var(--card)}
tr.avgrow td{background:var(--head)}
table.tall{width:100%;max-width:860px;table-layout:fixed}
table.tall col.c-model{width:210px}table.tall col.c-met{width:64px}
table.tall td,table.tall th{padding:4px 10px}
table.tall tbody tr:hover{background:transparent}
table.tall tr.m1 td{border-bottom:none;padding-top:7px}table.tall tr.m2 td{padding-bottom:7px;border-bottom:1px solid var(--line)}
table.tall td.ds{vertical-align:middle;font-weight:500;border-bottom:1px solid var(--line);border-right:1px solid var(--line)}
table.tall td.met{color:var(--mut);font-size:12px;text-align:left}table.tall th:nth-child(2){text-align:left}
table.tall tr.sep td{text-align:left;background:var(--head);color:var(--mut);font-size:12.5px;padding:6px 10px;border-bottom:1px solid var(--line)}
table.tall td.avg,table.tall th.avg{border-left:1px solid var(--line);background:var(--head)}
table.tall tbody tr.part td{color:var(--mut)}table.tall tbody tr.part td.ds span{color:var(--mut)}td.avg{font-weight:500}u{text-decoration-thickness:1.5px;text-underline-offset:2px}ul{margin:6px 0 0 18px;padding:0}li{margin:2px 0}
"""


def main():
    ch, r, main_rows, lt, chip = load()
    c = cell_means(main_rows)
    lc = cell_means(lt)
    # board latency (7 channels): chosen setting; channel-independent full-buffer ports at H720 = 1 ch x 7
    b29 = pd.read_csv(os.path.join(BOARD, "board_latency_20260929.csv"))
    bc = pd.read_csv(os.path.join(BOARD, "board_latency_candidates_20261007.csv"))
    lat = {}
    for reg in POOL + PART:
        b = R2B[reg]
        setting = (P.mfg_main_config_id(reg) or "").replace("cand:", "")
        cand = bc[(bc.model == b) & ((bc.config == setting) | bc.same_as.fillna("").str.split(";").apply(lambda s: setting in s))]
        src = cand if len(cand) else b29[(b29.model == b) & (b29.direct_dft == 0)]
        get = lambda H, D: src[(src.H == H) & (src.D == D) & (src.status == "PASS")].latency_ms
        h96 = get(96, 7)
        h720 = get(720, 7)
        if not len(h720) and len(get(720, 1)):
            h720 = get(720, 1) * 7
        lat[reg] = (f"{h96.iloc[0]:.0f}" if len(h96) else "✗", f"{h720.iloc[0]:.0f}" if len(h720) else "✗")
    stream = pd.read_csv(os.path.join(BOARD, "board_stream_20261012.csv"))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    secs = [
        ("setup-models", "설정 A. 비교 모델",
         '<p class="note">채널 처리 = 한 채널의 예측이 다른 채널 값을 읽으면 "섞음" (값을 바꿔 확인한 결과; 가중치 공유는 섞기가 아님). '
         '파라미터 = 고른 설정의 실제 학습 모델. 칩 안 칸 = 고른 설정으로 24칸마다 펌웨어 빌드 (128 KB flash / 20 KB SRAM). '
         '커밋 = docs/ENV.md, setup_baselines.sh 가 받는 공식 저장소.</p>' + sec_models(c, lc, chip)),
        ("setup-settings", "설정 B. 베이스라인 설정 고르기",
         '<p class="note">순위 = 모든 적격 후보 중 검증 평균 순위 (낮을수록 좋음; ①에서 빠진 후보를 빼고 다시 매겨도 고른 설정이 1등). '
         '정확도만으로 골랐다면 = ① 없이 ②만 적용한 결과 (괄호 = 그 설정의 칩 안 칸). 출처 = docs/chosen_mfg.json · chosen_ltsf.json · '
         'docs/EXPERIMENT_PROTOCOL.md §5.</p>'
         + sec_settings()),
        ("setup-data", "설정 C. 데이터",
         '<p class="note">행 수·기간·결측은 data/&lt;이름&gt;/data.csv · mask.csv 에서 이 페이지를 만들 때 계산. 결측 = 채점에서 빠지는 칸 비율 '
         '(전체 칸 대비). H 실제 시간 = H × 간격. LTSF 행 수는 공개 파일 기준. 출처·채널 선택 = utils/prep_manufacturing.py.</p>'
         + sec_data()),
        ("setup-training", "설정 D. 공통 학습 조건",
         '<p class="note">값의 출처 = run.py (학습 명령) · scripts/train.py (최적화·조기 종료·지표) · docs/EXPERIMENT_PROTOCOL.md §3.</p>'
         + sec_training()),
        ("setup-chip", "설정 E. 칩 판정 기준", '<p class="note">출처 = deployment/board/chip_verdict.py · board_run.py · '
         'deployment/board/README.md (공통 C 규칙).</p>' + sec_chip_rule()),
        ("setup-eval", "설정 F. 평가 · 순위 규칙", sec_eval_rule()),
        ("setup-posthoc", "설정 G. 사후 변경 · 예외 (공개할 것)", '<p class="note">출처 = docs/EXPERIMENT_PROTOCOL.md §5 · §6 · §10.</p>'
         + sec_posthoc()),
        ("mfg", "표 1. 산업 데이터 (mfg-6, L 96) — 칩 풀 비교",
         '<p class="note">굵게 = 칩 풀 안 최고. 평균 순위·1등 칸은 칸(데이터 × H)마다 풀 8개 안에서 매김. 보드 지연 = NUCLEO-F103RB 72 MHz,'
         ' 7채널 1회 예측 (H720 의 채널 독립 포트는 1채널 측정 × 7). ✗ = 칩 밖.</p>' + sec_mfg_summary(ch, c, chip, lat)),
        ("mfg-detail", "표 2. 산업 데이터 상세 (데이터 × H)",
         '<p class="note">열 = 데이터(채널 수 D) × H, 행 = 모델 × (MSE, MAE). <b>굵게</b> = 그 데이터·H 에서 칩 풀 안 1등, '
         '<u>밑줄</u> = 2등. 빨강 ✗ = 그 칸에서 칩을 넘음(값은 참고, 순위 제외).</p>'
         + sec_pairs(c, chip, MFG, POOL, PART)),
        ("ltsf", "표 3. 일반 벤치마크 (LTSF-8, L 96) — 같은 풀",
         '<p class="note">저자 공식 설정. 순위는 mfg 와 같은 풀 8개 안에서. LTSF 는 채널 수가 커서(321 · 862) 데이터별 칩 판정은 하지 않음.</p>'
         + sec_ltsf_summary(lc)),
        ("ltsf-detail", "표 4. LTSF 상세 (데이터 × H)",
         '<p class="note">열 = 데이터 × H, 행 = 모델 × (MSE, MAE). <b>굵게</b> = 풀 안 1등, <u>밑줄</u> = 2등. 구분선 아래 = 칩 풀 밖 경량 모델(순위 제외).</p>'
         + sec_pairs(lc, None, LTSF, POOL, PART)),
        ("heavy", "표 5. 서버급 참고 모델 (칩에 안 들어감 · 정확도 천장)",
         '<p class="note">mfg 는 각 모델 기본 설정, LTSF 는 공식 설정.</p>' + sec_heavy(c, lc)),
        ("chip", "표 6. 칩 판정과 보드 실측",
         '<p class="note">칸마다 펌웨어 전체 빌드(실제 채널 수, 골든 벡터 제외). 출력을 채널 하나씩 내보내는 포트는 그 기준. '
         '오른쪽 = 가장 빠듯한 칸을 실제 보드에 올려 출력·스택·지연 확인(10-12).</p>' + sec_chip(chip, stream)),
        ("mixing", "표 7. 섞기 효과와 대조군 (mfg, L 96, H 4개)",
         '<p class="note">개선 = 칸마다 (없음 − 있음) / 있음 × 100 의 24칸 평균.</p>' + sec_mixing(r)),
        ("ablation", "표 8. FEATHer 부품 제거 실험 (ablation)", sec_ablation(r)),
        ("lookback", "표 9. 입력 길이 (mfg, 평균 MSE) — 서버 실행 중",
         '<p class="note">회색 n/24 = 아직 다 안 끝남(고른 설정 기준).</p>' + sec_lookback(r, ch)),
        ("redesign", "개발 중 · 재설계 1단계 (검증 손실만, 논문 미포함)",
         '<p class="note">사전 등록 docs/prereg_redesign_1012.md. 테스트는 3단계에서 고른 하나만 봄.</p>' + sec_redesign(r)),
    ]
    def notes_below(h):
        # every section opens with its explanatory <p class="note">; show it under the table instead
        if h.startswith('<p class="note">'):
            k = h.index("</p>") + 4
            return h[k:] + h[:k].replace('class="note"', 'class="note below"')
        return h

    body = "".join(f'<h2 id="{a}">{esc(t)}</h2>{notes_below(h)}' for a, t, h in secs)
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FEATHer Results</title><style>{CSS}</style></head><body><main>
<h1>FEATHer 실험 결과표</h1><div class="meta">생성 {now}</div>
{body}</main></body></html>"""
    open(OUT, "w", encoding="utf-8").write(page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
