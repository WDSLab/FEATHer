# -*- coding: utf-8 -*-
"""
Compact results TABLE (not the line-by-line coverage dump).

Prints one matrix: rows = model (or config), columns = dataset x horizon,
cells = mean metric over seeds. Sorted by row mean; best per column marked *.

Works for the main sweep (rows = model) and the FEATHer arch searches
(single model -> rows = config exp_tag, auto-detected).

Usage:
    python tools/audit/results_table.py                         # fcst_results.csv, MSE
    python tools/audit/results_table.py --metric MAE
    python tools/audit/results_table.py --data Steel,WindSCADA,EWELD
    python tools/audit/results_table.py --csv results/feather_mfg_search.csv
    python tools/audit/results_table.py --csv results/feather_mfg_search_mse.csv --metric MSE
"""

import argparse
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Compact results matrix table")
    p.add_argument("--csv", default="results/fcst_results.csv")
    p.add_argument("--metric", default="MSE",
                   choices=["MSE", "MAE", "RMSE", "CORR", "R2", "val_loss"])
    p.add_argument("--exp_tag", default=None, help="filter to one exp_tag")
    p.add_argument("--model", default=None, help="filter to one model")
    p.add_argument("--data", default=None, help="comma list to filter/order columns")
    p.add_argument("--horizons", default="96,192,336,720")
    p.add_argument("--rows", default="auto", choices=["auto", "model", "exp_tag"],
                   help="what indexes the rows (auto: exp_tag if 1 model else model)")
    p.add_argument("--sort", default="mean", choices=["mean", "params"])
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[missing] {args.csv}")
        return
    df = pd.read_csv(args.csv)
    if df.empty:
        print(f"[empty] {args.csv}")
        return

    if args.exp_tag:
        df = df[df.exp_tag == args.exp_tag]
    if args.model:
        df = df[df.model == args.model]
    if args.data:
        wanted = args.data.split(",")
        df = df[df.data.isin(wanted)]
    H = [int(x) for x in args.horizons.split(",")]
    df = df[df.pred_len.isin(H)]
    if df.empty:
        print("No rows match the filter.")
        return

    rows = args.rows
    if rows == "auto":
        rows = "exp_tag" if df.model.nunique() == 1 else "model"

    # column order: requested data order (or sorted), then horizons
    data_order = (args.data.split(",") if args.data
                  else sorted(df.data.unique()))
    data_order = [d for d in data_order if d in set(df.data)]

    piv = df.pivot_table(index=rows, columns=["data", "pred_len"],
                         values=args.metric, aggfunc="mean")
    piv = piv.reindex(columns=pd.MultiIndex.from_tuples(
        [(d, h) for d in data_order for h in H if (d, h) in piv.columns]))
    piv["MEAN"] = piv.mean(axis=1)

    # param + seed-count annotation
    par = df.groupby(rows).num_params.max()
    nseed = df.groupby(rows).seed.nunique().max()

    if args.sort == "params":
        piv = piv.assign(_p=par).sort_values("_p").drop(columns="_p")
    else:
        piv = piv.sort_values("MEAN")

    # winners per column (lower is better except CORR/R2)
    higher_better = args.metric in ("CORR", "R2")
    winner = {c: (piv[c].idxmax() if higher_better else piv[c].idxmin())
              for c in piv.columns}

    # render: cells to string, mark best with trailing *
    disp = piv.copy()
    for c in disp.columns:
        col = []
        for idx, v in disp[c].items():
            if pd.isna(v):
                col.append("-")
            else:
                s = f"{v:.4f}"
                col.append(s + "*" if winner[c] == idx else s + " ")
        disp[c] = col
    disp.insert(0, "params", [int(par.get(i, 0)) for i in disp.index])

    # flatten multi-index columns to 'Steel@96' etc.
    flat = []
    for c in disp.columns:
        if isinstance(c, tuple):
            flat.append(f"{c[0][:5]}@{c[1]}")
        else:
            flat.append(c)
    disp.columns = flat

    print("=" * 100)
    print(f"  {args.metric}  |  {args.csv}  |  rows={rows}  |  seeds/cell<= {nseed}"
          f"  |  * = best in column")
    print("=" * 100)
    print(disp.to_string())
    print("\n(lower is better)" if not higher_better else "\n(higher is better)")


if __name__ == "__main__":
    main()
