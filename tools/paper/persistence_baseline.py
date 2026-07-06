# -*- coding: utf-8 -*-
"""
Naive copy-last persistence baseline for the result tables.

Both JMS tables carry a persistence row (decided 2026-07-02: cheap, and it
pre-empts the "is this task trivial?" reviewer attack — persistence is what
exposed the original PMSM problem). This tool computes that row
REPRODUCIBLY under the exact evaluation protocol of the worker:

  - same window enumeration (data_provider, flag="test", deterministic,
    unit-aware for CMAPSS/CMAPSS3/PMSM — windows never cross a unit),
  - same standardization (scaler fitted on the train split by the Dataset),
  - same target tensor (batch_y[:, :, :n_features][:, -pred_len:]),
  - forecast = last observed input step, repeated across the horizon.

MSE/MAE are reported over ALL channels (how models are scored) and for the
target channel alone (last column; the audit convention).

Usage:
    python tools/paper/persistence_baseline.py                # mfg + pdm
    python tools/paper/persistence_baseline.py --group mfg
    python tools/paper/persistence_baseline.py --data Steel --pred_len 96
Output: results/persistence_baseline.csv (overwritten each run — the
computation is deterministic and cheap) + a printed table.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils import data_factory  # noqa: E402

MFG = [("Steel", [96, 192, 336, 720]),
       ("GasTurbine", [96, 192, 336, 720]),
       ("WindSCADA", [96, 192, 336, 720])]
PDM = [("CMAPSS", [24, 48, 96]),
       ("CMAPSS3", [24, 48, 96]),
       ("PMSM", [24, 48, 96])]

OUTPUT = os.path.join(ROOT, "results", "persistence_baseline.csv")


def persistence_metrics(data_name, seq_len, pred_len, batch_size=256):
    """Copy-last persistence MSE/MAE on the test split (standardized)."""
    df, _, _ = data_factory.data_select(data_name, os.path.join(ROOT, "data/"))
    n_features = df.shape[1] - 1

    _, loader = data_factory.data_provider(
        os.path.join(ROOT, "data/"), data_name, "M", batch_size,
        seq_len, seq_len, pred_len, "test",
    )

    se_all = ae_all = se_tgt = ae_tgt = 0.0
    n_all = n_tgt = 0
    n_windows = 0
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y = batch[0].float(), batch[1].float()
            true = batch_y[:, :, :n_features][:, -pred_len:]
            pred = batch_x[:, -1:, :n_features].expand(-1, pred_len, -1)
            err = (pred - true).numpy()
            se_all += float((err ** 2).sum())
            ae_all += float(np.abs(err).sum())
            n_all += err.size
            err_t = err[:, :, -1]
            se_tgt += float((err_t ** 2).sum())
            ae_tgt += float(np.abs(err_t).sum())
            n_tgt += err_t.size
            n_windows += err.shape[0]

    return dict(
        data=data_name, seq_len=seq_len, pred_len=pred_len,
        n_windows=n_windows, n_channels=n_features,
        MSE=se_all / n_all, MAE=ae_all / n_all,
        MSE_target=se_tgt / n_tgt, MAE_target=ae_tgt / n_tgt,
    )


def main():
    p = argparse.ArgumentParser(description="Copy-last persistence baseline")
    p.add_argument("--group", type=str, default="all",
                   choices=["mfg", "pdm", "all"])
    p.add_argument("--data", type=str, default="",
                   help="single dataset (overrides --group)")
    p.add_argument("--pred_len", type=int, default=0,
                   help="single horizon (0 = dataset defaults)")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--output", type=str, default=OUTPUT)
    args = p.parse_args()

    if args.data:
        default = dict(MFG + PDM).get(args.data, [96, 192, 336, 720])
        todo = [(args.data, [args.pred_len] if args.pred_len else default)]
    else:
        todo = {"mfg": MFG, "pdm": PDM, "all": MFG + PDM}[args.group]

    rows = []
    for data_name, horizons in todo:
        for h in horizons:
            r = persistence_metrics(data_name, args.seq_len, h)
            rows.append(r)
            print(f"  {data_name:>10} | H={h:<4} | windows={r['n_windows']:>6} | "
                  f"MSE={r['MSE']:.4f} MAE={r['MAE']:.4f} | "
                  f"target MSE={r['MSE_target']:.4f} MAE={r['MAE_target']:.4f}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
