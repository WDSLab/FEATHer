# -*- coding: utf-8 -*-
"""Prove that the DEPLOYED C code runs the model the paper reports.

The golden vectors show the port computes the same function as PyTorch on one
synthetic window with random weights.  That is necessary but not sufficient for
the paper's claim, which is stronger:

    the accuracy in Table 1 is produced by the code that runs on the MCU

This closes the gap in two steps, both on REAL data with the TRAINED weights:

  (a) checkpoint -> PyTorch test MSE, compared against the sweep's CSV row.
      Establishes that this checkpoint is the model the table reports.
  (b) N real test windows dumped to binary; the C port replays them and the
      two prediction sets are compared elementwise.
      Establishes that the port computes the same thing on real inputs.

Targets differ from the golden test on purpose: masked/imputed cells and the
worker's exact metric aggregation stay on the PyTorch side, so the C harness
only has to reproduce PREDICTIONS - no metric logic is duplicated in C, where
it could silently diverge.

    python deployment/board/verify_deployed_accuracy.py --ckpt <path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.FEATHer.FEATHer import FEATHer     # noqa: E402
from utils import data_factory                    # noqa: E402
from export_golden_feather2 import ARCH           # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "golden"


def build(pred_len: int, d_model: int, period: int, ckpt: str) -> FEATHer:
    m = FEATHer(seq_len=96, pred_len=pred_len, d_model=d_model,
                use_norm=True, period=period, **ARCH).eval()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match the frozen architecture\n"
                         f"  missing:    {missing}\n  unexpected: {unexpected}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="MetroPT3")
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--period", type=int, default=24)
    ap.add_argument("--windows", type=int, default=200,
                    help="how many real test windows to replay through C")
    ap.add_argument("--csv", default="results/probes/feather_ckpt.csv",
                    help="sweep row to check the checkpoint against")
    ap.add_argument("--mlp_rank", type=int, default=ARCH["mlp_rank"],
                    help="residual MLP rank of the checkpoint (re-frozen model: 4)")
    ap.add_argument("--seed", type=int, default=None,
                    help="sweep seed of the checkpoint; default = parsed from '_s<seed>' "
                         "in the file name.  The CSV has one row per seed, so matching "
                         "without it compares against an arbitrary seed.")
    ap.add_argument("--exp_tag", default=None, help="restrict the CSV row to this exp_tag")
    a = ap.parse_args()
    ARCH["mlp_rank"] = a.mlp_rank
    if a.seed is None:
        import re
        m_ = re.search(r"_s(\d+)\.pth$", a.ckpt)
        a.seed = int(m_.group(1)) if m_ else None

    test_data, test_loader = data_factory.data_provider(
        "data/", a.data, "M", 32, 96, 48, a.pred_len, "test")
    D = next(iter(test_loader))[0].shape[-1]
    m = build(a.pred_len, D, a.period, a.ckpt)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{a.data}  D={D}  H={a.pred_len}  period={a.period}  "
          f"params={n_params}")

    # ---- (a) does this checkpoint reproduce the reported test MSE? ----
    se = cnt = 0.0
    keep_x, keep_p = [], []
    with torch.no_grad():
        for bx, by, _, _ in test_loader:
            bx, by = bx.float(), by.float()
            out = m(bx[:, :, :D])
            yt = by[:, :, :D][:, -a.pred_len:].numpy()
            yp = out[:, -a.pred_len:].numpy()
            diff = yp - yt                       # NaN where target imputed
            fin = np.isfinite(diff)
            se += float(np.sum(diff[fin] ** 2))
            cnt += int(fin.sum())
            if len(keep_x) * bx.shape[0] < a.windows:
                keep_x.append(bx[:, :, :D].numpy())
                keep_p.append(yp)
    mse = se / max(1.0, cnt)
    print(f"\n(a) PyTorch test MSE (observed cells only) = {mse:.6f}")

    csv_path = _ROOT / a.csv
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[(df.data == a.data) & (df.pred_len == a.pred_len)]
        if a.seed is not None and "seed" in df:
            row = row[row.seed == a.seed]
        if a.exp_tag is not None and "exp_tag" in df:
            row = row[row.exp_tag == a.exp_tag]
        if len(row) > 1:
            raise SystemExit(f"{len(row)} CSV rows match - pass --seed / --exp_tag")
        if len(row):
            ref = float(row.MSE.iloc[0])
            rel = abs(mse - ref) / ref * 100
            ok = rel < 0.5
            print(f"    sweep CSV row (seed {a.seed}) = {ref:.6f}   "
                  f"({rel:.3f}% {'OK' if ok else 'MISMATCH'})")
            if not ok:
                raise SystemExit(
                    "the checkpoint does not reproduce the swept number - the "
                    "exported weights would not be the model the table reports")
        else:
            print(f"    (no matching row in {a.csv})")
    else:
        print(f"    (CSV {a.csv} not found - skipping the table check)")

    # ---- (b) dump real windows + PyTorch predictions for the C harness ----
    X = np.concatenate(keep_x)[:a.windows].astype(np.float32)
    P = np.concatenate(keep_p)[:a.windows].astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_x.bin").write_bytes(np.ascontiguousarray(X).tobytes())
    (OUT_DIR / "real_p.bin").write_bytes(np.ascontiguousarray(P).tobytes())
    with (OUT_DIR / "real_ref.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/verify_deployed_accuracy.py"
                " - do not edit. */\n")
        f.write(f"/* {a.data} test windows, ckpt={Path(a.ckpt).name} */\n")
        f.write("#ifndef REAL_REF_H\n#define REAL_REF_H\n")
        f.write(f"#define REAL_N {len(X)}\n#define REAL_D {D}\n"
                f"#define REAL_L 96\n#define REAL_H {a.pred_len}\n")
        f.write(f"static const double REAL_PT_MSE = {mse:.10f};\n")
        f.write("#endif /* REAL_REF_H */\n")
    print(f"\n(b) wrote {len(X)} real windows -> golden/real_x.bin, real_p.bin,"
          f" real_ref.h")
    print(f"    next: export the weights with --ckpt, then run the C replay:")
    print(f"      python deployment/board/export_golden_feather2.py "
          f"--pred_len {a.pred_len} --d_model {D} --mlp_rank {a.mlp_rank} --ckpt {a.ckpt}")
    print(f"      cd deployment/board && make realcheck_r "
          f"FEA2_GEOM=\"-DFEA2_MLP_R={a.mlp_rank} -DFEA2_H={a.pred_len} -DFEA2_D={D}\"")


if __name__ == "__main__":
    main()
