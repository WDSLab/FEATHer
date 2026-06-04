"""Sweep all (model, dataset, horizon) cells and write edge_estimates.csv.

Usage:
    python -m deployment.cortex_m3.run --check
    python -m deployment.cortex_m3.run --model FEATHer --data ETTh1 --pred_len 96
    python -m deployment.cortex_m3.run                          # full sweep
    python -m deployment.cortex_m3.run --bit_width 32           # fp32 study

Each row of ``results/edge_estimates.csv`` is one EdgeEstimate; we use
the same append-and-dedup convention as ``results/fcst_results.csv``
so the file can be re-run incrementally.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import fields
from pathlib import Path

import torch

# Allow running as a script without -m by adding repo root to sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines import get_model, list_models                 # noqa: E402
from deployment.cortex_m3.estimator import (                  # noqa: E402
    EdgeEstimate,
    estimate_model,
)


# ---------------------------------------------------------------------
# Defaults consistent with the manuscript benchmark
# ---------------------------------------------------------------------
DEFAULT_MODELS = [
    "DiPE_Linear", "SparseTSF", "FEATHer", "FITS", "DLinear",
    "PatchTST", "TimeMixer", "TQNet", "LMS_AutoTSF", "TimesNet",
    "iTransformer", "MDMLP_EIA",
]
DEFAULT_DATASETS = [
    # (name, n_features) — matches baselines/utils/data_factory
    ("ETTh1", 7), ("ETTh2", 7), ("ETTm1", 7), ("ETTm2", 7),
    ("Weather", 21), ("Exchange", 8),
    ("Electricity", 321), ("Traffic", 862),
]
DEFAULT_HORIZONS = [96, 192, 336, 720]
DEFAULT_SEQ_LEN = 96
DEFAULT_BIT_WIDTH = 8

CSV_PATH = _ROOT / "results" / "edge_estimates.csv"
COLUMNS = [f.name for f in fields(EdgeEstimate)]


# ---------------------------------------------------------------------
def _existing_keys(csv_path: Path) -> set:
    """Resume key set so reruns skip cells already computed."""
    if not csv_path.exists():
        return set()
    keys = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            keys.add((row["model"], row["data"],
                      int(row["pred_len"]), int(row["bit_width"])))
    return keys


def _append_row(csv_path: Path, est: EdgeEstimate) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(est.as_row())


def _build_args_namespace(model_name: str, seq_len: int, pred_len: int,
                          n_features: int) -> argparse.Namespace:
    """Build a minimal Namespace expected by baseline wrappers."""
    from baselines import get_method_defaults
    base = get_method_defaults(model_name)
    args = argparse.Namespace(
        model=model_name,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=seq_len // 2,
        enc_in=n_features, dec_in=n_features, c_out=n_features,
        features="M", freq="h", embed="timeF",
        use_norm=True,
        # FEATHer-specific defaults; harmless for other wrappers
        d_state=8, kernel_size=7, period=12, num_bands=3,
        use_topk_gate=False, topk=2,
        lambda_spec=0.01,
    )
    for k, v in base.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------
def run_cell(model_name: str, data_name: str, n_features: int,
             seq_len: int, pred_len: int, bit_width: int) -> EdgeEstimate:
    args = _build_args_namespace(model_name, seq_len, pred_len, n_features)
    model = get_model(model_name, args, n_features, seq_len, pred_len)
    # Some baselines (e.g. MDMLP_EIA) hard-code .to('cuda') in their
    # internal buffers. Use CUDA when available so those builds run;
    # the cost analysis is device-agnostic since we only inspect
    # tensor shapes, not numeric values.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return estimate_model(
        model,
        model_name=model_name,
        data_name=data_name,
        seq_len=seq_len,
        pred_len=pred_len,
        n_features=n_features,
        bit_width=bit_width,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Cortex-M3 deployment estimator sweep.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--pred_len", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--bit_width", type=int, default=DEFAULT_BIT_WIDTH,
                    choices=[8, 16, 32])
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--check", action="store_true",
                    help="print missing cells but do not run them")
    ap.add_argument("--exclude", default="",
                    help="comma-separated model names to skip")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    existing = _existing_keys(csv_path)

    models = [args.model] if args.model else list(DEFAULT_MODELS)
    excluded = set(args.exclude.split(",")) if args.exclude else set()
    models = [m for m in models if m not in excluded]
    datasets = ([(args.data,
                  dict(DEFAULT_DATASETS).get(args.data, 7))]
                if args.data else DEFAULT_DATASETS)
    horizons = [args.pred_len] if args.pred_len else DEFAULT_HORIZONS

    todo = []
    for m in models:
        for d, nf in datasets:
            for h in horizons:
                key = (m, d, h, args.bit_width)
                if key not in existing:
                    todo.append((m, d, nf, h))

    if args.check:
        print(f"{len(todo)} cells pending (csv: {csv_path})")
        for cell in todo[:25]:
            print("  ", cell)
        if len(todo) > 25:
            print(f"  ... and {len(todo) - 25} more")
        return

    for m, d, nf, h in todo:
        try:
            est = run_cell(m, d, nf, args.seq_len, h, args.bit_width)
        except Exception as e:
            print(f"  [FAIL] {m} {d} H={h}: {type(e).__name__}: {e}")
            continue
        _append_row(csv_path, est)
        print(f"  [ok ] {m:14s} {d:12s} H={h:3d} "
              f"params={est.num_params:>8d} "
              f"peak_ram={est.peak_ram_bytes/1024:6.1f}KB "
              f"flash={est.flash_bytes/1024:6.1f}KB "
              f"lat={est.latency_ms:8.2f}ms "
              f"O={est.deploy_16k}{est.deploy_32k}{est.deploy_64k}")


if __name__ == "__main__":
    main()
