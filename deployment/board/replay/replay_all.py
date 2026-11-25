# -*- coding: utf-8 -*-
"""Board measurement of every tuned industrial pick, the deployed way: the TRAINED weights of the pick's final
run (seed 2025) in the streaming firmware (main.c.replay, no golden vectors in flash), real test windows sent over the
serial link, the board's forecasts scored against PyTorch's.  One row per cell in results/board/board_replay_<board>.csv:
latency (mean / min / max over the windows), flash and RAM of the firmware, MSE board vs PyTorch on the same windows.

Per cell: the checkpoint (fetched from the training machine when it is not here) -> replay_flash.py (build + flash) -> replay.py
--windows N (each window gets only its own L = 96 steps) -> summary row.  Cells already in the CSV are skipped; cells
whose final run has not finished are skipped and picked up by a later pass.

    python deployment/board/replay/replay_all.py --port COM3            # every cell that has a checkpoint
    python deployment/board/replay/replay_all.py --port COM3 --limit 2  # a trial
    python deployment/board/replay/replay_all.py --board f411 --port COM4   # F411 (latency only)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent / "tools"), str(ROOT / "tools" / "results")]
import protocol as P  # noqa: E402
import chip_verdict as cv  # noqa: E402

SEED = 2025
REMOTE = ""                                         # optional: ssh alias of the machine that holds the checkpoints
REMOTE_REPO = ""                                    # and the path of the repository there
FIELDS = ["board", "tier", "model", "trial", "data", "H", "D", "seed", "windows", "status", "mse_board", "mse_torch",
          "maxabs_vs_torch", "latency_ms_mean", "latency_ms_min", "latency_ms_max", "flash_B", "ram_B", "timestamp"]


def ckpt_path(tier, model, trial, data, H):
    return ROOT / "results" / "checkpoints" / f"final__{tier}__{trial}" / model / f"{data}_H{H}_L96_s{SEED}.pth"


def fetch(path: Path) -> bool:
    """The checkpoint here, copied from the finals machine when missing; False when it is not there either."""
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    if not REMOTE:
        return False
    rel = path.relative_to(ROOT).as_posix()
    r = subprocess.run(["scp", "-q", f"{REMOTE}:{REMOTE_REPO}/{rel}", str(path)], capture_output=True)
    return r.returncode == 0 and path.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--board", default="f103", choices=["f103", "f411"], help="f411: NUCLEO-F411RE on COM4")
    ap.add_argument("--tiers", default="1k,chip")
    ap.add_argument("--models", default=None, help="comma list of registry models (default: every model of the tier)")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="stop after N measured cells (0 = all)")
    ap.add_argument("--out", default=None, help="default results/board/board_replay_<board>.csv")
    a = ap.parse_args()
    a.out = a.out or str(ROOT / "results" / "board" / f"board_replay_{a.board}.csv")
    py = sys.executable
    env = {**os.environ, "PYTHONUTF8": "1"}
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out)
        done = set(zip(d.tier, d.model, d.trial, d.data, d.H.astype(int)))
    wdir = ROOT / "results" / "board" / ("replay_windows" if a.board == "f103" else f"replay_windows_{a.board}")
    wdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for key, pick in sorted(P.tuned().items()):
        tier, model, data, H = key.split("|")
        H = int(H)
        if tier not in a.tiers.split(",") or data not in P.MFG or model not in P.tune_models(tier) \
                or (a.models and model not in a.models.split(",")):
            continue
        cell = (tier, model, pick["trial"], data, H)
        if cell in done:
            continue
        tag = f"{tier:<4} {model:<13} {pick['trial']:<4} {data:<12} H={H:<3}"
        ck = ckpt_path(*cell)
        if not fetch(ck):
            print(f"{tag} no final checkpoint yet", flush=True)
            continue
        b = subprocess.run([py, "-W", "ignore", str(ROOT / "deployment/board/replay/replay_flash.py"), "--tier", tier,
                            "--model", model, "--data", data, "--H", str(H), "--ckpt", str(ck), "--board", a.board,
                            "--port", a.port],
                           cwd=ROOT, capture_output=True, text=True, errors="replace", env=env)
        m = re.search(r"flash_B=(\d+) ram_B=(\d+)", b.stdout)
        if b.returncode != 0 or not m or "flashed" not in b.stdout:
            print(f"{tag} BUILD/FLASH FAILED: {(b.stdout + b.stderr).strip()[-300:]}", flush=True)
            continue
        wcsv = wdir / f"{tier}_{model}_{data}_H{H}_{pick['trial']}_s{SEED}.csv"
        if wcsv.exists():
            wcsv.unlink()                               # a cell is measured in one go on the firmware just flashed
        r = subprocess.run([py, "-W", "ignore", str(ROOT / "deployment/board/replay/replay.py"), "--tier", tier,
                            "--model", model, "--data", data, "--H", str(H), "--port", a.port, "--board", a.board,
                            "--windows", str(a.windows), "--out", str(wcsv)],
                           cwd=ROOT, capture_output=True, text=True, errors="replace", env=env)
        if r.returncode != 0 or not wcsv.exists():
            print(f"{tag} REPLAY FAILED: {(r.stdout + r.stderr).strip()[-300:]}", flush=True)
            continue
        w = pd.read_csv(wcsv)
        mse_b, mse_t = w.se_board.sum() / w.n_obs.sum(), w.se_torch.sum() / w.n_obs.sum()
        status = "PASS" if abs(mse_b / mse_t - 1) < 1e-5 else "CHECK"
        row = dict(board=a.board, tier=tier, model=model, trial=pick["trial"], data=data, H=H, D=cv.DATA_D[data],
                   seed=SEED, windows=len(w), status=status, mse_board=mse_b, mse_torch=mse_t,
                   maxabs_vs_torch=w.maxabs_vs_torch.max(), latency_ms_mean=w.latency_ms.mean(),
                   latency_ms_min=w.latency_ms.min(), latency_ms_max=w.latency_ms.max(),
                   flash_B=int(m.group(1)), ram_B=int(m.group(2)), timestamp=dt.datetime.now().isoformat(timespec="seconds"))
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        print(f"{tag} {status} {row['latency_ms_mean']:9.2f} ms  flash {row['flash_B'] / 1024:.1f} KB  RAM "
              f"{row['ram_B'] / 1024:.2f} KB  max|board-torch| {row['maxabs_vs_torch']:.1e}", flush=True)
        n += 1
        if a.limit and n >= a.limit:
            break


if __name__ == "__main__":
    main()
