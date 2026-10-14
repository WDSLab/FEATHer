# -*- coding: utf-8 -*-
"""Streaming replay of a test split through the board.

The deployed loop, end to end: the PC plays the sensors - it sends the test split of
one dataset to the NUCLEO one time step at a time (exactly the scaled values the
PyTorch evaluation feeds the model) - and every `--every` steps asks for a forecast.
The board keeps the last L = 96 steps itself (main.c.stream), forecasts with the
TRAINED weights and streams the forecast back.  The PC scores it against the true
future (observed cells only, as in every table) and against the PyTorch forecast of
the same window.

    # no board: PyTorch stands in for the board (checks windows, bookkeeping, scoring;
    # with --every 1 the replay must reproduce the checkpoint's CSV test MSE exactly)
    python deployment/board/replay.py --data MetroPT3 --H 96 --sim --every 1

    # board: install the firmware first, then replay (needs `pip install pyserial`)
    python deployment/board/board_prep.py --model feather_mix --H 96 --D 7 \
        --stream_ckpt results/checkpoints/candR4/FEATHer/MetroPT3_H96_L96_s2025.pth
    (CubeIDE: Build, Run)   then
    python deployment/board/replay.py --data MetroPT3 --H 96 --port COM3 --every 60

    # real-time segment: one step per sampling interval (or a fraction of it)
    python deployment/board/replay.py ... --port COM3 --every 60 --realtime 1.0 --max_steps 120

Writes results/board_replay_<data>_H<H>_s<seed>_<mode>.csv (one row per forecast)
and prints a summary: forecasts, MSE board vs PyTorch, max |board - PyTorch|,
latency, and the duty cycle / energy per day at the dataset's real sampling interval
(from the board current measurements: active 29.75 mA, WFI sleep 10.38 mA - the sleep
figure is the whole NUCLEO board, not a deployment figure, so energy/day is reported
as compute-only and as an upper bound with the board's sleep current).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "dev"))
os.chdir(REPO)
from worker import load_worker            # noqa: E402
from sensor_dropout import build_args        # noqa: E402  (same rebuild as the dropout check)

SAMPLING_S = {"MetroPT3": 60, "Steel": 900, "EWELD": 900, "N2O": 300, "WindSCADA": 600,
              "BlastFurnace": 3600}           # Table 2 intervals
CPU_HZ = 72_000_000
V, I_ACTIVE_mA, I_SLEEP_mA = 3.3, 29.75, 10.38   # results/board/board_current_20260929.csv
L = 96


class TorchBoard:
    """Stands in for the board: same protocol semantics, PyTorch forward on the window."""

    def __init__(self, model, D):
        self.m, self.D = model, D
        self.win = np.zeros((L, D), np.float32)

    def handshake(self):
        return L, self.m.pred_len, self.D, self.m.period

    def push(self, x):
        self.win = np.roll(self.win, -1, axis=0)
        self.win[-1] = x

    def forecast(self):
        with torch.no_grad():
            y = self.m(torch.from_numpy(self.win[None]))
        return y[0].numpy().astype(np.float64), 0


class SerialBoard:
    """main.c.stream over the ST-LINK virtual COM port (binary protocol, see the firmware)."""

    def __init__(self, port, D, H, P):
        import serial                             # pip install pyserial (only needed here)
        self.s = serial.Serial(port, 115200, timeout=10)
        self.D, self.H, self.P = D, H, P
        time.sleep(0.2)
        self.s.reset_input_buffer()

    def _read(self, n):
        b = self.s.read(n)
        if len(b) != n:
            raise TimeoutError(f"board sent {len(b)}/{n} bytes")
        return b

    def handshake(self):
        self.s.write(b"P")
        if self._read(1) != b"p":
            raise RuntimeError("no handshake - is main.c.stream flashed?")
        return struct.unpack("<4H", self._read(8))

    def push(self, x):
        self.s.write(b"S" + np.asarray(x, "<f4").tobytes())
        if self._read(1) != b"k":
            raise RuntimeError("sample not acknowledged")

    def forecast(self):
        self.s.write(b"F")
        if self._read(1) != b"Y":
            raise RuntimeError("bad forecast tag")
        raw = np.frombuffer(self._read(4 * self.H * self.D), "<f4")
        cyc = struct.unpack("<I", self._read(4))[0]
        # emission order (channel d, output period om, phase ph) -> (H, D)
        y = raw.reshape(self.D, self.H // self.P, self.P).reshape(self.D, self.H).T
        return y.astype(np.float64), cyc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="MetroPT3")
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--every", type=int, default=60, help="forecast every N steps (window stride)")
    ap.add_argument("--sim", action="store_true", help="PyTorch stands in for the board")
    ap.add_argument("--port", default=None, help="board COM port, e.g. COM3")
    ap.add_argument("--realtime", type=float, default=0.0,
                    help="seconds of wall time per step (0 = as fast as the link allows)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N steps (0 = whole split)")
    a = ap.parse_args()
    if a.sim == bool(a.port):
        ap.error("give exactly one of --sim / --port")

    worker = load_worker()
    with contextlib.redirect_stdout(io.StringIO()):
        args = build_args(worker, "mix", a.data, a.H, a.seed)
        df, _, _ = worker.data_factory.data_select(a.data, args.root_path)
        test, _ = worker.data_factory.data_provider(
            args.root_path, a.data, args.features, args.batch_size, 96, args.label_len, a.H, "test")
    D = df.shape[1] - 1
    m = worker.get_model("FEATHer", args, n_features=D, seq_len=96, pred_len=a.H).float()
    ckpt = worker.checkpoint_path(args.exp_tag, "FEATHer", a.data, a.H, 96, a.seed)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    if m.period != 24:
        raise SystemExit("export_golden_feather2.py exports period 24 only (MetroPT3, Steel, EWELD, N2O)")

    # the stream = the test split's input series; window i covers stream[i : i + L]
    X = np.stack([np.asarray(test[i][0])[0] for i in range(len(test))]
                 + list(np.asarray(test[len(test) - 1][0])[1:]))[:, :D].astype(np.float32)
    n_win = len(test)
    assert np.array_equal(X[n_win - 1:n_win - 1 + L], np.asarray(test[n_win - 1][0])[:, :D].astype(np.float32))

    board = TorchBoard(m, D) if a.sim else SerialBoard(a.port, D, a.H, m.period)
    geo = tuple(int(v) for v in board.handshake())
    if geo != (L, a.H, D, m.period):
        raise SystemExit(f"board geometry {geo} != expected {(L, a.H, D, m.period)} - re-run board_prep.py")

    rows, se_b = [], [0.0, 0]
    t_wall = time.time()
    steps = len(X) if not a.max_steps else min(a.max_steps, len(X))
    for t in range(steps):
        t0 = time.time()
        board.push(X[t])
        i = t - L + 1                                   # window index ending at step t
        if i >= 0 and i < n_win and i % a.every == 0:
            yb, cyc = board.forecast()
            bx, by, _, _ = test[i]
            truth = np.asarray(by, np.float64)[-a.H:, :D]
            with torch.no_grad():
                yt = m(torch.from_numpy(np.asarray(bx, np.float32)[None, :, :D]))[0].numpy().astype(np.float64)
            fin = np.isfinite(truth)
            e_b = np.where(fin, (yb - truth) ** 2, 0.0)
            e_t = np.where(fin, (yt - truth) ** 2, 0.0)
            se_b[0] += e_b.sum(); se_b[1] += int(fin.sum())
            rows.append(dict(window=i, cycles=cyc, latency_ms=cyc / CPU_HZ * 1e3,
                             maxabs_vs_torch=float(np.abs(yb - yt).max()),
                             se_board=e_b.sum(), se_torch=e_t.sum(), n_obs=int(fin.sum())))
        if a.realtime > 0:
            time.sleep(max(0.0, a.realtime - (time.time() - t0)))
    r = pd.DataFrame(rows)
    mode = "sim" if a.sim else "board"
    out = os.path.join("results", f"board_replay_{a.data}_H{a.H}_s{a.seed}_{mode}_every{a.every}.csv")
    r.to_csv(out, index=False)

    mse_b = r.se_board.sum() / r.n_obs.sum()
    mse_t = r.se_torch.sum() / r.n_obs.sum()
    print(f"{a.data} H={a.H} seed={a.seed} mode={mode}: {steps} steps streamed, {len(r)} forecasts "
          f"(every {a.every} steps) in {time.time() - t_wall:.1f} s wall")
    print(f"  MSE board {mse_b:.6f} | PyTorch same windows {mse_t:.6f} | max |board - PyTorch| "
          f"{r.maxabs_vs_torch.max():.2e}")
    if a.every == 1 and steps == len(X):
        ref = pd.read_csv("results/_archive/pre_raw_20261007/feather_mfg_candR4.csv")
        ref = ref[(ref.exp_tag == "candR4") & (ref.data == a.data) & (ref.pred_len == a.H) & (ref.seed == a.seed)]
        print(f"  all {n_win} windows: CSV test MSE {ref.MSE.iloc[0]:.6f}  (rel diff {abs(mse_b / ref.MSE.iloc[0] - 1):.1e})")
    if not a.sim:
        lat = r.latency_ms
        dt_s = SAMPLING_S[a.data] * a.every               # time between forecasts in the field
        duty = lat.mean() / 1e3 / dt_s
        e_fc = V * I_ACTIVE_mA * 1e-3 * lat.mean() / 1e3   # J per forecast
        per_day = 86400 / dt_s
        print(f"  latency mean {lat.mean():.1f} ms, max {lat.max():.1f} ms | one forecast every {dt_s:.0f} s "
              f"in the field -> duty cycle {duty * 100:.3f} %")
        print(f"  energy/day: compute {e_fc * per_day:.2f} J ({e_fc * per_day / 3.6:.2f} mWh); "
              f"with the NUCLEO sleep current {V * I_SLEEP_mA * 1e-3 * 86400 * (1 - duty) + e_fc * per_day:.0f} J "
              "(board upper bound)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
