# -*- coding: utf-8 -*-
"""Streaming replay of a test split through the board.

The deployed loop, end to end: the PC plays the sensors - it sends the test split of
one dataset to the NUCLEO one time step at a time (exactly the scaled values the
PyTorch evaluation feeds the model) - and every `--every` steps asks for a forecast.
The board keeps the last L = 96 steps itself (main.c.replay), forecasts with the
TRAINED weights and streams the forecast back.  The PC scores it against the true
future (observed cells only, as in every table) and against the PyTorch forecast of
the same window.  The tuned pick of the model at that H in the tier (--tier 1k | chip), its replay_weights.py
checkpoint.

    # no board: PyTorch stands in for the board (checks windows, bookkeeping, scoring;
    # with --every 1 the replay must reproduce the test MSE of results/mfg_final.csv)
    python deployment/board/replay/replay.py --model FEATHer_core --data MetroPT3 --H 96 --sim --every 60 --out <scratch csv>

    # board: install the firmware first, then replay (needs `pip install pyserial`)
    python deployment/board/replay/replay_flash.py --model FEATHer_core --H 96
    python deployment/board/replay/replay.py --model FEATHer_core --data MetroPT3 --H 96 --port COM3 --every 60

    # real-time segment: one step per sampling interval (or a fraction of it)
    python deployment/board/replay/replay.py ... --port COM3 --every 60 --realtime 1.0 --max_steps 120

Writes results/board/board_replay_<model>_<data>_H<H>_s<seed>_every<every>.csv (one row per forecast, flushed; a
run that stops resumes after its last forecast: the board gets L fresh steps first).  Prints a summary: forecasts, MSE
board vs PyTorch, max |board - PyTorch|, latency, and the duty cycle / energy per day at the dataset's real sampling
interval (rough board currents: active 29.75 mA, WFI sleep 10.38 mA - the sleep figure is the whole NUCLEO board, so
energy/day is reported as compute-only and as an upper bound with the board's sleep current).  These are rough values for
this summary only; the power reported in the paper is measured with deployment/board/tools/power_run.py.
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import train as T                         # noqa: E402
from baselines import get_model           # noqa: E402
from utils import data_factory            # noqa: E402
from replay_weights import SEED, checkpoint, hparams, official, worker_args   # noqa: E402

SAMPLING_S = {"MetroPT3": 60, "Steel": 900, "EWELD": 900, "N2O": 300, "WindSCADA": 600,
              "BlastFurnace": 3600}           # sampling interval of each dataset (s)
CPU_HZ = {"f103": 72_000_000, "f411": 84_000_000}
V, I_ACTIVE_mA, I_SLEEP_mA = 3.3, 29.75, 10.38   # F103 board current, active / WFI sleep
L = 96


class TorchBoard:
    """Stands in for the board: same protocol semantics, PyTorch forward on the window."""

    def __init__(self, model, D, H, seg):
        self.m, self.D, self.H, self.seg = model, D, H, seg
        self.win = np.zeros((L, D), np.float32)

    def handshake(self):
        return L, self.H, self.D, self.seg

    def push(self, x):
        self.win = np.roll(self.win, -1, axis=0)
        self.win[-1] = x

    def forecast(self):
        with torch.no_grad():
            y = self.m(torch.from_numpy(self.win[None]))
        return y[0, -self.H:].numpy().astype(np.float64), 0


class SerialBoard:
    """main.c.replay over the ST-LINK virtual COM port (binary protocol, see the firmware)."""

    def __init__(self, port, D, H, seg):
        import serial                             # pip install pyserial (only needed here)
        self.s = serial.Serial(port, 115200, timeout=120)   # the slowest models take over 10 s per forecast
        self.D, self.H, self.seg = D, H, seg
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
            raise RuntimeError("no handshake - is main.c.replay flashed?")
        return struct.unpack("<4H", self._read(8))

    def push(self, x):
        self.s.write(b"S" + np.asarray(x, "<f4").tobytes())
        if self._read(1) != b"k":
            raise RuntimeError("sample not acknowledged")

    def forecast(self):
        self.s.write(b"F")
        if self._read(1) != b"Y":
            raise RuntimeError("bad forecast tag")
        # per channel, H x (u16 t, f32 value) in the port's own order
        raw = np.frombuffer(self._read(6 * self.H * self.D), np.dtype([("t", "<u2"), ("v", "<f4")]))
        y = np.empty((self.H, self.D), np.float32)
        for d in range(self.D):
            r = raw[d * self.H:(d + 1) * self.H]
            if not np.array_equal(np.sort(r["t"]), np.arange(self.H)):
                raise RuntimeError(f"channel {d}: time indices are not 0..{self.H - 1}")
            y[r["t"], d] = r["v"]
        cyc = struct.unpack("<Q", self._read(8))[0]    # u64 (main.c.replay)
        return y.astype(np.float64), cyc


def build_model(model, data, H, tier="chip"):
    hp = hparams(model, data, H, tier)
    sys.argv = ["train.py"] + worker_args(model, data, H, os.devnull, tier) + ["--gpu", "0"]
    args = T.parse_args()
    T._apply_overrides(args)
    with contextlib.redirect_stdout(io.StringIO()):
        df, _, _ = data_factory.data_select(data, args.root_path)
        test, _ = data_factory.data_provider(args.root_path, data, args.features, args.batch_size, 96,
                                             min(args.label_len, 96), H, "test")
    D = df.shape[1] - 1
    m = get_model(hp["model"], args, n_features=D, seq_len=96, pred_len=H).float()
    m.load_state_dict(torch.load(checkpoint(data, model, H, tier), map_location="cpu"))
    m.eval()
    hook = getattr(m, "_test_start", None)
    if hook is not None:
        hook()
    return m, test, D, 0            # SEG of the handshake: every port sends indexed values (FEATHer included)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry model of the size study")
    ap.add_argument("--data", default="MetroPT3")
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--tier", default="chip", choices=["1k", "chip"])
    ap.add_argument("--every", type=int, default=60, help="forecast every N steps (window stride)")
    ap.add_argument("--sim", action="store_true", help="PyTorch stands in for the board")
    ap.add_argument("--port", default=None, help="board COM port, e.g. COM3")
    ap.add_argument("--board", default="f103", choices=["f103", "f411"], help="CPU clock for the latency")
    ap.add_argument("--realtime", type=float, default=0.0,
                    help="seconds of wall time per step (0 = as fast as the link allows)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N steps (0 = whole split)")
    ap.add_argument("--windows", type=int, default=0,
                    help="measurement mode: N windows evenly over the test split, each preceded by only its own L steps "
                         "(the board forecasts from its last L steps, so the forecast is the same; 0 = stream it all)")
    ap.add_argument("--out", default=None, help="output CSV (required with --sim: sim results stay out of results/board)")
    a = ap.parse_args()
    if a.sim == bool(a.port):
        ap.error("give exactly one of --sim / --port")
    if a.sim and not a.out:
        ap.error("--sim needs --out (a scratch file)")

    m, test, D, seg = build_model(a.model, a.data, a.H, a.tier)
    # the stream = the test split's input series; window i covers stream[i : i + L]
    X = np.stack([np.asarray(test[i][0])[0] for i in range(len(test))]
                 + list(np.asarray(test[len(test) - 1][0])[1:]))[:, :D].astype(np.float32)
    n_win = len(test)
    assert np.array_equal(X[n_win - 1:n_win - 1 + L], np.asarray(test[n_win - 1][0])[:, :D].astype(np.float32))

    board = TorchBoard(m, D, a.H, seg) if a.sim else SerialBoard(a.port, D, a.H, seg)
    geo = tuple(int(v) for v in board.handshake())
    if geo != (L, a.H, D, seg):
        raise SystemExit(f"board geometry {geo} != expected {(L, a.H, D, seg)} - flash this cell with replay_flash.py")

    out = a.out or os.path.join("results", "board", f"board_replay_{a.tier}_{a.model}_{a.data}_H{a.H}_s{SEED}_every{a.every}.csv")
    cols = ["window", "cycles", "latency_ms", "maxabs_vs_torch", "se_board", "se_torch", "n_obs"]
    first = 0
    if os.path.exists(out):
        done = pd.read_csv(out)
        if len(done):
            first = int(done.window.iloc[-1]) + a.every      # L fresh steps fill the board's window again
            print(f"resuming {out}: {len(done)} forecasts done, next window {first}", flush=True)
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    if new:
        f.write(",".join(cols) + "\n"); f.flush(); os.fsync(f.fileno())

    def score(i):
        yb, cyc = board.forecast()
        bx, by, _, _ = test[i]
        truth = np.asarray(by, np.float64)[-a.H:, :D]
        with torch.no_grad():
            yt = m(torch.from_numpy(np.asarray(bx, np.float32)[None, :, :D]))[0, -a.H:].numpy().astype(np.float64)
        fin = np.isfinite(truth)
        e_b = np.where(fin, (yb - truth) ** 2, 0.0)
        e_t = np.where(fin, (yt - truth) ** 2, 0.0)
        f.write(",".join(repr(float(v)) if k else str(int(v)) for k, v in enumerate(
            (i, cyc, cyc / CPU_HZ[a.board] * 1e3, np.abs(yb - yt).max(), e_b.sum(), e_t.sum(), int(fin.sum())))) + "\n")
        f.flush(); os.fsync(f.fileno())

    t_wall = time.time()
    steps = len(X) if not a.max_steps else min(a.max_steps, len(X))
    if a.windows:
        done_w = set(pd.read_csv(out).window) if not new else set()
        for i in sorted({int(round(v)) for v in np.linspace(0, n_win - 1, a.windows)} - done_w):
            for t in range(i, i + L):                   # window i = stream[i : i + L]
                board.push(X[t])
            score(i)
        steps = first = 0
    for t in range(first, steps):
        t0 = time.time()
        board.push(X[t])
        i = t - L + 1                                   # window index ending at step t
        if i >= first and i < n_win and i % a.every == 0:
            score(i)
        if a.realtime > 0:
            time.sleep(max(0.0, a.realtime - (time.time() - t0)))
    f.close()
    r = pd.read_csv(out)
    mode = "sim" if a.sim else "board"
    mse_b = r.se_board.sum() / r.n_obs.sum()
    mse_t = r.se_torch.sum() / r.n_obs.sum()
    print(f"{a.tier} {a.model} {a.data} H={a.H} seed={SEED} mode={mode}: {steps - first} steps streamed, {len(r)} forecasts "
          f"(every {a.every} steps) in {time.time() - t_wall:.1f} s wall")
    print(f"  MSE board {mse_b:.6f} | PyTorch same windows {mse_t:.6f} | max |board - PyTorch| "
          f"{r.maxabs_vs_torch.max():.2e}")
    if a.every == 1 and steps == len(X):
        ref = official(a.model, a.data, a.H, a.tier).MSE
        print(f"  all {n_win} windows: CSV test MSE {ref:.6f}  (rel diff {abs(mse_b / ref - 1):.1e})")
    if not a.sim:
        lat = r.latency_ms
        dt_s = SAMPLING_S[a.data] * a.every               # time between forecasts in the field
        duty = lat.mean() / 1e3 / dt_s
        e_fc = V * I_ACTIVE_mA * 1e-3 * lat.mean() / 1e3   # J per forecast
        per_day = 86400 / dt_s
        print(f"  latency mean {lat.mean():.1f} ms, min {lat.min():.1f} ms, max {lat.max():.1f} ms | one forecast every "
              f"{dt_s:.0f} s in the field -> duty cycle {duty * 100:.3f} %")
        if a.board == "f103":                             # the current was measured on the F103 only
            print(f"  energy/day: compute {e_fc * per_day:.2f} J ({e_fc * per_day / 3.6:.2f} mWh); "
                  f"with the NUCLEO sleep current {V * I_SLEEP_mA * 1e-3 * 86400 * (1 - duty) + e_fc * per_day:.0f} J "
                  "(board upper bound)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
