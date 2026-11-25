# -*- coding: utf-8 -*-
"""Supply current of the F103 with the X-NUCLEO-LPM01A (PowerShield) in place of JP6 (IDD).

Wiring: JP6 off, CN14 pin 3 (VOUT) -> JP6 MCU side, CN14 pin 1 -> GND; ST-LINK still on its own USB.
For each chip pick (MetroPT3 H96 / H720, the real-time cells) the harness is built with --power active
(back-to-back inference, random golden weights), flashed, and the PowerShield gives 10 static-mode
readings at 3.3 V.  Idle (WFI) once.  Resumable.

    python deployment/board/tools/power_run.py --lpm COM6 --out results/_incoming/.../board_current_lpm01a.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
import statistics as st
import tempfile
import time
from pathlib import Path

import serial

import board_prep as bp
import board_run as br
import chip_verdict as cv

MODELS = "FEATHer_core,FITS,MixLinear,TimeBase,PhaseFormer,CMoS,SparseTSF"
COLS = ["rep", "mode", "tier", "model", "trial", "data", "H", "D", "volt_V", "n", "static_errors", "mean_mA", "sd_mA",
        "min_mA", "max_mA", "flash_B", "timestamp"]
SAMPLE = re.compile(r"^(\d{3,4})-(\d{2})$")


class Shield:
    """Static mode (the 10 mA ceiling of dynamic mode is below the ~30 mA active current): one averaged
    value per 'start', so N starts give N values.  Static mode must be set before the output turns on."""

    def __init__(self, port: str):
        self.s = serial.Serial(port, 3686400, timeout=1)
        for c in ("htc", "acqmode stat", "volt 3300m", "trigdelay 1", "pwrend on"):
            self.cmd(c)

    def cmd(self, c: str) -> str:
        self.s.reset_input_buffer()
        self.s.write((c + "\n").encode())
        time.sleep(0.5)
        return self.s.read(self.s.in_waiting or 1).decode(errors="replace")

    def acquire(self, n: int = 10) -> list[float]:
        vals, self.raw, self.errors = [], "", 0
        for _ in range(n):
            self.s.reset_input_buffer()
            self.s.write(b"start\n")
            buf, t0 = "", time.time()
            while time.time() - t0 < 6 and "Acquisition completed" not in buf:
                buf += self.s.read(self.s.in_waiting or 1).decode(errors="replace")
            self.raw += buf
            self.errors += "error" in buf.lower()
            vals += [int(m[1]) * 10.0 ** -int(m[2]) * 1e3 for line in buf.replace("\r", "").split("\n")
                     if (m := SAMPLE.match(line.strip()))]
        return vals

    def close(self):
        self.cmd("hrc")
        self.s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lpm", default="COM6")
    ap.add_argument("--port", default="COM3", help="ST-LINK virtual COM of the F103 (picks the programmer)")
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=1, help="whole sweep repeated (each cell built and flashed again)")
    ap.add_argument("--test", action="store_true", help="one acquisition on whatever runs now, print and stop")
    a = ap.parse_args()
    sh = Shield(a.lpm)
    if a.test:
        mA = sh.acquire(5)
        print([round(v, 3) for v in mA], "errors", sh.errors)
        return sh.close()
    cells = [c for c in br.tuned_cells(a.models.split(","), [96, 720], ["chip"]) if c["data"] == "MetroPT3"]
    jobs = [(rep, m, c) for rep in range(1, a.reps + 1) for m, c in [("active", c) for c in cells] + [("idle", cells[0])]]
    done = set()
    if a.out.exists():
        with a.out.open(encoding="utf-8") as f:
            done = {(int(r["rep"]), r["mode"], r["model"], r["trial"], int(r["H"])) for r in csv.DictReader(f)}
    sh.acquire(1)                                            # power the target so the programmer can attach
    with tempfile.TemporaryDirectory() as tmp:
        proj, hal = Path(tmp) / "proj", Path(tmp) / "hal"
        shutil.copytree(bp.DEFAULT_PROJECT, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        for rep, mode, c in jobs:
            if (rep, mode, c["model"], c["trial"], c["H"]) in done:
                continue
            D = cv.DATA_D[c["data"]]
            bp.install(proj, c["bm"], c["H"], D, mode, config=c["config"], geo=c["geo"],
                       stream_ch=c["bm"] in bp.STREAM_CH, data=c["data"] if c["bm"] in bp.CONFIG_MODELS else None)
            elf, fl, _ = br.build(proj, hal)
            br.flash_board(elf, a.port)
            time.sleep(3)
            mA = sh.acquire()
            if len(mA) < 8:
                print(f"{mode} {c['model']} H={c['H']}: only {len(mA)} samples\n{sh.raw[-400:]}", flush=True)
                continue
            row = dict(rep=rep, mode=mode, tier=c["tier"], model=c["model"], trial=c["trial"], data=c["data"], H=c["H"], D=D,
                       volt_V=3.3, n=len(mA), static_errors=sh.errors, mean_mA=f"{st.mean(mA):.3f}", sd_mA=f"{st.stdev(mA):.3f}",
                       min_mA=f"{min(mA):.3f}", max_mA=f"{max(mA):.3f}", flash_B=fl,
                       timestamp=dt.datetime.now().isoformat(timespec="seconds"))
            new = not a.out.exists()
            with a.out.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"rep{rep} {mode:<6} {c['model']:<13} {c['trial']} H={c['H']:<3} {row['mean_mA']} mA (sd {row['sd_mA']})",
                  flush=True)
    sh.close()


if __name__ == "__main__":
    main()
