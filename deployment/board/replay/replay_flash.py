# -*- coding: utf-8 -*-
"""Build (and flash) the streaming replay firmware (main.c.replay) with a model's trained checkpoint.

Same toolchain, flags and scratch-copy discipline as board_run.py (the CubeIDE project itself is
never modified); then drive the board with replay.py.

    python deployment/board/replay/replay_flash.py --model FEATHer_core --H 96              # checkpoint of replay_weights.py
    python deployment/board/replay/replay_flash.py --model FEATHer_core --H 96 --build_only # no board needed
"""
from __future__ import annotations

import argparse
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import board_prep as bp
import board_run as br
import chip_verdict as cv
from replay_weights import checkpoint, pick


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry model of the size study")
    ap.add_argument("--data", default="MetroPT3")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--tier", default="chip", choices=["1k", "chip"])
    ap.add_argument("--ckpt", default=None, help="default: replay_weights.py's checkpoint")
    ap.add_argument("--board", default="f103", choices=["f103", "f411"], help="f411: its own project and flags")
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--build_only", action="store_true", help="report flash / RAM, do not flash")
    ap.add_argument("--port", default=None, help="flash the board behind this COM port (needed with two boards)")
    a = ap.parse_args()
    if a.board == "f411":
        br.use_f411()
    project = a.project or (br.F411["project"] if a.board == "f411" else bp.DEFAULT_PROJECT)
    flash_max, ram_max = br.BOARD_MEM[a.board]
    ckpt = str(Path(a.ckpt or checkpoint(a.data, a.model, a.H, a.tier)).resolve())
    c = br.cell(a.tier, a.model, a.data, a.H, pick(a.model, a.data, a.H, a.tier))
    trial, bm = c["trial"], c["bm"]
    D = cv.DATA_D[a.data]
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        shutil.copytree(project, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        bp.install(proj, bm, a.H, D, "none", replay_ckpt=ckpt, config=c["config"], geo=c["geo"],
                   data=a.data if bm in bp.CONFIG_MODELS else None)
        if a.board == "f411":
            br.splice_main(proj)
        elf, fl, ram = br.build(proj, Path(tmp) / "hal")
        fits = fl <= flash_max and ram <= ram_max
        print(f"{a.tier} {a.model} {trial} {a.data} H={a.H} D={D}: flash {fl / 1024:.1f} KB, RAM {ram / 1024:.2f} KB "
              f"(flash_B={fl} ram_B={ram})"
              f"{'' if fits else '  DOES NOT FIT'}", flush=True)
        if not fits:
            sys.exit(1)
        if not a.build_only:
            br.flash_board(elf, a.port)
            print("flashed", flush=True)


if __name__ == "__main__":
    main()
