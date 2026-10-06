# -*- coding: utf-8 -*-
"""Build + flash the streaming replay firmware (main.c.stream) with a trained checkpoint (2026-10-06).

Same toolchain, flags and scratch-copy discipline as board_run.py (the CubeIDE project itself is
never modified); then drive the board with replay.py.

    python deployment/board/stream_flash.py --H 96 \
        --ckpt results/checkpoints/candR4/FEATHer/MetroPT3_H96_L96_s2025.pth
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import board_prep as bp
import board_run as br


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--D", type=int, default=7)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--project", type=Path, default=bp.DEFAULT_PROJECT)
    a = ap.parse_args()
    ckpt = str(Path(a.ckpt).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        shutil.copytree(a.project, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        bp.install(proj, "feather_mix", a.H, a.D, "none", stream_ckpt=ckpt)
        elf, fl, ram = br.build(proj, Path(tmp) / "hal")
        br.flash_board(elf)
    print(f"flashed streaming FEATHer-mix H={a.H} D={a.D} ({Path(ckpt).name}): "
          f"flash {fl / 1024:.1f} KB, RAM {ram / 1024:.2f} KB")


if __name__ == "__main__":
    main()
