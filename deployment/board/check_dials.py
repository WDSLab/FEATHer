# -*- coding: utf-8 -*-
"""Golden tests of the chip-tier structural values (10-14) - C vs PyTorch on random weights.

`make test` covers the frozen / default exports only (one golden per port).  The search space
(configs/experiments/size.yaml) also needs every structural value the ports can be built at:
FEATHer num_bands / point_kernel / use_norm / gate none / mlp_rank / head_seg / agg_shared (mix
and CI), TimeBase use_period_norm = 0, DiPE individual_* and use_freq_w / use_time_w = False.
Each case exports its golden, builds the port's x86 test with the matching macros (WSL) and runs
it; golden/ is restored afterwards so `make test` is unaffected.  Cases run one at a time
(golden/ is shared).

    python deployment/board/check_dials.py              # every case
    python deployment/board/check_dials.py --only dipe  # cases whose label contains "dipe"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_prep as bp  # noqa: E402


def _flags(lines: list[str]) -> list[str]:
    """'#define X v' -> '-DX=v'"""
    return ["-D{}={}".format(*ln.split()[1:3]) for ln in lines]


def fea(ci: bool, H: int, D: int, P: int = 24, **knobs):
    geo = {"period": P, **knobs}
    args = ["--pred_len", str(H), "--d_model", str(D), "--period", str(P)] + bp.fea2_export_args(geo)
    width = ["-DFEA2_D=1", f"-DFEA2CI_D={D}"] if ci else [f"-DFEA2_D={D}"]
    flags = width + [f"-DFEA2_H={H}", f"-DFEA2_PERIOD={P}",
                     f"-DFEA2_MLP_R={bp.fea2_knobs(geo)['mlp_rank']}"] + _flags(bp.fea2_macros(geo))
    tag = " ".join(f"{k}={v}" for k, v in knobs.items())
    if ci:
        return (f"FEATHer-CI  H={H} D={D} {tag}", "feather2ci", args, [("feather2ci", "FEA2_GEOM", flags)])
    return (f"FEATHer-mix H={H} D={D} {tag}", "feather2", args,
            [("feather2", "FEA2_GEOM", flags), ("feather2r", "FEA2_GEOM", flags)])


def cfg(model: str, stem: str, macro: str, H: int, D: int, config: str):
    args = ["--pred_len", str(H), "--d_model", str(D), "--config", config, "--data", "Steel"]
    return (f"{model} H={H} D={D} {config}", stem, args,
            [(stem, f"{macro}_GEOM", [f"-D{macro}_H={H}", f"-D{macro}_D={D}"])])


CASES = (
    [fea(ci, H, D, **k) for ci in (False, True) for H, D, k in (
        (96, 5, dict(num_bands=1)), (720, 5, dict(num_bands=2)),
        (96, 4, dict(point_kernel=1)), (720, 4, dict(point_kernel=5)),
        (96, 6, dict(use_norm=0)), (720, 6, dict(gate_mode="none")),
        (96, 3, dict(mlp_rank=2)), (720, 7, dict(mlp_rank=9)),
        (96, 5, dict(num_bands=2, gate_mode="none")),
        (720, 3, dict(num_bands=1, point_kernel=7, use_norm=0, mlp_rank=2, head_seg=12, agg_shared=1)))]
    + [fea(True, 720, 5, head_seg=12, agg_shared=1), fea(True, 96, 4, head_seg=8)]
    + [cfg("TimeBase", "timebase", "TB", H, D, c) for H, D, c in (
        (96, 5, "ETTh1+timebase_use_period_norm=0"),
        (720, 5, "ETTh1+timebase_use_period_norm=0+timebase_individual=1"),
        (720, 3, "ETTh1+timebase_use_period_norm=0+timebase_basis_num=3+timebase_period_len=12"))]
    + [cfg("DiPE", "dipe", "DIPE", H, D, c) for H, D, c in (
        (96, 5, "default+dipe_individual_c=1"), (720, 5, "default+dipe_individual_f=1"),
        (96, 4, "default+dipe_individual_t=1"), (720, 4, "default+dipe_use_freq_w=0"),
        (96, 6, "default+dipe_use_time_w=0"),
        (720, 3, "default+dipe_individual_c=1+dipe_individual_f=1+dipe_individual_t=1+dipe_num_experts=3"),
        (96, 5, "default+dipe_num_experts=2+dipe_individual_f=1+dipe_use_time_w=0+dipe_use_revin=0"),
        (720, 6, "default+dipe_use_freq_w=0+dipe_use_time_w=0+dipe_individual_c=1"))])


def run_case(label, stem, args, builds) -> list[tuple[str, bool, str]]:
    r = subprocess.run([sys.executable, str(HERE / f"export_golden_{stem}.py")] + args, cwd=bp.ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    if r.returncode != 0:
        return [(f"{label} [export]", False, (r.stdout + r.stderr)[-400:])]
    out = []
    for target, var, flags in builds:
        w = bp.wsl(f"make -B {target} {var}='{' '.join(flags)}' 2>&1", HERE)
        ok = w.returncode == 0 and "FAIL" not in w.stdout
        out.append((f"{label} [{target}]", ok, "" if ok else w.stdout[-600:]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run the cases whose label contains this (case-insensitive)")
    a = ap.parse_args()
    cases = [c for c in CASES if a.only.lower() in c[0].lower()]
    res = []
    with tempfile.TemporaryDirectory() as tmp:
        bak = Path(tmp) / "golden"
        shutil.copytree(bp.GOLDEN, bak)
        try:
            for c in cases:
                for name, ok, why in run_case(*c):
                    res.append(ok)
                    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)
                    if why:
                        print("      " + why.replace("\n", "\n      "))
        finally:     # restore file by file (Dropbox: never remove the directory)
            for f in bak.iterdir():
                shutil.copy2(f, bp.GOLDEN / f.name)
            for f in bp.GOLDEN.iterdir():
                if not (bak / f.name).exists():
                    f.unlink()
    print(f"\n{sum(res)}/{len(res)} PASS")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
