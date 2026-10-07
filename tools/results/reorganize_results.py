"""One-time reorganisation of results/ (2026-10-07).  Dry run by default; --apply moves and rewrites.

results/
  raw_mfg.csv  raw_ltsf.csv  (select_mfg.csv)      the managed result files (tools/results/build_raw.py)
  board/      chip and board measurements
  analysis/   tables computed from other results
  dev/        dev-loop experiments (not paper numbers)
  _incoming/  new run shards before they are merged into raw_*.csv
  _archive/pre_raw_20261007/   the CSVs that raw_*.csv was built from (and the old merge outputs)

Files are moved with os.replace (never deleted).  Code references "results/<name>" (and "{R}/<name>",
"results" / "<name>") are rewritten to the new location.  Tools that still read the pre-raw files
(tables_cf.py, merge_raw.py, table_deploy.py, ...) now read them from the archive folder, so the
current manuscript tables rebuild with identical numbers until they are ported to raw_*.csv.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
ARC = "_archive/pre_raw_20261007"

MOVE = {
    "board": ["board_current_20260929.csv", "board_footprint.csv", "board_footprint_r4_frozen.csv",
              "board_footprint_recompute_fair.csv", "board_footprint_recompute_r4.csv",
              "board_footprint_rsweep_1004.csv", "board_latency_20260929.csv",
              "board_replay_MetroPT3_H720_s2025_board_every60.csv",
              "board_replay_MetroPT3_H96_s2025_board_every60.csv",
              "board_replay_MetroPT3_H96_s2025_sim_every1.csv"],
    "analysis": ["complexity.csv", "ops_vs_latency_20261006.csv", "sram_vs_lookback_20261006.csv",
                 "coupling_vs_mixing_20261006.csv", "kpi_steel.csv", "linear_ceiling.csv",
                 "linear_ceiling_mix.csv", "mfg_r2.csv", "ltsf_r2.csv"],
    "dev": ["dev_lookback.csv", "dev_nearfar.csv", "dev_sensor_dropout.csv"],
    ARC: ["mfg_baselines.csv", "ltsf_baselines.csv", "feather_mfg.csv", "feather_ltsf.csv",
          "feather_mfg_candR4.csv", "feather_mfg_candR4_ci.csv", "feather_ltsf_candR4.csv",
          "feather_ltsf_candR4_mix.csv", "feather_mfg_lookback_L192.csv", "feather_mfg_lookback_L336.csv",
          "feather_ci_mfg_lookback_L192.csv", "feather_ci_mfg_lookback_L336.csv",
          "stsf_mlp_mfg_lookback_L192.csv", "stsf_mlp_mfg_lookback_L336.csv", "cmos_mfg.csv",
          "new3_mfg.csv", "timebase_lr_mfg.csv", "ablation.csv", "control_sparsetsf_mix.csv",
          "control_front_none.csv", "mfg_raw.csv", "ltsf_raw.csv"],
}
DIRS = ["board", "analysis", "dev", "_incoming", ARC]

# dynamic paths that a plain filename replacement cannot see
EXTRA = [
    ("tools/paper/table_deploy.py", 'f"{R}/{group}_raw.csv"', f'f"{{R}}/{ARC}/{{group}}_raw.csv"'),
    ("tools/paper/merge_raw.py", 'f"{R}/{name}_baselines.csv"', f'f"{{R}}/{ARC}/{{name}}_baselines.csv"'),
    ("tools/paper/merge_raw.py", 'f"{R}/feather_{name}.csv"', f'f"{{R}}/{ARC}/feather_{{name}}.csv"'),
    ("tools/paper/merge_raw.py", 'f"{R}/{name}_raw.csv"', f'f"{{R}}/{ARC}/{{name}}_raw.csv"'),
    ("manuscript/version/_build/tables_cf.py", 'f"results/{f}_mfg_lookback_L{L}.csv"',
     f'f"results/{ARC}/{{f}}_mfg_lookback_L{{L}}.csv"'),
    ("run_forecast.py", 'ABLATION_CSV = "results/ablation.csv"', 'ABLATION_CSV = "results/_incoming/ablation.csv"'),
    ("tools/dev/dev_summary.py", '"results/dev_loop.csv"', '"results/dev/dev_loop.csv"'),
    ("tools/dev/run_dev.py", '"results/dev_loop.csv"', '"results/dev/dev_loop.csv"'),
    ("tools/results/build_raw.py", 'R = os.path.join(ROOT, "results")',
     f'R = os.path.join(ROOT, "results")\nSRC = os.path.join(R, *"{ARC}".split("/"))   # pre-raw CSVs (moved 10-07)'),
    ("tools/results/build_raw.py", "p = os.path.join(R, name)", "p = os.path.join(SRC, name)"),
]
SKIP_DIRS = {"_archive", "results", ".git", "__pycache__", "node_modules"}
EXTS = (".py", ".sh", ".cmd", ".ps1")


def code_files():
    for d, sub, files in os.walk(ROOT):
        sub[:] = [s for s in sub if s not in SKIP_DIRS and not s.endswith("-main")]
        for f in files:
            if f.endswith(EXTS):
                yield os.path.join(d, f)


def main(apply):
    plan = [(n, dst) for dst, names in MOVE.items() for n in names]
    missing = [n for n, _ in plan if not os.path.exists(os.path.join(RES, n))]
    if missing:
        print("NOT FOUND (skipped):", missing)
    # code rewrites
    edits = {}
    for path in code_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        if rel == "tools/results/reorganize_results.py":
            continue
        s0 = s = open(path, encoding="utf-8").read()
        for n, dst in plan:
            s = re.sub(rf"(results/(?:\.\./)*){re.escape(n)}\b", rf"\g<1>{dst}/{n}", s)   # results/<n>
            s = s.replace(f"{{R}}/{n}", f"{{R}}/{dst}/{n}")                                 # f"{R}/<n>"
            s = s.replace(f'"results" / "{n}"', f'"results" / "{dst}" / "{n}"')              # pathlib
        for f, a, b in EXTRA:
            if rel == f:
                s = s.replace(a, b)
        if s != s0:
            edits[rel] = (path, s, sum(1 for x, y in zip(s0.splitlines(), s.splitlines()) if x != y))
    for rel, (_, _, k) in sorted(edits.items()):
        print(f"  edit {rel}  ({k} lines)")
    for n, dst in plan:
        if n not in missing:
            print(f"  move results/{n} -> results/{dst}/{n}")
    if not apply:
        print("(dry run; --apply to move and rewrite)")
        return
    for d in DIRS:
        os.makedirs(os.path.join(RES, d), exist_ok=True)
    for n, dst in plan:
        if n not in missing:
            os.replace(os.path.join(RES, n), os.path.join(RES, dst, n))
    for rel, (path, s, _) in edits.items():
        open(path, "w", encoding="utf-8", newline="").write(s)
    print("applied.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
