"""Plan-driven runner (2026-10-07): run only the cells the protocol needs and the results do not have yet.

    python run_plan.py --stage select --check                 # what is missing, nothing runs
    python run_plan.py --stage select --ngpu 1 --jobs_per_gpu 3
    python run_plan.py --stage ltsf --ngpu 2 --jobs_per_gpu 3   (server)
    python tools/results/merge_incoming.py                    # shards -> raw_*.csv / select_mfg.csv

Stages (EXPERIMENT_PROTOCOL.md):
    ltsf      every model x 8 LTSF datasets x 4 H x 3 seeds, L 96, official settings      -> raw_ltsf.csv
    select    industrial candidates x 6 datasets x 4 H x seed 2025, L 96                   -> select_mfg.csv
    main      industrial chosen setting x 3 seeds, L 96 (models without a choice are skipped) -> raw_mfg.csv
    lookback  industrial chosen setting x 3 seeds, L 48 / 192 / 336 (chip models + FEATHer) -> raw_mfg.csv

A cell is done when raw_*.csv, select_mfg.csv or a not-yet-merged shard in results/_incoming/ holds a row
with the same (role, config_id, model, data, pred_len, seq_len, seed).  New rows are written to
results/_incoming/<host>_<stage>.csv with exp_tag "<role>|<config_id>"; merge_incoming.py splits it back.
"""
import argparse
import glob
import os
import socket
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "results"))
import protocol as P                                   # noqa: E402
from utils.dispatch_queue import run_on_gpus           # noqa: E402

RES = os.path.join(ROOT, "results")
KEY = ["role", "config_id", "model", "data", "pred_len", "seq_len", "seed"]


def plan(stage, models=None):
    """List of cells: dict(role, config_id, model, group, data, pred_len, seq_len, seed)."""
    cells = []
    add = lambda **c: cells.append(c)
    if stage == "ltsf":
        for m in (models or P.FEATHER + P.LIGHT + P.HEAVY):
            for d in P.LTSF:
                for h in P.HS:
                    for s in P.SEEDS:
                        add(role="reference" if m in P.HEAVY else "main", config_id=P.ltsf_config_id(m, d),
                            model=m, group="ltsf", data=d, pred_len=h, seq_len=96, seed=s)
    elif stage == "select":
        for m in (models or list(P.CANDIDATES)):
            cands = P.CANDIDATES[m]
            if len(cands) < 2:
                continue
            for c in cands:
                for d in P.MFG:
                    for h in P.HS:
                        add(role="select", config_id="cand:" + c, model=m, group="mfg", data=d, pred_len=h,
                            seq_len=96, seed=P.SELECT_SEED)
    elif stage in ("main", "lookback"):
        pool = models or (P.FEATHER + P.LIGHT + P.HEAVY if stage == "main" else P.FEATHER + P.ON_CHIP + ["CycleNet", "DLinear"])
        Ls = [96] if stage == "main" else [48, 192, 336]
        for m in pool:
            cfg = P.mfg_main_config_id(m)
            if cfg is None:
                print(f"  [skip] {m}: industrial setting not chosen yet (run --stage select first)")
                continue
            role = "reference" if m in P.HEAVY else ("main" if stage == "main" else "lookback")
            for L in Ls:
                for d in P.MFG:
                    for h in P.HS:
                        for s in P.SEEDS:
                            add(role=role, config_id=cfg, model=m, group="mfg", data=d, pred_len=h, seq_len=L, seed=s)
    else:
        raise ValueError(stage)
    return cells


def done_set():
    frames = []
    for f in ["raw_mfg.csv", "raw_ltsf.csv", "select_mfg.csv"]:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, usecols=lambda c: c in KEY))
    for p in glob.glob(os.path.join(RES, "_incoming", "*.csv")):
        d = pd.read_csv(p, on_bad_lines="skip")
        if "exp_tag" in d and len(d):
            sp = d.exp_tag.astype(str).str.split("|", n=1, expand=True)
            if sp.shape[1] == 2:
                d["role"], d["config_id"] = sp[0], sp[1]
                frames.append(d[KEY])
    if not frames:
        return set()
    a = pd.concat(frames, ignore_index=True).dropna(subset=KEY)
    # a main run of a candidate setting at the selection seed is also its selection run (same cell)
    m = a[(a.role == "main") & a.config_id.str.startswith("cand:") & (a.seed == P.SELECT_SEED) & (a.seq_len == 96)]
    a = pd.concat([a, m.assign(role="select")], ignore_index=True)
    return set(zip(a.role, a.config_id, a.model, a.data, a.pred_len.astype(int), a.seq_len.astype(int),
                   a.seed.astype(int)))


def jobs_for(missing, args, out_csv):
    groups = {}
    for c in missing:
        k = (c["role"], c["config_id"], c["model"], c["group"], c["data"], c["pred_len"], c["seq_len"])
        groups.setdefault(k, []).append(c["seed"])
    jobs = []
    for (role, cfg, m, g, d, h, L), seeds in sorted(groups.items(), key=lambda kv: (kv[0][4] != "MetroPT3", kv[0])):
        lr, loss, ov = P.resolve(m, g, d, h, cfg)
        ovs = ";".join(f"{k}={v}" for k, v in sorted(ov.items()))
        cmd = [sys.executable, "scripts/benchmarks/run_forecast.py", "--model", P.registry_name(m),
               "--data", d, "--pred_len", str(h), "--seq_len", str(L),
               "--seeds", ",".join(str(s) for s in sorted(seeds)), "--exp_tag", f"{role}|{cfg}",
               "--results_csv", out_csv, "--num_epochs", "50", "--patience", "10", "--batch_size", "32",
               "--lr", str(lr), "--loss", loss, "--result_model_name", m]
        if ovs:
            cmd += ["--model_overrides", ovs]
        jobs.append((f"{role}|{cfg} | {m} | {d} | H={h} | L={L} | seeds={sorted(seeds)} | lr={lr}", cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ltsf", "select", "main", "lookback"])
    ap.add_argument("--models", default="", help="comma list (default: the stage's pool)")
    ap.add_argument("--data", default="", help="comma list of datasets (default: all of the stage)")
    ap.add_argument("--configs", default="", help="comma list of config_ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N worker invocations (testing)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--jobs_per_gpu", type=int, default=3)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0])
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()] or None
    cells = plan(a.stage, models)
    if a.data:
        keep = {x.strip() for x in a.data.split(",")}
        cells = [c for c in cells if c["data"] in keep]
    if a.configs:
        keep = {x.strip() for x in a.configs.split(",")}
        cells = [c for c in cells if c["config_id"] in keep]
    done = done_set()
    missing = [c for c in cells if tuple(c[k] for k in KEY) not in done]
    df = pd.DataFrame(missing or [{}])
    print(f"stage {a.stage}: {len(cells)} cells planned, {len(cells) - len(missing)} done, {len(missing)} missing")
    if missing:
        print(df.groupby(["model", "config_id"]).size().to_string())
    if a.check or not missing:
        return
    os.makedirs(os.path.join(RES, "_incoming"), exist_ok=True)
    out = os.path.join("results", "_incoming", f"{a.host}_{a.stage}.csv")
    jobs = jobs_for(missing, a, out)
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"\n>>> {len(jobs)} worker invocations -> {out}")
    failed = run_on_gpus(jobs, a.ngpu, base_gpu=a.gpu, jobs_per_gpu=a.jobs_per_gpu)
    if failed:
        print(f"[WARN] {len(failed)} invocation(s) failed; their cells stay missing - re-run to retry.")


if __name__ == "__main__":
    main()
