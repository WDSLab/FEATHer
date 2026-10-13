"""Plan-driven runner (2026-10-07): run only the cells the protocol needs and the results do not have yet.

    python run.py --stage select --check                 # what is missing, nothing runs
    python run.py --stage select --ngpu 1 --jobs_per_gpu 3
    python run.py --stage ltsf --ngpu 2 --jobs_per_gpu 3   (server)
    python tools/results/merge_incoming.py                    # shards -> raw_*.csv / select_mfg.csv

Stages (docs/EXPERIMENT_PROTOCOL.md):
    ltsf      every model x 8 LTSF datasets x 4 H x 3 seeds, L 96, official settings      -> raw_ltsf.csv
              (cells without an official setting run once their setting is chosen by ltsf_select)
    ltsf_select  no-official LTSF cells: the model's published settings x 4 H x seed 2025 -> select_ltsf.csv
    select    industrial candidates x 6 datasets x 4 H x seed 2025, L 96                   -> select_mfg.csv
    main      industrial chosen setting x 3 seeds, L 96 (models without a choice are skipped) -> raw_mfg.csv
    lookback  industrial chosen setting x 3 seeds, L 48 / 192 / 336 (FEATHer + chip models + DLinear,
              CycleNet as in the earlier Table 11 pool) -> raw_mfg.csv
    ablation  FEATHer arms (protocol.ABLATION_PLAN: arm x horizons) x 6 datasets x 3 seeds          -> raw_mfg.csv
    control   SparseTSF(+MLP) + mixing bus, FEATHer without bands/gate x 6 x 4 H x 3 seeds          -> raw_mfg.csv
    redesign  10-12 FEATHer-mix candidates (protocol.REDESIGN_STAGE1/2) x 6 x 4 H x seed 2025        -> raw_mfg.csv
    size_select  10-13 size-matched baselines (1K / chip-max tier, protocol.SIZE_*) x lr x0.1/x1/x10 x 6 x 4 H x
              seed 2025 -> raw_mfg.csv (role sizesel); tools/results/size_verdict.py --write picks the lr
    size      the chosen lr x 3 seeds (seed 2025 reused from size_select)                        -> raw_mfg.csv

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
os.chdir(ROOT)                      # the worker path and results/_incoming/ are relative to the repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "results"))
import protocol as P                                   # noqa: E402
from utils.dispatch_queue import run_on_gpus           # noqa: E402

RES = os.path.join(ROOT, "results")
KEY = P.KEY


def plan(stage, models=None):
    """List of cells: dict(role, config_id, model, group, data, pred_len, seq_len, seed)."""
    cells = []
    add = lambda **c: cells.append(c)
    if stage == "ltsf":
        pending = []
        for m in (models or P.FEATHER + P.LIGHT + P.HEAVY):
            for d in P.LTSF:
                cfg = P.ltsf_config_id(m, d)
                if cfg is None:                      # no official setting and not chosen yet
                    pending.append(f"{m}|{d}")
                    continue
                for h in P.HS:
                    for s in P.SEEDS:
                        add(role="reference" if m in P.HEAVY else "main", config_id=cfg,
                            model=m, group="ltsf", data=d, pred_len=h, seq_len=96, seed=s)
        if pending:
            print(f"  [skip] {len(pending)} LTSF cells without an official setting wait for --stage ltsf_select: "
                  + ", ".join(pending))
    elif stage == "ltsf_select":
        for m in (models or P.LIGHT):
            for d in P.LTSF:
                cands = P.ltsf_candidates(m, d)
                if len(cands) < 2:
                    continue
                for c in cands:
                    for h in P.HS:
                        add(role="select", config_id=c, model=m, group="ltsf", data=d, pred_len=h,
                            seq_len=96, seed=P.SELECT_SEED)
    elif stage == "select":
        for m in (models or list(P.CANDIDATES)):
            cands = P.CANDIDATES.get(m, [])
            if len(cands) < 2:
                continue
            for c in cands:
                for d in P.MFG:
                    for h in P.HS:
                        add(role="select", config_id="cand:" + c, model=m, group="mfg", data=d, pred_len=h,
                            seq_len=96, seed=P.SELECT_SEED)
    elif stage == "ablation":
        for arm, hs in P.ABLATION_PLAN.items():
            if models and arm not in models:
                continue
            for d in P.MFG:
                for h in hs:
                    for s in P.SEEDS:
                        add(role="ablation", config_id="abl:" + arm, model="FEATHer", group="mfg", data=d,
                            pred_len=h, seq_len=96, seed=s)
    elif stage == "control":
        for arm in P.CONTROL_ARMS:
            if models and arm not in models:
                continue
            cfg = P.control_config_id(arm)
            if cfg is None:
                print(f"  [skip] {arm}: host {P.CONTROL_ARMS[arm][2]} has no chosen setting yet")
                continue
            for d in P.MFG:
                for h in P.HS:
                    for s in P.SEEDS:
                        add(role="control", config_id=cfg, model=P.control_row_model(arm), group="mfg",
                            data=d, pred_len=h, seq_len=96, seed=s)
    elif stage == "redesign":
        for cfg in P.REDESIGN_STAGE1 + P.REDESIGN_STAGE2:
            if models and cfg not in models:
                continue
            for d in P.MFG:
                for h in P.HS:
                    add(role="redesign", config_id=cfg, model="FEATHer_mix", group="mfg", data=d,
                        pred_len=h, seq_len=96, seed=P.SELECT_SEED)
    elif stage in ("size_select", "size"):
        # 10-13 size-matched comparison: lr multiplier chosen on validation (size_select, seed 2025), then the
        # chosen multiplier x 3 seeds (size; its seed-2025 cells are the selection runs, see done_set)
        for tier in P.SIZE_TIERS:
            for m in P.SIZE_DIALS:
                if models and m not in models:
                    continue
                if stage == "size_select":
                    mults = P.SIZE_LR_MULT
                else:
                    mult = P.chosen_size().get(f"{tier}|{m}")
                    if mult is None:
                        print(f"  [skip] {tier} {m}: lr not chosen yet (size_select + size_verdict.py --write)")
                        continue
                    mults = [mult]
                for mult in mults:
                    for d in P.MFG:
                        cfg = P.size_config_id(tier, m, d, mult)
                        if cfg is None:
                            print(f"  [skip] {tier} {m} {d}: no dial value in docs/size_dials.json yet")
                            continue
                        for h in P.HS:
                            for s in ([P.SELECT_SEED] if stage == "size_select" else P.SEEDS):
                                add(role="sizesel" if stage == "size_select" else "size", config_id=cfg, model=m,
                                    group="mfg", data=d, pred_len=h, seq_len=96, seed=s)
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


def _result_fields():
    """Column order of the worker's result rows (scripts/train.py CSV_FIELDS), read without importing torch."""
    import ast
    tree = ast.parse(open(os.path.join(ROOT, "scripts", "train.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CSV_FIELDS":
            return [e.value for e in node.value.elts]
    raise RuntimeError("CSV_FIELDS not found in scripts/train.py")


def done_set():
    frames = []
    for f in ["raw_mfg.csv", "raw_ltsf.csv", "select_mfg.csv", "select_ltsf.csv"]:
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
    # a size-matched selection run (seed 2025) is also the seed-2025 run of the same config in the size stage
    a = pd.concat([a, a[a.role == "sizesel"].assign(role="size")], ignore_index=True)
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
        cmd = [sys.executable, "scripts/train.py", "--model", P.registry_name(m, cfg),
               "--data", d, "--pred_len", str(h), "--seq_len", str(L),
               "--seeds", ",".join(str(s) for s in sorted(seeds)), "--exp_tag", f"{role}|{cfg}",
               "--results_csv", out_csv, "--num_epochs", "50", "--patience", "10", "--batch_size", "32",
               "--lr", str(lr), "--loss", loss, "--result_model_name", m]
        if ovs:
            cmd += ["--model_overrides", ovs]
        cmd += P.cli_flags(m, cfg)
        if args.save_model:
            cmd += ["--save_model"]
        jobs.append((f"{role}|{cfg} | {m} | {d} | H={h} | L={L} | seeds={sorted(seeds)} | lr={lr}", cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ltsf", "ltsf_select", "select", "main", "lookback", "ablation", "control", "size_select", "size",
                                                    "redesign"])
    ap.add_argument("--models", default="", help="comma list (default: the stage's pool)")
    ap.add_argument("--data", default="", help="comma list of datasets (default: all of the stage)")
    ap.add_argument("--configs", default="", help="comma list of config_ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N worker invocations (testing)")
    ap.add_argument("--save_model", action="store_true", help="also save each trained state dict "
                    "(results/checkpoints/<exp_tag>/<model>/...; needed for board export and replay)")
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
    if not os.path.exists(out):
        # write the header before 3-6 workers start appending (otherwise a fast worker can append its
        # row between another worker's create and header write)
        open(out, "w", newline="").write(",".join(_result_fields()) + chr(10))
    jobs = jobs_for(missing, a, out)
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"\n>>> {len(jobs)} worker invocations -> {out}")
    failed = run_on_gpus(jobs, a.ngpu, base_gpu=a.gpu, jobs_per_gpu=a.jobs_per_gpu)
    if failed:
        print(f"[WARN] {len(failed)} invocation(s) failed; their cells stay missing - re-run to retry.")


if __name__ == "__main__":
    main()
