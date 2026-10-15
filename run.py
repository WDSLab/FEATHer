"""Plan-driven runner: run only the cells the protocol needs and the results do not have yet.

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
              CycleNet) -> raw_mfg.csv
    ablation  FEATHer arms (protocol.ABLATION_PLAN: arm x horizons) x 6 datasets x 3 seeds          -> raw_mfg.csv
    control   SparseTSF(+MLP) + mixing bus, FEATHer without bands/gate x 6 x 4 H x 3 seeds          -> raw_mfg.csv
    size_select  size-matched comparison (1K / chip-max tier): the drawn random-search trials (docs/size_trials.json,
              40 per model) x 6 x H96/H720 x seed 2025, short search budget (size.yaml select_budget)
              -> raw_mfg.csv (role sizesel);
              tools/results/size_verdict.py --write picks one trial per tier, model and dataset
    size      the chosen trial x 4 H x 3 seeds, full protocol (50 epochs / patience 10) -> raw_mfg.csv

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


def grid(role, config_id, model, group, datas, hs, seeds, seq_len=96):
    """One cell per (dataset, horizon, seed)."""
    return [dict(role=role, config_id=config_id, model=model, group=group, data=d, pred_len=h, seq_len=seq_len,
                 seed=s) for d in datas for h in hs for s in seeds]


def plan(stage, models=None, tiers=None):
    """List of cells: dict(role, config_id, model, group, data, pred_len, seq_len, seed)."""
    cells = []
    one_seed = [P.SELECT_SEED]
    if stage == "ltsf":
        pending = []
        for m in (models or P.FEATHER + P.LIGHT + P.HEAVY):
            for d in P.LTSF:
                cfg = P.ltsf_config_id(m, d)
                if cfg is None:                      # no official setting and not chosen yet
                    pending.append(f"{m}|{d}")
                else:
                    cells += grid("reference" if m in P.HEAVY else "main", cfg, m, "ltsf", [d], P.HS, P.SEEDS)
        if pending:
            print(f"  [skip] {len(pending)} LTSF cells without an official setting wait for --stage ltsf_select: "
                  + ", ".join(pending))
    elif stage == "ltsf_select":
        for m in (models or P.LIGHT):
            for d in P.LTSF:
                cands = P.ltsf_candidates(m, d)
                for c in (cands if len(cands) >= 2 else []):
                    cells += grid("select", c, m, "ltsf", [d], P.HS, one_seed)
    elif stage == "select":
        for m in (models or list(P.CANDIDATES)):
            cands = P.CANDIDATES.get(m, [])
            for c in (cands if len(cands) >= 2 else []):
                cells += grid("select", "cand:" + c, m, "mfg", P.MFG, P.HS, one_seed)
    elif stage == "ablation":
        for arm, hs in P.ABLATION_PLAN.items():
            if not models or arm in models:
                cells += grid("ablation", "abl:" + arm, "FEATHer", "mfg", P.MFG, hs, P.SEEDS)
    elif stage == "control":
        for arm in P.CONTROL_ARMS:
            if models and arm not in models:
                continue
            cfg = P.control_config_id(arm)
            if cfg is None:
                print(f"  [skip] {arm}: host {P.CONTROL_ARMS[arm][2]} has no chosen setting yet")
            else:
                cells += grid("control", cfg, P.control_row_model(arm), "mfg", P.MFG, P.HS, P.SEEDS)
    elif stage in ("size_select", "size"):
        chosen = P.chosen_size()
        # FEATHer runs only when named (--models FEATHer_mix,FEATHer_CI): its trials are fixed now, run after the baselines
        for tier in (tiers or P.SIZE_TIERS):
            for m in (models or [m for m in P.SIZE_MODELS if m not in P.FEATHER]):
                trials = P.size_trials(tier, m)
                if not trials:
                    print(f"  [skip] {tier} {m}: no trials (not drawn yet, or no setting fits this tier)")
                    continue
                for d in P.MFG:
                    if stage == "size_select":
                        pick, role, hs, seeds = trials, "sizesel", P.SIZE_SELECT_HS, one_seed
                    else:
                        pick = [t for t in trials if t["id"] == chosen.get(f"{tier}|{m}|{d}")]
                        role, hs, seeds = "size", P.HS, P.SEEDS
                    for t in pick:
                        cells += grid(role, P.size_config_id(tier, t["id"], t["knobs"][d]), m, "mfg", [d], hs, seeds)
    elif stage in ("main", "lookback"):
        pool = models or (P.FEATHER + P.LIGHT + P.HEAVY if stage == "main"
                          else P.FEATHER + P.ON_CHIP + ["CycleNet", "DLinear"])
        for m in pool:
            cfg = P.mfg_main_config_id(m)
            if cfg is None:
                print(f"  [skip] {m}: industrial setting not chosen yet (run --stage select first)")
                continue
            role = "reference" if m in P.HEAVY else stage
            for L in ([96] if stage == "main" else [48, 192, 336]):
                cells += grid(role, cfg, m, "mfg", P.MFG, P.HS, P.SEEDS, seq_len=L)
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


def done_runs():
    """(role, config_id, model, data, pred_len, seq_len, seed) -> set of the "hparams" values recorded for that cell
    (None = an older row that recorded no hyperparameters)."""
    frames = []
    for f in ["raw_mfg.csv", "raw_ltsf.csv", "select_mfg.csv", "select_ltsf.csv"]:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, usecols=lambda c: c in KEY + ["hparams"]))
    for p in glob.glob(os.path.join(RES, "_incoming", "*.csv")):
        d = pd.read_csv(p, on_bad_lines="skip")
        if "exp_tag" in d and len(d):
            d["role"], d["config_id"] = zip(*[str(t).split("|", 1) if "|" in str(t) else (t, "") for t in d.exp_tag])
            frames.append(d[[c for c in KEY + ["hparams"] if c in d]])
    if not frames:
        return {}
    a = pd.concat(frames, ignore_index=True).dropna(subset=KEY)
    if "hparams" not in a:
        a["hparams"] = None
    # a main run of a candidate setting at the selection seed is also its selection run (same cell)
    m = a[(a.role == "main") & a.config_id.str.startswith("cand:") & (a.seed == P.SELECT_SEED) & (a.seq_len == 96)]
    a = pd.concat([a, m.assign(role="select")], ignore_index=True)
    done = {}
    for row in a.itertuples(index=False):
        key = (row.role, row.config_id, row.model, row.data, int(row.pred_len), int(row.seq_len), int(row.seed))
        done.setdefault(key, set()).add(row.hparams if isinstance(row.hparams, str) and row.hparams else None)
    return done


def is_done(cell, done):
    """Done = a row for the cell exists, recorded with today's hyperparameters (or before they were recorded)."""
    seen = done.get(tuple(cell[k] for k in KEY), set())
    return None in seen or cell["hparams"] in seen


def shard_path(host, stage):
    """results/_incoming/<host>_<stage>.csv, or <host>_<stage>_2.csv ... when that file has an older column layout."""
    fields = _result_fields()
    for n in range(1, 100):
        p = os.path.join("results", "_incoming", f"{host}_{stage}" + ("" if n == 1 else f"_{n}") + ".csv")
        if not os.path.exists(p) or open(p, newline="").readline().strip().split(",") == fields:
            return p
    raise RuntimeError("no free shard name")


# industrial sets, smallest first (rows of data.csv), so the small datasets finish and can be looked at early
SMALL_FIRST = {d: i for i, d in enumerate(["BlastFurnace", "Steel", "WindSCADA", "EWELD", "N2O", "MetroPT3"])}


def jobs_for(missing, args, out_csv):
    groups = {}
    for c in missing:
        k = (c["role"], c["config_id"], c["model"], c["group"], c["data"], c["pred_len"], c["seq_len"])
        groups.setdefault(k, []).append(c["seed"])
    jobs = []
    for (role, cfg, m, g, d, h, L), seeds in sorted(groups.items(), key=lambda kv: (SMALL_FIRST.get(kv[0][4], -1), kv[0])):
        lr, loss, ov = P.resolve(m, g, d, h, cfg)
        hp = P.hparams(m, g, d, h, cfg, role)
        epochs, patience = P.train_budget(role)
        ovs = ";".join(f"{k}={v}" for k, v in sorted(ov.items()))
        cmd = [sys.executable, "scripts/train.py", "--model", P.registry_name(m, cfg),
               "--data", d, "--pred_len", str(h), "--seq_len", str(L),
               "--seeds", ",".join(str(s) for s in sorted(seeds)), "--exp_tag", f"{role}|{cfg}",
               "--results_csv", out_csv, "--num_epochs", str(epochs), "--patience", str(patience), "--batch_size", "32",
               "--lr", str(lr), "--loss", loss, "--result_model_name", m, "--hparams", hp]
        if ovs:
            cmd += ["--model_overrides", ovs]
        cmd += P.cli_flags(m, cfg)
        if args.save_model:
            cmd += ["--save_model"]
        jobs.append((f"{role}|{cfg} | {m} | {d} | H={h} | L={L} | seeds={sorted(seeds)} | lr={lr}", cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ltsf", "ltsf_select", "select", "main", "lookback", "ablation", "control", "size_select", "size"])
    ap.add_argument("--models", default="", help="comma list (default: the stage's pool)")
    ap.add_argument("--data", default="", help="comma list of datasets (default: all of the stage)")
    ap.add_argument("--configs", default="", help="comma list of config_ids (default: all)")
    ap.add_argument("--tier", default="", help="size stages only: comma list of tiers (1k,chip; default: all)")
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
    cells = plan(a.stage, models, [t.strip() for t in a.tier.split(",") if t.strip()] or None)
    if a.data:
        keep = {x.strip() for x in a.data.split(",")}
        cells = [c for c in cells if c["data"] in keep]
    if a.configs:
        keep = {x.strip() for x in a.configs.split(",")}
        cells = [c for c in cells if c["config_id"] in keep]
    for c in cells:
        c["hparams"] = P.hparams(c["model"], c["group"], c["data"], c["pred_len"], c["config_id"], c["role"])
    done = done_runs()
    missing = [c for c in cells if not is_done(c, done)]
    df = pd.DataFrame(missing or [{}])
    print(f"stage {a.stage}: {len(cells)} cells planned, {len(cells) - len(missing)} done, {len(missing)} missing")
    if missing:
        by_cfg = df.groupby(["model", "config_id"]).size()
        print((by_cfg if len(by_cfg) <= 60 else df.groupby("model").size()).to_string())
    if a.check or not missing:
        return
    os.makedirs(os.path.join(RES, "_incoming"), exist_ok=True)
    out = shard_path(a.host, a.stage)
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
