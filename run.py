"""Plan-driven runner: run only what the protocol needs and the results do not have yet.

    python run.py --stage tune --tier chip --check          # which tuning studies are finished / missing, nothing runs
    python run.py --stage tune --tier chip --ngpu 2 --jobs_per_gpu 3
    python tools/results/tune.py --collect                  # finished studies -> configs/experiments/tuned.json
    python run.py --stage final --tier chip
    python tools/results/merge_incoming.py                  # shards -> results/*.csv

Stages (configs/experiments/size.yaml):
    tune      one Optuna study per (tier, model, dataset, H) (tools/results/tune.py: size.yaml tune, 40 trained trials,
              seed 2025) -> results/tune/ journals + {mfg,ltsf}_tune.csv; a job = one study
    final     each tuned pick (tuned.json) x size.yaml final seeds, final epochs / patience, models saved
              (results/checkpoints/final__<tier>__<trial>/) -> {mfg,ltsf}_final.csv
    ablation  FEATHer_core's tuned pick with one part changed (size.yaml ablation) where the pick has that part,
              1K + chip, every H, final seeds -> mfg_ablation.csv

A cell (final / ablation) is done when a result file (protocol.RESULT_FILES) or a not-yet-merged shard in
results/_incoming/ holds a row with the same protocol.KEY; a study is done when its journal has `tune.trials` trained
trials (or it stopped: no setting fits).  New rows go to results/_incoming/<host>_<stage>.csv in the final layout
(protocol.RESULT_FIELDS).  The config_id of a planned cell is internal: it tells protocol.resolve() the settings.
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
# industrial sets, smallest first (rows of data.csv), so the small datasets finish and can be looked at early
SMALL_FIRST = {d: i for i, d in enumerate(["BlastFurnace", "Steel", "WindSCADA", "EWELD", "N2O", "MetroPT3"])}


def grid(role, config_id, model, group, datas, hs, seeds, arm="-", seq_len=96):
    """One cell per (dataset, horizon, seed)."""
    return [dict(role=role, config_id=config_id, model=model, group=group, data=d, pred_len=h, seq_len=seq_len,
                 seed=s, arm=arm) for d in datas for h in hs for s in seeds]


def studies(models=None, tiers=None):
    """Tuning studies: dict(tier, model, data, H), small datasets first."""
    out = [dict(tier=t, model=m, data=d, H=h) for t in (tiers or P.SIZE_TIERS)
           for m in (models or P.tune_models(t)) if m in P.tune_models(t) for d in P.size_data(t) for h in P.HS]
    return sorted(out, key=lambda s: (P.SIZE_TIERS.index(s["tier"]), SMALL_FIRST.get(s["data"], -1), s["data"],
                                      s["model"], s["H"]))


def plan(stage, models=None, tiers=None):
    """List of cells: dict(role, config_id, model, group, data, pred_len, seq_len, seed, arm)."""
    cells = []
    seeds = [int(s) for s in P.FINAL["seeds"]]
    if stage == "final":
        missing = 0
        for s in studies(models, tiers):
            pick = P.tuned_pick(s["tier"], s["model"], s["data"], s["H"])
            if pick is None:
                missing += 1
                continue
            group = P.tier_group(s["tier"])
            cells += grid("final", P.size_config_id(s["tier"], pick["trial"], pick["knobs"]), s["model"], group,
                          [s["data"]], [s["H"]], seeds)
        if missing:
            print(f"  [skip] {missing} studies not tuned yet (tuned.json; tools/results/tune.py --collect)")
    elif stage == "ablation":
        for s in studies(models or [P.FEATHER_CORE], tiers or ["1k", "chip"]):
            if P.SIZE_ABLATION_PLACES and f"{s['tier']}|{s['data']}" not in P.SIZE_ABLATION_PLACES:
                continue
            pick = P.tuned_pick(s["tier"], s["model"], s["data"], s["H"])
            if pick is None:
                print(f"  [skip] {s['tier']} {s['model']} {s['data']} H{s['H']}: not tuned yet")
                continue
            k = pick["knobs"]
            for arm, change in P.SIZE_ABLATION.items():
                if any(k.get(a) != v for a, v in change.items()):
                    cells += grid("ablation", P.size_config_id(s["tier"], pick["trial"], {**k, **change}), s["model"],
                                  "mfg", [s["data"]], [s["H"]], seeds, arm=arm)
    else:
        raise ValueError(stage)
    return cells


def done_runs():
    """Set of protocol.KEY tuples that have a row (result files + shards not merged yet)."""
    frames = [P.read_results(p) for p in [os.path.join(RES, f) for f in P.RESULT_FILES]
              + glob.glob(os.path.join(RES, "_incoming", "*.csv")) if os.path.exists(p)]
    if not frames:
        return set()
    a = pd.concat(frames, ignore_index=True).dropna(subset=P.KEY)
    return {(r.stage, r.tier, r.trial, r.arm, r.model, r.data, int(r.H), int(r.seed), r.settings)
            for r in a.itertuples(index=False)}


def is_done(cell, done):
    return tuple(cell[k] for k in P.KEY) in done


def shard_path(host, stage):
    """results/_incoming/<host>_<stage>.csv, or <host>_<stage>_2.csv ... when that file has an older column layout."""
    for n in range(1, 100):
        p = os.path.join("results", "_incoming", f"{host}_{stage}" + ("" if n == 1 else f"_{n}") + ".csv")
        if not os.path.exists(p) or open(p, newline="").readline().strip().split(",") == P.RESULT_FIELDS:
            return p
    raise RuntimeError("no free shard name")


def jobs_for(missing, args, out_csv):
    groups = {}
    for c in missing:
        k = (c["role"], c["config_id"], c["model"], c["group"], c["data"], c["pred_len"], c["seq_len"], c["arm"])
        groups.setdefault(k, []).append(c["seed"])
    jobs = []
    epochs, patience = P.train_budget()
    for (role, cfg, m, g, d, h, L, arm), seeds in sorted(groups.items(),
                                                         key=lambda kv: (SMALL_FIRST.get(kv[0][4], -1), kv[0])):
        lr, loss, ov = P.resolve(m, g, d, h, cfg)
        lab = P.labels(role, cfg, m, d, arm)
        ovs = ";".join(f"{k}={v}" for k, v in sorted(ov.items()))
        cmd = [sys.executable, "scripts/train.py", "--model", P.registry_name(m),
               "--data", d, "--pred_len", str(h), "--seq_len", str(L),
               "--seeds", ",".join(str(s) for s in sorted(seeds)),
               *[x for k in ("stage", "tier", "trial", "arm", "settings") for x in (f"--{k}", lab[k])],
               "--results_csv", out_csv, "--num_epochs", str(epochs), "--patience", str(patience), "--batch_size", "32",
               "--lr", str(lr), "--loss", loss, "--result_model_name", m]
        if ovs:
            cmd += ["--model_overrides", ovs]
        if role == "final" or args.save_model:          # tuned finals: the board export and the replay read them
            cmd += ["--save_model"]
        jobs.append((f"{role}|{cfg} | {m} | {d} | H={h} | L={L} | seeds={sorted(seeds)} | lr={lr}", cmd))
    return jobs


def run_tune(a, models, tiers):
    import tune as TU                                   # optuna / torch only for this stage
    todo, n_done = [], 0
    for s in studies(models, tiers):
        if a.data and s["data"] not in a.data:
            continue
        st = TU.study_state(s["tier"], s["model"], s["data"], s["H"])
        if st["finished"]:
            n_done += 1
        else:
            todo.append((s, st))
    print(f"stage tune: {n_done + len(todo)} studies, {n_done} finished, {len(todo)} to run "
          f"({sum(st['trained'] > 0 for _, st in todo)} started)")
    if todo:
        df = pd.DataFrame([{**s, "trained": st["trained"]} for s, st in todo])
        print(df.groupby(["tier", "model"]).agg(studies=("H", "size"), trained=("trained", "sum")).to_string())
    if a.check or not todo:
        return
    out = shard_path(a.host, "tune")
    jobs = [(f"tune | {s['tier']} | {s['model']} | {s['data']} | H={s['H']}",
             [sys.executable, "tools/results/tune.py", "--tier", s["tier"], "--model", s["model"], "--data", s["data"],
              "--H", str(s["H"]), "--out", out]) for s, _ in todo]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"\n>>> {len(jobs)} studies -> {out}")
    failed = run_on_gpus(jobs, a.ngpu, base_gpu=a.gpu, jobs_per_gpu=a.jobs_per_gpu)
    if failed:
        print(f"[WARN] {len(failed)} studies stopped with an error; re-run to resume them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "final", "ablation"])
    ap.add_argument("--models", default="", help="comma list (default: the stage's pool)")
    ap.add_argument("--data", default="", help="comma list of datasets (default: all of the stage)")
    ap.add_argument("--tier", default="", help="tune / final / ablation: comma list of tiers (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N worker invocations (testing)")
    ap.add_argument("--save_model", action="store_true", help="also save each trained state dict for the ablation "
                    "stage (results/checkpoints/<stage>__<tier>__<trial>__<arm>/<model>/...; finals always save)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--jobs_per_gpu", type=int, default=3)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0])
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()] or None
    tiers = [t.strip() for t in a.tier.split(",") if t.strip()] or None
    a.data = {x.strip() for x in a.data.split(",") if x.strip()}
    if a.stage == "tune":
        return run_tune(a, models, tiers)
    cells = plan(a.stage, models, tiers)
    if a.data:
        cells = [c for c in cells if c["data"] in a.data]
    for c in cells:
        c.update(P.labels(c["role"], c["config_id"], c["model"], c["data"], c["arm"]), H=c["pred_len"])
    done = done_runs()
    missing = [c for c in cells if not is_done(c, done)]
    df = pd.DataFrame(missing or [{}])
    print(f"stage {a.stage}: {len(cells)} cells planned, {len(cells) - len(missing)} done, {len(missing)} missing")
    if missing:
        by = df.groupby(["tier", "model"]).size() if "tier" in df else df.groupby("model").size()
        print(by.to_string())
    if a.check or not missing:
        return
    os.makedirs(os.path.join(RES, "_incoming"), exist_ok=True)
    out = shard_path(a.host, a.stage)
    if not os.path.exists(out):
        # write the header before 3-6 workers start appending (otherwise a fast worker can append its
        # row between another worker's create and header write)
        open(out, "w", newline="").write(",".join(P.RESULT_FIELDS) + chr(10))
    jobs = jobs_for(missing, a, out)
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"\n>>> {len(jobs)} worker invocations -> {out}")
    failed = run_on_gpus(jobs, a.ngpu, base_gpu=a.gpu, jobs_per_gpu=a.jobs_per_gpu)
    if failed:
        print(f"[WARN] {len(failed)} invocation(s) failed; their cells stay missing - re-run to retry.")


if __name__ == "__main__":
    main()
