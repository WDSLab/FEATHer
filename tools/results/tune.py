"""Tuning runner: one Optuna study per (tier, model, dataset, H) (configs/experiments/size.yaml `tune`).

    python tools/results/tune.py --tier chip --model CMoS --data Steel --H 96 --gpu 0     # run / resume one study
    python tools/results/tune.py --collect                                                # picks -> tuned.json

One study, in one process:
  0. sampler = Optuna's TPE with its defaults (set_sampler)
  1. draw a setting (tune_space.Space.sample) within the tier (tune_budget.Budget: 1K count, or the flash proxy)
  2. chip / ltsf tiers: build the firmware for real (tune_budget.Budget.check; golden/ is locked, so studies running
     side by side build one at a time)
  3. a draw that cannot be built, does not fit or cannot be trained is a REJECT: not trained, not counted, told to Optuna
     as PRUNED without values (TPE learns to avoid the region); `tune.max_rejects` rejects in a row stop the study
  4. train it in this process (scripts/train.py train_single, seed `tune.seed`, `tune.budget` epochs / patience), the
     best validation loss so far reported each epoch to the successive-halving pruner (`tune.pruner`)
  5. outcome, each one COUNTED towards `tune.trials`:
       done      finished training: value = best validation MSE
       pruned    stopped early by the pruner (Brigato et al.: a pruned trial is one of the budget's trials)
       diverged  no epoch gave a finite validation loss: value = inf, the worst score; the pruner
                 stops such a run at its first rung, so it is usually stored as PRUNED with outcome "diverged"
  The pick = the done trial with the lowest validation loss (test is never read here).

Records: the Optuna journal results/tune/<tier>/<model>/<data>_H<H>.log (resume after a crash: the trial that was
running is closed as FAIL, not counted, and drawn again; the sampler is seeded with `tune.seed` + the number of trials
so far (set_sampler), so a resumed study does not redraw the settings already tried and is not draw-for-draw the same
as an uninterrupted one) and one CSV row per trained trial (stage "tune", arm = outcome,
MSE / MAE empty).  --collect writes configs/experiments/tuned.json from the finished studies.
"""
import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import socket
import sys
import tempfile
import time
import traceback
from datetime import datetime

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal._file import BaseJournalFileLock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), HERE]
import protocol as P   # noqa: E402
import tune_budget as TB   # noqa: E402
import tune_space as S   # noqa: E402
import train as T   # noqa: E402

STUDIES = os.path.join(ROOT, "results", "tune")
DONE, PRUNED, FAIL, RUNNING = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                               optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)


class _OneWriter(BaseJournalFileLock):
    """No journal lock: a study is run by one process only (_study_guard), and Optuna's lock files would be created
    and removed on every write (slow on synced folders)."""
    def acquire(self):
        return True

    def release(self):
        pass


def journal_path(tier, model, data, H):
    return os.path.join(STUDIES, tier, model, f"{data}_H{H}.log")


def load_study(path, create=False):
    if create:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    storage = JournalStorage(JournalFileBackend(path, lock_obj=_OneWriter()))
    return optuna.create_study(
        storage=storage, study_name="tune", direction="minimize", load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=int(P.TUNE["pruner"]["min_epochs"]),
                                                      reduction_factor=int(P.TUNE["pruner"]["reduction"])))


def set_sampler(study, tpe):
    """Optuna's TPE with its defaults (rejects count as trials for it: 10 random draws, then gamma on all trials),
    seeded with tune.seed + the trials so far, so a resumed study does not redraw tried settings.  Returns the TPE
    instance to keep (its random stream continues across draws).
    In a check without training, leaving rejects out of TPE's count needed more draws for 40 trained trials, so the
    default stays."""
    if tpe is None:
        tpe = optuna.samplers.TPESampler(seed=int(P.TUNE["seed"]) + len(study.trials))
    study.sampler = tpe
    return tpe


def study_state(tier, model, data, H):
    """{"finished": bool, "trained": n} of one study (run.py); finished = `tune.trials` trained, or stopped."""
    path = journal_path(tier, model, data, H)
    if not os.path.exists(path):
        return {"finished": False, "trained": 0}
    st = optuna.load_study(study_name="tune", storage=JournalStorage(JournalFileBackend(path, lock_obj=_OneWriter())))
    n = len(counted(st.trials))
    return {"finished": n >= int(P.TUNE["trials"]) or bool(st.user_attrs.get("stopped")), "trained": n}


@contextlib.contextmanager
def _study_guard(path):
    """Refuse to run a study that another process is running (non-blocking OS lock, freed if the holder dies)."""
    lock = os.path.join(tempfile.gettempdir(), f"feather_tune_{hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:10]}.lock")
    with open(lock, "a+b") as f:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"{path}: this study is already running in another process")
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def is_reject(t):
    return t.state == PRUNED and "reject" in t.user_attrs


def counted(trials):
    """Trained trials: done, pruned by the pruner, diverged."""
    return [t for t in trials if t.state == DONE or (t.state == PRUNED and not is_reject(t))]


def rejects_in_a_row(trials):
    n = 0
    for t in reversed([t for t in trials if t.state != FAIL]):
        if not is_reject(t):
            break
        n += 1
    return n


def pick(trials):
    """The done trial with the lowest validation loss (ties: the earlier trial), or None."""
    done = [t for t in trials if t.state == DONE and math.isfinite(t.value)]
    return min(done, key=lambda t: (t.value, t.number)) if done else None


def worker_args(tier, model, group, data, H, tid, knobs, gpu):
    """scripts/train.py arguments of one trial, resolved as run.py resolves a planned cell (same settings in the
    final runs)."""
    lr, loss, ov = P.resolve(model, group, data, H, P.size_config_id(tier, tid, knobs))
    b = P.TUNE["budget"]
    argv = ["--model", P.registry_name(model), "--data", data, "--pred_len", str(H), "--seq_len", "96",
            "--num_epochs", str(b["epochs"]), "--patience", str(b["patience"]), "--batch_size", "32",
            "--lr", str(lr), "--loss", loss, "--result_model_name", model, "--gpu", str(gpu)]
    if ov:
        argv += ["--model_overrides", ";".join(f"{k}={v}" for k, v in sorted(ov.items()))]
    args = T.parse_args(argv)
    T._apply_overrides(args)
    return args


def run_study(tier, model, data, H, gpu, out_csv):
    group = P.tier_group(tier)
    if model not in P.tune_models(tier):     # e.g. LTSF FEATHer_core / no-MLP journals, superseded by the single FEATHer_CI study
        raise SystemExit(f"{model} is not tuned on the {tier} tier (protocol.tune_models)")
    assert data in P.size_data(tier) and H in P.HS, (tier, model, data, H)
    path = journal_path(tier, model, data, H)
    n_trials, max_rejects = int(P.TUNE["trials"]), int(P.TUNE["max_rejects"])
    seed = int(P.TUNE["seed"])
    with _study_guard(path):
        study = load_study(path, create=True)
        for t in study.trials:                          # a trial left running by a crash: closed, not counted
            if t.state == RUNNING:
                study.tell(t.number, state=FAIL)
                print(f"  closed trial {t.number} left running by an earlier process (not counted)")
        if len(counted(study.trials)) >= n_trials or study.user_attrs.get("stopped"):
            return report(study, tier, model, data, H)
        t0 = time.time()
        budget = TB.Budget(tier, model, group, data, H)
        space = S.Space(model, group, data, H, cap=budget.cap)
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
        print(f"{tier} {model} {data} H{H}: cap {budget.cap} stored numbers, size keys {space.size_keys} "
              f"({time.time() - t0:.0f} s)", flush=True)
        tpe = None
        while len(counted(study.trials)) < n_trials:
            if rejects_in_a_row(study.trials) >= max_rejects:
                study.set_user_attr("stopped", f"{max_rejects} rejects in a row")
                break
            tpe = set_sampler(study, tpe)
            trial = study.ask()
            try:
                knobs, n = space.sample(trial)
                build = budget.check(knobs)
            except S.Reject as e:
                trial.set_user_attr("reject", e.why)
                study.tell(trial, state=PRUNED)
                continue
            tid = f"t{len(counted(study.trials)) + 1:02d}"
            trial.set_user_attr("tid", tid)
            if build:
                trial.set_user_attr("flash_B", build.get("flash_B"))
                trial.set_user_attr("ram_B", build.get("ram_B"))
                trial.set_user_attr("ram_stream_B", build.get("ram_stream_B"))   # the SRAM check of emit-API ports
            args = worker_args(tier, model, group, data, H, tid, knobs, gpu)
            seen = {}

            def on_epoch(epoch, best_val, best_epoch, params):
                seen.update(best_val=best_val, best_epoch=best_epoch, params=params)
                # step = epochs done, so the pruner's rungs fall after 3, 6, 12 epochs (min_epochs 3, reduction 2)
                trial.report(best_val if math.isfinite(best_val) else math.inf, epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            t1 = time.time()
            try:
                r = T.train_single(args, seed, device, on_epoch=on_epoch, run_test=False)
                outcome, val = ("done", r["val_loss"]) if math.isfinite(r["val_loss"]) else ("diverged", math.inf)
                trial.set_user_attr("outcome", outcome)
                study.tell(trial, val)
            except optuna.TrialPruned:
                r = dict(val_loss=seen.get("best_val", math.nan), best_epoch=seen.get("best_epoch", -1),
                         params=seen.get("params", -1))
                # no finite loss yet: a diverged run, stopped at the first rung because inf is the worst score
                outcome = "pruned" if math.isfinite(r["val_loss"]) else "diverged"
                trial.set_user_attr("outcome", outcome)
                study.tell(trial, state=PRUNED)
            except torch.cuda.OutOfMemoryError:
                study.tell(trial, state=FAIL)           # the machine, not the setting: stop, run again later
                raise
            except Exception as e:                      # builds but cannot be trained: a reject, not counted
                traceback.print_exc()
                trial.set_user_attr("reject", f"train_error: {type(e).__name__}: {str(e)[:200]}")
                study.tell(trial, state=PRUNED)
                print(f"  reject (train error) {knobs}", flush=True)
                continue
            finally:
                torch.cuda.empty_cache()
            T.append_result_row(out_csv, {
                "stage": "tune", "tier": tier, "trial": tid, "arm": outcome, "model": model, "data": data, "H": H,
                "seed": seed, "MSE": math.nan, "MAE": math.nan,
                "val_loss": round(r["val_loss"], 6) if math.isfinite(r["val_loss"]) else math.nan,
                "best_epoch": r["best_epoch"], "params": r["params"], "host": socket.gethostname().split(".")[0],
                "time": datetime.now().strftime("%Y%m%d_%H%M%S"), "settings": P.settings_str(knobs)})
            draws = len([t for t in study.trials if t.state != FAIL])
            print(f"  {tid} {outcome:<8} val {r['val_loss']:.4f} ({time.time() - t1:.0f} s)  stored {n}  "
                  f"draws {draws}  {P.settings_str(knobs)}", flush=True)
        return report(study, tier, model, data, H)


def report(study, tier, model, data, H):
    ts = study.trials
    c = counted(ts)
    out = {k: sum(t.user_attrs.get("outcome") == k for t in c) for k in ("done", "pruned", "diverged")}
    rej = {}
    for t in ts:
        if is_reject(t):
            w = t.user_attrs["reject"].split(":")[0]
            rej[w] = rej.get(w, 0) + 1
    p = pick(ts)
    print(f"{tier} {model} {data} H{H}: {len(c)} trained {out}, rejects {rej}"
          + (f", stopped ({study.user_attrs['stopped']})" if study.user_attrs.get("stopped") else "")
          + (f"\n  pick {p.user_attrs['tid']} val {p.value:.4f} {P.settings_str(p.user_attrs['knobs'])}" if p else
             "\n  no pick"), flush=True)
    return p


def collect():
    """Finished studies -> configs/experiments/tuned.json ("<tier>|<model>|<data>|<H>" -> trial, knobs, val_loss).
    Picks already in tuned.json whose journal is not on this machine (tuned elsewhere) are kept; copy that machine's
    results/tune/ journals here to re-check them."""
    out, open_ = {}, []
    here = set()
    for path in sorted(glob.glob(os.path.join(STUDIES, "*", "*", "*_H*.log"))):
        tier, model = path.split(os.sep)[-3:-1]
        data, H = os.path.basename(path)[:-4].rsplit("_H", 1)
        if model not in P.tune_models(tier) or data not in P.size_data(tier):
            continue                                   # a model / dataset left out of the tier (its journal kept)
        st = optuna.load_study(study_name="tune", storage=JournalStorage(JournalFileBackend(path, lock_obj=_OneWriter())))
        key = f"{tier}|{model}|{data}|{H}"
        here.add(key)
        p = pick(st.trials)
        if not study_state(tier, model, data, int(H))["finished"]:
            open_.append(f"{key} ({len(counted(st.trials))} trained)")
        elif p is None:
            open_.append(f"{key} (finished without a pick: {st.user_attrs.get('stopped', 'all diverged')})")
        else:
            out[key] = {"trial": p.user_attrs["tid"], "knobs": p.user_attrs["knobs"], "val_loss": round(p.value, 6)}
    kept = {k: v for k, v in P.tuned().items() if k not in here
            and k.split("|")[1] in P.tune_models(k.split("|")[0]) and k.split("|")[2] in P.size_data(k.split("|")[0])}
    if kept:
        print(f"  kept {len(kept)} picks with no journal on this machine (tuned elsewhere)")
    out.update(kept)
    with open(P.TUNED_FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(sorted(out.items())), f, indent=1)
        f.write("\n")
    print(f"{len(out)} picks -> {P.TUNED_FILE}")
    for s in open_:
        print("  not finished:", s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=P.SIZE_TIERS)
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--H", type=int)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("results", "_incoming", f"{socket.gethostname().split('.')[0]}_tune.csv"))
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.collect:
        return collect()
    if not (a.tier and a.model and a.data and a.H):
        ap.error("--tier, --model, --data and --H (or --collect)")
    os.chdir(ROOT)                     # data/ and results/ are relative to the repo root
    run_study(a.tier, a.model, a.data, a.H, a.gpu, a.out)


if __name__ == "__main__":
    main()
