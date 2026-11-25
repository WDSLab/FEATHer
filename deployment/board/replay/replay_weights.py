# -*- coding: utf-8 -*-
"""Weights of the settings the streaming replay runs: the tuned pick of every model in a tier (1k or chip;
protocol.tuned_pick, one per dataset and H), seed 2025.  The final runs save them (--save_model ->
results/checkpoints/final__<tier>__<trial>/<model>/...), so the replay uses the very model of the reported numbers; a
cell without that checkpoint is retrained here with the final budget (worker results to a scratch CSV, never to the
managed result files; the test MSE is compared with the same cell of results/mfg_final.csv).

    python deployment/board/replay/replay_weights.py --scratch <folder>
        -> results/checkpoints/replay__<data>/<model>/<data>_H<H>_L96_s2025.pth   (retrained cells only)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "results"))
import protocol as P   # noqa: E402

MODELS = [P.FEATHER_CORE, "SparseTSF", "FITS", "MixLinear", "TimeBase", "CMoS", "PhaseFormer"]
SEED = 2025


def pick(model, data, h, tier="chip"):
    p = P.tuned_pick(tier, model, data, h)
    assert p, f"no {tier}-tier pick for {model} {data} H{h} in {P.TUNED_FILE}"
    return p


def official(model, data, h, tier="chip"):
    fin = P.read_results(ROOT / "results" / "mfg_final.csv")
    t = pick(model, data, h, tier)["trial"]
    r = fin[(fin.stage == "final") & (fin.tier == tier) & (fin.trial == t) & (fin.arm == "-") & (fin.model == model)
            & (fin.data == data) & (fin.H == h) & (fin.seed == SEED)]
    assert len(r) == 1, (model, data, h, len(r))
    return r.iloc[0]


def hparams(model, data, h, tier="chip"):
    """Every setting of the pick (the final runs' settings, protocol.hparams)."""
    p = pick(model, data, h, tier)
    return json.loads(P.hparams(model, "mfg", data, h, P.size_config_id(tier, p["trial"], p["knobs"])))


def checkpoint(data, model, h, tier="chip"):
    """The final run's checkpoint of the pick if it was saved, else the retrained one."""
    name = f"{data}_H{h}_L96_s{SEED}.pth"
    final = ROOT / "results" / "checkpoints" / f"final__{tier}__{pick(model, data, h, tier)['trial']}" / model / name
    return final if final.exists() else ROOT / "results" / "checkpoints" / f"replay__{tier}__{data}" / model / name


def worker_args(model, data, h, csv, tier="chip"):
    hp = hparams(model, data, h, tier)
    s = ";".join(f"{k}={v}" for k, v in sorted(hp["settings"].items()))
    return (["--model", hp["model"], "--data", data, "--pred_len", str(h), "--seq_len", "96", "--seeds", str(SEED),
             "--stage", "replay", "--trial", data, "--results_csv", str(csv), "--num_epochs", str(P.FINAL["epochs"]),
             "--patience", str(P.FINAL["patience"]),
             "--batch_size", "32", "--lr", str(hp["lr"]), "--loss", hp["loss"], "--result_model_name", model]
            + (["--model_overrides", s] if s else []))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="MetroPT3")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    csv = scratch / "replay_weights.csv"
    trained = []
    for model in a.models.split(","):
        for h in (96, 720):
            if checkpoint(a.data, model, h).exists():
                print("checkpoint", checkpoint(a.data, model, h), flush=True)
                continue
            with (scratch / f"{model}_{a.data}_H{h}.log").open("w") as log:
                subprocess.run([sys.executable, "scripts/train.py"] + worker_args(model, a.data, h, csv) + ["--save_model"],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            print("trained", model, h, flush=True)
            trained.append((model, h))
    new = pd.read_csv(csv) if trained else None
    for model, h in trained:
        o, n = official(model, a.data, h), new[(new.model == model) & (new.H == h)].iloc[-1]
        print(f"{model:<14} H={h:<3} official MSE {o.MSE:.6f}  retrained {n.MSE:.6f}  rel {abs(n.MSE / o.MSE - 1):.1e}", flush=True)


if __name__ == "__main__":
    main()
