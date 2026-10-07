"""Build the single raw result files from the scattered CSVs (one-time migration, 2026-10-07).

    results/raw_mfg.csv    paper-grade runs on the industrial datasets (main / lookback / ablation /
                           control / reference), only rows whose setting conforms to EXPERIMENT_PROTOCOL.md
    results/raw_ltsf.csv   same for the standard benchmarks
    results/_archive/legacy_rows_20261007.csv   every row that does NOT conform (kept, never deleted)

Every row gets three columns on top of the worker's columns:
    role       main | lookback | ablation | control | reference
    config_id  name of the setting the row was trained with (the run planner matches on it)
    source     file the row came from

config_id names (see EXPERIMENT_PROTOCOL.md):
    frozen          FEATHer frozen configuration (09-23 R4)
    cand:<DS>       a baseline candidate = the authors' final setting for LTSF dataset <DS>
    cand:default    the default setting the authors present (config file / PyTorch class default)
    official        LTSF: the authors' per-dataset setting for that dataset
    official-exc    LTSF: the published setting failed the health check and was re-selected on validation
    ref-default     server-class reference models on the industrial data (default settings, not selected)
    abl:<arm> / ctrl:<arm>   ablation / control arms

Read-only on the sources: nothing is moved or deleted here.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results")
SRC = os.path.join(R, "_archive", "pre_raw_20261007")
HEAVY = {"PatchTST", "TimeMixer", "TQNet", "LMS_AutoTSF", "TimesNet", "iTransformer", "MDMLP_EIA"}


def mfg_config(model, H):
    """config_id of a baseline row trained with the setting used on the industrial data until 10-07,
    or None when that setting is not one of the protocol's candidates."""
    if model in HEAVY:
        return "ref-default"
    return {
        "SparseTSF": "cand:ETTh1",             # lr 0.02 = ETTh1 (and 4 other datasets) final script
        "SparseTSF_MLP": "cand:Electricity",   # lr 0.02, d_model 128 = mlp/electricity.sh
        "DLinear": "cand:ETTh1",               # lr 0.005
        "DiPE_Linear": "cand:default",         # base.yaml: 1 expert, loss_alpha 0, RevIN on
        "CMoS": "cand:default",                # config.toml
        "SAMformer": "cand:default",           # PyTorch class default: lr 1e-3, rho 0.5
        "PhaseFormer": "cand:ETTh1" if H <= 336 else None,   # H720 official = the wide config
        "TimeBase": "cand:ETTh1" if H == 96 else None,       # ETTh1-H96 values were used at every H
        "CycleNet": None,                      # lr 0.005; protocol = L96 script lr 0.01
        "MixLinear": None,                     # alpha 0.5 / lr 0.03 is not any final setting
        "FITS": None,                          # cut_freq L//4 instead of the official formula
    }.get(model)


def ltsf_config(model, data, H):
    if model in HEAVY:
        return "official"
    if model == "SparseTSF":
        return "official"
    if model == "SparseTSF_MLP":
        return "official" if data == "Electricity" else None      # other datasets used the linear lr
    if model == "DLinear":
        if data == "ETTh2":
            return "official-exc"                                  # 0.05 diverged; 5e-3 chosen on validation
        if data == "Weather" or (data == "ETTm2" and H >= 336):
            return None
        return "official"
    # 10-06 additions (server run, results/new4_ltsf.csv). Exchange has no official setting -> the protocol
    # uses the ETTh1 setting; the server ran Exchange with other values except where noted.
    if model == "CMoS":
        return None if data == "Exchange" else "official"          # Exchange ran the repo default
    if model == "TimeBase":
        return None if (data == "Exchange" and H > 96) else "official"   # Exchange = ETTh1-H96 values at every H
    if model == "SAMformer":
        return "official"                                          # incl. Exchange (official exchange_rate)
    if model == "PhaseFormer":                                     # only the ETTh1 H<=336 config was right
        return "official" if (data in ("ETTh1", "Exchange") and H <= 336) else None
    return None              # CycleNet, DiPE_Linear, MixLinear, FITS: settings differ from the scripts


def load(name, **kw):
    p = os.path.join(SRC, name)                    # pre-raw CSVs were moved here on 10-07
    if not os.path.exists(p):
        p = os.path.join(R, name)                  # e.g. _archive/baselines/fcst_results_baselines.csv
    df = pd.read_csv(p, **kw)
    df["source"] = name
    return df


def tag(df, role, cfg):
    df = df.copy()
    df["role"] = role
    df["config_id"] = [cfg(r) for r in df.itertuples()]
    return df


def main():
    mfg, ltsf = [], []

    # ---- industrial: main (L96)
    b = load("mfg_baselines.csv")
    mfg.append(tag(b, "main", lambda r: mfg_config(r.model, r.pred_len)))
    for f in ("cmos_mfg.csv", "new3_mfg.csv"):
        d = load(f)
        mfg.append(tag(d[d.exp_tag == "main"], "main", lambda r: mfg_config(r.model, r.pred_len)))
    for f, m in (("feather_mfg_candR4.csv", "FEATHer_mix"), ("feather_mfg_candR4_ci.csv", "FEATHer_CI")):
        d = load(f)
        d["model"] = m
        mfg.append(tag(d, "main", lambda r: "frozen"))

    # ---- industrial: lookback (L192 / L336 only; Aug L48 / L720 rows are re-run under the protocol)
    a = load(os.path.join("_archive", "baselines", "fcst_results_baselines.csv"))
    a = a[(a.exp_tag == "lookback") & a.seq_len.isin([192, 336])]
    mfg.append(tag(a, "lookback", lambda r: mfg_config(r.model, r.pred_len)))
    for f, m in (("feather_mfg_lookback_L192.csv", "FEATHer_mix"), ("feather_mfg_lookback_L336.csv", "FEATHer_mix"),
                 ("feather_ci_mfg_lookback_L192.csv", "FEATHer_CI"), ("feather_ci_mfg_lookback_L336.csv", "FEATHer_CI")):
        d = load(f)
        d["model"] = m
        mfg.append(tag(d, "lookback", lambda r: "frozen"))
    for f in ("stsf_mlp_mfg_lookback_L192.csv", "stsf_mlp_mfg_lookback_L336.csv"):
        mfg.append(tag(load(f), "lookback", lambda r: mfg_config(r.model, r.pred_len)))

    # ---- industrial: ablation and control arms
    d = load("ablation.csv")
    mfg.append(tag(d, "ablation", lambda r: f"abl:{r.exp_tag}"))
    d = load("control_sparsetsf_mix.csv")
    mfg.append(tag(d, "control", lambda r: f"ctrl:{r.model}"))
    d = load("control_front_none.csv")
    mfg.append(tag(d, "control", lambda r: "ctrl:FEATHer_frontnone"))

    # ---- standard benchmarks
    b = load("ltsf_baselines.csv")
    ltsf.append(tag(b, "main", lambda r: ltsf_config(r.model, r.data, r.pred_len)))
    if os.path.exists(os.path.join(SRC, "new4_ltsf_server_20261007.csv")):     # server snapshot (10-07)
        d = load("new4_ltsf_server_20261007.csv")
        ltsf.append(tag(d[d.exp_tag == "main"], "main", lambda r: ltsf_config(r.model, r.data, r.pred_len)))
    for f, m in (("feather_ltsf_candR4.csv", "FEATHer_CI"), ("feather_ltsf_candR4_mix.csv", "FEATHer_mix")):
        d = load(f)
        d["model"] = m
        ltsf.append(tag(d, "main", lambda r: "frozen"))

    key = ["role", "config_id", "model", "data", "pred_len", "seq_len", "seed"]
    out, legacy = {}, []
    for name, parts in (("raw_mfg.csv", mfg), ("raw_ltsf.csv", ltsf)):
        df = pd.concat(parts, ignore_index=True)
        df.loc[df.role.isin(["main", "lookback"]) & df.model.isin(HEAVY), "role"] = "reference"
        bad = df[df.config_id.isna()].sort_values("timestamp").drop_duplicates(
            ["role", "model", "data", "pred_len", "seq_len", "seed"], keep="last")
        legacy.append(bad.assign(raw_file=name))
        good = df[df.config_id.notna()].copy()
        n0 = len(good)
        good = good.sort_values("timestamp").drop_duplicates(key, keep="last")   # resume duplicates
        cols = ["role", "config_id"] + [c for c in good.columns if c not in ("role", "config_id", "source")] + ["source"]
        out[name] = good[cols].sort_values(key).reset_index(drop=True)
        print(f"{name}: {len(out[name])} rows ({n0 - len(good)} duplicates dropped), "
              f"{len(bad)} non-conforming -> legacy")
        print(out[name].groupby(["role", "seq_len"]).size().to_string())
        mm = out[name][out[name].role.isin(["main", "reference"])]
        print(mm.groupby(["model", "config_id"]).size().to_dict(), "\n")

    if "--write" in sys.argv:
        for name, df in out.items():
            df.to_csv(os.path.join(R, name), index=False)
        lg = pd.concat(legacy, ignore_index=True)
        lg.to_csv(os.path.join(R, "_archive", "legacy_rows_20261007.csv"), index=False)
        print("written:", list(out), "+ _archive/legacy_rows_20261007.csv")
    else:
        print("(dry run; add --write to write the files)")
    lg = pd.concat(legacy, ignore_index=True)
    print("legacy rows by model:", lg.groupby(["raw_file", "model"]).size().to_dict())


if __name__ == "__main__":
    main()
