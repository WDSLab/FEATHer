"""Reproducible preprocessing for the 5 manufacturing LTSF datasets (JMS pivot).

Each recipe reads the RAW source already in data/<name>/ and writes the
cleaned data/<name>/data.csv that utils/data_factory.py loads. Idempotent:
re-running regenerates data.csv from raw without touching the raw files.

RAW PROVENANCE (download once, place in data/<name>/):
  Steel       UCI #851  -> Steel_industry_data.csv   (direct UCI zip)
  GasTurbine  UCI #551  -> gt_2011.csv .. gt_2015.csv (direct UCI zip)
  TEP         github.com/anasouzac/new_tep_datasets/matlab_data_1year.csv -> raw.csv
  WindSCADA   Zenodo rec 5841834, Kelmarsh_SCADA_2017_3083.zip
              -> Turbine_Data_Kelmarsh_1_2017-..csv (saved as turbine1_2017.csv)
              (Zenodo Cloudflare-blocks scripted downloads -> download via browser)
  PMSM        Kaggle wkirgsn/electric-motor-temperature -> measures_v2.csv
              (Kaggle needs auth -> download via browser/CLI)
  CMAPSS      NASA C-MAPSS FD001 train_FD001.txt -> raw/train_FD001.txt
              (mirror: github.com/cyrilli/TurboEngine_Dataset_NASA/CMAPSSData)
              SEPARATE short-horizon PdM section, NOT the main [96..720] table
              (run-to-failure trajectories ~200 cycles are too short for long
              horizons; engine-aware windowing via Dataset_CMAPSS).

Run:  python tools/prep_manufacturing.py            # all
      python tools/prep_manufacturing.py Steel CMAPSS  # subset
"""
import os
import sys
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _w(df, name):
    out = os.path.join(DATA, name, "data.csv")
    assert df.isna().sum().sum() == 0, f"{name}: NaN present"
    df.to_csv(out, index=False)
    print(f"{name:10s} -> data.csv  rows={len(df):>7d}  D={df.shape[1]-1}")


def steel():
    # UCI #851 DAEWOO Steel (Gwangyang, KR), 15-min, full-year 2018.
    df = pd.read_csv(os.path.join(DATA, "Steel", "Steel_industry_data.csv"))
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="raise")
    chans = ["Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
             "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
             "Usage_kWh"]  # dropped NSM (clock ramp) + 3 categoricals; target=Usage_kWh last
    out = df[["date"] + chans].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    _w(out, "Steel")


def gasturbine():
    # UCI #551 combined-cycle gas turbine (TR), hourly, 2011-2015. No native
    # timestamp -> synthetic hourly index. target=NOX last.
    files = [os.path.join(DATA, "GasTurbine", f"gt_{y}.csv") for y in range(2011, 2016)]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.insert(0, "date", pd.date_range("2011-01-01 00:00:00", periods=len(df), freq="h")
              .strftime("%Y-%m-%d %H:%M:%S"))
    _w(df, "GasTurbine")  # NOX already last column in source


def tep():
    # Tennessee Eastman Process, continuous 1-yr sim, 3-min. Drop index+STATUS
    # and the 2 zero-variance manipulated vars XMV(5)/XMV(9). target=XMEAS(1).
    df = pd.read_csv(os.path.join(DATA, "TEP", "raw.csv"))
    df = df.drop(columns=["Unnamed: 0", "STATUS", "XMV(5)", "XMV(9)"])
    df.insert(0, "date", pd.date_range("2020-01-01 00:00:00", periods=len(df), freq="3min")
              .strftime("%Y-%m-%d %H:%M:%S"))
    # move XMEAS(1) to last as target
    cols = [c for c in df.columns if c not in ("date", "XMEAS(1)")]
    _w(df[["date"] + cols + ["XMEAS(1)"]], "TEP")


def windscada():
    # Kelmarsh Turbine 1, 2017, 10-min. 9-line metadata header (skiprows=9);
    # keep 14 physical channels (orig names carry mojibake -> rename ASCII).
    df = pd.read_csv(os.path.join(DATA, "WindSCADA", "turbine1_2017.csv"),
                     skiprows=9, low_memory=False)
    df = df.rename(columns={df.columns[0]: "date"})
    excl = ["standard deviation", "minimum", "maximum", ", min", ", max", ", std",
            "stddev", "density adjusted", "cascading"]
    wants = ["Wind speed (m/s)", "Power (kW)", "Rotor speed (RPM)", "Generator RPM (RPM)",
             "Gearbox speed (RPM)", "Front bearing temperature", "Rear bearing temperature",
             "Nacelle ambient temperature", "Generator bearing rear temperature",
             "Generator bearing front temperature", "Rotor bearing temp", "Wind direction",
             "Nacelle position", "Blade angle (pitch position) A"]
    chosen = []
    for kw in wants:
        m = [c for c in df.columns
             if c.lower().startswith(kw.lower()) and not any(e in c.lower() for e in excl)]
        if m:
            chosen.append(m[0])
    sub = df[["date"] + chosen].copy()
    sub[chosen] = sub[chosen].interpolate(limit=12, limit_direction="both").ffill().bfill()
    pcol = [c for c in chosen if c.lower().startswith("power")][0]
    sub = sub[["date"] + [c for c in chosen if c != pcol] + [pcol]]
    sub.columns = ["date"] + [f"ch{i}" for i in range(len(chosen) - 1)] + ["Power_kW"]
    _w(sub, "WindSCADA")


def pmsm():
    # Paderborn PMSM test bench. REBUILT 2026-07-02 for the short-horizon PdM
    # section: the old single-session 2 Hz series made H=96 span 48 s of slow
    # rotor-thermal drift (copy-last persistence MSE 0.004 -> trivial). Now:
    # ALL 69 sessions, 30-s MEAN aggregation (mean, not decimation -> no
    # aliasing on the fast current/torque channels; matches the 10-min-mean
    # SCADA convention), C-MAPSS-style ['unit', <chans>, pm] schema so
    # Dataset_CMAPSS windows within a session and splits by session (the
    # sessions are independent test-bench drive cycles). Horizons [24,48,96]
    # = 12/24/48 min of motor-thermal forecasting. target=pm last.
    p = pd.read_csv(os.path.join(DATA, "PMSM", "measures_v2.csv"))
    chans = [c for c in p.columns if c not in ("profile_id", "pm")]
    parts = []
    for pid, g in p.groupby("profile_id", sort=True):
        g = g.reset_index(drop=True)
        agg = g[chans + ["pm"]].groupby(g.index // 60).mean()  # 60 x 0.5s = 30s
        agg.insert(0, "unit", pid)
        parts.append(agg)
    out = pd.concat(parts, ignore_index=True)
    _w(out, "PMSM")


def cmapss3():
    # NASA C-MAPSS FD003 — same single-operating-condition setup as FD001 but
    # TWO fault modes (HPC degradation + fan degradation) and longer
    # trajectories (median 220 / max 525 cycles -> ~3x the H=96 windows of
    # FD001). Vetted 2026-07-02: FD003's constant sensors {s1,s5,s16,s18,s19}
    # are a subset of the channels FD001 already drops, so the SAME 14-channel
    # set is kept -> identical D=14 / target=s11, directly comparable columns.
    # (FD002/FD004 rejected: 6 operating conditions driven by the unobserved
    # flight profile -> regime switches are unpredictable; D=21 breaks sub-1K.)
    cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(os.path.join(DATA, "CMAPSS", "raw", "train_FD003.txt"),
                     sep=r"\s+", header=None, names=cols)
    keep = ["s2", "s3", "s4", "s7", "s8", "s9", "s12", "s13", "s14",
            "s15", "s17", "s20", "s21", "s11"]  # s11 last = target (as FD001)
    assert all(df[c].std() > 1e-6 for c in keep), "FD003: kept channel is constant"
    os.makedirs(os.path.join(DATA, "CMAPSS3"), exist_ok=True)
    _w(df[["unit"] + keep].copy(), "CMAPSS3")


def cmapss():
    # NASA C-MAPSS FD001 (turbofan run-to-failure sim, single operating
    # condition). 100 engines, ~200 cycles each. SEPARATE short-horizon PdM
    # section, NOT the main [96..720] table (trajectories too short for long
    # horizons -> Dataset_CMAPSS windows WITHIN an engine and splits by engine
    # so no window crosses an engine boundary). Keep the 14 informative sensors
    # (constant / near-constant s1,s5,s6,s10,s16,s18,s19 dropped -> D=14, sub-1K
    # holds) + the per-engine `unit` id (consumed by the loader, not a channel).
    # target = s11 (degradation-monotonic). No real timestamps.
    cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(os.path.join(DATA, "CMAPSS", "raw", "train_FD001.txt"),
                     sep=r"\s+", header=None, names=cols)
    keep = ["s2", "s3", "s4", "s7", "s8", "s9", "s12", "s13", "s14",
            "s15", "s17", "s20", "s21", "s11"]  # s11 last = target
    _w(df[["unit"] + keep].copy(), "CMAPSS")


RECIPES = {"Steel": steel, "GasTurbine": gasturbine, "TEP": tep,
           "WindSCADA": windscada, "PMSM": pmsm, "CMAPSS": cmapss,
           "CMAPSS3": cmapss3}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(RECIPES)
    for name in todo:
        RECIPES[name]()
