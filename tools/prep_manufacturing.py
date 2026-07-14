"""Reproducible preprocessing for the 5 manufacturing LTSF datasets (JMS pivot).

Each recipe reads the RAW source already in data/<name>/ and writes the
cleaned data/<name>/data.csv that utils/data_factory.py loads. Idempotent:
re-running regenerates data.csv from raw without touching the raw files.

RAW PROVENANCE (download once, place in data/<name>/):
  Steel       UCI #851  -> Steel_industry_data.csv   (direct UCI zip)
  GasTurbine  UCI #551  -> gt_2011.csv .. gt_2015.csv (direct UCI zip)
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


def blastfurnace():
    # MISIS blast-furnace process sensors (Mendeley 6d7jbc7tb5, Data in Brief
    # 2025, DOI 10.1016/j.dib.2025.112159). Two sheets in the raw .xlsx:
    #   'data' = 27 CONTINUOUS hourly process channels (blast flow/pressures,
    #            hot/cold-blast + top/peripheral-gas temps, CO2/H2, ore-to-coke
    #            ratio), gapless 2013-01-01..2016-05-18 (29,602 rows, 0 NaN).
    #   'Si'   = hot-metal silicon content (= PRODUCT QUALITY, the source
    #            paper's forecasting target), measured PER-CAST -> irregular
    #            (~every 1.8 h, 16,589 rows).
    # Build one multivariate hourly series:
    #   (1) winsorize each process channel to its [0.5, 99.5] pct — the raw
    #       carries gross sensor faults (a shared ~3311 temperature error
    #       SENTINEL, physical-impossible negatives, zero-dropouts); tail-clip
    #       touches 0.84% of cells and DELETES nothing (a MAD rule over-flagged
    #       low-variance channels e.g. Pt 48%). Cleaned copy-last persistence
    #       MSE 1.37->3.40 over H=96..720 = non-trivial (cf. GasTurbine 1.3-1.6).
    #   (2) asof-merge the most recent Si cast onto each hour (forward-carry;
    #       leading pre-first-cast hours back-filled). Si placed LAST -> target.
    # D=27 -> above sub-1K (STRESS/breadth tier like Weather/Electricity; the
    # sub-1K claim stays on the D<=14 core Steel/GasTurbine/WindSCADA).
    xl = pd.ExcelFile(os.path.join(DATA, "BlastFurnace", "raw",
                                   "1_blast_furnace_data_first_dataset.xlsx"))
    proc = xl.parse("data")
    si = xl.parse("Si")
    proc["dt"] = pd.to_datetime(proc["dt"])
    si["dt"] = pd.to_datetime(si["dt"])
    chans = [c for c in proc.columns if c != "dt"]
    for c in chans:
        lo, hi = proc[c].quantile(0.005), proc[c].quantile(0.995)
        proc[c] = proc[c].clip(lo, hi)
    proc = proc.sort_values("dt").reset_index(drop=True)
    si = si.sort_values("dt").reset_index(drop=True)
    merged = pd.merge_asof(proc, si[["dt", "Si"]], on="dt", direction="backward")
    merged["Si"] = merged["Si"].bfill()  # leading hours before the first cast
    merged["date"] = merged["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    _w(merged[["date"] + chans + ["Si"]].copy(), "BlastFurnace")


def eweld():
    # EWELD (Liu et al., Scientific Data 10:615, 2023, DOI 10.1038/
    # s41597-023-02503-6; Figshare 10.6084/m9.figshare.21893808) — 386
    # industrial/commercial 15-min smart-meter load series (2016-2022) with
    # 5,741 labeled extreme-weather events. Each user file = [Time, Value(kWh)]
    # = a single load channel. We build a manufacturing multivariate load-bank
    # from Section C (Manufacturing) METAL divisions (C24 basic metals + C25
    # fabricated metals — domain-aligned with Steel/BlastFurnace): the 14
    # lowest-ID users that fully cover a gap-free 3-year window 2018-01-01..
    # 2020-12-31 (>99% complete on the 15-min grid; residual forward-filled).
    # D=14 -> sub-1K CORE. 15-min sampling -> daily(96)/weekly/ANNUAL multi-
    # period (6-year source gives a real annual harmonic) -> showcases the
    # multi-band head; the marked extreme-weather windows feed the real
    # distribution-shift robustness story (vs synthetic corruption).
    # NOTE: strong daily cycle -> copy-last-STEP persistence 1.27-1.47 (non-
    # trivial) but seasonal copy-last-DAY ~0.55 -> MUST report a seasonal-naive
    # baseline; the hard/robustness value lives in the extreme-weather windows.
    # target = last selected user (all channels forecast in M mode).
    import zipfile
    S, E = pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31 23:45")
    grid = pd.date_range(S, E, freq="15min")
    z = zipfile.ZipFile(os.path.join(DATA, "EWELD", "raw", "EWELD.zip"))
    cand = sorted(
        [n for n in z.namelist() if n.endswith(".csv") and "/C/" in n
         and ("C24 Manufacture of basic metals" in n
              or "C25 Manufacture of fabricated metal" in n)],
        key=lambda n: int(n.split("/")[-1].replace("U", "").replace(".csv", "")))
    cols = {}
    for n in cand:
        df = pd.read_csv(z.open(n)); df["Time"] = pd.to_datetime(df["Time"])
        if df["Time"].min() <= S and df["Time"].max() >= E:
            s = df.set_index("Time")["Value"].reindex(grid)
            if s.isna().mean() < 0.01:
                cols[n.split("/")[-1].replace(".csv", "")] = s.ffill().bfill().values
        if len(cols) >= 14:
            break
    out = pd.DataFrame(cols, index=grid).reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    _w(out, "EWELD")


def n2o():
    # Full-scale wastewater-treatment plant (Avedore, DK) process + N2O
    # emission monitoring (Mendeley 10.17632/xmbxhscgpr.1; arXiv:2407.05959,
    # 2024, "...Modeling and Forecasting of N2O..."). SCADA export = 24 value
    # channels + 24 quality flags on a 1-min grid, but each sensor reports only
    # every ~2-4 min -> ~40% NaN at 1-min. Keep the 14 CONTINUOUS sensor
    # channels (blower + per-tank airflow, N2O/NH4/NO3/O2/SS/temperature/PO4,
    # inlet flow); DROP the quality flags, O2/phase SETPOINTS, PROCESSPHASE /
    # PHASECODE (categorical), valve PCT, inlet STATE. Resample to 5-min MEAN
    # -> residual NaN 40%->5.8% (interpolate), ~210k rows over 2 yr
    # (2022-06..2024-06). Standardized copy-last persistence 1.10->1.53 over
    # H=96..720 = non-trivial. D=14 -> sub-1K CORE. Continuous INDUSTRIAL
    # PROCESS monitoring (bio-chemical); target = N2O (the source paper's
    # forecasting target = greenhouse-gas emission / process-quality variable).
    ren = {"BIOLOGY.BLOWERSTATION 1.Q.AIRFLOW value": "Blower_airflow",
           "BIOLOGY.LINE 3 TANK 1.NH4 value": "T1_NH4",
           "BIOLOGY.LINE 3 TANK 1.NO3 value": "T1_NO3",
           "BIOLOGY.LINE 3 TANK 1.O2 value": "T1_O2",
           "BIOLOGY.LINE 3 TANK 1.Q.AIRFLOW value": "T1_airflow",
           "BIOLOGY.LINE 3 TANK 1.SS value": "T1_SS",
           "BIOLOGY.LINE 3 TANK 1.TEMPERATURE value": "T1_temp",
           "BIOLOGY.LINE 3 TANK 2.O2 value": "T2_O2",
           "BIOLOGY.LINE 3 TANK 2.Q.AIRFLOW value": "T2_airflow",
           "BIOLOGY.LINE 3 TANK 2.SS value": "T2_SS",
           "BIOLOGY.LINE 3 TANK 2.TEMPERATURE value": "T2_temp",
           "BIOLOGY.LINE 3 TANK 1.PO4 value": "T1_PO4",
           "INLET.Q value": "Inlet_Q",
           "BIOLOGY.LINE 3 TANK 1.N2O value": "N2O"}   # N2O last = target
    df = pd.read_csv(os.path.join(DATA, "N2O", "raw", "aved_raw.csv"))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    g = (df.set_index("time")[list(ren)].rename(columns=ren)
           .resample("5min").mean().interpolate(limit_direction="both"))
    g = g.reset_index().rename(columns={"time": "date"})
    g["date"] = g["date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    _w(g[["date"] + [c for c in ren.values() if c != "N2O"] + ["N2O"]].copy(), "N2O")


def metropt3():
    # MetroPT-3 (UCI #791; Veloso et al., Scientific Data 9:764, 2022) —
    # metro-train Air Production Unit (APU) reciprocating air compressor. UCI
    # ships a ~10-s decimated stream (1,516,948 rows, 2020-02-01..2020-09-01,
    # 0 NaN). Keep the 7 ANALOG sensor channels (TP2/TP3/H1/DV_pressure/
    # Reservoirs pressures, Motor_current, Oil_temperature); DROP the 8 digital/
    # binary status signals (COMP/DV_eletric/Towers/MPG/LPS/Pressure_switch/
    # Oil_level/Caudal_impulses) — not continuous. Resample to 1-min MEAN (mild
    # anti-aliasing aggregation, same convention as the rebuilt PMSM / SCADA):
    # ~307k rows; standardized copy-last persistence 0.80->1.14 over H=96..720
    # = non-trivial (native 10-s also passes at ~1.4; the compressor duty-cycle
    # makes copy-last poor -> NOT the flat-plateau triviality trap that killed
    # PMSM-at-2Hz). D=7 -> sub-1K CORE (like ETT). Native task = anomaly
    # detection; the TIME benchmark casts it as forecasting (cite as precedent).
    # target = Oil_temperature (compressor thermal-health indicator).
    analog = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
              "Motor_current", "Oil_temperature"]
    df = pd.read_csv(os.path.join(DATA, "MetroPT3", "raw",
                                  "MetroPT3(AirCompressor).csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    g = (df.set_index("timestamp")[analog].resample("1min").mean()
           .interpolate(limit_direction="both"))
    g = g.reset_index().rename(columns={"timestamp": "date"})
    g["date"] = g["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    chans = [c for c in analog if c != "Oil_temperature"]
    _w(g[["date"] + chans + ["Oil_temperature"]].copy(), "MetroPT3")


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


RECIPES = {"Steel": steel, "GasTurbine": gasturbine,
           "WindSCADA": windscada, "PMSM": pmsm, "CMAPSS": cmapss,
           "CMAPSS3": cmapss3, "BlastFurnace": blastfurnace,
           "MetroPT3": metropt3, "N2O": n2o, "EWELD": eweld}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(RECIPES)
    for name in todo:
        RECIPES[name]()
