"""Reproducible preprocessing for the manufacturing datasets (JMS pivot).

9 recipes: the 6 MFG main-table sets (Steel / WindSCADA / BlastFurnace /
MetroPT3 / N2O / EWELD) + the 3 short-horizon PdM shelf sets (CMAPSS /
CMAPSS3 / PMSM, archived from the paper). Each recipe reads the RAW source
already in data/<name>/ and writes the cleaned data/<name>/data.csv that
utils/data_factory.py loads. Idempotent: re-running regenerates data.csv
from raw without touching the raw files.

GAP-FILL POLICY (audit P3, 2026-07-14): for the gridded sensor sets, short
gaps (<= ~1 h) are linearly interpolated; LONGER outages are held at the
last observation (no linear bridging across outages, no lookahead) and only
the leading edge is back-filled. `_fill_gaps` prints the gap audit on every
run so a changed raw re-download is visible. State this policy in the paper.

RAW PROVENANCE (download once, place in data/<name>/):
  Steel        UCI #851  -> Steel_industry_data.csv   (direct UCI zip)
  WindSCADA    Zenodo rec 5841834, Kelmarsh_SCADA_2017_3083.zip
               -> Turbine_Data_Kelmarsh_1_2017-..csv (saved turbine1_2017.csv)
               (Zenodo Cloudflare-blocks scripted downloads -> use a browser)
  BlastFurnace Mendeley 6d7jbc7tb5 (Data in Brief 2025)
               -> raw/1_blast_furnace_data_first_dataset.xlsx (needs openpyxl)
  MetroPT3     UCI #791 -> raw/MetroPT3(AirCompressor).csv
  N2O          Mendeley xmbxhscgpr (Avedore WWTP) -> raw/aved_raw.csv
  EWELD        Figshare 10.6084/m9.figshare.21893808 -> raw/EWELD.zip
  PMSM         Kaggle wkirgsn/electric-motor-temperature -> measures_v2.csv
               (Kaggle needs auth -> download via browser/CLI)
  CMAPSS(3)    NASA C-MAPSS train_FD001.txt / train_FD003.txt -> raw/
               (mirror: github.com/cyrilli/TurboEngine_Dataset_NASA/CMAPSSData)
               SEPARATE short-horizon PdM shelf, NOT the main [96..720] table
               (run-to-failure trajectories ~200 cycles are too short for long
               horizons; engine-aware windowing via Dataset_CMAPSS).

Run:  python tools/prep_manufacturing.py            # all
      python tools/prep_manufacturing.py Steel CMAPSS  # subset
"""
import os
import sys

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _w(df, name):
    out = os.path.join(DATA, name, "data.csv")
    assert df.isna().sum().sum() == 0, f"{name}: NaN present"
    df.to_csv(out, index=False)
    print(f"{name:10s} -> data.csv  rows={len(df):>7d}  D={df.shape[1]-1}")


def _fill_gaps(df, cols, name, interp_limit, step_desc, max_gap_bound=None):
    """AUDIT-FIX (2026-07-14, P3): explicit, bounded gap policy.

    Previously N2O/MetroPT3 ran an UNLIMITED whole-series linear interpolate
    (lookahead endpoints; multi-day outages bridged with fictional ramps)
    and WindSCADA's `interpolate(limit=12).ffill()` let the unlimited ffill
    defeat the limit silently. Policy now, per channel:
      - gaps <= interp_limit steps: linear interpolation (short sensor
        dropouts; both endpoints are at most interp_limit steps away);
      - gaps  > interp_limit steps: HOLD the last observation (sensor-hold,
        no lookahead, no linear bridging across outages);
      - leading edge (no prior observation): back-fill only.
    Prints the gap audit; `max_gap_bound` (steps) asserts a sanity ceiling
    so a different raw re-download cannot silently change the data shape.
    """
    isna = df[cols].isna()
    n_nan = int(isna.to_numpy().sum())
    worst = 0
    for c in cols:
        col_na = isna[c].to_numpy()
        if col_na.any():
            ch = np.diff(np.concatenate([[0], col_na.astype(int), [0]]))
            runs = np.where(ch == -1)[0] - np.where(ch == 1)[0]
            worst = max(worst, int(runs.max()))
    print(f"{name}: NaN cells={n_nan} "
          f"({100.0 * n_nan / max(1, isna.size):.2f}%), "
          f"max gap={worst} steps ({step_desc}) -> "
          f"interp<={interp_limit}, hold-last beyond, bfill lead")
    if max_gap_bound is not None:
        assert worst <= max_gap_bound, (
            f"{name}: max gap {worst} steps exceeds the audited bound "
            f"{max_gap_bound} — raw source changed? Re-vet before filling.")

    out = df.copy()
    interp = out[cols].interpolate(limit_direction="both")
    for c in cols:
        col_na = isna[c]
        # length of the NaN run each cell belongs to
        run_id = (col_na != col_na.shift()).cumsum()
        run_len = col_na.groupby(run_id).transform("sum")
        keep = col_na & (run_len <= interp_limit)      # short gaps -> linear
        out[c] = out[c].where(~keep, interp[c])
    out[cols] = out[cols].ffill().bfill()              # long gaps -> hold
    return out


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
    # 10-min grid: linear interp <= 12 steps (2 h); the worst audited outage
    # is 153 steps (25.5 h, audited 2026-07-14) -> hold-last, bound 288 (2 d).
    sub = _fill_gaps(sub, chosen, "WindSCADA", interp_limit=12,
                     step_desc="10-min", max_gap_bound=288)
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
    #   (1) winsorize each process channel to its [0.5, 99.5] pct computed on
    #       the TRAIN region only (first 70% rows, matching Dataset_Custom's
    #       0.7/0.1/0.2 split — audit P3: full-series quantiles were textbook
    #       preprocessing leakage, 2 scalars/channel). The raw carries gross
    #       sensor faults (a shared ~3311 temperature error SENTINEL,
    #       physically-impossible negatives, zero-dropouts); tail-clip touches
    #       ~1% of cells and DELETES nothing (a MAD rule over-flagged
    #       low-variance channels e.g. Pt 48%). Side benefit (audited
    #       2026-07-14): the Tp4 sentinel spikes live in the LATER rows, so
    #       train-region bounds clip them (hi 3313 -> 855) where full-series
    #       quantiles let them through. Cleaned copy-last persistence
    #       MSE 1.37->3.40 over H=96..720 = non-trivial.
    #   (2) asof-merge the most recent Si cast onto each hour (forward-carry;
    #       leading pre-first-cast hours back-filled). Si placed LAST -> target.
    # D=28 (27 process + Si) -> above sub-1K (STRESS/breadth tier like
    # Weather/Electricity; the sub-1K claim stays on the D<=14 core
    # Steel/WindSCADA/MetroPT3/N2O/EWELD).
    xl = pd.ExcelFile(os.path.join(DATA, "BlastFurnace", "raw",
                                   "1_blast_furnace_data_first_dataset.xlsx"))
    proc = xl.parse("data")
    si = xl.parse("Si")
    proc["dt"] = pd.to_datetime(proc["dt"])
    si["dt"] = pd.to_datetime(si["dt"])
    chans = [c for c in proc.columns if c != "dt"]
    proc = proc.sort_values("dt").reset_index(drop=True)
    # AUDIT-FIX (2026-07-14, P3): quantile bounds from the TRAIN region only
    # (first 70% rows = Dataset_Custom's train split); clip applied to the
    # full series. Full-series quantiles leaked 2 scalars/channel from
    # val/test AND missed the late Tp4 sentinel spikes (see note above).
    n_train = int(len(proc) * 0.7)
    for c in chans:
        lo = proc[c].iloc[:n_train].quantile(0.005)
        hi = proc[c].iloc[:n_train].quantile(0.995)
        proc[c] = proc[c].clip(lo, hi)
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
    # AUDIT-FIX (2026-07-14, P3): a different raw re-download (users added/
    # removed from the C24/C25 divisions) must fail loudly, not silently
    # change D and break every trained checkpoint's input width.
    assert len(cols) == 14, (
        f"EWELD: expected exactly 14 qualifying users, got {len(cols)} — "
        f"raw EWELD.zip contents changed?")
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
    # -> residual NaN ~5.8% (filled per the module gap policy: linear <=1 h,
    # hold-last beyond — see _fill_gaps), ~210k rows over 2 yr
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
           .resample("5min").mean())
    # 5-min grid: linear interp <= 12 steps (1 h). Long SCADA outages exist
    # (audited 2026-07-14: INLET.Q up to 26,643 steps = 92 d; airflows 66 d)
    # -> hold-last, NOT the previous unlimited linear bridge (audit P3).
    g = _fill_gaps(g, list(ren.values()), "N2O",
                   interp_limit=12, step_desc="5-min", max_gap_bound=30000)
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
    g = df.set_index("timestamp")[analog].resample("1min").mean()
    # 1-min grid: linear interp <= 60 steps (1 h). Recording gaps up to 48 h
    # exist (audited 2026-07-14) -> hold-last beyond 1 h (audit P3).
    g = _fill_gaps(g, analog, "MetroPT3",
                   interp_limit=60, step_desc="1-min", max_gap_bound=4320)
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


RECIPES = {"Steel": steel,
           "WindSCADA": windscada, "PMSM": pmsm, "CMAPSS": cmapss,
           "CMAPSS3": cmapss3, "BlastFurnace": blastfurnace,
           "MetroPT3": metropt3, "N2O": n2o, "EWELD": eweld}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(RECIPES)
    for name in todo:
        RECIPES[name]()
