"""Reproducible preprocessing for the six industrial datasets.

One recipe per dataset (Steel / WindSCADA / BlastFurnace / MetroPT3 / N2O /
EWELD).  Each reads the raw source in data/<name>/ and writes the cleaned
data/<name>/data.csv that utils/data_factory.py loads.  Idempotent: re-running
regenerates data.csv without touching the raw files.

Gap-fill policy: on the gridded sensor sets every gap is held at the last
observation (causal, no interpolation or lookahead; leading edge back-filled),
and a companion mask.csv (1 = observed, 0 = imputed) is written next to
data.csv.  The loader NaN-s imputed TARGET cells so the loss and test metrics
use observed values only; the filled values are still model INPUT.
`_fill_gaps` prints the gap audit on every run so a changed raw download is
visible.  MetroPT3 / BlastFurnace raw are gap-free per their sources: gaps here
are resampling / logger-off time, not sensor NaNs.

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

Run:  python utils/prep_manufacturing.py            # all
      python utils/prep_manufacturing.py Steel N2O  # subset
"""
import os
import sys

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _w(df, name, mask=None):
    out = os.path.join(DATA, name, "data.csv")
    assert df.isna().sum().sum() == 0, f"{name}: NaN present"
    df.to_csv(out, index=False)
    msg = f"{name:10s} -> data.csv  rows={len(df):>7d}  D={df.shape[1]-1}"
    if mask is not None:
        # mask.csv (1=observed, 0=imputed) is aligned BY POSITION to the written
        # data.csv channel columns, so `mask` must already be in that order.
        chan_cols = list(df.columns[1:])                    # drop 'date'
        m = np.asarray(mask).astype("int8")
        assert m.shape == (len(df), len(chan_cols)), (
            f"{name}: mask shape {m.shape} != data "
            f"{(len(df), len(chan_cols))} - column/row misalignment")
        mdf = pd.DataFrame(m, columns=chan_cols)
        mdf.insert(0, "date", df["date"].values)
        mdf.to_csv(os.path.join(DATA, name, "mask.csv"), index=False)
        n_imp = int((m == 0).sum())
        msg += (f"  | mask.csv imputed={n_imp} "
                f"({100.0 * n_imp / max(1, m.size):.2f}%)")
    print(msg)


def _fill_gaps(df, cols, name, interp_limit, step_desc, max_gap_bound=None):
    """Fill the NaN cells of `cols`; returns (filled df, observed mask 1/0).

    Per channel: gaps <= interp_limit steps are linearly interpolated, longer
    gaps hold the last observation (no lookahead), the leading edge is
    back-filled.  All callers pass interp_limit=0 (every gap hold-last).
    Prints the gap audit; `max_gap_bound` (steps) asserts a ceiling so a
    different raw download cannot silently change the data.
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
            f"{max_gap_bound} - raw source changed? Re-vet before filling.")

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
    # every imputed cell is marked unobserved; utils/data_loader.py masks only
    # the TARGET with it (the input keeps the filled values)
    observed = (~isna).astype("int8")                  # DataFrame, columns=cols
    return out, observed


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
    # 10-min grid; all gaps hold-last (worst audited 153 steps = 25.5 h; bound 288 = 2 d).
    sub, obs = _fill_gaps(sub, chosen, "WindSCADA", interp_limit=0,
                          step_desc="10-min", max_gap_bound=288)
    pcol = [c for c in chosen if c.lower().startswith("power")][0]
    order = [c for c in chosen if c != pcol] + [pcol]
    sub = sub[["date"] + order]
    sub.columns = ["date"] + [f"ch{i}" for i in range(len(chosen) - 1)] + ["Power_kW"]
    _w(sub, "WindSCADA", mask=obs[order].values)


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
    #       the TRAIN region only (first 70% rows = Dataset_Custom's train
    #       split; full-series quantiles would leak val/test).  The raw carries
    #       gross sensor faults (a shared ~3311 temperature error sentinel,
    #       impossible negatives, zero-dropouts); the clip touches ~1% of cells
    #       and deletes nothing (a MAD rule over-flagged low-variance channels).
    #       The Tp4 sentinel spikes sit in the later rows, so train-region
    #       bounds clip them where full-series quantiles would not.
    #   (2) asof-merge the most recent Si cast onto each hour (forward-carry;
    #       leading pre-first-cast hours back-filled). Si placed LAST -> target.
    # D=28 (27 process + Si).
    xl = pd.ExcelFile(os.path.join(DATA, "BlastFurnace", "raw",
                                   "1_blast_furnace_data_first_dataset.xlsx"))
    proc = xl.parse("data")
    si = xl.parse("Si")
    proc["dt"] = pd.to_datetime(proc["dt"])
    si["dt"] = pd.to_datetime(si["dt"])
    chans = [c for c in proc.columns if c != "dt"]
    proc = proc.sort_values("dt").reset_index(drop=True)
    # quantile bounds from the train region only, clip applied to the full series
    n_train = int(len(proc) * 0.7)
    for c in chans:
        lo = proc[c].iloc[:n_train].quantile(0.005)
        hi = proc[c].iloc[:n_train].quantile(0.995)
        proc[c] = proc[c].clip(lo, hi)
    si = si.sort_values("dt").reset_index(drop=True)
    merged = pd.merge_asof(proc, si[["dt", "Si"]], on="dt", direction="backward")
    # Rows before the first cast have no backward Si match; their bfill borrows
    # the future first cast, so mask ONLY those leading Si cells.  The
    # between-cast forward-carry is a causal step target and the process
    # channels are gapless, so both stay observed.
    si_observed = merged["Si"].notna().values
    merged["Si"] = merged["Si"].bfill()  # leading hours before the first cast
    merged["date"] = merged["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    obs = np.ones((len(merged), len(chans) + 1), dtype="int8")  # Si is last col
    obs[:, -1] = si_observed.astype("int8")
    _w(merged[["date"] + chans + ["Si"]].copy(), "BlastFurnace",
       mask=obs if (obs == 0).any() else None)


def eweld():
    # EWELD (Liu et al., Scientific Data 10:615, 2023, DOI 10.1038/
    # s41597-023-02503-6; Figshare 10.6084/m9.figshare.21893808): 386
    # industrial/commercial 15-min smart-meter load series (2016-2022).  Each
    # user file = [Time, Value(kWh)] = one load channel.  We build a
    # multichannel load bank from the Section C METAL divisions (C24 basic
    # metals + C25 fabricated metals): the 14 lowest-ID users that cover
    # 2018-01-01..2020-12-31 with >99% of the 15-min grid (residual filled and
    # masked).  D=14; target = last selected user.
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
    obs = {}                       # observed (pre-fill)
    for n in cand:
        df = pd.read_csv(z.open(n)); df["Time"] = pd.to_datetime(df["Time"])
        if df["Time"].min() <= S and df["Time"].max() >= E:
            s = df.set_index("Time")["Value"].reindex(grid)
            if s.isna().mean() < 0.01:
                key = n.split("/")[-1].replace(".csv", "")
                cols[key] = s.ffill().bfill().values
                obs[key] = (~s.isna()).values          # 1=observed, 0=filled
        if len(cols) >= 14:
            break
    # a different raw download (users added/removed) must fail loudly, not
    # silently change D and every trained checkpoint's input width
    assert len(cols) == 14, (
        f"EWELD: expected exactly 14 qualifying users, got {len(cols)} - "
        f"raw EWELD.zip contents changed?")
    out = pd.DataFrame(cols, index=grid).reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # mask the residual filled cells (<1%/channel) like the gap-filled sensor
    # sets; column order matches `out` (dict insertion order)
    mask = np.column_stack([obs[k] for k in cols]).astype("int8")
    _w(out, "EWELD", mask=mask if (mask == 0).any() else None)


def n2o():
    # Full-scale wastewater-treatment plant (Avedore, DK) process + N2O
    # emission monitoring (Mendeley 10.17632/xmbxhscgpr.1; arXiv:2407.05959).
    # SCADA export = 24 value channels + 24 quality flags on a 1-min grid, but
    # each sensor reports only every ~2-4 min (~40% NaN at 1-min).  Keep the
    # 14 continuous sensor channels (blower + per-tank airflow, N2O/NH4/NO3/
    # O2/SS/temperature/PO4, inlet flow); drop quality flags, setpoints,
    # categorical phase codes, valve PCT and inlet state.  Resample to 5-min
    # mean -> residual NaN ~5.8% (hold-last, see _fill_gaps), ~210k rows
    # (2022-06..2024-06).  D=14; target = N2O (the source paper's target).
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
    # 5-min grid; all gaps hold-last (long SCADA outages: INLET.Q up to
    # 26,643 steps = 92 d, airflows 66 d).
    g, obs = _fill_gaps(g, list(ren.values()), "N2O",
                        interp_limit=0, step_desc="5-min", max_gap_bound=30000)
    g = g.reset_index().rename(columns={"time": "date"})
    g["date"] = g["date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    order = [c for c in ren.values() if c != "N2O"] + ["N2O"]
    _w(g[["date"] + order].copy(), "N2O", mask=obs[order].values)


def metropt3():
    # MetroPT-3 (UCI #791; Veloso et al., Scientific Data 9:764, 2022):
    # metro-train Air Production Unit reciprocating air compressor.  UCI ships
    # a ~10-s stream (1,516,948 rows, 2020-02-01..2020-09-01, 0 NaN).  Keep
    # the 7 analog channels (TP2/TP3/H1/DV_pressure/Reservoirs pressures,
    # Motor_current, Oil_temperature); drop the 8 digital status signals.
    # Resample to 1-min mean (mild anti-aliasing): ~307k rows.  D=7.
    # Native task = anomaly detection; the TIME benchmark casts it as
    # forecasting.  target = Oil_temperature (compressor thermal health).
    analog = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
              "Motor_current", "Oil_temperature"]
    df = pd.read_csv(os.path.join(DATA, "MetroPT3", "raw",
                                  "MetroPT3(AirCompressor).csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    g = df.set_index("timestamp")[analog].resample("1min").mean()
    # 1-min grid; all gaps hold-last (logger-off gaps up to 48 h = unobserved
    # time, not sensor NaNs).
    g, obs = _fill_gaps(g, analog, "MetroPT3",
                        interp_limit=0, step_desc="1-min", max_gap_bound=4320)
    g = g.reset_index().rename(columns={"timestamp": "date"})
    g["date"] = g["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    chans = [c for c in analog if c != "Oil_temperature"]
    order = chans + ["Oil_temperature"]
    _w(g[["date"] + order].copy(), "MetroPT3", mask=obs[order].values)


RECIPES = {"Steel": steel, "WindSCADA": windscada, "BlastFurnace": blastfurnace,
           "MetroPT3": metropt3, "N2O": n2o, "EWELD": eweld}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(RECIPES)
    for name in todo:
        RECIPES[name]()
