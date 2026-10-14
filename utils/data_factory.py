"""Datasets of the paper: where each one is read from, and the PyTorch loaders built on it.

LTSF-8 come from darts (Electricity from data/electricity.csv); the six industrial sets from data/<name>/data.csv
(written by utils/prep_manufacturing.py, with an optional mask.csv of observed cells).
"""
import os

import pandas as pd
from torch.utils.data import DataLoader

from utils.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

# name: source ("darts" | "csv" = <root>/<name>/data.csv | a file under <root>), target column (None = the last
# column), time-feature frequency (pandas alias: "min" minute, "h" hour, "d" day), Dataset class.  Time encoding:
# "timeF" everywhere except Exchange ("learned").  Minute sets must use "min": pandas reads "m" as MONTH.
DATASETS = {
    "ETTh1":        ("darts", "OT", "h", Dataset_ETT_hour),
    "ETTh2":        ("darts", "OT", "h", Dataset_ETT_hour),
    "ETTm1":        ("darts", "OT", "min", Dataset_ETT_minute),
    "ETTm2":        ("darts", "OT", "min", Dataset_ETT_minute),
    "Weather":      ("darts", "T (degC)", "min", Dataset_Custom),
    "Exchange":     ("darts", "7", "d", Dataset_Custom),
    "Traffic":      ("darts", None, "h", Dataset_Custom),          # all 862 channels (LTSF convention)
    "Electricity":  ("electricity.csv", None, "h", Dataset_Custom),
    # industrial (see utils/prep_manufacturing.py for sources and preprocessing)
    "Steel":        ("csv", "Usage_kWh", "min", Dataset_Custom),        # UCI #851, 15-min plant energy, D=6
    "WindSCADA":    ("csv", "Power_kW", "min", Dataset_Custom),         # Kelmarsh turbine 1, 10-min SCADA, D=14
    "BlastFurnace": ("csv", "Si", "h", Dataset_Custom),               # MISIS blast furnace, hourly, D=28
    "MetroPT3":     ("csv", "Oil_temperature", "min", Dataset_Custom),  # UCI #791 compressor, 1-min, D=7
    "N2O":          ("csv", "N2O", "min", Dataset_Custom),              # Avedore WWTP, 5-min, D=14
    "EWELD":        ("csv", None, "min", Dataset_Custom),               # EWELD metals load, 15-min, D=14
}
DARTS = {"ETTh1": "ETTh1Dataset", "ETTh2": "ETTh2Dataset", "ETTm1": "ETTm1Dataset", "ETTm2": "ETTm2Dataset",
         "Weather": "WeatherDataset", "Exchange": "ExchangeRateDataset", "Traffic": "TrafficDataset"}


def _read(data, root_path):
    """The dataset as a DataFrame with a leading 'date' column, NaN rows dropped."""
    source = DATASETS[data][0]
    if source == "darts":                       # imported only here: the industrial sets do not need darts
        import darts.datasets
        df = getattr(darts.datasets, DARTS[data])().load().to_dataframe()
        df.insert(0, "date", df.index)
    else:
        df = pd.read_csv(os.path.join(root_path, data, "data.csv") if source == "csv" else root_path + source)
        if "date" not in df.columns:
            df.insert(0, "date", df.index)
    return df.dropna().reset_index(drop=True)


def data_select(data, root_path):
    """(DataFrame, freq, embed) of a dataset."""
    _, _, freq, _ = DATASETS[data]
    return _read(data, root_path), freq, ("learned" if data == "Exchange" else "timeF")


def data_provider(root_path, data, features, batch_size, seq_len, label_len, pred_len, flag):
    """(Dataset, DataLoader) for one split (flag = train / val / test)."""
    _, target, freq, Data = DATASETS[data]
    df, _, embed = data_select(data, root_path)
    kwargs = dict(df=df, data_name=data, flag=flag, size=[seq_len, label_len, pred_len], features=features,
                  target=target if target is not None else df.columns[-1], timeenc=int(embed == "timeF"),
                  freq=freq)
    # observed-cell mask of the gap-filled industrial sets (WindSCADA / N2O / MetroPT3): imputed target cells are
    # NaN-ed by Dataset_Custom so that loss and metrics score observed values only (data.csv itself has no NaN)
    mask_path = os.path.join(root_path, data, "mask.csv")
    if Data is Dataset_Custom and os.path.exists(mask_path):
        kwargs["mask_df"] = pd.read_csv(mask_path).reset_index(drop=True)
    data_set = Data(**kwargs)
    print(flag, len(data_set))
    train = flag not in ("test", "val")         # evaluation splits: fixed order, every sample kept
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=train, num_workers=0,
                        drop_last=train)
    return data_set, loader
