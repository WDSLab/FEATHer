from utils.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS, Dataset_CMAPSS
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os


# darts is only needed for the canonical benchmark datasets
# (ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic).
# Importing lazily lets local-CSV datasets (SML, AirQuality, Volatility, etc.)
# work without darts installed.
def _darts():
    from darts.datasets import (
        ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset,
        WeatherDataset, ExchangeRateDataset, ElectricityDataset, TrafficDataset,
    )
    return {
        "ETTh1": ETTh1Dataset, "ETTh2": ETTh2Dataset,
        "ETTm1": ETTm1Dataset, "ETTm2": ETTm2Dataset,
        "Weather": WeatherDataset, "Exchange": ExchangeRateDataset,
        "Electricity": ElectricityDataset, "Traffic": TrafficDataset,
    }

# Active scope only: LTSF-8 (darts ETT/Weather/Exchange/Traffic + local
# electricity.csv) + manufacturing 6 (Steel/WindSCADA/BlastFurnace/MetroPT3/N2O/
# EWELD, locked 2026-07-14) + short-horizon PdM shelf (CMAPSS/CMAPSS3/PMSM) +
# nrel (solar, kept). Removed loaders (PEMS/METR/ninja/era_ssr/KnowAir/astd/
# AirQuality/Volatility/SML/TEP/PM/GasTurbine) — re-add from git history if needed.
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Weather': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'Steel': Dataset_Custom,
    'WindSCADA': Dataset_Custom,
    'BlastFurnace': Dataset_Custom,
    'MetroPT3': Dataset_Custom,
    'N2O': Dataset_Custom,
    'EWELD': Dataset_Custom,
    'PMSM': Dataset_CMAPSS,   # rebuilt 2026-07-02: 69 sessions, unit-aware
    'CMAPSS': Dataset_CMAPSS,
    'CMAPSS3': Dataset_CMAPSS,
    'nrel': Dataset_Custom,
}

# features - Multi-Uni(MS) or Multi-Multi(M) or Uni-Uni(S)
def data_provider(root_path, data, features, batch_size, seq_len, label_len, pred_len, flag, starting_percent=0, percent=100, train_shuffle_flag=True):
    Data = data_dict[data]
    
    if data == 'ETTh1':
        df = _darts()["ETTh1"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        target = 'OT'

    elif data == 'ETTh2':
        df = _darts()["ETTh2"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        target = 'OT'

    elif data == 'ETTm1':
        df = _darts()["ETTm1"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        target = 'OT'

    elif data == 'ETTm2':
        df = _darts()["ETTm2"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        target = 'OT'

    elif data == 'Traffic':
        # Full native 862 channels — LTSF benchmark convention (every
        # baseline's official script uses enc_in=862).
        df = _darts()["Traffic"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        target = df.columns[-1]

    elif data == 'Weather':
        df = _darts()["Weather"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        target = 'T (degC)'

    elif data == 'Exchange':
        df = _darts()["Exchange"]().load().to_dataframe()  
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'd'
        embed = 'learned'
        target = '7'

    elif data == 'Steel':
        # DAEWOO Steel Co. (Gwangyang, KR) — 15-min plant energy/power
        # telemetry, 6 continuous channels, full year 2018 (UCI #851).
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'Usage_kWh'
        freq = 'm'
        embed = 'timeF'

    elif data == 'WindSCADA':
        # Kelmarsh wind farm (UK, Senvion MM92), Turbine 1, 2017, 10-min
        # SCADA. 14 physical channels (wind/power/drivetrain temps/speeds/
        # pitch); short NaN gaps interpolated. Target = Power_kW.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'Power_kW'
        freq = 'm'
        embed = 'timeF'

    elif data == 'BlastFurnace':
        # MISIS blast-furnace process sensors (Mendeley 6d7jbc7tb5, Data in
        # Brief 2025). 27 continuous hourly process channels (blast pressures/
        # temps/gas/flows) + Si (hot-metal silicon content = product-quality
        # target, asof-merged from the irregular per-cast Si sheet). Gapless
        # 2013-2016; raw sensor-fault spikes winsorized in prep. D=27 -> above
        # sub-1K -> STRESS/breadth tier (like Weather/Electricity). Target = Si.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'Si'
        freq = 'h'
        embed = 'timeF'

    elif data == 'EWELD':
        # EWELD (Sci Data 2023) — manufacturing (Section-C metals) 15-min load
        # bank, 14 users over 2018-2020 (3 annual cycles). D=14 -> sub-1K core.
        # daily/weekly/annual multi-period; extreme-weather event windows =
        # real distribution-shift robustness. target = last user (M mode).
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'm'
        embed = 'timeF'

    elif data == 'N2O':
        # Avedore WWTP (DK) process + N2O monitoring (Mendeley xmbxhscgpr;
        # arXiv:2407.05959). 14 continuous sensor channels @ 5-min mean (~210k
        # rows, 2 yr). D=14 -> sub-1K core. Continuous industrial-process
        # monitoring (bio-chemical); target = N2O (emission/quality variable).
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'N2O'
        freq = 'm'
        embed = 'timeF'

    elif data == 'MetroPT3':
        # MetroPT-3 (UCI #791, Sci Data 2022) — metro-train APU air compressor.
        # 7 analog channels (pressures/motor-current/oil-temp), 1-min mean
        # aggregation in prep (~307k rows, Feb-Sep 2020). D=7 -> sub-1K core.
        # Native task = anomaly detection; cast as forecasting (TIME precedent).
        # Target = Oil_temperature (compressor thermal-health indicator).
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'Oil_temperature'
        freq = 'm'
        embed = 'timeF'

    elif data == 'PMSM':
        # Paderborn LEA PMSM test-bench, REBUILT 2026-07-02: all 69 drive-cycle
        # sessions, 30-s mean aggregation, C-MAPSS-style ['unit', chans, pm]
        # schema -> Dataset_CMAPSS windows within a session / splits by session.
        # Short-horizon PdM section [24,48,96] (12/24/48 min). Target = pm
        # (rotor permanent-magnet temperature, the canonical target).
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'pm'
        freq = 'm'
        embed = 'timeF'

    elif data in ('CMAPSS', 'CMAPSS3'):
        # NASA C-MAPSS FD001 / FD003 — turbofan run-to-failure trajectories
        # (SEPARATE short-horizon PdM section, not the main [96..720] table).
        # Engine-aware windowing + split-by-engine in Dataset_CMAPSS; the
        # leading 'unit' column is the engine id (consumed by the loader, not
        # a channel). Same 14 informative sensors in both subsets; target=s11.
        # FD003 = same single operating condition as FD001 but two fault modes
        # and longer trajectories. No real timestamps.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 's11'
        freq = 'h'
        embed = 'timeF'

    elif data == 'nrel':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = '136'
        freq = 'h'
        embed = 'nrel'
        
    elif data == 'Electricity':
        df = pd.read_csv(root_path + 'electricity.csv')
        df.insert(0, 'date', df.index) if 'date' not in df.columns else None
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        target = df.columns[-1]
        
    timeenc = 0 if embed != 'timeF' else 1

    if flag in ('test', 'val'):
        # Evaluation splits: deterministic order, keep every sample.
        shuffle_flag = False
        drop_last = False
        batch_size = batch_size
        freq = freq
    else:
        shuffle_flag = train_shuffle_flag
        drop_last = True
        batch_size = batch_size  # bsz for train
        freq = freq

    data_set = Data(
        df=df,
        data_name=data,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq,
        starting_percent=starting_percent, 
        percent=percent
    )
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)
    
    return data_set, data_loader

def data_select(data, root_path):
    
    if data == 'ETTh1':
        df = _darts()["ETTh1"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ETTh2':
        df = _darts()["ETTh2"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ETTm1':
        df = _darts()["ETTm1"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ETTm2':
        df = _darts()["ETTm2"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Traffic':
        # Full native 862 channels — LTSF benchmark convention.
        df = _darts()["Traffic"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Weather':
        df = _darts()["Weather"]().load().to_dataframe()
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Exchange':
        df = _darts()["Exchange"]().load().to_dataframe()  
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'd'
        embed = 'learned'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Steel':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'WindSCADA':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'BlastFurnace':
        # 29 cols (date + 27 process channels + Si) -> n_features = 28.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'EWELD':
        # 15 cols (date + 14 user load channels) -> n_features = 14.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'N2O':
        # 15 cols (date + 13 channels + N2O) -> n_features = 14.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'MetroPT3':
        # 8 cols (date + 6 channels + Oil_temperature) -> n_features = 7.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'PMSM':
        # 13 cols (unit + 12 channels) -> worker n_features = 12.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data in ('CMAPSS', 'CMAPSS3'):
        # 15 cols (unit + 14 sensors) -> worker n_features = shape[1]-1 = 14.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'nrel':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'nrel'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Electricity':
        df = pd.read_csv(root_path + 'electricity.csv')
        df.insert(0, 'date', df.index) if 'date' not in df.columns else None
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
        
    return df, freq, embed