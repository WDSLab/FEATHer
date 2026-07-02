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

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'PEMS03': Dataset_Custom,
    'PEMS04': Dataset_Custom,
    'PEMS07': Dataset_Custom,
    'PEMS08': Dataset_Custom,
    'Weather': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'AirQuality': Dataset_Custom,
    'Volatility': Dataset_Custom,
    'SML': Dataset_Custom,
    'Steel': Dataset_Custom,
    'GasTurbine': Dataset_Custom,
    'TEP': Dataset_Custom,
    'WindSCADA': Dataset_Custom,
    'PMSM': Dataset_CMAPSS,   # rebuilt 2026-07-02: 69 sessions, unit-aware
    'CMAPSS': Dataset_CMAPSS,
    'CMAPSS3': Dataset_CMAPSS,
    'PM': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'nrel': Dataset_Custom,
    'astd': Dataset_Custom,
    'PEMS_BAY': Dataset_Custom,
    'METR': Dataset_Custom,
    'ninja_pv': Dataset_Custom,
    'ninja_pv2': Dataset_Custom,
    'ninja_wind': Dataset_Custom,
    'era_ssr': Dataset_Custom,
    'KnowAir': Dataset_Custom,
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

    elif data == 'PEMS_BAY':
        df = pd.read_csv(root_path + 'PEMS_BAY/pems_bay.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'm'
        embed = 'timeF'

    elif data == 'PEMS03':
        df = pd.read_csv(root_path + 'PEMS/PEMS03.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'm'
        embed = 'timeF'

    elif data == 'PEMS04':
        df = pd.read_csv(root_path + 'PEMS/PEMS04.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'm'
        embed = 'timeF'

    elif data == 'PEMS07':
        df = pd.read_csv(root_path + 'PEMS/PEMS07.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'm'
        embed = 'timeF'

    elif data == 'PEMS08':
        df = pd.read_csv(root_path + 'PEMS/PEMS08.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'm'
        embed = 'timeF'

    elif data == 'METR':
        df = pd.read_csv(root_path + 'METR/metr_la.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'm'
        embed = 'timeF'

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

    elif data == 'AirQuality':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'NOx(GT)'
        freq = 'h'
        embed = 'timeF'

    elif data == 'Volatility':
        df = pd.read_csv(root_path + data + '.csv')
        #
        df = df[['close_price', 'Symbol', 'date']]
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        df['date'] = df['date'].dt.tz_localize(None)
        df = df.pivot(index='date', columns='Symbol', values='close_price')
        nan_counts = df.isna().sum()
        symbols_to_keep = nan_counts.sort_values(ascending=True).head(14).index   # Sort by ascending NaN count and select top 14 Symbols
        df = df[symbols_to_keep]    # Keep only selected Symbols
        #
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = '.SPX'
        freq = 'd'
        embed = 'timeF'

    elif data == 'SML':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.drop(['21:Exterior_Entalpic_turbo', '20:Exterior_Entalpic_2'
                      , '19:Exterior_Entalpic_1'], axis=1)
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = '3:Temperature_Comedor_Sensor'
        freq = 'm'
        embed = 'timeF'

    elif data == 'Steel':
        # DAEWOO Steel Co. (Gwangyang, KR) — 15-min plant energy/power
        # telemetry, 6 continuous channels, full year 2018 (UCI #851).
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'Usage_kWh'
        freq = 'm'
        embed = 'timeF'

    elif data == 'GasTurbine':
        # Combined-cycle gas turbine (Turkey, UCI #551) — 11 sensor channels,
        # hourly aggregates 2011-2015. No native timestamp (synthetic hourly
        # index in data.csv). Target = NOX emissions (PEMS forecasting).
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'NOX'
        freq = 'h'
        embed = 'timeF'

    elif data == 'TEP':
        # Tennessee Eastman Process — process-plant simulation, 52 vars
        # (41 XMEAS + 11 XMV), 3-min sampling, continuous one-year run.
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'XMEAS(1)'
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

    elif data == 'PM':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'pm2.5'
        freq = 'h'
        embed = 'timeF'

    elif data == 'nrel':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = '136'
        freq = 'h'
        embed = 'nrel'
        
    elif data == 'astd':
        df = np.load(root_path + 'astd_array_43.npy')[:, :, 0]    # co2
        df = pd.DataFrame(df, columns=[str(i) for i in range(df.shape[1])])   
        date_index = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
        date_index = pd.DataFrame({"date": date_index})
        df.insert(0, 'date', date_index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        df = df.iloc[500:]
        target = '42'
        freq = 'd'
        embed = 'astd'

    elif data == 'ninja_pv':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'sensor_99'
        freq = 'h'
        embed = 'ninja_pv'
        
    elif data == 'ninja_pv2':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'sensor_98'
        freq = 'h'
        embed = 'ninja_pv'

    elif data == 'ninja_wind':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'sensor_99'
        freq = 'h'
        embed = 'ninja_wind'

    elif data == 'era_ssr':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'h'
        embed = 'era_ssr'

    elif data == 'KnowAir':
        df = pd.read_csv(root_path + data + '.csv')
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= '2018-01-01') & (df['date'] <= '2020-12-31')
        df = df.loc[mask].reset_index(drop=True)
        target = df.columns[-1]
        freq = 'h'
        embed = 'KnowAir'

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
    elif data == 'PEMS_BAY':
        df = pd.read_csv(root_path + 'PEMS_BAY/pems_bay.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'PEMS03':
        df = pd.read_csv(root_path + 'PEMS/PEMS03.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'PEMS04':
        df = pd.read_csv(root_path + 'PEMS/PEMS04.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'PEMS07':
        df = pd.read_csv(root_path + 'PEMS/PEMS07.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'PEMS08':
        df = pd.read_csv(root_path + 'PEMS/PEMS08.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'METR':
        df = pd.read_csv(root_path + 'METR/metr_la.csv', index_col=0)
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
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
    elif data == 'AirQuality':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Volatility':
        df = pd.read_csv(root_path + data + '.csv')
        #
        df = df[['close_price', 'Symbol', 'date']]
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        df['date'] = df['date'].dt.tz_localize(None)
        df = df.pivot(index='date', columns='Symbol', values='close_price')
        nan_counts = df.isna().sum()
        symbols_to_keep = nan_counts.sort_values(ascending=True).head(14).index   # Sort by ascending NaN count and select top 14 Symbols
        df = df[symbols_to_keep]    # Keep only selected Symbols
        #
        df.insert(0, 'date', df.index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'd'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'SML':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.drop(['21:Exterior_Entalpic_turbo', '20:Exterior_Entalpic_2', '19:Exterior_Entalpic_1'], axis=1)
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
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
    elif data == 'GasTurbine':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'TEP':
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
    elif data == 'PMSM':
        # 13 cols (unit + 12 channels) -> worker n_features = shape[1]-1 = 12.
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
    elif data == 'PM':
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
    elif data == 'astd':
        df = np.load(root_path + 'astd_array_43.npy')[:, :, 0]    # co2
        df = pd.DataFrame(df, columns=[str(i) for i in range(df.shape[1])])   
        date_index = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
        date_index = pd.DataFrame({"date": date_index})
        df.insert(0, 'date', date_index)
        df = df.dropna()
        df = df.reset_index(drop=True)
        df = df.iloc[500:]
        freq = 'd'
        embed = 'astd'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ninja_pv':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'sensor_99'
        freq = 'h'
        embed = 'ninja_pv'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ninja_pv2':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'sensor_98'
        freq = 'h'
        embed = 'ninja_pv2'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ninja_wind':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = 'sensor_99'
        freq = 'h'
        embed = 'ninja_wind'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'era_ssr':
        df = pd.read_csv(root_path + data + '.csv')
        df = df.dropna()
        df = df.reset_index(drop=True)
        target = df.columns[-1]
        freq = 'h'
        embed = 'era_ssr'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'KnowAir':
        df = pd.read_csv(root_path + data + '.csv')
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= '2018-01-01') & (df['date'] <= '2020-12-31')
        df = df.loc[mask].reset_index(drop=True)
        target = df.columns[-1]
        freq = 'h'
        embed = 'KnowAir'
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