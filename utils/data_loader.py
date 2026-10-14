"""Sliding-window datasets for one split (train / val / test) of a multichannel series.

Each item = (input window, label + target window, their time marks).  The three classes differ only in where the
splits are cut (and Dataset_Custom puts the target column last and can hide imputed target cells).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

SPLITS = {"train": 0, "val": 1, "test": 2}


class _Windows(Dataset):
    minute_marks = False                      # timeenc=0 marks: add a quarter-hour column (ETT minute)

    def __init__(self, df, data_name, flag="train", size=None, features="S", target="OT", scale=True,
                 timeenc=0, freq="h", mask_df=None):
        self.seq_len, self.label_len, self.pred_len = size or (24 * 4 * 4, 24 * 4, 24 * 4)
        self.set_type = SPLITS[flag]
        self.data_name, self.features, self.target, self.scale = data_name, features, target, scale
        self.timeenc, self.freq = timeenc, freq
        self.df_raw = self._order_columns(df)
        self._read(mask_df)

    def _order_columns(self, df):
        return df

    def _borders(self, n_rows):
        """([train, val, test] start rows, [train, val, test] end rows); val / test start seq_len early."""
        raise NotImplementedError

    def _read(self, mask_df):
        starts, ends = self._borders(len(self.df_raw))
        border1, border2 = starts[self.set_type], ends[self.set_type]
        print("Current border1, border2 is:", border1, border2)

        cols = list(self.df_raw.columns[1:]) if self.features in ("M", "MS") else [self.target]
        values = self.df_raw[cols].values
        self.scaler = StandardScaler()
        if self.scale:                        # fitted on the train split only
            self.scaler.fit(values[starts[0]:ends[0]])
            values = self.scaler.transform(values)

        self.border1 = border1                # window w spans raw rows border1 + w ... (--dump_preds)
        self.data_x = values[border1:border2]
        self.data_y = values[border1:border2]
        self.data_stamp = self._marks(pd.to_datetime(self.df_raw["date"].values[border1:border2]))
        if mask_df is not None:               # observed-only scoring: imputed target cells -> NaN (input keeps them)
            assert len(mask_df) == len(self.df_raw), f"{self.data_name}: mask rows != data rows"
            observed = mask_df[cols].to_numpy().astype(bool)[border1:border2]
            self.data_y = np.asarray(self.data_y, dtype=float).copy()
            self.data_y[~observed] = np.nan

    def _marks(self, dates):
        if self.timeenc == 1:
            return time_features(dates, freq=self.freq).transpose(1, 0)
        marks = [dates.month, dates.day, dates.weekday, dates.hour]
        if self.minute_marks:
            marks.append(dates.minute // 15)
        return np.stack([np.asarray(m, dtype=np.int64) for m in marks], axis=1)

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[index:s_end], self.data_y[r_begin:r_end],
                self.data_stamp[index:s_end], self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(_Windows):
    """ETTh1/ETTh2: 12 / 4 / 4 months of hourly rows."""

    def _borders(self, n_rows):
        month = 30 * 24
        return [0, 12 * month - self.seq_len, 16 * month - self.seq_len], [12 * month, 16 * month, 20 * month]


class Dataset_ETT_minute(_Windows):
    """ETTm1/ETTm2: 12 / 4 / 4 months of 15-minute rows."""
    minute_marks = True

    def _borders(self, n_rows):
        month = 30 * 24 * 4
        return [0, 12 * month - self.seq_len, 16 * month - self.seq_len], [12 * month, 16 * month, 20 * month]


class Dataset_Custom(_Windows):
    """Every other dataset: target column last, contiguous 70 / 10 / 20 % split (Autoformer convention)."""

    def _order_columns(self, df):
        cols = [c for c in df.columns if c not in ("date", self.target)]
        return df[["date"] + cols + [self.target]]

    def _borders(self, n_rows):
        n_train, n_test = int(n_rows * 0.7), int(n_rows * 0.2)
        n_val = n_rows - n_train - n_test
        return [0, n_train - self.seq_len, n_rows - n_test - self.seq_len], [n_train, n_train + n_val, n_rows]
