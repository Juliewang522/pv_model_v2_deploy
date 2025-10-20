import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from utils.timefeatures import time_features
from utils.inverterfeatures import inverter_features
from data_provider.merge_pgsql_meteo import merge_pgsql_and_meteo

warnings.filterwarnings('ignore')

class Dataset_Influx_Meteo_Pred(Dataset):
    # 包含气象数据的预测推理(batch_y的时间戳比batch_x更长，data需要拼接未来时间戳)

    def __init__(self, size=None, scale=True, inverse=True,
                 flag='influx_pred', freq='t', scalers=None, t0 = None,
                 station_id=None, latitude = None, longitude = None, inv_sncodes = None):
        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 156
            self.label_len = 52
            self.pred_len = 52
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['influx_pred']

        self.station_id = station_id
        self.latitude = latitude
        self.longitude = longitude
        self.inv_sncodes = inv_sncodes

        self.freq = freq
        self.scale = scale
        self.inverse = inverse
        self.scalers = scalers
        self.t0 = t0
        self.t0_energy = None

        self.__read_data__()

    def __read_data__(self):

        if self.scalers is not None:
            self.feat_scaler = self.scalers["feat_scaler"]
            self.target_scaler = self.scalers["target_scaler"]
            df_raw, self.t0_energy = merge_pgsql_and_meteo(seq_len=self.seq_len, pred_len=self.pred_len,
                                                           longitude=self.longitude, latitude=self.latitude, t0= self.t0,
                                                           station_id = self.station_id, inv_sncodes = self.inv_sncodes)
            border1 = len(df_raw) - self.seq_len - self.pred_len
            border2 = len(df_raw)

        feature_cols = list(df_raw.columns[2:-1])
        self.target = df_raw.columns[-1]

        if self.scalers is None:
            self.feat_scaler.fit(df_raw[feature_cols].values)
            self.target_scaler.fit(df_raw[[self.target]].values)

        data_features = self.feat_scaler.transform(df_raw[feature_cols].values)
        data_target = self.target_scaler.transform(df_raw[[self.target]].values)

        data_inverter = inverter_features(df_raw['inverter_code'], self.station_id)
        data = np.concatenate([data_inverter, data_features, data_target], axis=1)

        df_stamp = df_raw[['date']][border1:border2]
        self.date = pd.to_datetime(df_stamp['date'].values)
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_raw.values[border1:border2, 1:]
        else:
            self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        if self.inverse:
            seq_y = self.data_x[r_begin:r_end]
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def get_scaler(self):
        return self.feat_scaler, self.target_scaler

    def get_t0_energy(self):
        return self.t0_energy

    def get_date(self):
        return self.date[-self.pred_len:]

    def get_attribute(self):
        return self.target

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)
