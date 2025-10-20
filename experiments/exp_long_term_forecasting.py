from data_provider.data_factory import data_provider
from model import Transformer
import torch
import os
from utils.pgsql import write_df_to_pg
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(object):

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        self.model_dict = {
            'Transformer': Transformer,
        }
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        # Transformer.Model(self.args).float()
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _modify_model_architecture(self):

        if hasattr(self.model.decoder, 'projection'):
            original_projection = self.model.decoder.projection
            in_features = original_projection.in_features
            out_features = self.args.c_out

            new_projection = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(negative_slope=0.001, inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features // 2, out_features)
            )

            # 替换原始投影层
            self.model.decoder.projection = new_projection

        # 将新的投影层移到正确的设备
        self.model.decoder.projection = self.model.decoder.projection.to(self.device)

    def load_model(self, setting):
        """只在程序启动时调用一次"""
        """ 
           scalers: 来自 ckpt 的 {'feat_scaler': feat_scaler, 'target_scaler': target_scaler}
        """
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, self.args.src_station + self.args.tgt_station + '_checkpoint.pth')
        self._modify_model_architecture()  # 迁移学习进行模型修改
        self.model.load_state_dict(torch.load(best_model_path,map_location=torch.device('cpu')))

        scaler_path = os.path.join(path, self.args.tgt_station + '_scalers.pth')
        self.scalers = torch.load(scaler_path, weights_only=False)

        self.model.eval()

    def _get_data(self, flag: str):
        data_set, data_loader = data_provider(self.args, flag, self.scalers)
        return data_set, data_loader

    def predict(self, setting):
        pred_data, pred_loader = self._get_data(flag='influx_pred')  # 光伏数据是pred,有气象数据是influx_pred

        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.with_meteo:
                    # decoder input(meteo)
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    nwp_forecast = batch_y[:, -self.args.pred_len:, self.args.met_start_idx:self.args.met_end_idx]
                    dec_inp[..., self.args.met_start_idx:self.args.met_end_idx] = nwp_forecast.float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    # decoder input(no meteo)
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()

                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)

                outputs = np.clip(outputs, 0, None)  # 或 outputs[outputs < 0] = 0
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './station_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        date =  pred_data.get_date()
        first_pred = preds[0].squeeze()

        pac_long = pd.DataFrame({
            'report_time': date,
            'station_id': self.args.station_id,
            'attribute': 'pac',
            'value': first_pred[:, 1],
        })

        eac_long = pd.DataFrame({
            'report_time': date,
            'station_id': self.args.station_id,
            'attribute': 'eac_today'
        })

        eac_long['report_time'] = pd.to_datetime(eac_long['report_time'])
        eac_long['value'] = (pac_long['value'].clip(lower=0)) * (15 / 60.0)

        # 按 逆变器 + 当日 分组后，按时间排序再累计
        eac_long = eac_long.sort_values(['report_time'])  # inverter_node
        eac_long['value'] = eac_long.groupby([eac_long['report_time'].dt.date])['value'].cumsum()

        # 仅对最近的当天增加发电量的起始值
        t0 = pred_data.get_t0_energy()
        print("当前时刻的累计发电量energy_t0:",t0)
        first_day = eac_long['report_time'].dt.floor('D').min()
        eac_long.loc[eac_long['report_time'].dt.floor('D') == first_day, 'value'] += t0

        df_long = pd.concat([pac_long, eac_long], ignore_index=True)
        df_long_pgsql = df_long[['report_time', 'station_id', 'attribute', 'value']]

        write_df_to_pg(df_long_pgsql)

        return

