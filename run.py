import argparse
import torch
from datetime import datetime, timedelta
import time
from types import SimpleNamespace
from utils.pgsql import reset_forecast_table
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import traceback

STATIONS = [

    {"name": "立能派2号厂房光伏发电站", "latitude": 23.67, "longitude": 116.16, "station_id": 1941310480876609537,
     "inv_sncodes": "6T2559050609", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
    {"name": "3号厂房光伏发电站", "latitude": 23.69, "longitude": 116.16, "station_id": 1923271358890020865,
     "inv_sncodes": "GR24C9052308", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
    {"name": "4号厂房光伏发电站", "latitude": 23.69, "longitude": 116.16, "station_id": 1941310656240459778,
     "inv_sncodes": "ES2540029169", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
    {"name": "5号厂房光伏发电站", "latitude": 23.68, "longitude": 116.18, "station_id": 1941310812125962241,
     "inv_sncodes": "ES2530073550", "enc_in": 30, "dec_in": 30, "met_start_idx": 20, "met_end_idx": 29},
    {"name": "惠州泽鑫", "latitude": 22.78, "longitude": 114.41, "station_id": 1,
     "inv_sncodes": "30090549A00248H06969", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
    {"name": "汤西敬老院", "latitude": 24.54, "longitude": 116.10, "station_id": 1923271358890020888,
     "inv_sncodes": "PFKQE5E05U", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
     {"name": "深圳塘头学校", "latitude": 22.66, "longitude": 113.91, "station_id": 1963133351044857857,
     "inv_sncodes": "GR2539079574", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
      {"name": "深圳兴围学校", "latitude": 22.42, "longitude": 114.02, "station_id": 1963133607132282882,
     "inv_sncodes": "2549101064", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
    {"name": "四百亩光伏发电站", "latitude": 23.68, "longitude": 116.18, "station_id": 1941310812125962333,
     "inv_sncodes": "NKETCJF0EC", "enc_in": 30, "dec_in": 30, "met_start_idx": 15, "met_end_idx": 29},
]

def build_args(base_args, station_cfg):
    """把 station 配置合并进 args"""
    args = SimpleNamespace(**vars(base_args))
    args.enc_in = station_cfg["enc_in"]
    args.dec_in = station_cfg["dec_in"]
    args.met_start_idx = station_cfg["met_start_idx"]
    args.met_end_idx = station_cfg["met_end_idx"]

    args.tgt_station = station_cfg["name"]
    args.latitude = station_cfg["latitude"]
    args.longitude = station_cfg["longitude"]
    args.station_id = station_cfg.get("station_id")
    args.inv_sncodes = station_cfg.get("inv_sncodes")
    return args

def generate_t0_list(start_date, end_date, time_points):
    """生成预测起始时间列表"""
    t0_list = []

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        for time_point in time_points:
            t0 = f"{date_str} {time_point}"
            t0_list.append(t0)
        current_date += timedelta(days=1)

    return t0_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--data', type=str, default='SolarPred')
    parser.add_argument('--src_station', type=str, default='立能派四百亩光伏发电站')
    parser.add_argument('--target', type=str, default='pac')
    parser.add_argument('--t0', type=str, default='2025-10-13 05:00:00')
    parser.add_argument('--predict_mode', type=str, choices=['online', 'scheduled'], default='scheduled')

    # scheduled 模式专用参数
    parser.add_argument('--start_date', type=str, default='2025-10-13',
                        help='起始日期 (scheduled模式), 格式: YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default='2025-10-20',
                        help='截止日期 (scheduled模式), 格式: YYYY-MM-DD')
    parser.add_argument('--time_points', type=str, default='08:00:00',
                        help='每日预测时间点 (scheduled模式), 多个时间点用逗号分隔, 格式: HH:MM:SS,HH:MM:SS')

    parser.add_argument('--c_out', type=int, default=3)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding')

    parser.add_argument('--seq_len', type=int, default=156)
    parser.add_argument('--label_len', type=int, default=52)
    parser.add_argument('--pred_len', type=int, default=52)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--inverse', action='store_true', default=True)
    parser.add_argument('--with_meteo', default=True)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    args = parser.parse_args()
    #reset_forecast_table()

    # 初始化所有电站的exp对象
    exps = []
    for station_cfg in STATIONS:
        station_args = build_args(args, station_cfg)
        setting = '{}_{}_sl{}_ll{}_pl{}'.format(
            station_args.src_station,
            station_args.model,
            station_args.seq_len,
            station_args.label_len,
            station_args.pred_len
        )

        exp = Exp_Long_Term_Forecast(station_args)
        exp.load_model(setting)  # 预先加载模型
        exps.append((setting, exp, station_args.tgt_station)) # 保存加载好的模型环境

    if args.predict_mode == 'scheduled':
        time_points = [tp.strip() for tp in args.time_points.split(',')]
        t0_list = generate_t0_list(args.start_date, args.end_date, time_points)
        print(f"[INFO] Scheduled mode detected, using time range:  {args.start_date} ~ {args.end_date}")

        for idx, t0 in enumerate(t0_list, 1):
            print(f"[任务 {idx}/{len(t0_list)}] 预测起始时间: {t0}")

            for setting, exp, tgt_station in exps:
                try:
                    exp.args.t0 = t0
                    print(f"[{t0}] 预测电站: {tgt_station}")
                    exp.predict(setting)
                except Exception as e:
                    print(f"[{t0}] 预测异常: {e}\n{traceback.format_exc()}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[任务 {idx}/{len(t0_list)}] 完成\n")

        print(f"[INFO] 所有 scheduled 预测任务完成!")

    elif args.predict_mode == 'online':
        print(f"[INFO] Real-time mode detected, using specified t0: {args.t0}")

        while True:
            print(f" 开始批量预测...")

            for setting, exp, tgt_station in exps:
                try:
                    print(f"[{args.t0}] 预测 {tgt_station}")
                    exp.predict(setting)
                except Exception as e:
                    print(f"[{args.t0}] 预测异常：{e}\n{traceback.format_exc()}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[{args.t0}] 本轮预测完成，等待下一周期...")
            sleep_seconds = 60 * 60 * 6
            time.sleep(sleep_seconds) # 6小时迭代一轮


