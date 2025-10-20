import pandas as pd
from data_provider.meteo_data import fetch_openmeteo_data
from typing import List
from utils.preddate import generate_pred_dates
from data_provider.optimized_pgsql_data import PGSQLDataFetcher


def _ordered_columns(
    date_col: str,
    inverter_col: str,
    influx_df: pd.DataFrame,
    meteo_df: pd.DataFrame,
    target_col: str = "eac_today", #eacToday
) -> List[str]:
    """Return columns in the requested order."""
    influx_other = [c for c in influx_df.columns if c not in {date_col, inverter_col, target_col}]
    meteo_cols = [c for c in meteo_df.columns if c != date_col]
    return [date_col, inverter_col, *influx_other, *meteo_cols, target_col]


def pad_pgsql(pgsql_df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
    pgsql_df = pgsql_df.copy()
    pgsql_df['date'] = pd.to_datetime(pgsql_df['date'])

    last_ts = pgsql_df['date'].iloc[-1]
    extra_dates = generate_pred_dates(last_ts, pred_len)
    pad_df = pd.DataFrame({
        'date': extra_dates,
    })

    num_cols = [c for c in pgsql_df.columns if c not in ('date', 'inverter_code')]
    for c in num_cols:
        pad_df[c] = 0.0

    if 'inverter_code' in pgsql_df.columns:
        inverter_code_value = pgsql_df['inverter_code'].iloc[0]
        pad_df['inverter_code'] = inverter_code_value

    padded_df = pd.concat([pgsql_df, pad_df], axis=0, ignore_index=True)
    return padded_df


def merge_pgsql_and_meteo(
    seq_len: int, pred_len: int,
    longitude: float, latitude: float,
    station_id: int, t0: str,
    inv_sncodes: str | list[str] = None,
) -> tuple[pd.DataFrame, float]:

    # 获取光伏数据
    with PGSQLDataFetcher() as fetcher:
        pgsql_df, t0_energy = fetcher.fetch_pgsql_sta_data(station_id = station_id, t0=t0,
                                                           limit_num = seq_len, sncode = inv_sncodes)

    pgsql_df = pad_pgsql(pgsql_df = pgsql_df, pred_len=pred_len)

    meteo_start_date = pgsql_df["date"].min().strftime("%Y-%m-%d")
    meteo_end_date = pgsql_df["date"].max().strftime("%Y-%m-%d")

    # 获取气象数据
    meteo_min15, _ = fetch_openmeteo_data(longitude = longitude, latitude = latitude,
                                          start_date = meteo_start_date, end_date = meteo_end_date)

    pgsql_df["date"] = pd.to_datetime(pgsql_df["date"]).dt.tz_localize(None)
    meteo_min15["date"] = pd.to_datetime(meteo_min15["date"]).dt.tz_localize(None)

    merged = pd.merge(pgsql_df, meteo_min15, on='date', how='left', validate='one_to_one')

    # 检测 NaN
    if merged.isna().any().any():
        nan_info = merged.isna().sum()
        print("⚠️ 警告: merge 后出现 NaN 值!")
        print("各列 NaN 数量如下:")
        print(nan_info[nan_info > 0])
        raise ValueError("程序暂停：merge 结果含 NaN，请检查数据对齐。")

    cols = {"date", "inverter_code", "pac_y"}
    middle_cols = [c for c in merged.columns if c not in cols]
    ordered_cols = ["date"] + ["inverter_code"] + middle_cols + ["pac_y"]
    merged = merged[ordered_cols]

    return merged, t0_energy

