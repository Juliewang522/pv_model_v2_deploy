import pandas as pd

def generate_pred_dates2(start_datetime, total_slots=58):
    start_datetime = pd.to_datetime(start_datetime)
    start_day = start_datetime.date()

    # 一天内的基础时间段（time对象）
    base_times = pd.date_range("05:30", "19:30", freq="5min").time  # 共29个点

    all_times = []
    day_offset = 0

    while len(all_times) < total_slots:
        current_day = start_day + pd.Timedelta(days=day_offset)

        # 拼接当前天的日期与时间组成完整时间戳
        day_times = pd.to_datetime([f"{current_day} {t}" for t in base_times])

        if day_offset == 0:
            # 第一天只保留 start_datetime 之后的时间点
            day_times = day_times[day_times > start_datetime]

        all_times.extend(day_times.tolist())
        day_offset += 1

    # 截取前 total_slots 个结果
    return pd.DatetimeIndex(all_times[:total_slots])


def generate_pred_dates(start_datetime, total_slots=98, *, freq="15min", include_start=False):
    """
    连续生成 5min 时间戳（全天 24h），自动跨天。
    - start_datetime: 起点（可以是 str/Timestamp）
    - total_slots: 需要的点数（一天 5min 共 288 个）
    - include_start: True 时包含 start_datetime 所在的整点；默认从“下一个 5min 整点”开始
    """
    ts = pd.to_datetime(start_datetime)
    step = pd.Timedelta(freq)

    # 起始点：包含当前整点或取下一个整点
    first = ts.floor(freq) if include_start else ts.floor(freq) + step

    return pd.date_range(start=first, periods=total_slots, freq=freq)


