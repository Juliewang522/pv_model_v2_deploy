import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# 此处已修改为虚拟的账户密码(仅为展示)
conn = psycopg2.connect(
        host="192.168.0.238",
        port="5892",
        dbname="ems",
        user="LNP",
        password="lnp@time",
    )

def reset_forecast_table():
    """清空 iot.pvs_forecast 表"""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE iot.pvs_forecast;")
    conn.commit()
    print("已清空 iot.pvs_forecast 表。")

def write_df_to_pg(df: pd.DataFrame):
    if df.empty:
        return 0

    df = df.copy()
    if "report_time" in df.columns:
        df["report_time"] = pd.to_datetime(df["report_time"], errors="coerce")
    df["value"] = df["value"].map(lambda x: None if pd.isna(x) else f"{x:.2f}")

    records = df.where(pd.notnull(df), None).to_numpy().tolist()
    cols = list(df.columns)
    col_list = ", ".join([f'"{c}"' for c in cols])
    fq_table = "iot.pvs_forecast"

    # 插入时键冲突，就改为更新
    sql = (
        f'INSERT INTO {fq_table} ({col_list}) VALUES %s '
        f'ON CONFLICT ("report_time","station_id","attribute") '
        f'DO UPDATE SET "value" = EXCLUDED."value";'
    )

    with conn.cursor() as cur:
        execute_values(cur, sql, records)
    conn.commit()
    return len(records)
