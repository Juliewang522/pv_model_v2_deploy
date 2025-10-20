import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


class PGSQLDataFetcher:

    def __init__(self, host="192.168.10.4", port="5432", dbname="ems",
                 user="LNP", password="lnp@timescaledb"):
        """初始化数据库连接"""
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )

    def try_sql(self, limit_num: int, sncode: str, start_date: str, end_date: str,
                                     end_hour: int = 19, end_minute: int = 30) -> str:
        return f"""
        SELECT *
        FROM (
        -- 1. 生成完整的时间序列（CTE）
            WITH time_series AS (
                SELECT 
                    time_bucket('15 minutes', ts) AS report_time
                FROM 
                    generate_series(
                        '{start_date} 00:00:00'::timestamp,
                        '{end_date} 23:45:00',
                        '15 minutes'::interval
                    ) AS ts
            ),
            -- 2. 查询原始数据并聚合
            raw_data AS (
                SELECT
                    time_bucket('15 minutes', report_time) AS report_time,
                    sncode,
                    ATTRIBUTE,
                    ROUND(AVG(NULLIF(VALUE, '')::numeric), 2) AS avg_value
                FROM
                    iot.pvs_inv_tm
                WHERE
                    sncode = '{sncode}'
                    AND report_time >= '{start_date} 00:00:00'
                    AND report_time < '{end_date} {end_hour:02d}:{end_minute:02d}:00'::timestamp + INTERVAL '15 minutes'
                    AND ATTRIBUTE in (
                          'grid_u_ab','grid_u_ca','grid_u_bc',
                          'grid_i_a', 'grid_i_b', 'grid_i_c',
                          'grid_u_a', 'grid_u_b', 'grid_u_c',
                          'fac','pac','pf','dc_total_p',
                          'rac','temp','eff','eac_today'
                      )
                GROUP BY
                    time_bucket('15 minutes', report_time),
                    sncode,
                    ATTRIBUTE
            )
            -- 3. 左连接时间序列和原始数据，补零
            SELECT
                ts.report_time AS time_stamp,
                COALESCE(rd.sncode, '{sncode}') AS inverter_code,
                COALESCE(rd.ATTRIBUTE, 'pac') AS ATTRIBUTE,
                COALESCE(rd.avg_value, 0) AS avg_value  -- 缺失值补0
            FROM
                time_series ts
            LEFT JOIN
                raw_data rd ON ts.report_time = rd.report_time
            ORDER BY
                ts.report_time
        ) sub
        ORDER BY time_stamp;
"""

    def _get_station_power_15min_sql(self, station_id: str, limit_num: int, start_date: str, end_date: str,
                                     end_hour: int = 19, end_minute: int = 30) -> str:
        return f"""
        SELECT *
        FROM (
            WITH time_windows AS (
                SELECT 
                    generate_series(
                        '{start_date} 00:00:00'::timestamp,
                        '{end_date} 23:45:00'::timestamp,
                        '15 minutes'::interval
                    ) AS window_start
            ),
            filtered_windows AS (
                SELECT 
                    window_start,
                    window_start + INTERVAL '15 minutes' AS window_end
                FROM time_windows
                WHERE 
                    EXTRACT(HOUR FROM window_start) >= 0
                    AND (
                        EXTRACT(HOUR FROM window_start) < 23
                        OR (EXTRACT(HOUR FROM window_start) = 23 AND EXTRACT(MINUTE FROM window_start) <= 45)
                    )
            ),
            raw_inv_data AS (
                SELECT 
                    DATE_TRUNC('hour', tm.report_time) AS hour,
                    (EXTRACT(MINUTE FROM tm.report_time)::int / 15) * 15 AS minute_group,
                    tm.report_time,
                    tm.sncode,
                    tm.attribute,
                    NULLIF(tm.value, '')::numeric AS val
                FROM 
                    iot.pvs_inv_tm tm
                WHERE 
                    tm.sncode IN (
                        SELECT si.inverter_code 
                        FROM pvs.solar_inverter si 
                        WHERE si.station_id = '{station_id}'
                    )
                    AND tm.attribute IN ('eac_today','pac')
                    AND tm.sncode <> '1024A6635644'
                    AND tm.report_time >= '{start_date} 00:00:00'
                    AND tm.report_time < '{end_date} {end_hour:02d}:{end_minute:02d}:00'::timestamp + INTERVAL '15 minutes'
            ),
            windowed_avg AS (
                SELECT
                    sncode,
                    (hour + (minute_group * INTERVAL '1 minute')) AS window_start,
                    MAX(val) FILTER (WHERE attribute='eac_today')  AS eac_today,
                    AVG(val) FILTER (WHERE attribute='pac') AS pac
                FROM
                    raw_inv_data
                GROUP BY
                    sncode, hour, minute_group
            )
            SELECT
                fw.window_start AS time_stamp,
                ROUND(COALESCE(SUM(wa.eac_today), 0), 2) AS eac_today_y,
                ROUND(COALESCE(SUM(wa.pac), 0), 2) AS pac_y
            FROM
                filtered_windows fw
            LEFT JOIN windowed_avg wa ON fw.window_start = wa.window_start
            WHERE fw.window_start <= '{end_date} {end_hour:02d}:{end_minute:02d}:00'   -- 把最后一个窗口排除出去
            GROUP BY
                fw.window_start
            ORDER BY fw.window_start DESC
            LIMIT {limit_num}
        ) sub
        ORDER BY time_stamp;
        """

    def _get_inverter_15min__detail_data_add(self, limit_num: int, sncode: str, start_date: str, end_date: str,
                                     end_hour: int = 19, end_minute: int = 30) -> str:
        return f"""
        SELECT *
        FROM (
            WITH time_windows AS (
                SELECT 
                    generate_series(
                        '{start_date} 00:00:00'::timestamp,
                        '{end_date} 23:45:00'::timestamp,
                        '15 minutes'::interval
                    ) AS window_start
            ),
            filtered_windows AS (
                SELECT 
                    window_start,
                    window_start + INTERVAL '15 minutes' AS window_end
                FROM time_windows
                WHERE 
                    EXTRACT(HOUR FROM window_start) >= 0
                    AND (
                        EXTRACT(HOUR FROM window_start) < 23
                        OR (EXTRACT(HOUR FROM window_start) = 23 AND EXTRACT(MINUTE FROM window_start) <= 45)
                    )
            ),
            raw_inv_data AS (
                SELECT 
                    DATE_TRUNC('hour', tm.report_time) AS hour,
                    (EXTRACT(MINUTE FROM tm.report_time)::int / 15) * 15 AS minute_group,
                    tm.report_time,
                    tm.sncode,
                    tm.attribute,
                    NULLIF(tm.value, '')::numeric AS val
                FROM 
                    iot.pvs_inv_tm tm
                WHERE 
                    tm.sncode = '{sncode}'
                    AND tm.sncode <> '1024A6635644'
                    AND tm.report_time >= '{start_date} 00:00:00'
                    AND tm.report_time < '{end_date} {end_hour:02d}:{end_minute:02d}:00'::timestamp + INTERVAL '15 minutes'
                    AND tm.attribute IN (
                      'grid_u_ab','grid_u_ca','grid_u_bc',
                      'grid_u_a','grid_u_c','grid_u_b',
                      'grid_i_a','grid_i_c','grid_i_b',
                      'fac','pac','pf','dc_total_p','rac',
                      'temp','eff','co2','eac_today'
                      )
            ),
            windowed_avg AS (
                SELECT
                    (hour + (minute_group * INTERVAL '1 minute')) AS time_stamp,
                    sncode AS inverter_code,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='dc_total_p'), 0), 2) AS dc_total_p,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='eac_today'), 0), 2)  AS eac_today,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='eff'), 0), 2)        AS eff,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='fac'), 0), 2)        AS fac,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_i_a'), 0), 2)   AS grid_i_a,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_i_c'), 0), 2)   AS grid_i_c,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_i_b'), 0), 2)   AS grid_i_b,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_u_a'), 0), 2)   AS grid_u_a,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_u_ab'), 0), 2)  AS grid_u_ab,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_u_b'), 0), 2)   AS grid_u_b,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_u_bc'), 0), 2)  AS grid_u_bc,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_u_c'), 0), 2)   AS grid_u_c,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='grid_u_ca'), 0), 2)  AS grid_u_ca,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='pac'), 0), 2)        AS pac,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='pf'), 0), 2)         AS pf,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='rac'), 0), 2)        AS rac,
                    ROUND(COALESCE(AVG(val) FILTER (WHERE attribute='temp'), 0), 2)       AS temp
                FROM
                    raw_inv_data
                GROUP BY
                    inverter_code, hour, minute_group
            )
            SELECT
                fw.window_start AS time_stamp,
                inv.inverter_code,
                
                ROUND(COALESCE(wa.grid_i_a, 0), 2)   AS grid_i_a,
                ROUND(COALESCE(wa.grid_i_b, 0), 2)   AS grid_i_b,
                ROUND(COALESCE(wa.grid_i_c, 0), 2)   AS grid_i_c,
                
                ROUND(COALESCE(wa.grid_u_a, 0), 2)   AS grid_u_a,
                ROUND(COALESCE(wa.grid_u_b, 0), 2)   AS grid_u_b,
                ROUND(COALESCE(wa.grid_u_c, 0), 2)   AS grid_u_c,
                
                ROUND(COALESCE(wa.grid_u_ab, 0), 2)  AS grid_u_ab,
                ROUND(COALESCE(wa.grid_u_bc, 0), 2)  AS grid_u_bc,
                ROUND(COALESCE(wa.grid_u_ca, 0), 2)  AS grid_u_ca,
                
                -- 功率因素与功率
                ROUND(COALESCE(wa.pf, 0), 2)         AS pf,
                ROUND(COALESCE(wa.fac, 0), 2)        AS fac,
                ROUND(COALESCE(wa.rac, 0), 2)        AS rac,
                ROUND(COALESCE(wa.pac, 0), 2)        AS pac,
                
                -- 直流功率、发电量、效率、温度
                ROUND(COALESCE(wa.dc_total_p, 0), 2) AS dc_total_p,
                ROUND(COALESCE(wa.eac_today, 0), 2)  AS eac_today,
                ROUND(COALESCE(wa.eff, 0), 2)        AS eff,
                ROUND(COALESCE(wa.temp, 0), 2)       AS temp
            FROM filtered_windows fw
            CROSS JOIN (SELECT DISTINCT inverter_code FROM windowed_avg) inv
            LEFT JOIN windowed_avg wa 
                ON fw.window_start = wa.time_stamp 
                AND inv.inverter_code = wa.inverter_code
            WHERE fw.window_start <= '{end_date} {end_hour:02d}:{end_minute:02d}:00'
            ORDER BY inv.inverter_code, fw.window_start DESC
            LIMIT {limit_num}
        ) sub
        ORDER BY time_stamp;
        """

    def fetch_inverter_15min_detail_data(self, limit_num: int, sncode: str, start_date: str, end_date: str,
                                       end_hour: int = 19, end_minute: int = 30) -> pd.DataFrame:
        """inv_feature--获取单个逆变器详细数据--version2.0(间隔采集)"""
        sql = self._get_inverter_15min__detail_data_add(limit_num, sncode, start_date, end_date, end_hour, end_minute)
        return pd.read_sql(sql, self.conn)

    def fetch_inverter_15min_detail_data_try(self, limit_num: int, sncode: str, start_date: str, end_date: str,
                                       end_hour: int = 19, end_minute: int = 30) -> pd.DataFrame:
        """inv_feature--获取单个逆变器详细数据--version2.0(间隔采集)"""
        sql = self.try_sql(limit_num, sncode, start_date, end_date, end_hour, end_minute)
        df = pd.read_sql(sql, self.conn)
        df = (
            df.pivot_table(
                index=["time_stamp", "inverter_code"],
                columns="attribute",
                values="avg_value"
            )
            .reset_index()
        )
        df =  df.sort_index(ascending=False).head(limit_num).sort_index()
        return df

    def fetch_station_15min_power_data(self, station_id: str, limit_num: int, start_date: str, end_date: str,
                                       end_hour: int = 19, end_minute: int = 30) -> pd.DataFrame:
        """sta_pac--获取电站15分钟间隔功率数据--version2.0(间隔采集)"""
        sql = self._get_station_power_15min_sql(station_id, limit_num, start_date, end_date, end_hour, end_minute)
        return pd.read_sql(sql, self.conn)

    def fetch_pgsql_sta_data(self, station_id: str, limit_num: int , t0: str, sncode:str) -> Tuple[pd.DataFrame, float]:

        print("开始获取数据...")

        dt = datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")

        inv = self.fetch_inverter_15min_detail_data(limit_num, start_date = dt.date() - timedelta(days=3),
                                                  end_date=dt.date(), end_hour=dt.hour, end_minute=dt.minute,sncode=sncode)
        sta = self.fetch_station_15min_power_data(station_id, limit_num, start_date = dt.date() - timedelta(days=3),
                                                  end_date=dt.date(), end_hour=dt.hour, end_minute=dt.minute)

        t0_energy = sta["eac_today_y"].iloc[-1]
        sta = sta.drop(columns=["eac_today_y"], errors="ignore")

        cols_keep_prod = [
            "time_stamp","inverter_code",
            'grid_i_a','grid_i_b','grid_i_c',
            'grid_u_a','grid_u_b', 'grid_u_c',
            "grid_u_ab","grid_u_bc","grid_u_ca",
            "dc_total_p","temp","eac_today","pac"
        ]

        inv = inv[cols_keep_prod]

        merged = pd.merge(inv, sta, on='time_stamp', how='left', validate='one_to_one')
        merged.rename(columns={"time_stamp": "date"}, inplace=True)

        print(f"数据获取完成，共{len(sta)}行")
        return merged, t0_energy

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()