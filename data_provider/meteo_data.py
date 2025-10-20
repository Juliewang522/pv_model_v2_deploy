import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from typing import Optional, Tuple

__all__ = [
    "fetch_openmeteo_data",
    "DEFAULT_MIN15_VARS",
    "DEFAULT_DAILY_VARS",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MIN15_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_direction_10m",
    "apparent_temperature",
    "cloud_cover",
    "rain",
    "sunshine_duration",
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]

DEFAULT_DAILY_VARS = ["sunrise", "sunset"]

# ---------------------------------------------------------------------------
# Client (cache + retry)
# ---------------------------------------------------------------------------
CACHE_EXPIRE = 3600  # seconds
_cache_session = requests_cache.CachedSession(".cache", expire_after=CACHE_EXPIRE)
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.2)
_client = openmeteo_requests.Client(session=_retry_session)


def fetch_openmeteo_data(
    *,
    latitude: float = 23.74,
    longitude: float = 116.18,
    timezone: str = "Asia/Singapore",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    past_days: int = 15,
    forecast_days: int = 2,
    models: str = "best_match",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if start_date and end_date:
        date_params = {"start_date": start_date, "end_date": end_date}
    else:
        date_params = {"past_days": past_days, "forecast_days": forecast_days}

    # url = "https://api.open-meteo.com/v1/forecast" (old)
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "minutely_15": DEFAULT_MIN15_VARS,
        "daily": DEFAULT_DAILY_VARS,
        "timeformat": "unixtime",
        "timezone": timezone,
        "models": models,
        **date_params,
    }

    responses = _client.weather_api(url, params=params)
    if not responses:
        raise RuntimeError("Open‑Meteo API returned no response.")

    response = responses[0]

    # ------------------------- 15‑minute data ------------------------------ #
    min15_block = response.Minutely15()
    if min15_block is None:
        raise RuntimeError("The chosen model does not provide 15‑minute data at this location.")

    # Extract variable arrays in **exactly** the same order as requested
    vars_np = [min15_block.Variables(i).ValuesAsNumpy() for i in range(min15_block.VariablesLength())]

    min15_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(min15_block.Time(), unit="s", utc=True)
                .tz_convert(timezone)
                .tz_localize(None),
                end=pd.to_datetime(min15_block.TimeEnd(), unit="s", utc=True)
                .tz_convert(timezone)
                .tz_localize(None),
                freq=pd.Timedelta(seconds=min15_block.Interval()),
                inclusive="left",
            )
        }
    )
    for name, arr in zip(DEFAULT_MIN15_VARS, vars_np):
        min15_df[name] = arr

    # --------------------------- daily data -------------------------------- #
    daily_block = response.Daily()
    vars_np_daily = [
        daily_block.Variables(i).ValuesInt64AsNumpy() if v_type == "int64" else daily_block.Variables(i).ValuesAsNumpy()
        for i, v_type in enumerate(["int64", "int64"])
    ]

    daily_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(daily_block.Time(), unit="s", utc=True)
                        .tz_convert(timezone)
                        .tz_localize(None),
                end=pd.to_datetime(daily_block.TimeEnd(), unit="s", utc=True)
                        .tz_convert(timezone)
                        .tz_localize(None),
                freq=pd.Timedelta(seconds=daily_block.Interval()),
                inclusive="left",
            )
        }
    )
    for name, arr in zip(DEFAULT_DAILY_VARS, vars_np_daily):
        daily_df[name] = arr

    return min15_df, daily_df


