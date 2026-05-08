#!/usr/bin/env python3
"""
爬取深圳历史天气数据（逐小时）
数据来源：Open-Meteo Historical Weather API（免费、无需注册）
基于 ERA5 再分析数据，覆盖全球。

深圳气象站：22.54°N, 114.06°E
爬取时间范围：2024-05 至 2026-05（约两年）
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fetch_openmeteo_historical(
    latitude: float = 22.54,
    longitude: float = 114.06,
    start_date: str = "2024-05-01",
    end_date: str = "2026-05-05",
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """
    从 Open-Meteo Historical Weather API 获取逐小时历史天气数据。

    获取的变量：
    - temperature_2m:          2米处气温 (°C)
    - relative_humidity_2m:    2米处相对湿度 (%)
    - dew_point_2m:            2米处露点温度 (°C)
    - apparent_temperature:    体感温度 (°C)
    - precipitation:           降水量 (mm)
    - rain:                    降雨量 (mm)
    - snowfall:                降雪量 (cm)
    - pressure_msl:            海平面气压 (hPa)
    - surface_pressure:        地表气压 (hPa)
    - cloud_cover:             总云量 (%)
    - wind_speed_10m:          10米处风速 (km/h)
    - wind_direction_10m:      10米处风向 (°)
    - wind_gusts_10m:          10米处阵风 (km/h)
    - shortwave_radiation:     短波辐射 (W/m²)
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
    }

    print(f"🌐 Fetching data from Open-Meteo...")
    print(f"   Location: ({latitude}, {longitude}) — Shenzhen")
    print(f"   Range: {start_date} → {end_date}")

    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch from Open-Meteo: {e}")
        # Fallback: try chunking by months
        print("🔄 Trying chunked download (month by month)...")
        return fetch_chunked(latitude, longitude, start_date, end_date, timezone)

    data = resp.json()

    if "hourly" not in data:
        print(f"❌ API error: {data.get('reason', 'Unknown error')}")
        return None

    # 解析数据
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)

    # 重命名列（英文 → 更可读的中文含义）
    df.rename(
        columns={
            "time": "datetime",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "dew_point_2m": "dew_point",
            "apparent_temperature": "feels_like",
            "precipitation": "precip",
            "rain": "rain",
            "pressure_msl": "pressure_msl",
            "surface_pressure": "pressure_surface",
            "cloud_cover": "cloud",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_dir",
            "wind_gusts_10m": "wind_gust",
        },
        inplace=True,
    )

    # 转换时间列
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    print(f"✅ Downloaded {len(df)} hourly records")
    print(f"   Columns: {list(df.columns)}")

    return df


def fetch_chunked(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
) -> pd.DataFrame:
    """
    分月下载，避免单次请求数据量过大。
    """
    from pandas.tseries.offsets import MonthEnd

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    all_dfs = []

    current = start
    while current < end:
        chunk_start = current.strftime("%Y-%m-%d")
        chunk_end = min(current + MonthEnd(1), end).strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "timezone": timezone,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "pressure_msl",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
            ],
        }

        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            df_chunk = pd.DataFrame(data["hourly"])
            all_dfs.append(df_chunk)
            print(f"   ✓ {chunk_start} → {chunk_end}: {len(df_chunk)} records")
        except Exception as e:
            print(f"   ⚠️ {chunk_start} → {chunk_end}: {e}")

        current = current + MonthEnd(1) + pd.Timedelta(days=1)
        time.sleep(0.5)  # 礼貌等待

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    df.rename(
        columns={
            "time": "datetime",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "dew_point_2m": "dew_point",
            "apparent_temperature": "feels_like",
            "precipitation": "precip",
            "rain": "rain",
            "pressure_msl": "pressure_msl",
            "surface_pressure": "pressure_surface",
            "cloud_cover": "cloud",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_dir",
            "wind_gusts_10m": "wind_gust",
        },
        inplace=True,
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 datetime 索引提取时间特征，帮助模型学习周期性。

    添加的特征：
    - hour_sin, hour_cos:  一天中的小时（循环编码）
    - day_sin, day_cos:    一年中的天数（循环编码）
    - month:               月份
    - is_weekend:          是否周末
    """
    df = df.copy()

    hour = df.index.hour
    day_of_year = df.index.dayofyear

    # 循环编码（正弦+余弦确保 0 和 23 小时、1 月 1 日和 12 月 31 日连续）
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(float)

    return df


def main():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(save_dir, exist_ok=True)

    # 爬取 2024-05-01 到 2026-05-05（约两年）
    print("=" * 60)
    print("🌤️ 深圳历史天气数据爬取")
    print("=" * 60)

    df = fetch_openmeteo_historical(
        latitude=22.54,
        longitude=114.06,
        start_date="2024-05-01",
        end_date="2026-05-05",
    )

    if df is None or len(df) == 0:
        print("❌ Failed to fetch data")
        return

    # 添加时间特征
    print("\n🔧 Adding time features...")
    df = add_time_features(df)

    # 保存
    save_path = os.path.join(save_dir, "shenzhen_weather_hourly.csv")
    df.to_csv(save_path)
    print(f"\n💾 Saved to {save_path}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

    # 数据质量报告
    print("\n📊 Data Quality Report:")
    print(f"   Date range: {df.index.min()} → {df.index.max()}")
    print(f"   Missing values:")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"     {col}: {n_missing} ({100*n_missing/len(df):.1f}%)")

    # 统计摘要
    print(f"\n📈 Statistics:")
    for col in ["temperature", "humidity", "pressure_msl", "wind_speed", "precip"]:
        if col in df.columns:
            print(
                f"   {col}: min={df[col].min():.1f}, max={df[col].max():.1f}, "
                f"mean={df[col].mean():.1f}, std={df[col].std():.1f}"
            )

    # 最近 24 小时数据快照
    print(f"\n🕐 Latest 5 records:")
    print(df.tail(5).to_string())


if __name__ == "__main__":
    main()
