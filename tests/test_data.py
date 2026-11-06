"""Tests for loading prices and working out delivery days."""

from datetime import date

import pandas as pd

from isem_forecast import data


def test_load_prices_keeps_one_region_and_the_price_series(tmp_path):
    times = pd.date_range("2026-11-01 23:00", periods=2, freq="30min", tz="UTC")
    rows = pd.DataFrame(
        {
            "timestamp_utc": list(times) * 3,
            "region": ["ROI", "ROI", "NI", "NI", "ROI", "ROI"],
            "series": ["dam_price_eur_mwh"] * 4 + ["dam_volume_mw"] * 2,
            "value": [100.0, 110.0, 101.0, 111.0, 4000.0, 4100.0],
        }
    )
    rows.to_parquet(tmp_path / "prices.parquet")

    prices = data.load_prices(tmp_path)

    assert prices.tolist() == [100.0, 110.0]
    assert prices.index.equals(pd.DatetimeIndex(times, name="timestamp_utc"))


def test_delivery_day_starts_at_23_00_irish_time():
    times = pd.DatetimeIndex(
        [
            "2026-11-01 22:30",  # winter: 22:30 Irish time, still 1 Nov
            "2026-11-01 23:00",  # winter: 23:00 Irish time, so 2 Nov
            "2026-07-01 21:30",  # summer: 22:30 Irish time, still 1 Jul
            "2026-07-01 22:00",  # summer: 23:00 Irish time, so 2 Jul
        ],
        tz="UTC",
    )
    assert list(data.delivery_day(times)) == [
        date(2026, 11, 1),
        date(2026, 11, 2),
        date(2026, 7, 1),
        date(2026, 7, 2),
    ]
