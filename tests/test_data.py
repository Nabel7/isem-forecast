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


def save_versions(folder, versions):
    """Write made-up per-farm wind forecasts: {publish time: MW for each of two farms}.

    Every version covers delivery days 2-4 Nov 2026, every 15 minutes.
    """
    folder.mkdir()
    times = pd.date_range("2026-11-01 23:00", "2026-11-04 22:45", freq="15min", tz="UTC")
    rows = []
    for published, mw in versions.items():
        for unit in ["GU_1", "GU_2"]:
            rows.append(
                pd.DataFrame(
                    {
                        "published_utc": pd.Timestamp(published, tz="UTC"),
                        "timestamp_utc": times,
                        "unit": unit,
                        "forecast_mw": mw,
                    }
                )
            )
    table = pd.concat(rows)
    for day, part in table.groupby(table["published_utc"].dt.strftime("%Y-%m-%d")):
        part.to_parquet(folder / f"{day}.parquet")


def test_wind_forecast_before_auction_uses_the_last_version_before_the_cutoff(tmp_path):
    save_versions(
        tmp_path / "wind_unit_forecasts",
        {
            "2026-11-01 06:15": 100.0,  # the morning before 2 Nov
            "2026-11-02 06:15": 200.0,  # the morning before 3 Nov
            "2026-11-02 12:15": 999.0,  # after 3 Nov's auction: must never be used
        },
    )

    table = data.wind_forecast_before_auction(tmp_path)
    per_day = table["forecast_mw"].groupby(data.delivery_day(table.index)).unique()

    # Two farms each forecast the MW given, so the all-island total is twice that.
    assert per_day[date(2026, 11, 2)].tolist() == [200.0]
    assert per_day[date(2026, 11, 3)].tolist() == [400.0]
    cutoffs = [data.auction_cutoff(day) for day in data.delivery_day(table.index)]
    assert (table["published_utc"] <= cutoffs).all()
    assert len(table[data.delivery_day(table.index) == date(2026, 11, 3)]) == 48


def test_wind_forecast_before_auction_falls_back_to_an_older_version(tmp_path):
    # Nothing came out on the morning of 3 Nov, so 4 Nov uses the version from 2 Nov.
    save_versions(tmp_path / "wind_unit_forecasts", {"2026-11-02 06:15": 50.0})

    table = data.wind_forecast_before_auction(tmp_path)
    four_nov = table[data.delivery_day(table.index) == date(2026, 11, 4)]

    assert four_nov["forecast_mw"].unique().tolist() == [100.0]
    used = pd.Timestamp("2026-11-02 06:15", tz="UTC")
    assert four_nov["published_utc"].unique().tolist() == [used]


def test_auction_cutoff_is_09_00_utc_the_day_before():
    assert data.auction_cutoff(date(2026, 11, 3)) == pd.Timestamp("2026-11-02 09:00", tz="UTC")


def test_load_wind_averages_15_minute_values_into_half_hours(tmp_path):
    times = pd.date_range("2026-11-01 23:00", periods=4, freq="15min", tz="UTC")
    rows = pd.DataFrame(
        {
            "timestamp_utc": list(times) * 2,
            "region": "ALL",
            "series": ["wind_actual_mw"] * 4 + ["wind_forecast_mw"] * 4,
            "value": [100.0, 200.0, 300.0, 400.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    rows.to_parquet(tmp_path / "wind.parquet")

    wind = data.load_wind(tmp_path)

    assert wind["actual_mw"].tolist() == [150.0, 350.0]
    assert wind["latest_forecast_mw"].tolist() == [1.0, 1.0]
