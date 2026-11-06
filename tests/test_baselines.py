"""Tests for the baseline forecasts, on made-up daily prices."""

import pandas as pd

from isem_forecast import baselines


def prices(values, start="2026-11-01"):
    """One price per day at midnight UTC, so "a day earlier" is easy to see."""
    times = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=times, dtype=float)


def test_same_time_yesterday_uses_the_price_24_hours_earlier():
    history = prices([10, 20, 30])
    targets = pd.DatetimeIndex(["2026-11-04"], tz="UTC")
    assert baselines.same_time_yesterday(history, targets).tolist() == [30.0]


def test_same_time_last_week_uses_the_price_7_days_earlier():
    history = prices([1, 2, 3, 4, 5, 6, 7])
    targets = pd.DatetimeIndex(["2026-11-08"], tz="UTC")
    assert baselines.same_time_last_week(history, targets).tolist() == [1.0]


def test_average_of_last_week_averages_the_last_7_days():
    history = prices([1, 2, 3, 4, 5, 6, 7])
    targets = pd.DatetimeIndex(["2026-11-08"], tz="UTC")
    assert baselines.average_of_last_week(history, targets).tolist() == [4.0]


def test_average_of_last_week_skips_missing_days():
    history = prices([1, 2, 3, 4, 5, 6, 7]).drop(pd.Timestamp("2026-11-07", tz="UTC"))
    targets = pd.DatetimeIndex(["2026-11-08"], tz="UTC")
    assert baselines.average_of_last_week(history, targets).tolist() == [3.5]  # mean of 1..6


def test_no_history_gives_no_forecast():
    targets = pd.DatetimeIndex(["2026-11-08"], tz="UTC")
    forecast = baselines.same_time_yesterday(prices([]), targets)
    assert forecast.isna().all()
