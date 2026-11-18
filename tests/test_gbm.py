"""Tests for the LightGBM features and forecaster, on made-up prices.

The features are the place a model could peek at the day it forecasts, so most tests
check that no row ever sees a price from its own trading day. The forecaster is tested
with a stand-in model, so these tests don't need LightGBM installed.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from isem_forecast.data import delivery_day
from isem_forecast.gbm import LightGBM, build_features


def days_of_prices(n, start="2026-11-01 23:00"):
    """Half-hourly prices for n delivery days (winter, so UTC = Irish time).

    Every half-hour of the k-th day costs 100 * k, so a price says which day it's from.
    """
    times = pd.date_range(start, periods=n * 48, freq="30min", tz="UTC")
    return pd.Series([100.0 * (i // 48 + 1) for i in range(len(times))], index=times)


def test_price_features_come_from_earlier_days():
    prices = days_of_prices(10)
    table = build_features(prices, prices.index)

    tenth_day = table.iloc[-48:]  # every day k costs 100k, so day 10 costs 1000
    assert (tenth_day["price_1d"] == 900).all()
    assert (tenth_day["price_2d"] == 800).all()
    assert (tenth_day["price_7d"] == 300).all()
    assert (tenth_day["yesterday_mean"] == 900).all()
    assert tenth_day["week_mean"].tolist() == pytest.approx([600.0] * 48)  # days 3 to 9
    assert table["price_1d"].iloc[:48].isna().all()  # nothing before day 1


def test_slot_counts_half_hours_from_23_00_irish_time():
    prices = days_of_prices(1)
    table = build_features(prices, prices.index)
    assert table["slot"].tolist() == list(range(48))


def test_on_the_long_clock_change_day_a_lag_never_comes_from_the_same_day():
    # 25 Oct 2026 has 50 half-hours. 24 hours before its last two is still 25 Oct.
    times = pd.date_range("2026-10-23 22:00", "2026-10-25 22:30", freq="30min", tz="UTC")
    prices = pd.Series(1.0, index=times)
    table = build_features(prices, times)

    day = np.array(delivery_day(times))
    last_day = table[day == date(2026, 10, 25)]
    assert len(last_day) == 50
    earlier = last_day.index - pd.Timedelta(days=1)
    same_day = np.array(delivery_day(earlier)) == date(2026, 10, 25)
    assert same_day.sum() == 2
    assert last_day["price_1d"][same_day].isna().all()


def test_wind_features_use_the_given_forecast():
    prices = days_of_prices(3)
    wind = pd.Series(np.arange(len(prices), dtype=float), index=prices.index)
    table = build_features(prices, prices.index, wind)

    assert table["wind"].tolist() == wind.tolist()
    assert table["wind_day_mean"].iloc[-1] == pytest.approx(wind.iloc[-48:].mean())
    assert table["wind_yesterday_mean"].iloc[-1] == pytest.approx(wind.iloc[48:96].mean())


class StandIn:
    """Remembers what it was trained on and always forecasts a change of +5."""

    made = []

    def __init__(self, objective, alpha, params):
        self.objective, self.alpha = objective, alpha
        StandIn.made.append(self)

    def fit(self, features, target):
        self.features, self.target = features, target
        return self

    def predict(self, features):
        return np.full(len(features), {None: 5.0, 0.1: 1.0, 0.9: 9.0}[self.alpha])


def split(prices, days_before):
    history = prices[: days_before * 48]
    targets = prices.index[days_before * 48 : (days_before + 1) * 48]
    return history, targets


def test_it_learns_only_from_the_past_and_adds_yesterdays_average_back():
    StandIn.made = []
    prices = days_of_prices(40)
    history, targets = split(prices, 35)

    forecast = LightGBM(make_model=StandIn)(history, targets)

    median = StandIn.made[0]
    assert median.objective == "l1"
    assert median.features.index.max() < targets[0]
    # It learns the change from yesterday's average: always +100 in these prices.
    assert (median.target == 100).all()
    assert (forecast == 3500 + 5).all()  # day 35's average, plus the forecast change


def test_no_forecast_until_there_are_enough_days():
    prices = days_of_prices(10)
    forecast = LightGBM(make_model=StandIn, min_days=28)(*split(prices, 9))
    assert forecast.isna().all()


def test_with_wind_only_it_trains_on_days_with_the_forecast():
    StandIn.made = []
    prices = days_of_prices(40)
    wind = pd.Series(500.0, index=prices.index[-31 * 48 :])  # the last 31 days only
    history, targets = split(prices, 39)

    LightGBM(wind, make_model=StandIn)(history, targets)
    assert len(set(delivery_day(StandIn.made[0].features.index))) == 30

    StandIn.made = []
    LightGBM(wind, wind_only=False, make_model=StandIn)(history, targets)
    assert len(set(delivery_day(StandIn.made[0].features.index))) == 38  # all with a day before


def test_ranges_are_kept_in_order():
    forecaster = LightGBM(make_model=StandIn)
    forecaster(*split(days_of_prices(40), 35))
    ranges = forecaster.ranges()
    assert (ranges["low"] <= ranges["median"]).all()
    assert (ranges["median"] <= ranges["high"]).all()
    assert ranges[["low", "median", "high"]].iloc[0].tolist() == [3501.0, 3505.0, 3509.0]
