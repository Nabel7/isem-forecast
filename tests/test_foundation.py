"""Tests for the Chronos-2 forecaster, with a stand-in for the model.

The real model is large and needs PyTorch, so these tests give the forecaster a fake one
that records what it was asked and answers with fixed numbers.
"""

import pandas as pd
import pytest

from isem_forecast.foundation import Chronos2, share_inside


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict_df(self, df, future_df=None, prediction_length=None, quantile_levels=None):
        self.calls.append((df, future_df))
        return pd.DataFrame({"0.1": [1.0] * prediction_length, "0.5": 2.0, "0.9": 3.0})


def ten_days():
    times = pd.date_range("2026-11-01 23:00", periods=10 * 48, freq="30min", tz="UTC")
    return pd.Series(range(len(times)), index=times, dtype=float)


def split(prices, days_of_history):
    history = prices[: days_of_history * 48]
    targets = prices.index[days_of_history * 48 : (days_of_history + 1) * 48]
    return history, targets


def test_it_shows_the_model_only_the_last_context_days():
    model = FakeModel()
    history, targets = split(ten_days(), 9)

    forecast = Chronos2(model, context_days=3)(history, targets)

    past, future = model.calls[0]
    assert len(past) == 3 * 48
    assert past["target"].tolist() == history[-3 * 48 :].tolist()
    assert past["timestamp"].dt.tz is None  # Chronos-2 wants plain timestamps
    assert future is None
    assert forecast.index.equals(targets)
    assert forecast.tolist() == [2.0] * 48  # the median


def test_covariates_go_in_as_past_and_future():
    model = FakeModel()
    prices = ten_days()
    wind = pd.Series(1000.0 + prices.values, index=prices.index)
    history, targets = split(prices, 9)

    Chronos2(model, context_days=2, covariates={"wind": wind})(history, targets)

    past, future = model.calls[0]
    assert past["wind"].tolist() == wind[history.index[-2 * 48 :]].tolist()
    assert future["wind"].tolist() == wind[targets].tolist()
    assert future["timestamp"].tolist() == targets.tz_convert(None).tolist()


def test_a_gap_in_a_covariate_skips_the_day():
    model = FakeModel()
    prices = ten_days()
    wind = pd.Series(1000.0, index=prices.index)
    history, targets = split(prices, 9)
    wind[targets[5]] = float("nan")  # one missing half-hour tomorrow

    forecast = Chronos2(model, context_days=2, covariates={"wind": wind})(history, targets)

    assert forecast.isna().all()
    assert model.calls == []  # the model wasn't asked


def test_ranges_keep_every_forecast_and_share_inside_counts_hits():
    forecaster = Chronos2(FakeModel(), context_days=1)
    prices = ten_days()
    for days in (8, 9):
        forecaster(*split(prices, days))

    ranges = forecaster.ranges()
    assert len(ranges) == 2 * 48
    assert ranges[["low", "median", "high"]].iloc[0].tolist() == [1.0, 2.0, 3.0]

    actual = pd.Series(2.0, index=ranges.index)
    actual.iloc[:24] = 10.0  # a quarter of the half-hours land outside the range
    assert share_inside(ranges, actual) == pytest.approx(0.75)
