"""Tests for the walk-forward evaluation, on three days of made-up half-hourly prices."""

from datetime import date

import pandas as pd
import pytest

from isem_forecast import evaluate


def three_days():
    """Half-hourly prices for delivery days 2-4 Nov 2026 (winter, so UTC = Irish time).

    Each day's price is its day number, e.g. every half-hour of 3 Nov costs 3.
    """
    times = pd.date_range("2026-11-01 23:00", periods=3 * 48, freq="30min", tz="UTC")
    values = [2.0] * 48 + [3.0] * 48 + [4.0] * 48
    return pd.Series(values, index=times)


def test_the_forecaster_never_sees_the_day_it_is_forecasting():
    seen = []

    def spy(history, targets):
        seen.append((history.index.max(), targets.min()))
        return pd.Series(0.0, index=targets)

    evaluate.walk_forward(three_days(), spy, date(2026, 11, 3), date(2026, 11, 4))

    assert len(seen) == 2
    for last_known, first_target in seen:
        assert last_known < first_target


def test_walk_forward_returns_every_half_hour_of_each_day():
    results = evaluate.walk_forward(
        three_days(), lambda h, t: pd.Series(0.0, index=t), date(2026, 11, 3), date(2026, 11, 4)
    )
    assert len(results) == 2 * 48
    assert set(results["delivery_day"]) == {date(2026, 11, 3), date(2026, 11, 4)}
    assert results["actual"].tolist() == [3.0] * 48 + [4.0] * 48


def test_summary_scores_mae_and_rmse():
    index = pd.date_range("2026-11-02", periods=2, freq="30min", tz="UTC")
    actual = pd.Series([10.0, 10.0], index=index)
    results = {
        "perfect": pd.DataFrame(
            {"delivery_day": date(2026, 11, 2), "actual": actual, "forecast": [10.0, 10.0]},
            index=index,
        ),
        "off": pd.DataFrame(
            {"delivery_day": date(2026, 11, 2), "actual": actual, "forecast": [7.0, 14.0]},
            index=index,
        ),
    }

    table = evaluate.summary(results)

    assert table.loc["perfect", "MAE"] == 0
    assert table.loc["off", "MAE"] == pytest.approx(3.5)  # misses of 3 and 4
    assert table.loc["off", "RMSE"] == pytest.approx(((9 + 16) / 2) ** 0.5)
    assert list(table.index) == ["perfect", "off"]  # best first


def test_every_method_is_scored_on_the_same_half_hours():
    index = pd.date_range("2026-11-02", periods=2, freq="30min", tz="UTC")
    actual = pd.Series([10.0, 10.0], index=index)
    results = {
        "complete": pd.DataFrame(
            {"delivery_day": date(2026, 11, 2), "actual": actual, "forecast": [10.0, 0.0]},
            index=index,
        ),
        "gappy": pd.DataFrame(
            {"delivery_day": date(2026, 11, 2), "actual": actual, "forecast": [10.0, None]},
            index=index,
        ),
    }
    # The second half-hour is skipped for both, so "complete" isn't penalised for it.
    assert evaluate.summary(results).loc["complete", "MAE"] == 0
