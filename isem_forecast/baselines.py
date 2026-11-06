"""Simple forecasts that any real model has to beat.

Every forecaster works the same way. It is given:
- history: every price known before the auction (a Series indexed by time)
- targets: the half-hours to forecast (the times of tomorrow's 48 half-hours)
and returns one forecast price for each target half-hour.

The day-ahead auction for delivery day D closes at 11:00 UTC on day D-1. The
prices for day D-1 are already known by then (they came out of the previous
day's auction), so a forecast for day D may use every price up to the end of D-1.
"""

import pandas as pd


def prices_days_earlier(history: pd.Series, targets: pd.DatetimeIndex, days: int) -> pd.Series:
    """For each target half-hour, the price at the same time `days` days earlier.

    Gives NaN (no value) where that earlier price isn't in the history.
    """
    earlier_times = targets - pd.Timedelta(days=days)
    earlier_prices = history.reindex(earlier_times)  # look up the price at each earlier time
    earlier_prices.index = targets  # label them with the times we're forecasting
    return earlier_prices


def same_time_yesterday(history: pd.Series, targets: pd.DatetimeIndex) -> pd.Series:
    """Each half-hour's price is the price at the same time yesterday."""
    return prices_days_earlier(history, targets, days=1)


def same_time_last_week(history: pd.Series, targets: pd.DatetimeIndex) -> pd.Series:
    """Each half-hour's price is the price 7 days earlier (same weekday, same time)."""
    return prices_days_earlier(history, targets, days=7)


def average_of_last_week(history: pd.Series, targets: pd.DatetimeIndex) -> pd.Series:
    """Each half-hour's price is the average of the same time on each of the last 7 days."""
    last_7_days = []
    for days in range(1, 8):  # 1, 2, ..., 7 days earlier
        last_7_days.append(prices_days_earlier(history, targets, days))

    side_by_side = pd.concat(last_7_days, axis=1)  # one column per day, one row per half-hour
    return side_by_side.mean(axis=1)  # average along each row; missing days are skipped


# Every method the evaluation runs: name -> forecaster. New models get added here.
BASELINES = {
    "yesterday": same_time_yesterday,
    "last_week": same_time_last_week,
    "week_average": average_of_last_week,
}
