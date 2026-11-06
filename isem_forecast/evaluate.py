"""Walk-forward evaluation: forecast one day at a time, using only the past.

For each delivery day, the forecaster sees only the prices from before that day
starts, makes its forecast, and is scored against what actually happened. It is
the closest thing to how a forecast would have been used live, and every model
is judged the same way, so their scores can be compared.
"""

from datetime import date

import pandas as pd

from .data import delivery_day


def walk_forward(prices: pd.Series, forecaster, start: date, end: date) -> pd.DataFrame:
    """Forecast every delivery day from start to end (inclusive), one day at a time.

    `forecaster` is any function taking (history, targets), like those in baselines.py.
    Returns one row per half-hour, indexed by UTC time, with columns
    delivery_day, actual and forecast.
    """
    day_of_each_half_hour = delivery_day(prices.index)
    every_day = sorted(day_of_each_half_hour.unique())  # each delivery day once, in order

    results = []
    for day in every_day:
        if day < start or day > end:
            continue

        targets = prices.index[day_of_each_half_hour == day]  # this day's half-hours
        history = prices[prices.index < targets[0]]  # only what was known before the day
        forecast = forecaster(history, targets)

        day_results = pd.DataFrame(
            {"delivery_day": day, "actual": prices[targets], "forecast": forecast}
        )
        results.append(day_results)

    return pd.concat(results)  # stack the days into one table


def errors(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Forecast minus actual: one column per method, one row per half-hour.

    `results` maps each method's name to its walk_forward() output.
    """
    columns = {}
    for name, method_results in results.items():
        columns[name] = method_results["forecast"] - method_results["actual"]
    table = pd.DataFrame(columns)

    # Keep only the half-hours every method forecast, so all are scored on the same ones.
    return table.dropna()


def summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each method's error in €/MWh, best first.

    MAE is the average size of the miss. RMSE is similar but punishes big misses more.
    """
    error = errors(results)
    mae = error.abs().mean()
    rmse = (error**2).mean() ** 0.5

    table = pd.DataFrame({"MAE": mae, "RMSE": rmse})
    return table.sort_values("MAE")


def mae_by_month(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean absolute error in €/MWh: one row per month, one column per method."""
    error = errors(results)
    month = [day.strftime("%Y-%m") for day in delivery_day(error.index)]
    return error.abs().groupby(month).mean()
