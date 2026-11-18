"""LightGBM: a model fitted on Irish data, as a forecaster for walk_forward.

LightGBM builds a few hundred small decision trees, one after another. Each tree is a
set of yes/no questions ("is it between 17:00 and 19:00?", "is the wind forecast below
1,500 MW?") ending in a number, and each new tree is fitted to what the trees before it
still got wrong. Added up, they give the forecast. This is "gradient boosting".

Unlike Chronos-2 it knows nothing in advance: it learns only from the rows it is given.
Each row is one half-hour, with what was known before that day's auction:

    slot, weekday                    the half-hour of the trading day, the day of the week
    price_1d, price_2d, price_7d     the price at the same time 1, 2 and 7 days earlier
    yesterday_mean, _min, _max       the day before's average, lowest and highest price
    week_mean                        the average over the 7 days before
    wind, wind_day_mean,             EirGrid's wind forecast for this half-hour and the
    wind_yesterday_mean              whole day, and the day before's (when given)

It forecasts the price *minus yesterday's average*, then adds the average back. Trees
can't forecast outside the values they were trained on, and prices roughly doubled over
the year; forecasting the change from yesterday keeps the answers in a familiar range.

Each day it is fitted again on every day before it, then forecasts the day.

Needs the `gbm` extra: pip install -e ".[gbm]"
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from .data import delivery_day

# Settings chosen before testing and not tuned on the results: small trees (15 leaves),
# a slow learning rate with enough trees, and some randomness (80% of rows and columns
# for each tree) so it doesn't lean too hard on any one thing.
PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}

LAGS = {"price_1d": 1, "price_2d": 2, "price_7d": 7}


def build_features(prices: pd.Series, times: pd.DatetimeIndex, wind: pd.Series | None = None):
    """One row of features for each half-hour in `times`, from `prices` and `wind` only.

    Every price feature comes from a delivery day *before* the row's own day, so a row
    never sees its own day's prices. Missing values are NaN; LightGBM copes with them.
    """
    day = np.array(delivery_day(times))
    table = pd.DataFrame(index=times)

    # The half-hour of the trading day: 23:00 Irish time is slot 0.
    local = times.tz_convert("Europe/Dublin") + pd.Timedelta(hours=1)
    table["slot"] = local.hour * 2 + local.minute // 30
    table["weekday"] = pd.to_datetime(day).day_of_week

    # The same time 1, 2 and 7 days earlier, but only if that was an earlier trading day
    # (on clock-change days, 24 hours earlier can still be the same trading day).
    for name, days in LAGS.items():
        earlier = times - pd.Timedelta(days=days)
        known = np.array(delivery_day(earlier)) < day
        table[name] = np.where(known, prices.reindex(earlier).values, np.nan)

    # Each whole day's average, lowest and highest price, looked up for the day before.
    per_day = prices.groupby(np.array(delivery_day(prices.index))).agg(["mean", "min", "max"])
    every_day = pd.date_range(per_day.index.min(), per_day.index.max()).date
    per_day = per_day.reindex(every_day)
    yesterday = np.array([d - timedelta(days=1) for d in day])
    table["yesterday_mean"] = per_day["mean"].reindex(yesterday).values
    table["yesterday_min"] = per_day["min"].reindex(yesterday).values
    table["yesterday_max"] = per_day["max"].reindex(yesterday).values
    week = per_day["mean"].rolling(7, min_periods=7).mean()  # the 7 days up to each day
    table["week_mean"] = week.reindex(yesterday).values

    if wind is not None:
        wind_per_day = wind.groupby(np.array(delivery_day(wind.index))).mean()
        table["wind"] = wind.reindex(times).values
        table["wind_day_mean"] = wind_per_day.reindex(day).values
        table["wind_yesterday_mean"] = wind_per_day.reindex(yesterday).values

    return table


class LightGBM:
    """A forecaster for walk_forward: fits LightGBM on every day before, then forecasts.

    wind          a wind forecast known before each auction (UTC index), or None
    wind_only     train only on half-hours that have the wind forecast; otherwise earlier
                  days are used too, with the wind features missing
    min_days      return no forecast (NaN) until there are this many days to learn from
    target        "change": learn the price minus yesterday's average (the default);
                  "price": learn the price itself
    make_model    builds an untrained model; the default is LightGBM (tests pass a stand-in)

    Besides the central forecast (fitted to the median), it fits two more models for the
    10% and 90% quantiles, so it gives a range like Chronos-2's. ranges() returns them.
    """

    def __init__(
        self,
        wind=None,
        wind_only=True,
        min_days=28,
        target="change",
        params=None,
        make_model=None,
    ) -> None:
        self.wind = wind
        self.wind_only = wind_only
        self.min_days = min_days
        self.target = target
        self.params = {**PARAMS, **(params or {})}
        self.make_model = make_model or _lightgbm
        self.forecasts = []
        self.last_models = None  # the models fitted for the latest day, to look inside

    def __call__(self, history: pd.Series, targets: pd.DatetimeIndex) -> pd.Series:
        train = build_features(history, history.index, self.wind)
        # What the model learns to forecast: the change from yesterday's average, or the price.
        base = train["yesterday_mean"] if self.target == "change" else 0.0
        change = history - base
        usable = change.notna()
        if self.wind is not None and self.wind_only:
            usable &= train["wind"].notna()
        train, change = train[usable], change[usable]

        future = build_features(history, targets, self.wind)
        days_to_learn_from = len(set(delivery_day(train.index)))
        missing = future["yesterday_mean"].isna().any()
        if self.wind is not None:
            missing = missing or future["wind"].isna().any()
        if days_to_learn_from < self.min_days or missing:
            return pd.Series(np.nan, index=targets)

        models = {
            "median": self.make_model("l1", None, self.params),  # l1: aims for the median
            "low": self.make_model("quantile", 0.1, self.params),
            "high": self.make_model("quantile", 0.9, self.params),
        }
        forecast = pd.DataFrame(index=targets)
        for name, model in models.items():
            model.fit(train, change)
            back = future["yesterday_mean"].values if self.target == "change" else 0.0
            forecast[name] = model.predict(future) + back

        # Three separately fitted models can cross; keep low <= median <= high.
        forecast["low"] = forecast[["low", "median"]].min(axis=1)
        forecast["high"] = forecast[["high", "median"]].max(axis=1)
        self.forecasts.append(forecast[["low", "median", "high"]])
        self.last_models = models
        return forecast["median"]

    def ranges(self) -> pd.DataFrame:
        """Every forecast so far: low (10%), median and high (90%) for each half-hour."""
        return pd.concat(self.forecasts)


def _lightgbm(objective: str, alpha: float | None, params: dict):
    # Imported here, so the rest of the package works without the extra.
    from lightgbm import LGBMRegressor

    extra = {"alpha": alpha} if alpha is not None else {}
    return LGBMRegressor(objective=objective, **extra, **params)
