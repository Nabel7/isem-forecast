"""Chronos-2, Amazon's pre-trained forecasting model, as a forecaster for walk_forward.

Chronos-2 was trained on a large collection of time series of every kind, so it can
forecast a series it has never seen without being trained on it ("zero-shot"). It is
given the recent price history and returns a range for each half-hour: the 10%, 50%
(median) and 90% quantiles. The median is the forecast that gets scored.

It can also take covariates: other series, such as a wind forecast, whose values are
known for the past *and* for the day being forecast. It learns from the recent history
how the price moved with them, and uses their values for tomorrow.

Needs the `foundation` extra (PyTorch and the chronos package, about 1 GB):

    pip install -e ".[foundation]"
"""

import pandas as pd

QUANTILES = [0.1, 0.5, 0.9]
HALF_HOURS_PER_DAY = 48


def load_model(device: str | None = None):
    """Load Chronos-2. The first time, it's downloaded from Hugging Face (about 500 MB).

    `device` is "mps" (a Mac's graphics chip), "cuda" (an Nvidia card) or "cpu"; by
    default the fastest one available.
    """
    # PyTorch and LightGBM each bring their own copy of OpenMP, the library that spreads
    # work over several cores. On a Mac, if PyTorch's loads first, LightGBM crashes the
    # moment it uses several cores. Loading LightGBM's first avoids that.
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pass

    # Imported here, not at the top, so the rest of the package works without the extra.
    import torch
    from chronos import Chronos2Pipeline

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)


class Chronos2:
    """A forecaster for walk_forward, backed by Chronos-2.

    model         from load_model()
    context_days  how many days of price history it is shown
    covariates    name -> a series known ahead of the auction, indexed by UTC time, e.g.
                  {"wind": wind_forecast_before_auction()["forecast_mw"]}. Its values over
                  the history are given as the past, its values for the targets as the future.

    Each call returns the median. The full range of every forecast so far is kept, and
    ranges() returns it.
    """

    def __init__(self, model, context_days: int = 28, covariates: dict | None = None):
        self.model = model
        self.context_days = context_days
        self.covariates = covariates or {}
        self.forecasts = []  # one table (low, median, high) per day forecast

    def __call__(self, history: pd.Series, targets: pd.DatetimeIndex) -> pd.Series:
        recent = history[-self.context_days * HALF_HOURS_PER_DAY :]

        # Chronos-2 reads tables: which series, when, and the value. It wants timestamps
        # without a time zone; dropping UTC's label changes nothing, the steps stay 30 minutes.
        past = pd.DataFrame(
            {"item_id": "ROI", "timestamp": recent.index.tz_convert(None), "target": recent.values}
        )
        future = None
        if self.covariates:
            future = pd.DataFrame({"item_id": "ROI", "timestamp": targets.tz_convert(None)})
            for name, values in self.covariates.items():
                past[name] = values.reindex(recent.index).values
                future[name] = values.reindex(targets).values

            # A gap in a covariate: skip the day (NaN) rather than forecast with a hole in it.
            if past.isna().any().any() or future.isna().any().any():
                return pd.Series(float("nan"), index=targets)

        answer = self.model.predict_df(
            past, future_df=future, prediction_length=len(targets), quantile_levels=QUANTILES
        )
        forecast = pd.DataFrame(
            {
                "low": answer["0.1"].values,
                "median": answer["0.5"].values,
                "high": answer["0.9"].values,
            },
            index=targets,
        )
        self.forecasts.append(forecast)
        return forecast["median"]

    def ranges(self) -> pd.DataFrame:
        """Every forecast so far: low (10%), median and high (90%) for each half-hour."""
        return pd.concat(self.forecasts)


def share_inside(ranges: pd.DataFrame, actual: pd.Series) -> float:
    """How often the actual price landed inside the 10%-90% range: ideally 0.8."""
    actual = actual.reindex(ranges.index)
    inside = (actual >= ranges["low"]) & (actual <= ranges["high"])
    return float(inside.mean())
