"""The backup wind forecast: ENTSO-E's, for a day SEMO's hasn't covered in time.

isem-data saves ENTSO-E's day-ahead wind forecast for Ireland's and Northern Ireland's
grids (dataset entsoe_wind_forecasts), with the time each value was first seen. That is
what makes it usable honestly: only values first seen before the cut-off are used.

Three adjustments, each recorded with the forecast:
- **Northern Ireland estimated.** ENTSO-E has sometimes had Ireland's forecast two days
  ahead but Northern Ireland's only one. Then Northern Ireland's hours are Ireland's times
  Northern Ireland's share over the last RECENT_DAYS days with both.
- **Scaled to SEMO's level.** The models learned from EirGrid's per-farm forecast on SEMO,
  which reads about 6% higher than ENTSO-E's on average (Aug to Nov 2026). So ENTSO-E's
  forecast is scaled by the ratio of the two over the last RECENT_DAYS days with both.
- **Hourly to half-hourly.** Each hour is used for both of its half-hours.
"""

from pathlib import Path

import pandas as pd

DATASET = "entsoe_wind_forecasts"
RECENT_DAYS = 28


def known_by(when: pd.Timestamp, data_dir: Path, column: str = "forecast_mw") -> pd.DataFrame:
    """Each hour's ENTSO-E forecast as it stood at `when`, one column per region (IE, NI).
    With column="first_seen_utc", when each of those values was first seen instead."""
    path = data_dir / f"{DATASET}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["IE", "NI"])
    saved = pd.read_parquet(path)
    saved = saved[saved["first_seen_utc"] <= when].sort_values("first_seen_utc")
    newest = saved.groupby(["timestamp_utc", "region"]).tail(1)
    table = newest.pivot(index="timestamp_utc", columns="region", values=column)
    return table.reindex(columns=["IE", "NI"]).sort_index()


def ratio(top: pd.Series, bottom: pd.Series) -> float | None:
    both = pd.concat([top, bottom], axis=1).dropna()
    if both.empty or both.iloc[:, 1].sum() <= 0:
        return None
    return float(both.iloc[:, 0].sum() / both.iloc[:, 1].sum())


def entsoe_backup(
    targets: pd.DatetimeIndex, semo: pd.Series, seen_by: pd.Timestamp, data_dir: Path
) -> tuple[pd.Series, dict] | None:
    """The backup forecast (MW) for the half-hours `targets`, from ENTSO-E's values first
    seen by `seen_by`, or None if Ireland's isn't there for every hour. `semo` is the
    SEMO-based forecast the models use (half-hourly), for the scaling."""
    table = known_by(seen_by, data_dir)
    hours = targets.floor("h")
    ie = table["IE"].reindex(hours)
    if table.empty or ie.isna().any():
        return None
    recent = table[
        (table.index < hours[0]) & (table.index >= hours[0] - pd.Timedelta(days=RECENT_DAYS))
    ]

    ni = table["NI"].reindex(hours)
    ni_estimated = bool(ni.isna().any())
    if ni_estimated:
        share = ratio(recent["NI"], recent["IE"])
        if share is None:
            return None
        ni = ni.fillna(ie * share)

    semo_hourly = semo.resample("1h").mean()
    scale = ratio(semo_hourly.reindex(recent.index), recent["IE"] + recent["NI"]) or 1.0
    values = pd.Series(((ie + ni) * scale).to_numpy(), index=targets)
    # ENTSO-E doesn't say when it published a forecast; the newest time one of the values
    # used was first seen stands in for it (it was published by then, so before the cut-off).
    seen = known_by(seen_by, data_dir, "first_seen_utc")
    used = [seen["IE"].reindex(hours)] + ([] if ni_estimated else [seen["NI"].reindex(hours)])
    first_seen = max(pd.Timestamp(s.max()) for s in used)
    info = {
        "wind_source": "ENTSO-E (Northern Ireland estimated)" if ni_estimated else "ENTSO-E",
        "wind_forecast_published_utc": f"{first_seen:%Y-%m-%dT%H:%MZ}",
        "entsoe_ni_estimated": ni_estimated,
        "entsoe_scale_to_semo": round(scale, 3),
    }
    return values, info
