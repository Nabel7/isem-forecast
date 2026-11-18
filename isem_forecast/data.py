"""Load the data collected by the isem-data project.

isem-forecast doesn't download anything itself: it reads the Parquet files that
isem-data saves: day-ahead prices, and wind (EirGrid's measured output, and every version
of its forecast). By default it looks in ../isem-data/data, so the two projects
should sit side by side in the same folder.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# isem-data's data folder, next to this project: <parent folder>/isem-data/data
THIS_PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = THIS_PROJECT.parent / "isem-data" / "data"


def load_prices(data_dir: Path = DATA_DIR, region: str = "ROI") -> pd.Series:
    """Half-hourly day-ahead prices (€/MWh) for one region, indexed by UTC time."""
    table = pd.read_parquet(data_dir / "prices.parquet")

    # The file also holds volumes and the other region; keep just this region's prices.
    is_price = table["series"] == "dam_price_eur_mwh"
    is_region = table["region"] == region
    prices = table[is_price & is_region]

    prices = prices.set_index("timestamp_utc")["value"]
    return prices.sort_index().rename("price")


def delivery_day(timestamps: pd.DatetimeIndex) -> pd.Index:
    """The trading day each half-hour belongs to.

    An I-SEM trading day runs from 23:00 to 23:00 Irish time, so the half-hour
    starting at 23:00 on 1 Nov belongs to 2 Nov.
    """
    irish_time = timestamps.tz_convert("Europe/Dublin")

    # Push every time forward one hour: 23:00 on 1 Nov becomes 00:00 on 2 Nov,
    # so its date is now the trading day. Works in summer and winter alike.
    shifted = irish_time + pd.Timedelta(hours=1)
    return pd.Index(shifted.date, name="delivery_day")


# ---- Wind ------------------------------------------------------------------------------------

# The day-ahead auction for delivery day D closes at 11:00 Irish time on D-1: 10:00 UTC in
# summer, 11:00 UTC in winter. A forecast counts as known before the auction if it was
# published by 09:00 UTC on D-1. That is at least an hour early in either season, and still
# safe if SEMO's publish times turn out to be an hour off (isem-data's forecasts.py says why
# they might be).
CUTOFF_HOUR_UTC = 9

# If no version came out that morning (SEMO sometimes skips or delays one), use the newest
# older version, as long as it is at most this much older. It's still a forecast a trader had.
OLDEST_VERSION = pd.Timedelta(days=2)


def load_wind(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """All-island wind every half-hour (MW), indexed by UTC time.

    actual_mw            what the wind farms produced (after any dispatch-down)
    latest_forecast_mw   EirGrid's forecast, in its latest revision

    The latest revision was made after the auction, so it's fine for looking back but a
    price forecast must not use it: that would be peeking. Use
    wind_forecast_before_auction() for that.
    """
    table = pd.read_parquet(data_dir / "wind.parquet")
    table = table[table["region"] == "ALL"]
    wide = table.pivot(index="timestamp_utc", columns="series", values="value")

    # The file has a value every 15 minutes; prices are half-hourly, so average each pair.
    half_hourly = wide.resample("30min").mean()
    names = {"wind_actual_mw": "actual_mw", "wind_forecast_mw": "latest_forecast_mw"}
    return half_hourly.rename(columns=names)[list(names.values())]


def auction_cutoff(day: date) -> pd.Timestamp:
    """The latest publish time a forecast may have to count as known before `day`'s auction."""
    day_before = pd.Timestamp(day - timedelta(days=1), tz="UTC")
    return day_before + pd.Timedelta(hours=CUTOFF_HOUR_UTC)


def wind_forecast_versions(data_dir: Path = DATA_DIR) -> dict[pd.Timestamp, pd.Series]:
    """Every saved version of EirGrid's wind forecast, keyed by when it was published.

    Each version is the all-island total (MW) for every half-hour it covers. The versions
    come from SEMO's per-farm forecasts (report BM-012), which isem-data has kept, every
    version, since August 2026.
    """
    versions = {}
    for path in sorted((data_dir / "wind_unit_forecasts").glob("*.parquet")):
        table = pd.read_parquet(path, columns=["published_utc", "timestamp_utc", "forecast_mw"])
        for published, rows in table.groupby("published_utc"):
            total = rows.groupby("timestamp_utc")["forecast_mw"].sum()  # add up the farms
            versions[published] = total.resample("30min").mean()  # 15-minute -> half-hourly
    return versions


def wind_forecast_before_auction(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """For each delivery day, EirGrid's wind forecast as it stood before that day's auction.

    One row per half-hour, indexed by UTC time:
        forecast_mw     the all-island wind forecast (MW)
        published_utc   when the version used was published

    Days with no usable version are left out.
    """
    versions = wind_forecast_versions(data_dir)
    if not versions:
        return pd.DataFrame(columns=["forecast_mw", "published_utc"])

    times = sorted(versions)
    first = (times[0] + pd.Timedelta(days=1)).date()  # the first day a version came before
    last = max(delivery_day(series.index).max() for series in versions.values())

    days = []
    for day in pd.date_range(first, last).date:
        cutoff = auction_cutoff(day)
        usable = [t for t in times if cutoff - OLDEST_VERSION <= t <= cutoff]
        if not usable:
            continue
        published = max(usable)  # the newest version a trader could have used
        version = versions[published]
        forecast = version[delivery_day(version.index) == day]
        days.append(pd.DataFrame({"forecast_mw": forecast, "published_utc": published}))

    table = pd.concat(days)
    table.index.name = "timestamp_utc"
    return table
