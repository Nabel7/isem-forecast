"""Load the data collected by the isem-data project.

isem-forecast doesn't download anything itself: it reads the Parquet files that
isem-data saves. By default it looks in ../isem-data/data, so the two projects
should sit side by side in the same folder.
"""

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
