"""The morning forecast: a delivery day's prices, made before its auction.

python -m isem_forecast.live --out ~/Developer/isem-live-forecasts/forecasts
python -m isem_forecast.live --day 2026-11-22 --out ...      # a particular day

Everything else in this package looks back: walk-forward tests over days already past,
with rules about what each forecast may see. This makes the forecast for real, each
morning, so it can be saved and time-stamped before the auction and scored afterwards.

What it uses, as it stands on the morning of D-1 (the day before delivery day D):

- Prices: SEMO's half-hours up to D-2. D-1's own prices were set by yesterday's auction,
  but SEMO only publishes them at about 01:00 after D-1; ENTSO-E has them, hourly, so each
  hour goes in as two equal half-hours. (In the wind test that costs Chronos-2 and LightGBM
  almost nothing: see the README.) If SEMO already has D-1, as when forecasting a past
  day, SEMO's are used.
- Wind: EirGrid's forecast for D in the newest version that is in the archive and was
  published by 09:00 UTC on D-1 (data.auction_cutoff, as in the tests). SEMO's report site
  only lists each version about a day late, so live this is usually the version from the
  evening of D-2, where the tests mostly had the one from 06:15 on D-1. Twelve hours older
  costs almost nothing (see the README). The version used is recorded with the forecast.
  If no version covers all of D, it's made at once with the backup, ENTSO-E's wind
  forecast (backup_wind.py: values first seen by then, scaled to SEMO's level), and
  counts like any other day; `wind_source` in the .json says which was used. With
  neither, it's made without wind: the two wind models give no forecast (left empty, and
  listed in the .json), and yesterday and week_average are still saved. Such a day is
  shown but never counts towards the score. (SEMO listed no new version for two days or
  more on 20-23 Nov and again from 23 Nov 2026.)

It writes two files to --out, named after the day:

    2026-11-22.csv    timestamp_utc, method, low, median, high (one row per half-hour
                      and method; low and high are the 10% and 90% of the range, for the
                      two models that give one)
    2026-11-22.json   when it was made, the cut-off, whether it was made in time, which
                      wind forecast and prices went in, and the versions of the models

Save early, replace later. GitHub starts the overnight runs hours late, at times no one
can predict, so the first run that can make a forecast saves one, even with the backup
wind or none. A later run before the cut-off replaces it only if it has a better wind
input (WIND_INPUTS: none, then ENTSO-E's, then SEMO's), and the .json lists what it
replaced (`replaced`). Which forecast stands is decided by the input alone, never by the
prices forecast, and nothing is replaced after the cut-off: the record is only worth
something if nothing in it can be redone after the fact. (On 26 Nov 2026 the runs came
at 02:23, 03:43, 05:21 and 09:47 UTC, none in the hours when the old rule allowed the
backup, and 27 Nov got no forecast.)
"""

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from .backup_wind import entsoe_backup
from .baselines import BASELINES
from .data import (
    DATA_DIR,
    THIS_PROJECT,
    auction_cutoff,
    load_prices,
    wind_forecast_before_auction,
)

LOCAL_TIME = "Europe/Dublin"
CONTEXT_DAYS = 28  # as in the tests (site.py)

# The methods, in the order they're written. The two baselines are the bar to beat.
METHODS = ["chronos2_wind", "lightgbm_wind", "yesterday", "week_average"]
COLUMNS = ["timestamp_utc", "method", "low", "median", "high"]


WIND_METHODS = ["chronos2_wind", "lightgbm_wind"]

# The wind inputs, worst first. A saved forecast is replaced before the cut-off only by one
# with a better input (see the top of this file).
WIND_INPUTS = ["none", "ENTSO-E", "SEMO"]


class NotReady(RuntimeError):
    """Something the forecast needs isn't in the archive yet."""


def half_hours_of(day: date) -> pd.DatetimeIndex:
    """Delivery day `day`'s half-hours in UTC: 23:00 Irish time the evening before to
    23:00 on the day (46, 48 or 50 of them)."""
    start = pd.Timestamp(day).tz_localize(LOCAL_TIME) - pd.Timedelta(hours=1)
    # Midnight the next day minus an hour, not midnight plus 23 hours: on the day the
    # clocks go back those differ, and the day has 25 hours.
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(LOCAL_TIME) - pd.Timedelta(hours=1)
    return pd.date_range(start, end, freq="30min", inclusive="left").tz_convert("UTC")


def load_entsoe_hours(data_dir: Path = DATA_DIR) -> pd.Series:
    """ENTSO-E's hourly day-ahead prices (isem-data's entsoe_prices), indexed by UTC hour."""
    path = data_dir / "entsoe_prices.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    table = pd.read_parquet(path)
    table = table[table["series"] == "dam_price_eur_mwh"]
    return table.set_index("timestamp_utc")["value"].sort_index()


def history_before(day: date, data_dir: Path = DATA_DIR) -> tuple[pd.Series, dict]:
    """Every half-hourly price known on the morning before `day`'s auction, and where the
    newest day came from."""
    prices = load_prices(data_dir)
    yesterday, before = (
        half_hours_of(day - timedelta(days=1)),
        half_hours_of(day - timedelta(days=2)),
    )
    if not before.isin(prices.index).all():
        raise NotReady(f"SEMO's prices for {day - timedelta(days=2)} aren't in the archive yet")
    if yesterday.isin(prices.index).all():
        source = "SEMO (half-hourly)"
    else:
        hours = load_entsoe_hours(data_dir).reindex(yesterday.floor("h"))
        if hours.isna().any():
            raise NotReady(
                f"ENTSO-E's prices for {day - timedelta(days=1)} aren't in the archive yet"
            )
        prices = pd.concat(
            [prices[prices.index < yesterday[0]], pd.Series(hours.values, yesterday)]
        )
        source = "ENTSO-E (hourly, each hour as two equal half-hours)"
    history = prices[prices.index < half_hours_of(day)[0]].rename("price")
    return history, {
        "newest_price_day": str(day - timedelta(days=1)),
        "newest_price_source": source,
    }


def make(
    day: date, data_dir: Path = DATA_DIR, model=None, now=None, make_gbm=None
) -> tuple[pd.DataFrame, dict]:
    """The forecast for `day`: a table (COLUMNS) and what went into it.

    `model` is Chronos-2 (loaded if not given) and `make_gbm` builds LightGBM's models
    (LightGBM's own if not given); tests pass stand-ins for both."""
    now = now or pd.Timestamp.now(tz="UTC")
    cutoff = auction_cutoff(day)
    targets = half_hours_of(day)
    history, prices_used = history_before(day, data_dir)

    honest = wind_forecast_before_auction(data_dir)
    wind_for_day = honest.reindex(targets)
    have_wind = not wind_for_day["forecast_mw"].isna().any()
    known = honest["forecast_mw"]
    wind_used = {
        "wind_source": "SEMO" if have_wind else None,
        "wind_forecast_published_utc": (
            f"{wind_for_day['published_utc'].iloc[0]:%Y-%m-%dT%H:%MZ}" if have_wind else None
        ),
    }
    if not have_wind:
        backup = entsoe_backup(targets, known, min(now, cutoff), data_dir)
        if backup is not None:
            values, wind_used = backup
            known = values.combine_first(known.drop(targets, errors="ignore")).sort_index()
            have_wind = True

    empty = pd.DataFrame(
        {"low": float("nan"), "median": float("nan"), "high": float("nan")}, index=targets
    )
    parts = []
    if have_wind:
        # Imported here so the package's other commands work without these extras installed.
        from .foundation import Chronos2, load_model
        from .gbm import LightGBM

        chronos = Chronos2(model or load_model(), CONTEXT_DAYS, {"wind": known})
        gbm = LightGBM(known, make_model=make_gbm)
        for name, forecaster in [("chronos2_wind", chronos), ("lightgbm_wind", gbm)]:
            forecaster(history, targets)
            band = forecaster.ranges().reindex(targets) if forecaster.forecasts else None
            if band is None:  # the model declined (a gap in its inputs): say so, don't guess
                band = empty
            parts.append(band.assign(method=name))
    else:  # no wind forecast by now: the wind models give none, the rest still count
        parts += [empty.assign(method=name) for name in WIND_METHODS]
    for name in ["yesterday", "week_average"]:
        median = BASELINES[name](history, targets)
        parts.append(
            pd.DataFrame(
                {"median": median, "low": float("nan"), "high": float("nan")}, index=targets
            ).assign(method=name)
        )
    table = pd.concat(parts).rename_axis("timestamp_utc").reset_index()[COLUMNS]

    meta = {
        "day": str(day),
        "made_at_utc": f"{now:%Y-%m-%dT%H:%M:%SZ}",
        "cutoff_utc": f"{cutoff:%Y-%m-%dT%H:%MZ}",
        "in_time": bool(now <= cutoff),
        **wind_used,
        **prices_used,
        "methods_without_forecast": sorted(
            m for m in METHODS if table.loc[table["method"] == m, "median"].isna().any()
        ),
        "context_days": CONTEXT_DAYS,
        "versions": versions(),
    }
    return table, meta


def versions() -> dict:
    """The model packages' versions and this code's commit, so any forecast can be redone."""
    found = {"model": "amazon/chronos-2"}
    for package in ["chronos-forecasting", "lightgbm", "torch"]:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=THIS_PROJECT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        found["isem_forecast_commit"] = commit
    except (OSError, subprocess.CalledProcessError):
        pass
    return found


def wind_input(meta: dict) -> str:
    """Which of WIND_INPUTS a forecast was made with, from its .json. (Forecasts saved
    before `wind_source` was recorded all had SEMO's.)"""
    if meta.get("wind_forecast_published_utc") is None:
        return "none"
    return "SEMO" if meta.get("wind_source") in (None, "SEMO") else "ENTSO-E"


def saved_meta(day: date, out: Path) -> dict | None:
    """The .json of the forecast already saved for `day` in `out`, if there is one."""
    info = out / f"{day}.json"
    return json.loads(info.read_text()) if info.exists() else None


def can_improve(saved: dict, now) -> bool:
    """Could a forecast made `now` still replace `saved`? Only before the cut-off, and
    only if `saved` lacks SEMO's wind forecast."""
    return now < pd.Timestamp(saved["cutoff_utc"]) and wind_input(saved) != WIND_INPUTS[-1]


def better(meta: dict, saved: dict) -> bool:
    """Is `meta`'s wind input better than `saved`'s, and is it still before the cut-off?"""
    return can_improve(saved, pd.Timestamp(meta["made_at_utc"])) and WIND_INPUTS.index(
        wind_input(meta)
    ) > WIND_INPUTS.index(wind_input(saved))


def save(table: pd.DataFrame, meta: dict, out: Path) -> tuple[Path, Path]:
    """Write the day's two files. If the day already has a forecast, replaces it only if
    this one is better (see `better`), listing the one it replaced in `replaced`."""
    out.mkdir(parents=True, exist_ok=True)
    csv, info = out / f"{meta['day']}.csv", out / f"{meta['day']}.json"
    if csv.exists() or info.exists():
        saved = saved_meta(meta["day"], out)
        if saved is None or not better(meta, saved):
            raise FileExistsError(
                f"{meta['day']} already has a forecast in {out}; it is replaced only before "
                "the cut-off, by one with a better wind input"
            )
        meta = {
            **meta,
            "replaced": saved.get("replaced", [])
            + [{"made_at_utc": saved["made_at_utc"], "wind_input": wind_input(saved)}],
        }
    table.to_csv(csv, index=False, float_format="%.2f", date_format="%Y-%m-%dT%H:%MZ")
    info.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    return csv, info


def tomorrow_in_ireland(now: datetime | None = None) -> date:
    now = now or datetime.now(timezone.utc)
    return pd.Timestamp(now).tz_convert(LOCAL_TIME).date() + timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast a delivery day's prices before its auction."
    )
    parser.add_argument("--day", type=date.fromisoformat, help="delivery day (default: tomorrow)")
    parser.add_argument("--out", type=Path, required=True, help="folder to save the forecast in")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="isem-data's data folder")
    parser.add_argument(
        "--check",
        action="store_true",
        help="only say whether there's work to do: exit 0 if so, 3 if the saved forecast stands",
    )
    args = parser.parse_args()
    day = args.day or tomorrow_in_ireland()
    out = args.out.expanduser()
    saved = saved_meta(day, out)
    if saved is not None and not can_improve(saved, pd.Timestamp.now(tz="UTC")):
        print(f"{day} already has a forecast in {out}; nothing to do")
        sys.exit(3 if args.check else 0)
    if args.check:
        print(f"{day}: " + ("no forecast yet" if saved is None else "can still be improved"))
        return
    try:
        table, meta = make(day, args.data_dir.expanduser())
    except NotReady as reason:
        sys.exit(f"Not ready to forecast {day}: {reason}")
    if saved is not None and not better(meta, saved):
        print(
            f"{day}: the saved forecast ({wind_input(saved)} wind) stands; this one has no "
            f"better wind input ({wind_input(meta)})"
        )
        return
    csv, _ = save(table, meta, out)
    if saved is not None:
        print(
            f"{day}: replaced the forecast made {saved['made_at_utc']} "
            f"({wind_input(saved)} wind) with one using {wind_input(meta)} wind"
        )
    late = "" if meta["in_time"] else f" (LATE: after the {meta['cutoff_utc']} cut-off)"
    medians = table.groupby("method")["median"].mean().reindex(METHODS)
    print(f"{day}: saved {csv}{late}")
    if meta["wind_forecast_published_utc"] is None:
        print("No wind forecast: saved without the wind models, so this day won't count.")
    elif meta["wind_source"] != "SEMO":
        print(f"SEMO's wind forecast was missing: used the backup, {meta['wind_source']}.")
    print(
        "Average forecast, €/MWh: "
        + ", ".join(f"{m} {v:.2f}" if v == v else f"{m} none" for m, v in medians.items())
    )


if __name__ == "__main__":
    main()
