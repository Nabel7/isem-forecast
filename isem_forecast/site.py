"""Run the forecast tests and write the small data files Open Power Lab's forecast page reads.

python -m isem_forecast.site --out ../dispatch-down-tracker/src/forecasts/data/prices

Two tests, both walk-forward (each day forecast using only what was known before its
auction), every method scored on the same half-hours:

1. The year: Chronos-2 and LightGBM on prices alone, against the simple baselines.
2. With wind: both given EirGrid's wind forecast as it stood before each auction, from
   when isem-data started keeping every version (August 2026).

Files written:
    scores.csv         test, method, mae, rmse, inside_range (the models with a range)
    monthly.csv        test, month, method, mae
    daily.csv          the wind test: day, method, mae
    halfhours.csv      the wind test, every half-hour: actual, forecasts, ranges, wind forecast
    meta.json          dates, settings, what peeking at a later wind forecast would score,
                       and two checks on LightGBM's set-up

Needs the `foundation` and `gbm` extras. Takes about 15 minutes on an M1 Pro, nearly all of
it refitting LightGBM every day of the year.
"""

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .baselines import BASELINES
from .data import (
    CUTOFF_HOUR_UTC,
    DATA_DIR,
    delivery_day,
    load_prices,
    load_wind,
    wind_forecast_before_auction,
)
from .evaluate import errors, summary, walk_forward
from .foundation import Chronos2, load_model, share_inside
from .gbm import LightGBM

CONTEXT_DAYS = 28  # a month of history: in the first look, more barely helped

# The names the page shows.
LABELS = {
    "chronos2_wind": "Chronos-2 + wind forecast",
    "chronos2": "Chronos-2, prices only",
    "lightgbm_wind": "LightGBM + wind forecast",
    "lightgbm": "LightGBM, prices only",
    "week_average": "Average of the last 7 days",
    "yesterday": "Same time yesterday",
    "last_week": "Same time last week",
}


def run_test(prices, methods: dict, start: date, end: date) -> dict[str, pd.DataFrame]:
    results = {}
    for name, method in methods.items():
        began = time.monotonic()
        results[name] = walk_forward(prices, method, start, end)
        print(f"  {name}: {start} to {end}, {time.monotonic() - began:.0f} s", flush=True)
    return results


def scores(test: str, results: dict, ranged: dict, prices) -> pd.DataFrame:
    table = summary(results).reset_index(names="method")
    table.insert(0, "test", test)
    common = errors(results).index  # the half-hours every method forecast
    table["inside_range"] = [
        share_inside(ranged[m].ranges().reindex(common), prices) if m in ranged else None
        for m in table["method"]
    ]
    return table


def by_month(test: str, results: dict) -> pd.DataFrame:
    error = errors(results).abs()
    month = [day.strftime("%Y-%m") for day in delivery_day(error.index)]
    table = error.groupby(month).mean().rename_axis("month").reset_index()
    table = table.melt(id_vars="month", var_name="method", value_name="mae")
    table.insert(0, "test", test)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the data for the site's forecast page.")
    parser.add_argument("--out", type=Path, required=True, help="folder to write into")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="isem-data's data folder")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    prices = load_prices(data_dir)
    wind = load_wind(data_dir)
    honest = wind_forecast_before_auction(data_dir)
    model = load_model()
    days = delivery_day(prices.index)
    end = max(days)

    # 1. The year, prices only. Start a week in, as for the baselines; LightGBM needs a
    # month to learn from, so the first month is left out for every method.
    year_start = min(days) + timedelta(days=7)
    year_ranged = {"chronos2": Chronos2(model, CONTEXT_DAYS), "lightgbm": LightGBM()}
    year = run_test(prices, {**BASELINES, **year_ranged}, year_start, end)

    # 2. With wind. Its first day is the first with a whole month of honest wind forecasts
    # behind it; days with a gap are skipped by every method alike.
    wind_start = min(delivery_day(honest.index)) + timedelta(days=CONTEXT_DAYS)
    known = honest["forecast_mw"]
    wind_ranged = {
        "chronos2_wind": Chronos2(model, CONTEXT_DAYS, {"wind": known}),
        "chronos2": Chronos2(model, CONTEXT_DAYS),
        "lightgbm_wind": LightGBM(known),
        "lightgbm": LightGBM(),
    }
    methods = {"week_average": BASELINES["week_average"], **wind_ranged}
    wind_test = run_test(prices, methods, wind_start, end)

    # Two checks on LightGBM's set-up, scored on the same days: train on every day, with
    # the wind missing before August; forecast the price itself, not the change.
    checks = {
        "train on every day since November 2025": LightGBM(known, wind_only=False),
        "forecast the price itself": LightGBM(known, target="price"),
    }
    check_results = {
        name: run_test(prices, {name: check}, wind_start, end) for name, check in checks.items()
    }

    # The trap: the same model given EirGrid's *latest* wind forecast, revised after the auction.
    # That file misses a few values around the October clock change; without filling them,
    # every day whose month of history touches the gap would be skipped.
    latest = wind["latest_forecast_mw"].interpolate(limit=4, limit_area="inside")
    peeking = Chronos2(model, CONTEXT_DAYS, {"wind": latest})
    peek = walk_forward(prices, peeking, wind_start, end)

    # Scores and monthly errors for both tests.
    both = [
        scores("year", year, year_ranged, prices),
        scores("wind", wind_test, wind_ranged, prices),
    ]
    pd.concat(both).to_csv(out / "scores.csv", index=False, float_format="%.4f")
    pd.concat([by_month("year", year), by_month("wind", wind_test)]).to_csv(
        out / "monthly.csv", index=False, float_format="%.3f"
    )

    # The wind test, day by day and half-hour by half-hour.
    error = errors(wind_test)
    common = error.index
    daily = error.abs().groupby(delivery_day(common)).mean().rename_axis("day").reset_index()
    daily.melt(id_vars="day", var_name="method", value_name="mae").to_csv(
        out / "daily.csv", index=False, float_format="%.3f"
    )

    halfhours = pd.DataFrame({"actual": prices.reindex(common)}, index=common)
    halfhours["week_average"] = wind_test["week_average"]["forecast"].reindex(common)
    for name, model_run in wind_ranged.items():
        band = model_run.ranges().reindex(common)
        halfhours[name] = band["median"]
        halfhours[f"{name}_low"] = band["low"]
        halfhours[f"{name}_high"] = band["high"]
    halfhours["wind_forecast_mw"] = honest["forecast_mw"].reindex(common)
    halfhours.rename_axis("timestamp_utc").reset_index().to_csv(
        out / "halfhours.csv", index=False, float_format="%.2f", date_format="%Y-%m-%dT%H:%MZ"
    )

    # Which wind forecast version each day used: normally the one from about 06:15 UTC the
    # day before; a few days fell back to an older one.
    used = honest["published_utc"].groupby(delivery_day(honest.index)).first()
    fallbacks = [
        {"day": str(day), "published_utc": f"{t:%Y-%m-%dT%H:%MZ}"}
        for day, t in used.items()
        if t.date() != day - timedelta(days=1) and wind_start <= day <= end
    ]

    peek_error = (peek["forecast"] - peek["actual"]).reindex(common).abs()
    if peek_error.isna().any():  # scored on fewer days than the others: not comparable
        raise SystemExit(f"Peeking test missing {peek_error.isna().sum()} half-hours")
    peek_error = peek_error.mean()

    def mae_on_common(result):
        miss = (result["forecast"] - result["actual"]).reindex(common).abs()
        if miss.isna().any():
            raise SystemExit("A check on LightGBM was scored on fewer days than the others")
        return round(float(miss.mean()), 3)

    check_scores = {name: mae_on_common(r[name]) for name, r in check_results.items()}
    meta = {
        "made": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%MZ}",
        "model": "amazon/chronos-2",
        "context_days": CONTEXT_DAYS,
        "labels": LABELS,
        "lightgbm_checks": check_scores,
        "year": {
            "first_day": str(min(delivery_day(errors(year).index))),
            "last_day": str(end),
            "half_hours": len(errors(year)),
        },
        "wind": {
            "first_day": str(min(delivery_day(common))),
            "last_day": str(max(delivery_day(common))),
            "half_hours": len(common),
            "days": len(set(delivery_day(common))),
            "cutoff_hour_utc": CUTOFF_HOUR_UTC,
            "fallback_versions": fallbacks,
            "peeking_mae": round(float(peek_error), 3),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pd.read_csv(out / "scores.csv").round(1).to_string(index=False))
    print(f"peeking at the latest wind forecast: MAE {peek_error:.1f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
