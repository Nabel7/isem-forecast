"""Score every forecasting method on the full price history.

python -m isem_forecast
python -m isem_forecast --data-dir ~/somewhere/isem-data/data
"""

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .baselines import BASELINES
from .data import DATA_DIR, delivery_day, load_prices
from .evaluate import errors, mae_by_month, summary, walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Score forecasts of I-SEM day-ahead prices.")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help=f"isem-data's data folder ({DATA_DIR})"
    )
    args = parser.parse_args()

    prices = load_prices(args.data_dir.expanduser())

    # Start a week in, so every method has the 7 days of history it needs on day one.
    days = delivery_day(prices.index)
    start = min(days) + timedelta(days=7)
    end = max(days)

    results = {}
    for name, forecaster in BASELINES.items():
        results[name] = walk_forward(prices, forecaster, start, end)

    pd.set_option("display.precision", 1)
    print(f"Delivery days {start} to {end}, ROI day-ahead price, €/MWh")
    print(f"{len(errors(results)):,} half-hours scored, the same ones for every method\n")
    print(summary(results).to_string())
    print("\nMAE by month:")
    print(mae_by_month(results).to_string())


if __name__ == "__main__":
    main()
