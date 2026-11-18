# isem-forecast

[![CI](https://github.com/Nabel7/isem-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Nabel7/isem-forecast/actions/workflows/ci.yml)

Forecasting Irish day-ahead electricity prices (I-SEM), one day at a time, using only what was known before the auction.

The data comes from [isem-data](https://github.com/Nabel7/isem-data): half-hourly SEMO day-ahead prices, EirGrid wind and demand, and company order results.

## Results so far

Every method forecasts tomorrow's 48 half-hourly prices and is scored **walk-forward**: for each delivery day it sees only the prices from before that day, as a trader would have before the auction (11:00 Irish time the day before). All methods are scored on the same half-hours.

ROI day-ahead price, delivery days 14 Nov 2025 to 4 Nov 2026 (17,086 half-hours). The average price over the period was about €147/MWh.

| Method | What it predicts | MAE (€/MWh) | RMSE (€/MWh) |
|---|---|---|---|
| `week_average` | the average of the same half-hour on each of the last 7 days | **36.0** | **52.4** |
| `yesterday` | the same half-hour yesterday | 39.9 | 58.6 |
| `last_week` | the same half-hour 7 days ago | 47.0 | 70.5 |

MAE is the average size of the miss; RMSE is similar but punishes big misses more. Lower is better.

**These are the baselines: the bar any real model has to beat.** Errors are smallest in calm months (August: €20 for `week_average`) and largest when prices move fast (October: €64).

## Running it

Put this project next to `isem-data`, with its data downloaded (`python -m isem_data --start 2025-11-07`):

```
Developer/
├── isem-data/       # collects the data into isem-data/data/
└── isem-forecast/   # reads it from there
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m isem_forecast        # prints the results table and the error by month
```

Use `--data-dir` if the data is somewhere else.

## How it works

```
isem_forecast/
├── data.py        # load prices from isem-data; work out each half-hour's delivery day
├── baselines.py   # the simple forecasts above
├── evaluate.py    # walk-forward testing and scoring
├── foundation.py  # Chronos-2 as a forecaster, with optional covariates (needs the foundation extra)
├── site.py        # runs the Chronos-2 tests and writes the data for Open Power Lab's forecast page
└── __main__.py    # the command line: python -m isem_forecast
```

- **A forecaster is a function** `(history, targets) -> forecast`: it gets the prices known before the auction and the half-hours to forecast, and returns one price per half-hour. New models plug into the same evaluation.
- **No peeking.** For delivery day D, the history stops at the end of day D-1, which was known before D's auction. A test checks that no forecaster ever sees the day it is forecasting.
- **Delivery days run 23:00 to 23:00 Irish time**, as I-SEM trading days do.
- **Inputs as they stood before the auction.** `wind_forecast_before_auction()` takes, for each delivery day, the newest version of EirGrid's wind forecast published by 09:00 UTC the day before, never a later revision.

## Notebooks

- [01 Chronos-2, a first look](notebooks/01_chronos2_first_look.ipynb): Amazon's pre-trained forecasting model, given only the price history and no training. In the same day-by-day test, delivery days 14 Nov 2025 to 14 Nov 2026, its average miss was €28.7/MWh with 28 days of history (€28.4 with 90 days), against €37.5 for `week_average`. It was ahead in every month. Its 80% range held the actual price about 73% of the time, so it's a little overconfident. Its worst days are sudden price falls on windy days, which prices alone can't foresee. Needs `pip install -e ".[foundation,notebooks]"`.
- [02 Chronos-2 with the wind forecast](notebooks/02_chronos2_with_wind.ipynb): the same model, also given EirGrid's wind forecast **as it stood before each auction**, built from every saved version of SEMO's per-farm forecasts. Delivery days 9 Sep to 14 Nov 2026 (67 days): average miss €36.5/MWh with wind, against €46.9 on prices alone and €66.3 for `week_average`. Using EirGrid's later, revised forecast would have scored €35.6: peeking flatters it only a little here. Giving it measured wind for the past instead of the forecast made it worse (€40.2).

## What's next

From the [data exploration](https://github.com/Nabel7/isem-data/blob/main/notebooks/01_prices_and_drivers.ipynb): net demand (demand minus wind) is the strongest single driver of price, and prices roughly doubled over the year at the same net demand, probably because of gas prices.

- [ ] A model using forecast wind and the calendar (hour, weekday)
- [ ] A demand forecast, to use forecast net demand
- [x] Forecasts as they stood at auction time, not later revisions (wind, notebook 02)
- [ ] A proper statistical model (LightGBM or LEAR) on the same days, with the same wind forecast
- [ ] Gas prices, to follow level shifts like 2026's

## Development

```bash
pre-commit install   # ruff lint + format on every commit
pytest -q            # tests use made-up prices and never need the real data
```

## License

MIT
