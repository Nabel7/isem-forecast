# Changelog

## 0.1.0 (Unreleased)

### Added
- Walk-forward evaluation of day-ahead price forecasts: each delivery day is forecast using only the prices from before it, and every method is scored on the same half-hours (MAE and RMSE, overall and by month).
- Three baselines: `yesterday`, `last_week` and `week_average`.
- `python -m isem_forecast` scores them all on the data collected by isem-data.
- A first look at Chronos-2 (Amazon's pre-trained forecaster), zero-shot on prices only: `notebooks/01_chronos2_first_look.ipynb`. The `foundation` and `notebooks` extras install what it needs.
- EirGrid's wind forecast as it stood before each auction: `wind_forecast_before_auction()`, built from every saved version of SEMO's per-farm forecasts (newest version published by 09:00 UTC the day before, never a later revision). Also `load_wind()` for measured wind and EirGrid's latest revision.
- `foundation.py`: Chronos-2 as a walk-forward forecaster, with optional covariates, keeping its 10%-90% range for every forecast.
- Chronos-2 with the wind forecast: `notebooks/02_chronos2_with_wind.ipynb`.
- `python -m isem_forecast.site --out DIR` writes the data for Open Power Lab's Price forecasts page.
