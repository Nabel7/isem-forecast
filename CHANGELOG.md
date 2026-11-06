# Changelog

## 0.1.0 (Unreleased)

### Added
- Walk-forward evaluation of day-ahead price forecasts: each delivery day is forecast using only the prices from before it, and every method is scored on the same half-hours (MAE and RMSE, overall and by month).
- Three baselines: `yesterday`, `last_week` and `week_average`.
- `python -m isem_forecast` scores them all on the data collected by isem-data.
