"""Tests for the morning forecast, on a small made-up archive.

Chronos-2 and LightGBM are replaced by stand-ins (as in test_foundation.py and
test_gbm.py), so the tests need neither model installed."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from isem_forecast import live

DAY = date(2026, 11, 22)  # the delivery day forecast in these tests
FIRST = DAY - timedelta(days=40)


class FakeTrees:
    """Stands in for LightGBM: always forecasts a change of +5 (range +1 to +9)."""

    def __init__(self, objective, alpha, params):
        self.alpha = alpha

    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.full(len(features), {None: 5.0, 0.1: 1.0, 0.9: 9.0}[self.alpha])


class FakeModel:
    def predict_df(self, df, future_df=None, prediction_length=None, quantile_levels=None):
        return pd.DataFrame({"0.1": [90.0] * prediction_length, "0.5": 100.0, "0.9": 110.0})


def price(t):  # a made-up price with a daily shape, so the models have something to learn
    return 100.0 + 30.0 * ((t.hour + 1) % 24 > 16)


def archive(
    tmp_path, semo_to=DAY - timedelta(days=2), entsoe_to=DAY - timedelta(days=1), wind_to=DAY
):
    """A data folder like isem-data's: SEMO's prices to `semo_to`, ENTSO-E's hours to
    `entsoe_to`, and a wind forecast version published at 06:14 UTC the day before each
    day to `wind_to`."""
    days = lambda last: [FIRST + timedelta(days=i) for i in range((last - FIRST).days + 1)]  # noqa: E731
    times = pd.DatetimeIndex([t for d in days(semo_to) for t in live.half_hours_of(d)])
    pd.DataFrame(
        {
            "timestamp_utc": times,
            "region": "ROI",
            "series": "dam_price_eur_mwh",
            "value": [price(t) for t in times],
        }
    ).to_parquet(tmp_path / "prices.parquet")

    hours = pd.DatetimeIndex(
        [t for d in days(entsoe_to) for t in live.half_hours_of(d) if t.minute == 0]
    )
    pd.DataFrame(
        {
            "timestamp_utc": hours,
            "region": "SEM",
            "series": "dam_price_eur_mwh",
            "value": [price(t) + 1 for t in hours],
        }
    ).to_parquet(tmp_path / "entsoe_prices.parquet")

    folder = tmp_path / "wind_unit_forecasts"
    folder.mkdir()
    for d in days(wind_to):
        published = pd.Timestamp(d - timedelta(days=1), tz="UTC") + pd.Timedelta(
            hours=6, minutes=14
        )
        quarter_hours = pd.date_range(
            live.half_hours_of(d)[0], periods=len(live.half_hours_of(d)) * 2, freq="15min"
        )
        pd.DataFrame(
            {
                "published_utc": published,
                "timestamp_utc": quarter_hours,
                "unit": "WF1",
                "forecast_mw": 1500.0,
            }
        ).to_parquet(folder / f"{published.date()}.parquet")
    return tmp_path


# --- the day's half-hours ---


def test_a_delivery_day_runs_23_00_to_23_00_irish_time():
    times = live.half_hours_of(date(2026, 11, 22))
    assert len(times) == 48
    assert times[0] == pd.Timestamp("2026-11-21 23:00", tz="UTC")
    assert len(live.half_hours_of(date(2026, 10, 25))) == 50  # clocks go back
    assert len(live.half_hours_of(date(2026, 3, 29))) == 46  # clocks go forward
    assert live.half_hours_of(date(2026, 8, 1))[0] == pd.Timestamp("2026-07-31 22:00", tz="UTC")


def test_tomorrow_is_by_the_date_in_ireland():
    assert live.tomorrow_in_ireland(pd.Timestamp("2026-11-21 08:00", tz="UTC")) == date(
        2026, 11, 22
    )
    # 23:30 UTC on 31 Jul is already 1 Aug in Ireland, so tomorrow is 2 Aug.
    assert live.tomorrow_in_ireland(pd.Timestamp("2026-07-31 23:30", tz="UTC")) == date(2026, 8, 2)


# --- the history it may see ---


def test_yesterday_comes_from_entso_e_hourly_when_semo_hasnt_published_it(tmp_path):
    history, used = live.history_before(DAY, archive(tmp_path))
    yesterday = live.half_hours_of(DAY - timedelta(days=1))
    assert used["newest_price_source"].startswith("ENTSO-E")
    assert history.index[-1] == yesterday[-1]  # nothing from the day itself
    assert (
        history[yesterday] == [price(t.floor("h")) + 1 for t in yesterday]
    ).all()  # each hour twice
    assert history.index.is_monotonic_increasing and not history.index.has_duplicates


def test_semos_half_hours_are_used_when_semo_has_yesterday(tmp_path):
    history, used = live.history_before(DAY, archive(tmp_path, semo_to=DAY))
    assert used["newest_price_source"].startswith("SEMO")
    assert history.index[-1] < live.half_hours_of(DAY)[0]  # SEMO's own day D is left out


def test_not_ready_without_entso_e_for_yesterday(tmp_path):
    with pytest.raises(live.NotReady, match="ENTSO-E"):
        live.history_before(DAY, archive(tmp_path, entsoe_to=DAY - timedelta(days=2)))


def test_not_ready_without_semo_for_the_day_before_yesterday(tmp_path):
    with pytest.raises(live.NotReady, match="SEMO"):
        live.history_before(DAY, archive(tmp_path, semo_to=DAY - timedelta(days=3)))


# --- the forecast ---


def test_the_forecast_has_every_method_for_every_half_hour(tmp_path):
    made = pd.Timestamp("2026-11-21 07:40", tz="UTC")
    table, meta = live.make(DAY, archive(tmp_path), model=FakeModel(), make_gbm=FakeTrees, now=made)

    assert list(table.columns) == live.COLUMNS
    assert table.groupby("method").size().to_dict() == {m: 48 for m in live.METHODS}
    chronos = table[table["method"] == "chronos2_wind"]
    assert (chronos[["low", "median", "high"]].to_numpy() == [90.0, 100.0, 110.0]).all()
    lightgbm = table[table["method"] == "lightgbm_wind"]
    assert lightgbm["median"].notna().all() and (lightgbm["low"] <= lightgbm["high"]).all()
    assert meta["in_time"] is True and meta["cutoff_utc"] == "2026-11-21T09:00Z"
    assert meta["wind_forecast_published_utc"] == "2026-11-21T06:14Z"
    assert meta["methods_without_forecast"] == []


def test_a_forecast_made_after_the_cut_off_says_so(tmp_path):
    late = pd.Timestamp("2026-11-21 09:05", tz="UTC")
    _, meta = live.make(DAY, archive(tmp_path), model=FakeModel(), make_gbm=FakeTrees, now=late)
    assert meta["in_time"] is False


def test_without_a_wind_forecast_it_waits_until_three_hours_before_the_cut_off(tmp_path):
    early = pd.Timestamp("2026-11-21 05:59", tz="UTC")  # the cut-off is 09:00
    with pytest.raises(live.NotReady, match="wind forecast"):
        live.make(
            DAY,
            archive(tmp_path, wind_to=DAY - timedelta(days=1)),
            model=FakeModel(),
            make_gbm=FakeTrees,
            now=early,
        )


@pytest.mark.parametrize("made", ["2026-11-21 06:00", "2026-11-21 08:59", "2026-11-21 10:00"])
def test_then_it_is_made_without_the_wind_models(tmp_path, made):
    def no_model():
        raise AssertionError("Chronos-2 shouldn't be loaded without a wind forecast")

    table, meta = live.make(
        DAY,
        archive(tmp_path, wind_to=DAY - timedelta(days=1)),
        model=None,
        make_gbm=no_model,
        now=pd.Timestamp(made, tz="UTC"),
    )
    assert table.groupby("method").size().to_dict() == {m: 48 for m in live.METHODS}
    by_method = table.groupby("method")["median"]
    assert by_method.count().to_dict() == {
        "chronos2_wind": 0,
        "lightgbm_wind": 0,
        "yesterday": 48,
        "week_average": 48,
    }
    assert meta["methods_without_forecast"] == ["chronos2_wind", "lightgbm_wind"]
    assert meta["wind_forecast_published_utc"] is None
    assert meta["in_time"] is (made < "2026-11-21 09:00")


def test_a_forecast_without_wind_saves_with_empty_wind_columns(tmp_path):
    table, meta = live.make(
        DAY,
        archive(tmp_path, wind_to=DAY - timedelta(days=1)),
        now=pd.Timestamp("2026-11-21 07:00", tz="UTC"),
    )
    csv, info = live.save(table, meta, tmp_path / "forecasts")
    saved = pd.read_csv(csv)
    assert saved.loc[saved["method"] == "chronos2_wind", "median"].isna().all()
    assert saved.loc[saved["method"] == "week_average", "median"].notna().all()
    assert '"wind_forecast_published_utc": null' in info.read_text()


def test_a_saved_forecast_is_never_replaced(tmp_path):
    table, meta = live.make(DAY, archive(tmp_path), model=FakeModel(), make_gbm=FakeTrees)
    out = tmp_path / "forecasts"
    live.save(table, meta, out)
    with pytest.raises(FileExistsError, match="never replaced"):
        live.save(table, meta, out)
    assert sorted(p.name for p in out.iterdir()) == ["2026-11-22.csv", "2026-11-22.json"]
