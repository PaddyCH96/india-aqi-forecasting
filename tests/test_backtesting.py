"""Tests for honest forecast evaluation.

The leakage checks matter most: the original features let the model read the
target off its own inputs, which produced sub-1% error that meant nothing.
These assertions fail if that regresses.
"""

import numpy as np
import pandas as pd
import pytest

from lib.backtesting import (
    KNOWN_AHEAD,
    make_direct_dataset,
    persistence_forecast,
    skill_score,
)
from lib.feature_engineering import (
    add_city_normalization,
    add_rolling_features,
    build_feature_pipeline,
)


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01-01", periods=400, freq="D")
    return pd.DataFrame({
        "city": "Testville",
        "date": dates,
        "aqi": 100 + 40 * np.sin(np.arange(400) / 30) + rng.normal(0, 10, 400),
        "pm2_5": rng.normal(60, 15, 400),
        "no2": rng.normal(30, 8, 400),
        "co": rng.normal(1.0, 0.3, 400),
    })


class TestNoTargetLeakage:
    def test_rolling_mean_excludes_current_day(self, series):
        """A window containing today leaks today's answer into its predictor."""
        out = add_rolling_features(series, columns=["aqi"], windows=[3])
        # Row 3's rolling mean must come from rows 0-2, never row 3.
        expected = series["aqi"].iloc[0:3].mean()
        assert out["aqi_roll3_mean"].iloc[3] == pytest.approx(expected)

    def test_rolling_mean_first_row_is_nan(self, series):
        out = add_rolling_features(series, columns=["aqi"], windows=[3])
        assert pd.isna(out["aqi_roll3_mean"].iloc[0])

    def test_city_zscore_is_not_a_copy_of_the_target(self, series):
        """The old formula correlated 1.000 with the target -- it was the answer."""
        out = add_city_normalization(series, target="aqi")
        corr = out["aqi_city_zscore"].corr(out["aqi"])
        assert abs(corr) < 0.98, f"z-score still tracks the target (r={corr:.3f})"

    def test_no_engineered_feature_is_perfectly_correlated(self, series):
        feat = build_feature_pipeline(series, target="aqi")
        numeric = feat.select_dtypes(include=[np.number]).drop(columns=["aqi"])
        for col in numeric.columns:
            corr = numeric[col].corr(feat["aqi"])
            if pd.notna(corr):
                assert abs(corr) < 0.99, f"{col} leaks the target (r={corr:.3f})"


class TestPersistenceBaseline:
    def test_repeats_last_observed_value(self, series):
        fc = persistence_forecast(series, horizon=5, target="aqi")
        assert len(fc) == 5
        assert fc["prediction"].nunique() == 1
        assert fc["prediction"].iloc[0] == pytest.approx(series["aqi"].iloc[-1])

    def test_dates_run_forward_from_the_last_day(self, series):
        fc = persistence_forecast(series, horizon=3, target="aqi")
        assert fc["date"].iloc[0] == series["date"].iloc[-1] + pd.Timedelta(days=1)

    def test_empty_history_is_handled(self):
        empty = pd.DataFrame({"city": [], "date": [], "aqi": []})
        assert persistence_forecast(empty, 3).empty


class TestDirectDataset:
    def test_target_is_taken_h_days_ahead(self, series):
        feat = build_feature_pipeline(series, target="aqi")
        names = ["pm2_5", "no2"]
        X, y = make_direct_dataset(feat, "aqi", horizon=7, feature_names=names)
        assert y.iloc[0] == pytest.approx(feat["aqi"].iloc[7])

    def test_calendar_features_describe_the_predicted_date(self, series):
        feat = build_feature_pipeline(series, target="aqi")
        names = [c for c in KNOWN_AHEAD if c in feat.columns]
        X, _ = make_direct_dataset(feat, "aqi", horizon=5, feature_names=names)
        # The month attached to row 0 should be the month five days later.
        assert X["month"].iloc[0] == pytest.approx(feat["month"].iloc[5])

    def test_delta_mode_targets_the_change(self, series):
        feat = build_feature_pipeline(series, target="aqi")
        X, y = make_direct_dataset(feat, "aqi", 3, ["pm2_5"], delta=True)
        change = feat["aqi"].iloc[3] - feat["aqi"].iloc[0]
        assert y.iloc[0] == pytest.approx(change)

    def test_rows_shrink_with_longer_horizon(self, series):
        feat = build_feature_pipeline(series, target="aqi")
        _, y1 = make_direct_dataset(feat, "aqi", 1, ["pm2_5"])
        _, y14 = make_direct_dataset(feat, "aqi", 14, ["pm2_5"])
        assert len(y14) < len(y1)


class TestSkillScore:
    def test_positive_when_model_beats_baseline(self):
        assert skill_score(8.0, 10.0) == pytest.approx(20.0)

    def test_negative_when_model_is_worse(self):
        assert skill_score(12.0, 10.0) == pytest.approx(-20.0)

    def test_none_when_baseline_is_unusable(self):
        assert skill_score(10.0, 0) is None
        assert skill_score(float("nan"), 10.0) is None
