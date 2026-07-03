import pandas as pd
import numpy as np
import pytest
from lib.metrics import (
    calc_mape,
    calc_rmse,
    calc_mae,
    evaluate_forecast,
    classify_model_quality,
)


class TestCalcMape:
    def test_perfect_prediction(self):
        actual = pd.Series([100.0, 110.0, 120.0])
        pred = actual.copy()
        assert calc_mape(actual, pred) == 0.0

    def test_known_values(self, sample_actual_predicted):
        actual, pred = sample_actual_predicted
        result = calc_mape(actual, pred)
        expected = (
            np.mean(np.abs((actual - pred) / actual)) * 100
        )
        assert abs(result - expected) < 1e-10

    def test_handles_zeros_in_actual(self):
        actual = pd.Series([0.0, 100.0, 200.0])
        pred = pd.Series([5.0, 105.0, 195.0])
        result = calc_mape(actual, pred)
        assert np.isinf(result) or np.isnan(result) is False

    def test_single_element(self):
        assert calc_mape(pd.Series([50.0]), pd.Series([55.0])) == 10.0

    def test_empty_series(self):
        assert np.isnan(calc_mape(pd.Series([], dtype=float), pd.Series([], dtype=float)))


class TestCalcRmse:
    def test_perfect_prediction(self):
        actual = pd.Series([100.0, 110.0, 120.0])
        pred = actual.copy()
        assert calc_rmse(actual, pred) == 0.0

    def test_known_values(self):
        actual = pd.Series([100.0, 110.0, 120.0])
        pred = pd.Series([95.0, 115.0, 118.0])
        expected = np.sqrt(np.mean([25.0, 25.0, 4.0]))
        assert abs(calc_rmse(actual, pred) - expected) < 1e-10

    def test_single_element(self):
        assert calc_rmse(pd.Series([100.0]), pd.Series([90.0])) == 10.0


class TestCalcMae:
    def test_perfect_prediction(self):
        actual = pd.Series([100.0, 110.0, 120.0])
        pred = actual.copy()
        assert calc_mae(actual, pred) == 0.0

    def test_known_values(self):
        actual = pd.Series([100.0, 110.0, 120.0])
        pred = pd.Series([95.0, 115.0, 118.0])
        expected = np.mean([5.0, 5.0, 2.0])
        assert abs(calc_mae(actual, pred) - expected) < 1e-10

    def test_single_element(self):
        assert calc_mae(pd.Series([100.0]), pd.Series([80.0])) == 20.0


class TestEvaluateForecast:
    def test_returns_all_metrics(self, sample_results):
        result = evaluate_forecast(sample_results)
        assert "mape" in result
        assert "rmse" in result
        assert "mae" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_matches_individual_calculations(self, sample_results):
        result = evaluate_forecast(sample_results)
        y, yhat = sample_results["y"], sample_results["yhat"]
        assert abs(result["mape"] - calc_mape(y, yhat)) < 1e-10
        assert abs(result["rmse"] - calc_rmse(y, yhat)) < 1e-10
        assert abs(result["mae"] - calc_mae(y, yhat)) < 1e-10

    def test_uses_correct_columns(self, sample_results):
        result = evaluate_forecast(sample_results)
        assert result["mape"] > 0


class TestClassifyModelQuality:
    @pytest.mark.parametrize("mape,expected_label,expected_color", [
        (10.0, "Excellent", "green"),
        (14.9, "Excellent", "green"),
        (15.0, "Good", "blue"),
        (19.9, "Good", "blue"),
        (20.0, "Moderate", "orange"),
        (29.9, "Moderate", "orange"),
        (30.0, "Poor", "red"),
        (50.0, "Poor", "red"),
    ])
    def test_classification(self, mape, expected_label, expected_color):
        label, color = classify_model_quality(mape)
        assert label == expected_label
        assert color == expected_color

    def test_zero_mape(self):
        assert classify_model_quality(0.0) == ("Excellent", "green")

    def test_edge_boundary_excellent_good(self):
        label, _ = classify_model_quality(14.999)
        assert label == "Excellent"
        label, _ = classify_model_quality(15.0)
        assert label == "Good"

    def test_edge_boundary_good_moderate(self):
        label, _ = classify_model_quality(19.999)
        assert label == "Good"
        label, _ = classify_model_quality(20.0)
        assert label == "Moderate"

    def test_edge_boundary_moderate_poor(self):
        label, _ = classify_model_quality(29.999)
        assert label == "Moderate"
        label, _ = classify_model_quality(30.0)
        assert label == "Poor"
