import numpy as np
import pytest
from lib.aqi import pm25_to_aqi, generate_synthetic_aqi


class TestPm25ToAqi:
    def test_none_input(self):
        assert pm25_to_aqi(None) is None

    def test_nan_input(self):
        assert pm25_to_aqi(float("nan")) is None

    def test_zero_pm25(self):
        assert pm25_to_aqi(0) == 0.0

    def test_first_breakpoint(self):
        result = pm25_to_aqi(30)
        assert abs(result - 50.1) < 0.01

    def test_second_breakpoint(self):
        result = pm25_to_aqi(60)
        assert abs(result - 100.1) < 0.01

    def test_third_breakpoint(self):
        result = pm25_to_aqi(90)
        assert abs(result - 199.9) < 0.1

    def test_fourth_breakpoint(self):
        result = pm25_to_aqi(120)
        assert abs(result - 299.9) < 0.1

    def test_fifth_breakpoint(self):
        result = pm25_to_aqi(250)
        assert abs(result - 400.1) < 0.1

    def test_above_250(self):
        result = pm25_to_aqi(300)
        expected = min(400 + (300 - 250) * 0.4, 500)
        assert abs(result - expected) < 0.1

    def test_capped_at_500(self):
        result = pm25_to_aqi(1000)
        assert result == 500.0

    def test_mid_range_values(self):
        result_40 = pm25_to_aqi(40)
        assert 50 < result_40 < 100
        result_75 = pm25_to_aqi(75)
        assert 100 < result_75 < 200

    @pytest.mark.parametrize("pm25,expected_range", [
        (10, (10, 30)),
        (50, (50, 100)),
        (100, (100, 250)),
        (200, (300, 400)),
        (400, (400, 500)),
    ])
    def test_value_ranges(self, pm25, expected_range):
        result = pm25_to_aqi(pm25)
        lo, hi = expected_range
        assert lo <= result <= hi, f"pm25={pm25} -> {result} not in [{lo}, {hi}]"

    def test_monotonically_increasing(self):
        values = np.linspace(0, 300, 100)
        results = [pm25_to_aqi(v) for v in values]
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1], (
                f"Not monotonic at index {i}: {results[i-1]} -> {results[i]}"
            )


class TestGenerateSyntheticAqi:
    def test_returns_correct_length(self, sample_dates):
        result = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=42)
        assert len(result) == 366

    def test_values_within_bounds(self, sample_dates):
        result = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=42)
        assert result.min() >= 30
        assert result.max() <= 400

    def test_deterministic_with_seed(self, sample_dates):
        r1 = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=42)
        r2 = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_different_results(self, sample_dates):
        r1 = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=42)
        r2 = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=99)
        assert not np.allclose(r1, r2)

    def test_higher_base_means_higher_aqi(self, sample_dates):
        r_low = generate_synthetic_aqi("City", sample_dates, 50, seed=42)
        r_high = generate_synthetic_aqi("City", sample_dates, 200, seed=42)
        assert r_high.mean() > r_low.mean()

    def test_has_seasonal_pattern(self, sample_dates):
        result = generate_synthetic_aqi("City", sample_dates, 100, seed=42)
        winter_months = result[:90]
        summer_months = result[172:263]
        assert abs(winter_months.mean() - summer_months.mean()) > 5

    def test_weekly_pattern_weekends_lower(self, sample_dates):
        result = generate_synthetic_aqi("City", sample_dates, 100, seed=42)
        weekend_vals = [result[i] for i, d in enumerate(sample_dates) if d.weekday() >= 5]
        weekday_vals = [result[i] for i, d in enumerate(sample_dates) if d.weekday() < 5]
        assert np.mean(weekend_vals) <= np.mean(weekday_vals)

    def test_returns_float64(self, sample_dates):
        result = generate_synthetic_aqi("City", sample_dates, 100, seed=42)
        assert result.dtype == np.float64
