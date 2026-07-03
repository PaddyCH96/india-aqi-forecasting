# Testing Guide

## Test Suite Overview

62 unit tests across 3 test files, covering all pure-computation modules in `lib/`.

| Test File | Module | Tests | Type |
|-----------|--------|-------|------|
| `tests/test_metrics.py` | `lib/metrics.py` | 27 | Unit (no DB needed) |
| `tests/test_aqi.py` | `lib/aqi.py` | 15 | Unit (no DB needed) |
| `tests/test_models.py` | `lib/models.py` | 9 | Unit (Prophet installed, no DB) |

### Coverage

- `lib/metrics.py`: 100% — MAPE, RMSE, MAE, evaluate_forecast, classify_model_quality
- `lib/aqi.py`: 100% — pm25_to_aqi (all breakpoints), generate_synthetic_aqi (shape, bounds, seasonality)
- `lib/models.py`: 100% — create_model, train_and_forecast, train_and_validate

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### With Coverage Report

```bash
pytest tests/ --cov=lib/ --cov-report=term --cov-report=html
open htmlcov/index.html
```

### Run Specific Test File

```bash
pytest tests/test_metrics.py -v
pytest tests/test_aqi.py -v
pytest tests/test_models.py -v
```

### Run Specific Test

```bash
pytest tests/test_aqi.py::TestPm25ToAqi::test_first_breakpoint -v
```

## Test Design

### Fixtures

Shared fixtures in `tests/conftest.py`:

| Fixture | Purpose |
|---------|---------|
| `sample_actual_predicted` | Two aligned pandas Series for metric tests |
| `sample_results` | DataFrame with `ds`, `y`, `yhat` columns |
| `sample_dates` | Full-year DatetimeIndex for synthetic AQI tests |

### No Database Required

All existing tests are pure computation and require no PostgreSQL connection. Tests in `test_models.py` require Prophet to be installed but do not touch the database.

### Key Test Patterns

**Boundary testing** (AQI breakpoints):
```python
def test_edge_boundary_excellent_good(self):
    label, _ = classify_model_quality(14.999)
    assert label == "Excellent"
    label, _ = classify_model_quality(15.0)
    assert label == "Good"
```

**Determinism** (synthetic data):
```python
def test_deterministic_with_seed(self, sample_dates):
    r1 = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=42)
    r2 = generate_synthetic_aqi("Delhi", sample_dates, 180, seed=42)
    np.testing.assert_array_equal(r1, r2)
```

**Monotonicity** (PM2.5→AQI conversion):
```python
def test_monotonically_increasing(self):
    values = np.linspace(0, 300, 100)
    results = [pm25_to_aqi(v) for v in values]
    for i in range(1, len(results)):
        assert results[i] >= results[i - 1]
```

## CI Integration

Tests run automatically via GitHub Actions on every push/PR to `main`:

- **OS:** ubuntu-latest
- **Python:** 3.11
- **Steps:** Install deps → `pytest tests/ --cov=lib/ --cov-report=xml`
- **Artifact:** Coverage report uploaded as `coverage-report`

Configuration in `.github/workflows/test.yml`.

## Adding Tests

1. Create `tests/test_<module>.py` following existing patterns.
2. Add fixtures to `tests/conftest.py` if needed.
3. Run `pytest tests/ -v` to verify.
4. Keep tests database-free unless testing DB-specific logic (use mocks).

## Known Limitations

- No integration tests (require running PostgreSQL). The `seed_data.py` and fetch scripts are tested only by manual execution.
- No API tests (FastAPI endpoints tested via manual curl/`/docs`).
- `lib/charts.py` is not unit tested (visual output is verified by inspection).
