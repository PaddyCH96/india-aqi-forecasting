# Handover Document

## Project Status

**India Air Quality Forecasting** is a production-ready data science project that forecasts AQI for 6 Indian cities through 2030 using Meta's Prophet model.

### What Works

- **Data ingestion:** Two fetch scripts pulling from OpenAQ and Open-Meteo APIs with synthetic data fallback.
- **Database:** PostgreSQL schema (`city_day` table) with parameterized queries (no SQL injection).
- **Forecasting:** Prophet with yearly seasonality, 6-year horizon, 95% confidence intervals.
- **Accuracy:** 13–16% MAPE on 6 cities (best: Mumbai 13.05%, Hyderabad 15.6%).
- **Dashboards:** Two Streamlit apps — 3-tab minimal and 4-tab feature-rich with multi-city comparison.
- **REST API:** FastAPI with endpoints for cities, forecast, and validation.
- **Containerization:** Docker Compose with PostgreSQL, seed, dashboard, and optional API services.
- **Tests:** 62 unit tests covering all lib modules, run via CI.
- **Documentation:** README, architecture, deployment, testing, handover, implementation plan.

### What Needs Attention

| Item | Priority | Notes |
|------|----------|-------|
| Model accuracy | Medium | Single regressor (yearly seasonality). Adding holiday effects (`add_country_holidays('IN')`) could reduce MAPE.|
| Real-time data | Low | Current data ends 2024. Needs scheduled fetch → retrain → update pipeline. |
| Tier-2 cities | Low | Pune, Lucknow, Jaipur have data but may have lower forecast confidence. |
| Integration tests | Low | Require PostgreSQL. Mock-based tests could supplement but aren't written. |
| Synthetic data quality | Low | 2020-2024 data is simulated for cities without real API coverage. |

### Key Decisions

1. **Keep both dashboards:** `dashboard_final.py` (minimal, 3-tab) and `dashboard_complete.py` (feature-rich, 4-tab) serve different user segments. Both share `lib/charts.py`.
2. **FastAPI over Flask:** Lighter weight, async support, auto-generated docs at `/docs`.
3. **Docker Compose over Kubernetes:** Appropriate for project scale. Single-node deployment is sufficient.
4. **Synthetic fallback:** Acceptable for demo/deployment but documented as limitation. Users should prefer real OpenAQ data when available.
5. **Prophet over ARIMA/XGBoost:** Proven accuracy for this dataset. Prophet handles seasonality and missing data well.

### Project Structure

```
india-air-quality/
├── lib/                # Shared library (8 modules)
├── scripts/            # 8 scripts (dashboards, fetch, pipeline, validation, API, seed)
├── tests/              # 62 unit tests across 3 files
├── docs/               # Architecture, deployment, testing, handover
├── data/raw/           # Raw CSV datasets (gitignored)
├── data/processed/     # Cleaned/synthetic data
├── outputs/            # Charts, CSV results, model config
├── notebooks/          # Jupyter notebooks for exploration
├── sql/                # Analytical SQL queries
├── .github/workflows/  # CI pipeline
├── Dockerfile          # Container definition
├── docker-compose.yml  # Multi-service orchestration
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore
├── README.md
└── implementation.md   # Detailed implementation plan
```

### Quick References

```bash
# Run dashboard
streamlit run scripts/dashboard_final.py

# Run API
uvicorn scripts.api:app --reload --port 8000

# Run tests
pytest tests/ -v --cov=lib/

# Deploy with Docker
docker compose up --build

# Deploy with Docker + API
docker compose --profile api up --build

# Seed database
python scripts/seed_data.py

# Batch forecast all cities
python scripts/multi_city_pipeline.py

# Run model validation
python scripts/validate_prophet.py
```

### Dependencies

See `requirements.txt`. Key additions beyond standard data science stack:
- `fastapi` + `uvicorn` for REST API
- `pytest-cov` for test coverage

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/cities` | GET | List available cities |
| `/forecast/{city}` | GET | Forecast for a city through 2030 |
| `/validate/{city}` | GET | Model validation metrics |

### Contact / Ownership

This project was built as part of an independent data science portfolio. For questions about the architecture or roadmap, refer to the `implementation.md` document or open an issue in the repository.

### Next Steps for a New Developer

1. Read `README.md` and `implementation.md` for context.
2. Run `pytest tests/ -v` to verify test suite.
3. Run `docker compose up --build` to see the full stack in action.
4. Explore the `lib/` package to understand core abstractions.
5. To improve the model, start with holiday effects and changepoint tuning (see `implementation.md`).
