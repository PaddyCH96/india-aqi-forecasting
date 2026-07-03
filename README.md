# India Air Quality Forecasting & Urban Risk Analysis

Forecast AQI up to 2030 for Indian cities using Prophet time-series models. Includes interactive dashboards, a REST API, Docker deployment, and CI/CD — built for urban planning, public health, and real estate risk assessment.

## Features

- **Dashboards** — Streamlit-based interactive visualizations (history, forecast, validation, multi-city comparison)
- **REST API** — FastAPI endpoints for forecasts, validation, and city metadata (auto-generated docs at `/docs`)
- **Batch Forecasting** — Multi-city Prophet pipeline with trend analysis (improving/worsening cities)
- **Model Validation** — 3-model comparison (full vs pre-COVID vs skip-COVID) with cross-validation
- **Data Ingestion** — Fetches from OpenAQ and Open-Meteo APIs with synthetic data fallback
- **Docker Deployment** — 4-service compose: PostgreSQL, seed, dashboard, optional API
- **CI Pipeline** — GitHub Actions runs 100 tests with 95% coverage on push
- **Shared Library** — `lib/` package as a single source of truth for all business logic

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Data Sources                                               │
│  ┌──────────┐  ┌────────────┐  ┌─────────────────────────┐ │
│  │ OpenAQ   │  │ Open-Meteo │  │ Synthetic (fallback)    │ │
│  └────┬─────┘  └─────┬──────┘  └───────────┬─────────────┘ │
│       │              │                      │               │
│       └──────────────┴──────────────────────┘               │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │  seed_data  │                          │
│                    └──────┬──────┘                          │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │  PostgreSQL │                          │
│                    │  (city_day) │                          │
│                    └──┬──────┬───┘                          │
│           ┌───────────┘      └───────────┐                  │
│           │                              │                  │
│    ┌──────▼──────┐              ┌────────▼────────┐       │
│    │  Dashboards │              │  FastAPI API    │       │
│    │  (Streamlit)│              │  /forecast       │       │
│    │  :8501      │              │  /validate       │       │
│    └─────────────┘              │  /cities/:health │       │
│                                 └────────┬────────┘       │
│                                          │                 │
│  User ◄──── Browser/curl ◄──────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- PostgreSQL (if running locally)
- Docker + Docker Compose (optional, for containerized deployment)

## Quick Start (Docker)

```bash
docker compose up --build
# Dashboard: http://localhost:8501
```

With REST API:

```bash
docker compose --profile api up --build
# Dashboard: http://localhost:8501
# API docs: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

## Local Setup

```bash
# 1. Clone and enter
git clone https://github.com/PaddyCH96/india-aqi-forecasting.git
cd india-air-quality

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure database
cp .env.example .env
# Edit .env if using a non-default PostgreSQL URL
# Default: postgresql://postgres:postgres@localhost:5432/india_air_quality

# 5. Create database and seed
createdb india_air_quality
python scripts/seed_data.py   # bootstraps from CSV or generates synthetic data
```

## Usage

### Dashboards

```bash
# Minimal 3-tab dashboard (History, Forecast, Validation)
streamlit run scripts/dashboard_final.py

# Feature-rich 4-tab dashboard (adds Multi-City Comparison)
streamlit run scripts/dashboard_complete.py
```

### REST API

```bash
uvicorn scripts.api:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service health + last data timestamp |
| `GET /cities` | List all cities with data summaries |
| `GET /forecast/{city}` | Prophet forecast for a city |
| `GET /validate/{city}` | Validation metrics (MAPE, RMSE, MAE) |

### Data Ingestion

```bash
# Fetch from OpenAQ API (real AQI data)
python scripts/fetch_openaq.py

# Fetch from Open-Meteo API (recent AQI data)
python scripts/fetch_recent_aqi.py
```

### Forecasting Pipelines

```bash
# Batch forecast for all eligible cities
python scripts/multi_city_pipeline.py

# Model validation (3-model comparison + cross-validation)
python scripts/validate_prophet.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=lib/ --cov-report=term

# Run specific test file
pytest tests/test_db.py -v
```

## Project Structure

```
india-air-quality/
├── lib/                    # Shared library (single source of truth)
│   ├── config.py           #   App configuration & constants
│   ├── db.py               #   Database connection & queries
│   ├── models.py           #   Prophet model wrappers
│   ├── metrics.py          #   MAPE, RMSE, MAE, evaluation
│   ├── aqi.py              #   PM2.5→AQI conversion & synthetic data
│   ├── charts.py           #   6 reusable chart functions
│   ├── logging.py          #   Logging setup
│   └── utils.py            #   Retry decorator, URL validation
├── scripts/                # Runnable scripts
│   ├── api.py              #   FastAPI REST API
│   ├── dashboard_final.py  #   3-tab Streamlit dashboard
│   ├── dashboard_complete.py # 4-tab feature-rich dashboard
│   ├── seed_data.py        #   Database bootstrap
│   ├── fetch_openaq.py     #   OpenAQ ingestion
│   ├── fetch_recent_aqi.py #   Open-Meteo ingestion
│   ├── multi_city_pipeline.py  # Batch forecasting
│   └── validate_prophet.py #   Model validation
├── tests/                  # 100 tests across 6 files
├── docs/                   # Architecture, deployment, testing, handover
├── notebooks/              # Jupyter notebooks
├── sql/                    # Analytical SQL queries
├── data/raw/               # Raw CSV datasets (gitignored)
├── data/processed/         # Generated data (gitignored)
├── outputs/                # Charts, forecasts (gitignored)
├── Dockerfile              # Python 3.11-slim container
├── docker-compose.yml      # 4 services (db, seed, dashboard, api)
├── .env.example            # Environment template
├── requirements.txt        # Pinned dependencies
└── release_report.md       # Release validation summary
```

## Configuration

All configuration in `lib/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_URL` | `postgresql://postgres:postgres@localhost:5432/india_air_quality` | Database URL |
| `TRAIN_CUTOFF` | `2023-01-01` | Train/test split date |
| `FORECAST_PERIODS` | 2190 | Forecast periods (6 years daily) |
| `FORECAST_YEARS` | 6 | Forecast horizon in years |
| `PROPHET_PARAMS` | `yearly_seasonality=True` | Default Prophet parameters |

Set via `.env`:
```env
DB_URL=postgresql://user:pass@host:5432/india_air_quality
```

## Deployment

### Docker (Production)

```bash
# Full stack
docker compose --profile api up --build -d

# Scale API
docker compose --profile api up --build -d --scale api=3
```

### Manual (Single Service)

```bash
# Dashboard
streamlit run scripts/dashboard_final.py --server.port 8501 --server.address 0.0.0.0

# API
uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Limitations

- **PostgreSQL dependency** — all dashboards and pipelines require a running Postgres instance
- **Synthetic data fallback** — when no real CSV is available, `seed_data.py` generates plausible synthetic data
- **Single-regressor model** — Prophet uses only yearly seasonality (weather regressors were tested but showed unstable correlations across the COVID period)
- **MAPE range** — current accuracy: Hyderabad 15.6%, Mumbai 13.05%
- **No real-time updates** — dashboards show static data until re-run

## Contributing

1. Branch off `main`: `git checkout -b feature/your-feature`
2. Make changes, add tests in `tests/`
3. Run tests: `pytest tests/ -v --cov=lib/`
4. Run linter: `ruff check lib/ scripts/ tests/`
5. Submit a pull request

## Key Insights

- Clear seasonal AQI spikes across most Tier-1 cities
- Long-term upward AQI trend in high-growth urban zones
- Hyderabad shows consistent upward AQI trend post-2018, indicating increasing environmental risk in high-growth corridors
- Significant variance across cities suggests localized policy intervention is required

## Related Resources

- `docs/architecture.md` — System architecture and data flow
- `docs/deployment.md` — Production deployment guide
- `docs/testing.md` — Testing strategy and coverage
- `docs/handover.md` — Full project handover documentation
- `release_report.md` — Release validation (100 tests, 95% coverage)
