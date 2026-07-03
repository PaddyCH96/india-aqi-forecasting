# Deployment Guide

## Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.10+ with PostgreSQL 16

---

## Option A: Docker Compose (Recommended)

### Quick Start

```bash
# Build and start all services
docker compose up --build

# Access the dashboard
open http://localhost:8501

# Access the REST API (requires api profile)
docker compose --profile api up --build
open http://localhost:8000/docs
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | PostgreSQL 16 with `india_air_quality` database |
| `seed` | — | One-shot container that seeds data on first run |
| `dashboard` | 8501 | Streamlit dashboard (3-tab) |
| `api` (profile: api) | 8000 | FastAPI REST API with auto-generated docs at `/docs` |

### Data Persistence

PostgreSQL data is stored in a Docker volume `pgdata`. To reset:

```bash
docker compose down -v
docker compose up --build
```

The `seed` service checks if the database already has data and skips seeding if it does. To force re-seed, delete the volume.

### Stopping

```bash
docker compose down
```

---

## Option B: Local Python

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure database
cp .env.example .env
# Edit .env with your PostgreSQL URL if not using localhost defaults

# Create the database
createdb india_air_quality

# Seed data
python scripts/seed_data.py
```

### Run Dashboard

```bash
streamlit run scripts/dashboard_final.py
# Opens at http://localhost:8501
```

### Run API

```bash
uvicorn scripts.api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### Run Batch Pipeline

```bash
python scripts/multi_city_pipeline.py
python scripts/validate_prophet.py
```

---

## Option C: Cloud Deployment

### Heroku / Render / Fly.io

1. Containerize using the provided Dockerfile.
2. Set environment variable `AQI_DB_URL` to your managed PostgreSQL instance.
3. For Heroku, use `heroku.yml`:
   ```yaml
   build:
     docker:
       web: Dockerfile
   ```
4. For seed data, run `python scripts/seed_data.py` as a one-off dyno/task.

### Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `AQI_DB_URL` | `postgresql://postgres@localhost:5432/india_air_quality` | Yes |

---

## Health Check

```bash
# Dashboard
curl http://localhost:8501/healthz  # Not applicable — Streamlit has no health endpoint

# API
curl http://localhost:8000/health
# {"status":"ok"}
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Could not connect to server" | PostgreSQL not running | `docker compose up db` or start local PostgreSQL |
| "Table 'city_day' does not exist" | No seed data | `python scripts/seed_data.py` |
| "Prophet installation failed" | Missing gcc | On macOS: `xcode-select --install`; on Linux: `apt-get install gcc` |
| Docker "port already allocated" | Local PostgreSQL running on 5432 | Stop local PostgreSQL or use `docker compose down` |
