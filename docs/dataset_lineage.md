# Dataset Lineage

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│                                                                 │
│  CPCB Raw CSVs          OpenAQ API         Open-Meteo API       │
│  (2015-2020)            (2020-2024)        (2020-2024)          │
│  ┌──────────────────┐   ┌──────────┐       ┌──────────────┐     │
│  │ city_day.csv     │   │ REST     │       │ REST          │     │
│  │ city_hour.csv    │   │ /v2/     │       │ /v1/          │     │
│  │ station_day.csv  │   │measure-  │       │ air-quality   │     │
│  │ station_hour.csv │   │ments     │       │               │     │
│  │ stations.csv     │   └────┬─────┘       └──────┬───────┘     │
│  └────────┬─────────┘        │                     │            │
│           │                  │                     │            │
│           ▼                  ▼                     ▼            │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                 INGESTION PIPELINES                     │     │
│  │                                                        │     │
│  │  init_db.py    seed_data.py    fetch_openaq.py         │     │
│  │  (full reset)  (incremental)   (OpenAQ live)          │     │
│  │                              fetch_recent_aqi.py       │     │
│  │                              (Open-Meteo live)         │     │
│  └───────────────────────┬────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              DATABASE (PostgreSQL)                      │     │
│  │                                                        │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │  city_measurements (fact table)                  │   │     │
│  │  │  ├── Real CPCB data:    29,531 rows  (75%)       │   │     │
│  │  │  ├── Synthetic data:     9,870 rows  (25%)       │   │     │
│  │  │  └── 26 cities, 2015-01-01 to 2024-12-31         │   │     │
│  │  ├──────────────────────────────────────────────────┤   │     │
│  │  │  stations (lookup) — 230 stations                │   │     │
│  │  └──────────────────────────────────────────────────┘   │     │
│  │                                                        │     │
│  │  Views:                                                 │     │
│  │    city_day — backward-compatible view                  │     │
│  └───────────────────────┬────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                 CONSUMERS                                │     │
│  │                                                        │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │     │
│  │  │dashboards│  │   API    │  │  ML Pipelines    │     │     │
│  │  │Streamlit │  │  FastAPI │  │  Prophet/XGBoost │     │     │
│  │  └──────────┘  └──────────┘  └──────────────────┘     │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Lineage by Dataset

### city_day.csv → city_measurements (CPCB source)

| Step | Process | Tool | Provenance |
|------|---------|------|------------|
| 1 | CPCB publishes daily city-level air quality | Government | Official Indian govt data |
| 2 | Downloaded as city_day.csv | Manual | File on disk |
| 3 | Loaded via init_db.py | Pandas + SQLAlchemy | `is_synthetic=FALSE`, `data_source='CPCB'` |
| 4 | Available in city_measurements | PostgreSQL | `ingested_at` timestamped |

**Row count**: 29,531 | **Hash**: (not computed) | **Update frequency**: Static (published 2020)

### city_hour.csv (Available, Not Currently Ingested)

| Step | Process | Tool | Notes |
|------|---------|------|-------|
| 1 | CPCB publishes hourly city-level air quality | Government | 707,875 rows |
| 2 | Downloaded as city_hour.csv | Manual | File on disk (63 MB) |
| 3 | Not currently ingested into DB | — | Available for future use |

**Status**: Available on disk, schema compatible with city_measurements. Add `data_source='CPCB_Hourly'` when ingested.

### station_day.csv → stations + city_measurements (Available, Not Fully Ingested)

| Step | Process | Tool | Notes |
|------|---------|------|-------|
| 1 | CPCB publishes daily station-level data | Government | 108,035 rows |
| 2 | Downloaded as station_day.csv | Manual | File on disk |
| 3 | station_day not yet ingested | — | Stations metadata loaded separately |
| 4 | stations.csv → stations table | init_db.py | 230 stations, 127 cities |

**Status**: Station data available. Join through `stations` table via `station_id`.

### Synthetic Data Generation

| Step | Process | Tool | Provenance |
|------|---------|------|------------|
| 1 | Generate seasonal + trend + noise AQI | lib/aqi.py | `is_synthetic=TRUE` |
| 2 | 6 cities, 2020-07-01 to 2024-12-31 | Pandas | `data_source='synthetic'` |
| 3 | Inserted via init_db.py/seed_data.py | SQLAlchemy | `ingested_at` timestamped |

**Row count**: 9,870 | **Algorithm**: `aqi = base + seasonal_sin + trend_linear + weekday_effect + gaussian_noise`

### OpenAQ API → city_measurements (Potential Future Source)

| Step | Process | Tool | Notes |
|------|---------|------|-------|
| 1 | GET /v2/measurements?city=X | requests | Real-time API |
| 2 | Aggregate hourly→daily | Pandas | Mean per day |
| 3 | Insert via fetch_openaq.py | insert_city_data | `data_source='OpenAQ'` |

**Status**: Script exists but API may require authentication for historical data.

## Data Provenance Rules

1. **Every row** in city_measurements must have a non-null `is_synthetic` and `data_source`
2. **Model training** defaults to `is_synthetic=FALSE` (real data only)
3. **Synthetic data** can be explicitly enabled via `use_synthetic=True` parameter
4. **Duplicate handling**: Unique constraint on (city, date, data_source) prevents re-insertion
5. **Freshness**: `ingested_at` timestamp on every row tracks when data entered the system

## Data Freshness

| Metric | Value |
|--------|-------|
| Latest real data | 2020-07-01 |
| Latest synthetic data | 2024-12-31 |
| Data gap (real) | 2020-07-01 to present (~6 years) |
| Data sources in DB | 2 (CPCB, synthetic) |

The 2020-2024 gap is the primary data limitation. Options to close it:
1. OpenAQ API (may require authentication)
2. Open-Meteo API (free, limited coverage)
3. CPCB live portal (manual download available)
4. Accept synthetic for demo purposes (clearly labeled)
