"""Tests for scripts/ingest_weather.py against a temporary SQLite database.

`scripts/` is not an importable package, so the module is loaded by path. Every
test builds its own engine under pytest's `tmp_path` - no test ever touches the
committed demo database.
"""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import text as sa_text

from lib.db import get_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "ingest_weather.py"
FIXTURE = Path(__file__).parent / "fixtures" / "open_meteo_era5_delhi_3day.json"

SCHEMA_COLUMNS = [
    "city", "date", "temp_mean_c", "temp_max_c", "temp_min_c",
    "precipitation_mm", "wind_speed_mean_kmh", "wind_speed_max_kmh",
    "wind_direction_deg", "relative_humidity_pct", "surface_pressure_hpa",
    "blh_mean_m", "blh_max_m", "blh_min_m",
    "grid_latitude", "grid_longitude", "grid_elevation_m",
    "is_synthetic", "data_source", "ingested_at",
]


def _load_module():
    """`scripts/` has no __init__.py, so load the script module by path."""
    spec = importlib.util.spec_from_file_location("ingest_weather", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload():
    """A fresh copy each call so mutating tests cannot leak state.

    Deliberately local to this module: `tests/test_open_meteo.py` has its own
    copy so the two test modules stay independently runnable.
    """
    return json.loads(FIXTURE.read_text())


def _engine(tmp_path):
    return get_engine(f"sqlite:///{tmp_path}/t.db")


def _weather_frame():
    from lib.providers.open_meteo import parse_era5_daily

    return parse_era5_daily(_payload(), "Delhi")


def _seed_city_measurements(engine):
    """Build a minimal `city_measurements` the way the real DB writes it.

    The real table is ~45% synthetic and every query in this phase is
    CPCB-scoped, so the fixture carries `data_source` / `is_synthetic` too.
    One city, three distinct dates.
    """
    pd.DataFrame({
        "city": ["Delhi"] * 3,
        "date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"]),
        "aqi": [180.0, 190.0, 200.0],
        "data_source": ["CPCB"] * 3,
        "is_synthetic": [0, 0, 0],
    }).to_sql("city_measurements", engine, if_exists="append", index=False)


class TestSchema:
    def test_schema(self, tmp_path):
        mod = _load_module()
        engine = _engine(tmp_path)

        mod.create_table(engine)
        mod.create_table(engine)  # idempotent

        with engine.connect() as conn:
            ddl = conn.execute(sa_text(
                "SELECT sql FROM sqlite_master WHERE name='city_weather'"
            )).scalar()
            indexes = [r[0] for r in conn.execute(sa_text(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )).fetchall()]

        assert ddl is not None
        assert "UNIQUE" in ddl
        for key in ("city", "date", "data_source"):
            assert key in ddl
        assert "DEFAULT 'open_meteo_era5'" in ddl
        for column in SCHEMA_COLUMNS:
            assert column in ddl, column
        assert "idx_city_weather_city_date" in indexes

    def test_city_measurements_ddl_untouched(self):
        """AC2: weather is a separate table; the ingest never alters or
        recreates `city_measurements`."""
        source = SCRIPT_PATH.read_text()
        assert "ALTER TABLE city_measurements" not in source
        assert "CREATE TABLE IF NOT EXISTS city_measurements" not in source


class TestIngestCity:
    def test_sqlite_date_join(self, tmp_path):
        """Landmine L1 - the highest-value test in this phase.

        `city_measurements.date` is SQLite TEXT holding
        '2015-01-01 00:00:00.000000'. A weather table written with bare ISO date
        strings joins to EVERY row in PostgreSQL and ZERO rows in SQLite, with no
        exception, no null, and no warning - a dialect-dependent wrong answer that
        reaches the hosted demo.
        """
        mod = _load_module()
        engine = _engine(tmp_path)
        _seed_city_measurements(engine)

        # Sanity: prove the landmine's pre-condition holds before testing the
        # defence. One city on three distinct dates -> 0 / 1 / 3.
        # If this fails, fix the fixture - never weaken the assertion, or
        # test_sqlite_date_join passes vacuously and the guard is gone.
        with engine.connect() as conn:
            assert conn.execute(sa_text(
                "SELECT typeof(date) FROM city_measurements LIMIT 1"
            )).scalar() == "text"
            assert conn.execute(sa_text(
                "SELECT COUNT(*) FROM city_measurements WHERE date = '2015-01-01'"
            )).scalar() == 0
            assert conn.execute(sa_text(
                "SELECT COUNT(*) FROM city_measurements WHERE date(date) = '2015-01-01'"
            )).scalar() == 1
            assert conn.execute(sa_text(
                "SELECT COUNT(*) FROM city_measurements "
                "WHERE date(date) IN ('2015-01-01','2015-01-02','2015-01-03')"
            )).scalar() == 3

        mod.create_table(engine)
        mod.ingest_city(engine, "Delhi", _weather_frame())

        with engine.connect() as conn:
            joined = conn.execute(sa_text("""
                SELECT COUNT(*) FROM city_measurements m
                JOIN city_weather w ON m.city = w.city AND m.date = w.date
            """)).scalar()

        assert joined > 0, "L1: weather dates did not join to city_measurements"
        assert joined == 3

    def test_insert_row_count_matches_frame(self, tmp_path):
        """Landmine L3: pandas 3.x changed to_sql/read_sql behaviour. Verify the
        row count, do not assume it."""
        mod = _load_module()
        engine = _engine(tmp_path)
        mod.create_table(engine)

        df = _weather_frame()
        inserted = mod.ingest_city(engine, "Delhi", df)

        assert inserted == len(df)
        with engine.connect() as conn:
            assert conn.execute(
                sa_text("SELECT COUNT(*) FROM city_weather")
            ).scalar() == len(df)

    def test_ingest_is_idempotent(self, tmp_path):
        """Delete-then-insert, not `ingest_hourly.py`'s bare append: a re-run
        must not duplicate rows nor raise IntegrityError."""
        mod = _load_module()
        engine = _engine(tmp_path)
        mod.create_table(engine)

        df = _weather_frame()
        mod.ingest_city(engine, "Delhi", df)
        mod.ingest_city(engine, "Delhi", df)

        with engine.connect() as conn:
            assert conn.execute(
                sa_text("SELECT COUNT(*) FROM city_weather")
            ).scalar() == len(df)

    def test_date_column_roundtrips_as_datetime(self, tmp_path):
        mod = _load_module()
        engine = _engine(tmp_path)
        mod.create_table(engine)

        df = _weather_frame()
        mod.ingest_city(engine, "Delhi", df)

        read_back = pd.read_sql(
            sa_text("SELECT date FROM city_weather ORDER BY date"), engine
        )
        assert list(pd.to_datetime(read_back["date"])) == list(
            pd.to_datetime(df["date"])
        )

        with engine.connect() as conn:
            raw = conn.execute(sa_text(
                "SELECT date FROM city_weather ORDER BY date LIMIT 1"
            )).scalar()
        assert str(raw) != "2015-01-01", (
            "L1: a bare ISO date string joins to zero rows in SQLite"
        )


class TestCacheRevalidation:
    def test_cached_payload_revalidates_timezone(self, tmp_path, monkeypatch):
        """Landmine L4's second door.

        The IST assertion lives inside `fetch_era5_archive`, so it never runs on
        a cache hit. A cache file recorded before `timezone` was pinned - or
        hand-edited - would flow into the table as UTC-aligned days with no error
        anywhere. The cache load path must revalidate.
        """
        mod = _load_module()

        cache_dir = tmp_path / "weather"
        cache_dir.mkdir()
        payload = _payload()
        payload["utc_offset_seconds"] = 0
        payload["timezone"] = "GMT"
        (cache_dir / "Delhi.json").write_text(json.dumps(payload))

        monkeypatch.setattr(mod, "CACHE_DIR", str(cache_dir))

        # A cache miss that silently falls through to the network must fail this
        # test rather than pass it.
        with patch(
            "lib.providers.open_meteo.requests.get",
            side_effect=AssertionError("network hit: the cache was not used"),
        ):
            with pytest.raises(ValueError, match="Refusing to ingest"):
                mod.load_or_fetch_payload("Delhi")

    @pytest.mark.parametrize("city", ["Mumbai", "Kolkata", "Chennai"])
    def test_cached_payload_revalidates_timezone_non_delhi(self, tmp_path, monkeypatch, city):
        """Same stale-cache guard as Delhi, but for other CITY_COORDS cities.

        Guards the fix that derives the expected timezone/offset from
        CITY_COORDS[city] instead of hardcoding Delhi's values.
        """
        mod = _load_module()

        cache_dir = tmp_path / "weather"
        cache_dir.mkdir()
        payload = _payload()
        payload["utc_offset_seconds"] = 0
        payload["timezone"] = "GMT"
        (cache_dir / f"{city}.json").write_text(json.dumps(payload))

        monkeypatch.setattr(mod, "CACHE_DIR", str(cache_dir))

        with patch(
            "lib.providers.open_meteo.requests.get",
            side_effect=AssertionError("network hit: the cache was not used"),
        ):
            with pytest.raises(ValueError, match="Refusing to ingest"):
                mod.load_or_fetch_payload(city)

    def test_expected_timezone_derives_per_city(self, tmp_path, monkeypatch):
        """Pin per-city lookup: patch Mumbai to a non-IST zone and prove the
        guard follows CITY_COORDS[city], not a shared IST constant."""
        from lib.config import CITY_COORDS

        mod = _load_module()

        monkeypatch.setitem(
            CITY_COORDS,
            "Mumbai",
            {"lat": 19.076, "lon": 72.8777,
             "timezone": "Asia/Dubai", "utc_offset_seconds": 14400},
        )

        cache_dir = tmp_path / "weather"
        cache_dir.mkdir()
        monkeypatch.setattr(mod, "CACHE_DIR", str(cache_dir))

        matching = _payload()
        matching["timezone"] = "Asia/Dubai"
        matching["utc_offset_seconds"] = 14400
        (cache_dir / "Mumbai.json").write_text(json.dumps(matching))

        with patch(
            "lib.providers.open_meteo.requests.get",
            side_effect=AssertionError("network hit: the cache was not used"),
        ):
            payload = mod.load_or_fetch_payload("Mumbai")
        assert payload["timezone"] == "Asia/Dubai"

        stale = _payload()  # fixture carries IST values
        (cache_dir / "Mumbai.json").write_text(json.dumps(stale))
        with pytest.raises(ValueError, match="Refusing to ingest"):
            mod.load_or_fetch_payload("Mumbai")
