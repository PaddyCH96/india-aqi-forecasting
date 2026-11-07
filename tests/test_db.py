"""Mock-based tests for lib/db.py query logic."""

from unittest.mock import MagicMock, patch

import pandas as pd

from lib.db import (
    get_engine,
    load_city_data,
    load_city_pollutants,
    load_city_weather,
    get_cities_with_recent_data,
    get_cities_with_data_summary,
    get_eligible_cities,
    count_recent_rows,
    insert_city_data,
    get_data_freshness,
)


def _empty_aqi_df():
    return pd.DataFrame({
        "ds": pd.Series(dtype=str),
        "y": pd.Series(dtype=float),
    })


class TestGetEngine:
    def test_returns_engine_with_default_url(self):
        """With no AQI_DB_URL set, config falls back to the bundled SQLite demo
        database when present, and to local PostgreSQL otherwise."""
        engine = get_engine()
        url = str(engine.url)
        assert url.startswith("sqlite:///") or url == (
            "postgresql://postgres@localhost:5432/india_air_quality"
        )

    def test_uses_custom_url(self):
        engine = get_engine("postgresql://user:pass@host:5432/testdb")
        url = str(engine.url)
        assert "user" in url
        assert "host" in url
        assert "testdb" in url


class TestLoadCityData:
    def test_returns_dataframe(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "ds": ["2023-01-01", "2023-01-02"],
                "y": [100.0, 110.0],
            })
            result = load_city_data(mock_engine, "Hyderabad")
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "ds" in result.columns
            assert "y" in result.columns

    def test_passes_params_correctly(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = _empty_aqi_df()
            load_city_data(mock_engine, "Mumbai")
            kwargs = mock_read_sql.call_args[1]
            assert kwargs["params"] == {"city": "Mumbai"}

    def test_empty_result(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = _empty_aqi_df()
            result = load_city_data(mock_engine, "UnknownCity")
            assert result.empty

    def test_cast_ds_to_datetime(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "ds": ["2023-01-01"],
                "y": [100.0],
            })
            result = load_city_data(mock_engine, "Delhi")
            assert pd.api.types.is_datetime64_any_dtype(result["ds"])

    def test_excludes_synthetic_by_default(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = _empty_aqi_df()
            load_city_data(mock_engine, "Delhi")
            sql_text = str(mock_read_sql.call_args[0][0])
            assert "AND NOT is_synthetic" in sql_text

    def test_includes_synthetic_when_requested(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = _empty_aqi_df()
            load_city_data(mock_engine, "Delhi", use_synthetic=True)
            sql_text = str(mock_read_sql.call_args[0][0])
            assert "AND NOT is_synthetic" not in sql_text


class TestLoadCityPollutants:
    def test_returns_all_pollutant_columns(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "date": ["2023-01-01"],
                "pm2_5": [80.0],
                "pm10": [120.0],
                "no2": [40.0],
                "aqi": [100.0],
                "aqi_bucket": ["Moderate"],
                "is_synthetic": [False],
                "data_source": ["CPCB"],
            })
            result = load_city_pollutants(mock_engine, "Delhi")
            assert "pm2_5" in result.columns
            assert "aqi_bucket" in result.columns
            assert len(result) == 1

    def test_excludes_synthetic_by_default(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({"date": []})
            load_city_pollutants(mock_engine, "Delhi")
            sql_text = str(mock_read_sql.call_args[0][0])
            assert "AND NOT is_synthetic" in sql_text


def _weather_df():
    return pd.DataFrame({
        "city": ["Delhi", "Delhi"],
        "date": ["2015-01-01", "2015-01-02"],
        "temp_mean_c": [14.2, 15.1],
        "blh_max_m": [1200.0, 980.0],
        "data_source": ["open_meteo_era5", "open_meteo_era5"],
    })


class TestLoadCityWeather:
    def test_passes_city_as_bound_parameter(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql", return_value=_weather_df()) as mock_read:
            load_city_weather(mock_engine, "Delhi")

        kwargs = mock_read.call_args[1]
        assert kwargs["params"] == {"city": "Delhi"}

    def test_does_not_interpolate_city_into_sql(self):
        """A city name reaching the query text verbatim would be an injection point."""
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql", return_value=_weather_df()) as mock_read:
            load_city_weather(mock_engine, "Delhi'; DROP TABLE city_weather; --")

        sql = str(mock_read.call_args[0][0])
        assert "DROP TABLE" not in sql
        assert ":city" in sql

    def test_returns_datetime_date_column(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql", return_value=_weather_df()):
            result = load_city_weather(mock_engine, "Delhi")

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_queries_city_weather_filtered_on_data_source(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql", return_value=_weather_df()) as mock_read:
            load_city_weather(mock_engine, "Delhi")

        sql = str(mock_read.call_args[0][0])
        assert "FROM city_weather" in sql
        assert "data_source = 'open_meteo_era5'" in sql
        assert "SELECT *" not in sql
        assert "ORDER BY date" in sql

    def test_selects_the_boundary_layer_height_measures(self):
        """Phase 3 chooses among all three BLH aggregates, so all three must be read."""
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql", return_value=_weather_df()) as mock_read:
            load_city_weather(mock_engine, "Delhi")

        sql = str(mock_read.call_args[0][0])
        for col in ("blh_mean_m", "blh_max_m", "blh_min_m"):
            assert col in sql


class TestGetCitiesWithRecentData:
    def test_returns_list(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "city": ["Delhi", "Hyderabad", "Mumbai"],
            })
            result = get_cities_with_recent_data(mock_engine)
            assert result == ["Delhi", "Hyderabad", "Mumbai"]

    def test_empty_result(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({"city": []})
            result = get_cities_with_recent_data(mock_engine)
            assert result == []


class TestGetCitiesWithDataSummary:
    def test_returns_summary_dataframe(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "city": ["Delhi", "Mumbai"],
                "days": [2000, 1800],
                "start_date": ["2015-01-01", "2015-06-01"],
                "end_date": ["2020-07-01", "2020-07-01"],
            })
            result = get_cities_with_data_summary(mock_engine)
            assert len(result) == 2
            assert "days" in result.columns

    def test_passes_min_days(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            get_cities_with_data_summary(mock_engine, min_days=500)
            kwargs = mock_read_sql.call_args[1]
            assert kwargs["params"] == {"min_days": 500}

    def test_default_min_days(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            get_cities_with_data_summary(mock_engine)
            kwargs = mock_read_sql.call_args[1]
            assert kwargs["params"] == {"min_days": 1000}


class TestGetEligibleCities:
    def test_returns_eligible(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "city": ["Delhi", "Hyderabad"],
                "total_days": [2000, 1800],
                "end_date": ["2020-07-01", "2020-07-01"],
            })
            result = get_eligible_cities(mock_engine)
            assert len(result) == 2
            assert "total_days" in result.columns

    def test_default_min_days(self):
        mock_engine = MagicMock()
        with patch("lib.db.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            get_eligible_cities(mock_engine)
            kwargs = mock_read_sql.call_args[1]
            assert kwargs["params"] == {"min_days": 1000}


class TestCountRecentRows:
    def test_returns_count(self):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = count_recent_rows(mock_engine)
        assert result == 42

    def test_returns_zero_on_none(self):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = count_recent_rows(mock_engine)
        assert result == 0

    def test_sql_contains_correct_date(self):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        count_recent_rows(mock_engine)
        executed_text = mock_conn.execute.call_args[0][0]
        assert "2020-07-01" in str(executed_text)

    def test_excludes_synthetic_by_default(self):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        count_recent_rows(mock_engine)
        executed_text = mock_conn.execute.call_args[0][0]
        assert "AND NOT is_synthetic" in str(executed_text)


class TestInsertCityData:
    def test_renames_pm25_column(self):
        mock_engine = MagicMock()
        df = pd.DataFrame({
            "date": ["2023-01-01"],
            "city": ["TestCity"],
            "pm25": [50.0],
        })

        captured = []
        def capture_self(self, *args, **kwargs):
            captured.append(self)

        with patch("lib.db.pd.DataFrame.to_sql", autospec=True, side_effect=capture_self):
            insert_city_data(mock_engine, df)

            assert len(captured) == 1
            inserted_df = captured[0]
            assert "pm2_5" in inserted_df.columns
            assert "pm25" not in inserted_df.columns

    def test_adds_provenance_columns(self):
        mock_engine = MagicMock()
        df = pd.DataFrame({
            "date": ["2023-01-01"],
            "city": ["TestCity"],
            "pm25": [50.0],
        })

        captured = []
        def capture_self(self, *args, **kwargs):
            captured.append(self)

        with patch("lib.db.pd.DataFrame.to_sql", autospec=True, side_effect=capture_self):
            insert_city_data(mock_engine, df, data_source="OpenAQ", is_synthetic=False)

            assert len(captured) == 1
            inserted_df = captured[0]
            assert inserted_df["data_source"].iloc[0] == "OpenAQ"
            assert not inserted_df["is_synthetic"].iloc[0]

    def test_calls_to_sql_with_append(self):
        mock_engine = MagicMock()
        df = pd.DataFrame({
            "date": ["2023-01-01"],
            "city": ["TestCity"],
            "pm25": [50.0],
        })

        with patch("lib.db.pd.DataFrame.to_sql", autospec=True) as mock_to_sql:
            insert_city_data(mock_engine, df)

            kwargs = mock_to_sql.call_args[1]
            assert kwargs["if_exists"] == "append"
            assert kwargs["index"] is False

    def test_keeps_pm2_5_if_already_present(self):
        mock_engine = MagicMock()
        df = pd.DataFrame({
            "date": ["2023-01-01"],
            "city": ["TestCity"],
            "pm2_5": [50.0],
            "aqi": [100.0],
        })

        captured = []
        def capture_self(self, *args, **kwargs):
            captured.append(self)

        with patch("lib.db.pd.DataFrame.to_sql", autospec=True, side_effect=capture_self):
            insert_city_data(mock_engine, df)

            assert len(captured) == 1
            inserted_df = captured[0]
            assert "pm2_5" in inserted_df.columns
            assert "pm25" not in inserted_df.columns


    def test_default_data_source_does_not_name_the_retired_openaq_source(self):
        """The OpenAQ fetch script is deleted this phase (its endpoint returns HTTP
        410 Gone) and never wrote a row under this default. A default that names it
        would be a lie the schema tells about its own provenance."""
        import inspect

        default = inspect.signature(insert_city_data).parameters["data_source"].default
        assert default != "OpenAQ"
        assert default == "unknown"

    def test_default_data_source_is_written_through_to_the_frame(self):
        mock_engine = MagicMock()
        df = pd.DataFrame({
            "date": ["2023-01-01"],
            "city": ["TestCity"],
            "pm25": [50.0],
        })

        captured = []
        def capture_self(self, *args, **kwargs):
            captured.append(self)

        with patch("lib.db.pd.DataFrame.to_sql", autospec=True, side_effect=capture_self):
            insert_city_data(mock_engine, df)

            assert captured[0]["data_source"].iloc[0] == "unknown"


class TestGetDataFreshness:
    def test_returns_summary_dict(self):
        mock_engine = MagicMock()
        mock_conn = MagicMock()

        scalar_results = iter([39401, 29531, 9870, 26, "2025-01-01", "2020-07-01"])

        def mock_scalar():
            return next(scalar_results)

        mock_cursor = MagicMock()
        mock_cursor.scalar = mock_scalar
        mock_cursor.fetchall.return_value = [("CPCB", 29531), ("synthetic", 9870)]

        def mock_execute(sql, *a, **kw):
            return mock_cursor

        mock_conn.execute = mock_execute
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = get_data_freshness(mock_engine)
        assert result["total_rows"] == 39401
        assert result["real_rows"] == 29531
        assert result["synthetic_rows"] == 9870
        assert result["cities"] == 26
