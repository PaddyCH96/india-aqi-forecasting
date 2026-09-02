"""Tests for lib/providers/open_meteo.py against a recorded ERA5 response.

Every test here is network-free: `requests.get` is patched and the payload comes
from `tests/fixtures/open_meteo_era5_delhi_3day.json`, a verbatim recording of the
keyless Open-Meteo archive API.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.providers.open_meteo import (
    DAILY_VARS,
    HOURLY_VARS,
    fetch_era5_archive,
    parse_era5_daily,
)

FIXTURE = Path(__file__).parent / "fixtures" / "open_meteo_era5_delhi_3day.json"

SCHEMA_COLUMNS = {
    "city", "date", "temp_mean_c", "temp_max_c", "temp_min_c",
    "precipitation_mm", "wind_speed_mean_kmh", "wind_speed_max_kmh",
    "wind_direction_deg", "relative_humidity_pct", "surface_pressure_hpa",
    "blh_mean_m", "blh_max_m", "blh_min_m",
    "grid_latitude", "grid_longitude", "grid_elevation_m",
    "is_synthetic", "data_source", "ingested_at",
}


def _payload():
    """A fresh copy each call so mutating tests cannot leak state."""
    return json.loads(FIXTURE.read_text())


def _mock_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class TestParseEra5Daily:
    def test_parse_returns_all_required_columns(self):
        df = parse_era5_daily(_payload(), "Delhi")

        assert SCHEMA_COLUMNS <= set(df.columns), SCHEMA_COLUMNS - set(df.columns)
        assert len(df) == 3
        assert df["city"].unique().tolist() == ["Delhi"]
        assert df["data_source"].unique().tolist() == ["open_meteo_era5"]
        assert not df["is_synthetic"].any()

        # The COLUMN_MAP rename actually happened - no raw API names survive.
        for raw in ("time", "boundary_layer_height", "temperature_2m_mean"):
            assert raw not in df.columns

    def test_blh_aggregation(self):
        payload = _payload()
        df = parse_era5_daily(payload, "Delhi")

        for _, row in df.iterrows():
            assert row["blh_max_m"] >= row["blh_mean_m"] >= row["blh_min_m"]

        first_day = payload["hourly"]["boundary_layer_height"][:24]
        assert df.loc[0, "blh_max_m"] == pytest.approx(max(first_day))
        assert df.loc[0, "blh_min_m"] == pytest.approx(min(first_day))
        assert df.loc[0, "blh_mean_m"] == pytest.approx(sum(first_day) / 24)

        # A naive daily-mean-only implementation would make all three identical.
        assert not (
            df["blh_max_m"].equals(df["blh_mean_m"])
            and df["blh_mean_m"].equals(df["blh_min_m"])
        )

    def test_24_hours_per_local_day(self):
        payload = _payload()

        counts = {}
        for stamp in payload["hourly"]["time"]:
            counts[stamp[:10]] = counts.get(stamp[:10], 0) + 1
        assert set(counts.values()) == {24}, counts

        seen_first = set()
        for stamp in payload["hourly"]["time"]:
            if stamp[:10] not in seen_first:
                assert stamp.endswith("T00:00"), stamp
                seen_first.add(stamp[:10])

        # Every parsed date was backed by a full 24 source hours - no partial-day
        # aggregate is silently produced.
        df = parse_era5_daily(payload, "Delhi")
        for value in df["date"]:
            assert counts[str(value)[:10]] == 24


class TestFetchEra5Archive:
    def test_rejects_non_ist_payload(self):
        """Landmine L4: an omitted timezone parameter silently yields UTC days,
        shifting every daily aggregate into the 05:30 -> 05:30 IST window with no
        error anywhere. The provider must refuse rather than ingest."""
        payload = _payload()
        payload["utc_offset_seconds"] = 0
        payload["timezone"] = "GMT"

        target = _mock_response(payload)
        with patch("lib.providers.open_meteo.requests.get", return_value=target):
            with pytest.raises(ValueError, match="Refusing to ingest"):
                fetch_era5_archive(28.6139, 77.209, "2015-01-01", "2015-01-03")

    def test_fetch_sends_ist_and_era5_params(self):
        ok = _mock_response(_payload())
        with patch("lib.providers.open_meteo.requests.get", return_value=ok) as mock_get:
            fetch_era5_archive(28.6139, 77.209, "2015-01-01", "2015-01-03")

        params = mock_get.call_args[1]["params"]
        assert params["timezone"] == "Asia/Kolkata"
        assert params["models"] == "era5"

        daily = params["daily"]
        daily_text = daily if isinstance(daily, str) else ",".join(daily)
        for name in DAILY_VARS:
            assert name in daily_text, name
        # Exact snake_case spelling, not the legacy windspeed_ form.
        assert "wind_speed_10m_mean" in daily_text
        assert "windspeed_10m_mean" not in daily_text
        # The API rejects a daily boundary layer variable - it is hourly-only.
        assert "boundary_layer_height" not in daily_text

        hourly = params["hourly"]
        hourly_text = hourly if isinstance(hourly, str) else ",".join(hourly)
        assert "boundary_layer_height" in hourly_text
        for name in HOURLY_VARS:
            assert name in hourly_text, name
