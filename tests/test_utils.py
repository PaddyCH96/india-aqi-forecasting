"""Tests for lib/utils.py."""

import pytest
from lib.utils import retry, validate_db_url


class TestRetry:
    def test_succeeds_first_attempt(self):
        call_count = 0

        @retry(max_attempts=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return 42

        assert succeed() == 42
        assert call_count == 1

    def test_retries_on_failure(self):
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        assert fail_twice() == "success"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        @retry(max_attempts=2, delay=0.01)
        def always_fail():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            always_fail()

    def test_only_retries_specified_exceptions(self):
        call_count = 0

        @retry(max_attempts=2, delay=0.01, exceptions=(ValueError,))
        def fail_with_type_error():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("retry this")
            raise TypeError("do not retry")

        with pytest.raises(TypeError, match="do not retry"):
            fail_with_type_error()
        assert call_count == 2

    def test_default_exceptions_catches_all(self):
        call_count = 0

        @retry(max_attempts=2, delay=0.01)
        def fail_wildly():
            nonlocal call_count
            call_count += 1
            raise KeyError("any exception")

        with pytest.raises(KeyError):
            fail_wildly()
        assert call_count == 2

    def test_preserves_return_value(self):
        @retry(max_attempts=3, delay=0.01)
        def return_value():
            return {"key": [1, 2, 3]}

        result = return_value()
        assert result == {"key": [1, 2, 3]}


class TestValidateDbUrl:
    def test_valid_postgres_url(self):
        assert validate_db_url("postgresql://user:pass@host:5432/db") is True

    def test_valid_postgresql_url(self):
        assert validate_db_url("postgresql://localhost/mydb") is True

    def test_raises_on_missing_prefix(self):
        with pytest.raises(ValueError, match="must start with 'postgresql://'"):
            validate_db_url("sqlite:///test.db")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="must start with 'postgresql://'"):
            validate_db_url("")

    def test_uses_default_url(self):
        assert validate_db_url() is True
