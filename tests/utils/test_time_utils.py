"""
tests/test_time_utils.py

Comprehensive tests for time utility functions in utils.time module.
Tests validate now_ms(), to_ms(), and to_iso() functions work correctly.
"""

import time
from datetime import datetime, timezone
from unittest.mock import patch

from utils.time import now_ms, to_iso, to_ms


class TestNowMs:
    """Test cases for now_ms() function."""

    def test_now_ms_returns_integer(self):
        """Test that now_ms() returns an integer."""
        result = now_ms()
        assert isinstance(result, int)

    def test_now_ms_reasonable_range(self):
        """Test that now_ms() returns a timestamp in reasonable range."""
        result = now_ms()

        # Should be between 2020 and 2030 in milliseconds
        # 2020-01-01 = 1577836800000
        # 2030-01-01 = 1893456000000
        assert 1577836800000 <= result <= 1893456000000

    def test_now_ms_increases_over_time(self):
        """Test that now_ms() increases over time."""
        time1 = now_ms()
        time.sleep(0.001)  # Small delay
        time2 = now_ms()

        assert time2 >= time1

    @patch("utils.time.datetime")
    def test_now_ms_uses_utc(self, mock_datetime):
        """Test that now_ms() uses UTC time."""
        # Mock datetime.utcnow() to return a specific time
        mock_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.utcnow.return_value = mock_dt

        result = now_ms()
        expected = int(mock_dt.timestamp() * 1000)

        assert result == expected
        mock_datetime.utcnow.assert_called_once()


class TestToMs:
    """Test cases for to_ms() function."""

    def test_to_ms_none(self):
        """Test to_ms() with None input."""
        assert to_ms(None) is None

    def test_to_ms_large_int(self):
        """Test to_ms() with large int (already in milliseconds)."""
        # Value > 1e12 should be treated as milliseconds
        large_ms = 1672574400000  # 2023-01-01 12:00:00 UTC in ms
        result = to_ms(large_ms)
        assert result == large_ms

    def test_to_ms_small_int(self):
        """Test to_ms() with small int (seconds, should convert to ms)."""
        seconds = 1672574400  # 2023-01-01 12:00:00 UTC in seconds
        result = to_ms(seconds)
        expected = seconds * 1000
        assert result == expected

    def test_to_ms_float_seconds(self):
        """Test to_ms() with float seconds."""
        seconds_float = 1672574400.123
        result = to_ms(seconds_float)
        expected = int(seconds_float * 1000)
        assert result == expected

    def test_to_ms_float_milliseconds(self):
        """Test to_ms() with float milliseconds."""
        ms_float = 1672574400000.5  # Already in ms range
        result = to_ms(ms_float)
        assert result == int(ms_float)

    def test_to_ms_datetime(self):
        """Test to_ms() with datetime object."""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = to_ms(dt)
        expected = int(dt.timestamp() * 1000)
        assert result == expected

    def test_to_ms_datetime_naive(self):
        """Test to_ms() with naive datetime (assumes UTC)."""
        dt = datetime(2023, 1, 1, 12, 0, 0)  # No timezone
        result = to_ms(dt)
        expected = int(dt.timestamp() * 1000)
        assert result == expected

    def test_to_ms_iso_string(self):
        """Test to_ms() with ISO format string."""
        iso_str = "2023-01-01T12:00:00Z"
        result = to_ms(iso_str)
        expected = 1672574400000  # Known value for this timestamp
        assert result == expected

    def test_to_ms_iso_string_with_microseconds(self):
        """Test to_ms() with ISO string containing microseconds."""
        iso_str = "2023-01-01T12:00:00.123456Z"
        result = to_ms(iso_str)
        # Should round/truncate microseconds
        assert result is not None
        assert isinstance(result, int)

    def test_to_ms_iso_string_without_z(self):
        """Test to_ms() with ISO string without Z suffix."""
        iso_str = "2023-01-01T12:00:00"
        result = to_ms(iso_str)
        # Without Z, it's parsed as local time (Asia/Bangkok UTC+7)
        # So 12:00:00 local becomes 05:00:00 UTC
        expected = 1672549200000  # 2023-01-01 05:00:00 UTC
        assert result == expected

    def test_to_ms_numeric_string_seconds(self):
        """Test to_ms() with numeric string representing seconds."""
        num_str = "1672574400"
        result = to_ms(num_str)
        expected = 1672574400000  # Converted to ms
        assert result == expected

    def test_to_ms_numeric_string_milliseconds(self):
        """Test to_ms() with numeric string representing milliseconds."""
        num_str = "1672574400000"
        result = to_ms(num_str)
        expected = 1672574400000
        assert result == expected

    def test_to_ms_invalid_string(self):
        """Test to_ms() with invalid string."""
        result = to_ms("invalid_timestamp")
        assert result is None

    def test_to_ms_empty_string(self):
        """Test to_ms() with empty string."""
        result = to_ms("")
        assert result is None

    def test_to_ms_unsupported_type(self):
        """Test to_ms() with unsupported type."""
        result = to_ms([1, 2, 3])  # List
        assert result is None

        result = to_ms({"key": "value"})  # Dict
        assert result is None

    def test_to_ms_boundary_values(self):
        """Test to_ms() with boundary values."""
        # Very small positive number
        result = to_ms(1)
        assert result == 1000

        # Very large number
        large_num = 999999999999999
        result = to_ms(large_num)
        assert result == large_num

        # Exactly at threshold (1e12) - since ts > 1e12 is False, it gets multiplied by 1000
        threshold = int(1e12)
        result = to_ms(threshold)
        assert result == threshold * 1000  # Gets multiplied by 1000

        # Just below threshold
        below_threshold = int(1e12) - 1
        result = to_ms(below_threshold)
        assert result == below_threshold * 1000

    def test_to_ms_zero(self):
        """Test to_ms() with zero."""
        result = to_ms(0)
        assert result == 0

    def test_to_ms_negative(self):
        """Test to_ms() with negative values."""
        result = to_ms(-1)
        assert result == -1000

        # For negative values, the threshold logic doesn't apply the same way
        # Since ts > 1e12 is False for negative values, they get multiplied by 1000
        large_negative = -1672574400000
        result = to_ms(large_negative)
        assert result == large_negative * 1000  # Gets multiplied by 1000


class TestToIso:
    """Test cases for to_iso() function."""

    def test_to_iso_epoch(self):
        """Test to_iso() with epoch timestamp."""
        result = to_iso(0)
        expected = "1970-01-01T00:00:00Z"
        assert result == expected

    def test_to_iso_known_timestamp(self):
        """Test to_iso() with known timestamp."""
        # 2023-01-01 12:00:00 UTC
        timestamp = 1672574400000
        result = to_iso(timestamp)
        expected = "2023-01-01T12:00:00Z"
        assert result == expected

    def test_to_iso_with_microseconds(self):
        """Test to_iso() with timestamp containing microseconds."""
        # 2023-01-01 12:00:00.123 UTC
        timestamp = 1672574400123
        result = to_iso(timestamp)
        expected = "2023-01-01T12:00:00.123000Z"  # Microseconds preserved
        assert result == expected

    def test_to_iso_current_time(self):
        """Test to_iso() with current time."""
        current_ms = now_ms()
        result = to_iso(current_ms)

        # Should be a valid ISO string
        assert isinstance(result, str)
        assert result.endswith("Z")
        assert "T" in result

        # Should be parseable back
        round_trip = to_ms(result)
        assert round_trip is not None
        # Allow small difference due to precision
        assert abs(round_trip - current_ms) < 2000  # 2 seconds tolerance

    def test_to_iso_future_timestamp(self):
        """Test to_iso() with future timestamp."""
        future_ms = now_ms() + 86400000  # +1 day
        result = to_iso(future_ms)

        assert isinstance(result, str)
        assert result.endswith("Z")
        assert "T" in result

    def test_to_iso_past_timestamp(self):
        """Test to_iso() with past timestamp."""
        past_ms = now_ms() - 86400000  # -1 day
        result = to_iso(past_ms)

        assert isinstance(result, str)
        assert result.endswith("Z")
        assert "T" in result

    def test_to_iso_invalid_input(self):
        """Test to_iso() with invalid input."""
        # Should fallback to current time for invalid inputs
        result = to_iso("invalid")
        assert isinstance(result, str)
        assert result.endswith("Z")

        result = to_iso(None)
        assert isinstance(result, str)
        assert result.endswith("Z")

        result = to_iso([1, 2, 3])
        assert isinstance(result, str)
        assert result.endswith("Z")

    def test_to_iso_negative_timestamp(self):
        """Test to_iso() with negative timestamp."""
        result = to_iso(-1000)
        # Should handle negative timestamps gracefully
        assert isinstance(result, str)
        assert result.endswith("Z")

    def test_to_iso_very_large_timestamp(self):
        """Test to_iso() with very large timestamp."""
        large_ts = 999999999999999
        result = to_iso(large_ts)
        # Should handle large timestamps gracefully
        assert isinstance(result, str)
        assert result.endswith("Z")


class TestRoundTripConversions:
    """Test cases for round-trip conversions between formats."""

    def test_now_ms_to_iso_to_ms(self):
        """Test round-trip conversion: now_ms -> to_iso -> to_ms."""
        original = now_ms()
        iso_str = to_iso(original)
        round_trip = to_ms(iso_str)

        assert round_trip is not None
        # Allow small difference due to precision loss in ISO format
        assert abs(round_trip - original) < 2000  # 2 seconds tolerance

    def test_known_timestamp_round_trip(self):
        """Test round-trip with known timestamp."""
        original = 1672574400000  # 2023-01-01 12:00:00 UTC
        iso_str = to_iso(original)
        round_trip = to_ms(iso_str)

        assert round_trip == original

    def test_datetime_round_trip(self):
        """Test round-trip with datetime object."""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ms = to_ms(dt)
        iso_str = to_iso(ms)
        round_trip = to_ms(iso_str)

        assert round_trip is not None
        assert abs(round_trip - ms) < 2000  # Allow small precision difference

    def test_seconds_to_ms_to_iso_to_ms(self):
        """Test round-trip: seconds -> ms -> iso -> ms."""
        original_seconds = 1672574400
        ms = to_ms(original_seconds)
        iso_str = to_iso(ms)
        final_ms = to_ms(iso_str)

        assert final_ms is not None
        # Should be close to original converted to ms
        assert abs(final_ms - (original_seconds * 1000)) < 2000


class TestIntegrationScenarios:
    """Test cases for realistic integration scenarios."""

    def test_timestamp_comparison(self):
        """Test comparing timestamps from different sources."""
        # Simulate getting timestamps from different sources
        time1 = now_ms()
        time.sleep(0.01)  # Small delay
        time2 = now_ms()

        assert time2 > time1
        assert time2 - time1 < 100  # Should be less than 100ms apart

    def test_iso_string_storage_and_retrieval(self):
        """Test storing timestamp as ISO and retrieving it."""
        original = now_ms()

        # Simulate storing as ISO string
        stored_iso = to_iso(original)

        # Simulate retrieving and converting back
        retrieved_ms = to_ms(stored_iso)

        assert retrieved_ms is not None
        assert abs(retrieved_ms - original) < 2000

    def test_mixed_timestamp_formats(self):
        """Test handling mixed timestamp formats in a collection."""
        timestamps = [
            1672574400,  # seconds as int
            1672574400000,  # milliseconds as int
            1672574400.123,  # seconds as float
            "1672574400000",  # milliseconds as string
            "2023-01-01T12:00:00Z",  # ISO string
        ]

        results = [to_ms(ts) for ts in timestamps]

        # All should convert to the same millisecond value
        expected_ms = 1672574400000
        for result in results:
            assert result is not None
            assert abs(result - expected_ms) < 2000  # Allow small precision differences

    def test_error_handling_in_batch_processing(self):
        """Test error handling when processing a batch of timestamps."""
        mixed_inputs = [
            1672574400000,  # Valid milliseconds
            "2023-01-01T12:00:00Z",  # Valid ISO
            "invalid",  # Invalid string
            None,  # None
            [1, 2, 3],  # Invalid type
            1672574400,  # Valid seconds
        ]

        results = [to_ms(ts) for ts in mixed_inputs]

        # Valid inputs should convert successfully
        assert results[0] == 1672574400000  # milliseconds
        assert results[1] == 1672574400000  # ISO string
        assert results[5] == 1672574400000  # seconds

        # Invalid inputs should return None
        assert results[2] is None  # invalid string
        assert results[3] is None  # None
        assert results[4] is None  # list
