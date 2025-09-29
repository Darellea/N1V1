"""
tests/test_time.py

Unit tests for timestamp utilities in utils/time.py
"""

import time
from datetime import datetime, timezone
from unittest.mock import patch

from utils.time import now_ms, to_iso, to_ms


class TestNowMs:
    """Test now_ms() function"""

    def test_now_ms_returns_int(self):
        """Test that now_ms returns an integer"""
        result = now_ms()
        assert isinstance(result, int)
        assert result > 0

    def test_now_ms_increasing(self):
        """Test that subsequent calls return increasing values"""
        first = now_ms()
        time.sleep(0.001)  # Small delay to ensure different timestamps
        second = now_ms()
        assert second > first

    @patch("utils.time.datetime")
    def test_now_ms_uses_utc(self, mock_datetime):
        """Test that now_ms uses UTC time"""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.utcnow.return_value = mock_now
        mock_datetime.utcfromtimestamp.return_value = mock_now

        result = now_ms()
        expected = int(mock_now.timestamp() * 1000)
        assert result == expected


class TestToMs:
    """Test to_ms() function"""

    def test_to_ms_none(self):
        """Test None input returns None"""
        assert to_ms(None) is None

    def test_to_ms_int_seconds(self):
        """Test int seconds conversion to ms"""
        assert to_ms(1234567890) == 1234567890000  # Seconds to ms

    def test_to_ms_int_milliseconds(self):
        """Test int ms remains as ms"""
        assert to_ms(1234567890123) == 1234567890123  # Already ms

    def test_to_ms_float_seconds(self):
        """Test float seconds conversion to ms"""
        assert to_ms(1234567890.123) == 1234567890123  # Seconds to ms

    def test_to_ms_float_milliseconds(self):
        """Test float ms remains as ms"""
        assert to_ms(1234567890123.0) == 1234567890123  # Already ms

    def test_to_ms_datetime(self):
        """Test datetime conversion to ms"""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        expected = int(dt.timestamp() * 1000)
        assert to_ms(dt) == expected

    def test_to_ms_datetime_naive(self):
        """Test naive datetime conversion (assumes local time)"""
        dt = datetime(2023, 1, 1, 12, 0, 0)  # No timezone
        # This test is more about ensuring no exception than exact value
        result = to_ms(dt)
        assert isinstance(result, int)
        assert result > 0

    def test_to_ms_iso_string(self):
        """Test ISO string conversion to ms"""
        iso_str = "2023-01-01T12:00:00Z"
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        expected = int(dt.timestamp() * 1000)
        assert to_ms(iso_str) == expected

    def test_to_ms_iso_string_with_timezone(self):
        """Test ISO string with timezone conversion"""
        iso_str = "2023-01-01T12:00:00+00:00"
        dt = datetime.fromisoformat(iso_str)
        expected = int(dt.timestamp() * 1000)
        assert to_ms(iso_str) == expected

    def test_to_ms_numeric_string_seconds(self):
        """Test numeric string seconds conversion"""
        assert to_ms("1234567890") == 1234567890000

    def test_to_ms_numeric_string_milliseconds(self):
        """Test numeric string ms conversion"""
        assert to_ms("1234567890123") == 1234567890123

    def test_to_ms_numeric_string_float(self):
        """Test float string conversion"""
        assert to_ms("1234567890.123") == 1234567890123

    def test_to_ms_invalid_string(self):
        """Test invalid string returns None"""
        assert to_ms("invalid") is None

    def test_to_ms_unsupported_type(self):
        """Test unsupported type returns None"""
        assert to_ms(["list"]) is None
        assert to_ms({"dict": "value"}) is None


class TestToIso:
    """Test to_iso() function"""

    def test_to_iso_valid_ms(self):
        """Test valid ms conversion to ISO"""
        # Test with a known timestamp
        test_ms = 1672574400000  # 2023-01-01T12:00:00Z
        result = to_iso(test_ms)
        assert result == "2023-01-01T12:00:00Z"

    def test_to_iso_current_time_fallback(self):
        """Test that invalid ms falls back to current time"""
        invalid_ms = -1  # Invalid timestamp
        result = to_iso(invalid_ms)
        # Should return current time in ISO format
        assert result.endswith("Z")
        assert "T" in result

    def test_to_iso_zero(self):
        """Test zero timestamp (epoch)"""
        result = to_iso(0)
        assert result == "1970-01-01T00:00:00Z"

    def test_to_iso_round_trip(self):
        """Test round-trip conversion ms -> ISO -> ms"""
        original_ms = now_ms()
        iso_str = to_iso(original_ms)
        round_trip_ms = to_ms(iso_str)
        # Allow for small differences due to datetime precision
        assert abs(round_trip_ms - original_ms) < 1000


class TestIntegration:
    """Integration tests for the time utilities"""

    def test_now_ms_and_to_iso_integration(self):
        """Test that now_ms and to_iso work together"""
        current_ms = now_ms()
        iso_str = to_iso(current_ms)

        # Verify ISO string format
        assert iso_str.endswith("Z")
        assert "T" in iso_str

        # Round-trip should be close to original
        round_trip_ms = to_ms(iso_str)
        assert abs(round_trip_ms - current_ms) < 1000

    def test_various_formats_to_ms(self):
        """Test conversion of various timestamp formats to ms"""
        test_cases = [
            # (input, expected_type, description)
            (1672574400, int, "seconds as int"),
            (1672574400000, int, "milliseconds as int"),
            (1672574400.0, float, "seconds as float"),
            (1672574400000.0, float, "milliseconds as float"),
            ("1672574400", str, "seconds as string"),
            ("1672574400000", str, "milliseconds as string"),
            ("2023-01-01T12:00:00Z", str, "ISO string"),
        ]

        for input_val, expected_type, description in test_cases:
            result = to_ms(input_val)
            assert isinstance(result, int), f"Failed for {description}"
            assert result > 0, f"Failed for {description}"

    def test_edge_cases(self):
        """Test edge cases for to_ms"""
        # Very large timestamp (year ~5138)
        large_ts = 99999999999999
        result = to_ms(large_ts)
        assert result == large_ts

        # Very small timestamp (should be treated as seconds)
        small_ts = 1000  # Definitely seconds
        result = to_ms(small_ts)
        assert result == small_ts * 1000

        # Boundary around the 1e12 threshold
        boundary = int(1e12)
        result_just_below = to_ms(boundary - 1)
        result_just_above = to_ms(boundary + 1)
        # Just below should be treated as seconds, just above as ms
        assert result_just_below == (boundary - 1) * 1000
        assert result_just_above == boundary + 1
