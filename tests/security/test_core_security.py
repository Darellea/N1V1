"""
Tests for core module security fixes.

This module tests the security improvements made to the core trading engine,
including input validation, secure defaults, and safe data processing.
"""


import pandas as pd
import pytest

from core.data_expansion_manager import DataExpansionManager
from core.data_processor import DataProcessor
from core.metrics_endpoint import MetricsEndpoint
from core.signal_processor import SignalProcessor


class TestDataProcessorSecurity:
    """Test security fixes in DataProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()

    def test_calculate_rsi_batch_input_validation(self):
        """Test input validation for calculate_rsi_batch."""
        # Test invalid data_dict type
        with pytest.raises(ValueError, match="data_dict must be a dictionary"):
            self.processor.calculate_rsi_batch("not_a_dict")

        # Test invalid period type
        with pytest.raises(ValueError, match="period must be a positive integer"):
            self.processor.calculate_rsi_batch({}, period="14")

        # Test negative period
        with pytest.raises(ValueError, match="period must be a positive integer"):
            self.processor.calculate_rsi_batch({}, period=-1)

        # Test period too large
        with pytest.raises(ValueError, match="period cannot exceed 1000"):
            self.processor.calculate_rsi_batch({}, period=2000)

    def test_calculate_rsi_batch_data_validation(self):
        """Test data validation in calculate_rsi_batch."""
        # Create test data
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                ]
            }
        )

        # Test with invalid symbol
        result = self.processor.calculate_rsi_batch({123: data}, period=14)
        assert 123 not in result  # Invalid symbol should be skipped

        # Test with non-DataFrame data
        result = self.processor.calculate_rsi_batch(
            {"TEST": "not_dataframe"}, period=14
        )
        assert "TEST" not in result  # Invalid data type should be skipped

        # Test with missing close column
        bad_data = pd.DataFrame({"open": [100, 101, 102]})
        result = self.processor.calculate_rsi_batch({"TEST": bad_data}, period=14)
        assert "TEST" not in result  # Missing column should be skipped

        # Test with insufficient data
        small_data = pd.DataFrame({"close": [100, 101]})
        result = self.processor.calculate_rsi_batch({"TEST": small_data}, period=14)
        assert "TEST" in result  # Should return original data for insufficient data


class TestDataExpansionManagerSecurity:
    """Test security fixes in DataExpansionManager."""

    @pytest.mark.asyncio
    async def test_collect_multi_pair_data_input_validation(self):
        """Test input validation for collect_multi_pair_data."""
        config = {"data_sources": [], "target_pairs": ["EUR/USD"], "timeframes": ["1h"]}
        manager = DataExpansionManager(config)

        # Test invalid target_samples type
        with pytest.raises(
            ValueError, match="target_samples must be a positive integer"
        ):
            await manager.collect_multi_pair_data("1000")

        # Test negative target_samples
        with pytest.raises(
            ValueError, match="target_samples must be a positive integer"
        ):
            await manager.collect_multi_pair_data(-100)

        # Test target_samples too large
        with pytest.raises(ValueError, match="target_samples cannot exceed 100,000"):
            await manager.collect_multi_pair_data(200000)


class TestSignalProcessorSecurity:
    """Test security fixes in SignalProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {}
        self.processor = SignalProcessor(self.config)

    @pytest.mark.asyncio
    async def test_generate_signals_input_validation(self):
        """Test input validation for generate_signals."""
        # Test invalid market_data type
        with pytest.raises(ValueError, match="market_data must be a dictionary"):
            await self.processor.generate_signals("not_a_dict")

        # Test empty market_data
        result = await self.processor.generate_signals({})
        assert result == []


class TestMetricsEndpointSecurity:
    """Test security fixes in MetricsEndpoint."""

    def test_secure_defaults(self):
        """Test that secure defaults are applied."""
        # Test with no config - should use secure defaults
        endpoint = MetricsEndpoint({})

        assert endpoint.enable_auth == True  # Secure default
        assert endpoint.enable_tls == True  # Secure default

        # Test with explicit config
        config = {"enable_auth": False, "enable_tls": False, "auth_token": "test_token"}
        endpoint = MetricsEndpoint(config)

        assert endpoint.enable_auth == False  # Explicit config overrides
        assert endpoint.enable_tls == False  # Explicit config overrides

    def test_auth_token_validation(self):
        """Test auth token validation."""
        # Test with auth enabled but no token
        config = {"enable_auth": True}
        endpoint = MetricsEndpoint(config)

        assert endpoint.enable_auth == True
        # Should have logged a warning about missing token

    def test_tls_validation(self):
        """Test TLS configuration validation."""
        # Test with TLS enabled but missing cert files
        config = {"enable_tls": True}
        endpoint = MetricsEndpoint(config)

        assert endpoint.enable_tls == False  # Should fallback to HTTP
        # Should have logged a warning about missing cert files


class TestInputValidationEdgeCases:
    """Test edge cases for input validation."""

    def test_data_processor_extreme_values(self):
        """Test DataProcessor with extreme values."""
        processor = DataProcessor()

        # Test with very large period (should be rejected)
        data = pd.DataFrame({"close": list(range(2000))})
        with pytest.raises(ValueError):
            processor.calculate_rsi_batch({"TEST": data}, period=1500)

    @pytest.mark.asyncio
    async def test_data_expansion_manager_extreme_values(self):
        """Test DataExpansionManager with extreme values."""
        config = {"data_sources": [], "target_pairs": ["EUR/USD"], "timeframes": ["1h"]}
        manager = DataExpansionManager(config)

        # Test with very large target_samples (should be rejected)
        with pytest.raises(ValueError):
            await manager.collect_multi_pair_data(500000)

    @pytest.mark.asyncio
    async def test_signal_processor_malformed_data(self):
        """Test SignalProcessor with malformed data."""
        processor = SignalProcessor({})

        # Test with None values in market_data
        malformed_data = {"TEST": None}
        result = await processor.generate_signals(malformed_data)
        # Should handle gracefully without crashing


if __name__ == "__main__":
    pytest.main([__file__])
