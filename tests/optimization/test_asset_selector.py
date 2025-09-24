"""
Test suite for Asset Selector module.

Tests asset selection, weighting, correlation filtering, and market data fetching.
"""

import pytest
import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import Dict, Any, List

from optimization.asset_selector import AssetSelector, ValidationAsset


class TestValidationAsset:
    """Test ValidationAsset dataclass and methods."""

    def test_validation_asset_creation(self):
        """Test basic ValidationAsset creation."""
        asset = ValidationAsset(
            symbol="BTC/USDT",
            name="Bitcoin",
            weight=1.0,
            required_history=1000,
            timeframe="1h"
        )

        assert asset.symbol == "BTC/USDT"
        assert asset.name == "Bitcoin"
        assert asset.weight == 1.0
        assert asset.required_history == 1000
        assert asset.timeframe == "1h"

    def test_validation_asset_defaults(self):
        """Test ValidationAsset with default values."""
        asset = ValidationAsset(symbol="ETH/USDT", name="Ethereum")

        assert asset.weight == 1.0
        assert asset.required_history == 1000
        assert asset.timeframe == "1h"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        asset = ValidationAsset(
            symbol="ADA/USDT",
            name="Cardano",
            weight=0.8,
            required_history=500,
            timeframe="4h"
        )

        expected = {
            'symbol': 'ADA/USDT',
            'name': 'Cardano',
            'weight': 0.8,
            'required_history': 500,
            'timeframe': '4h'
        }

        assert asset.to_dict() == expected

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'symbol': 'SOL/USDT',
            'name': 'Solana',
            'weight': 0.6,
            'required_history': 2000,
            'timeframe': '1d'
        }

        asset = ValidationAsset.from_dict(data)

        assert asset.symbol == 'SOL/USDT'
        assert asset.name == 'Solana'
        assert asset.weight == 0.6
        assert asset.required_history == 2000
        assert asset.timeframe == '1d'

    def test_from_dict_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {'symbol': 'DOT/USDT', 'name': 'Polkadot'}

        asset = ValidationAsset.from_dict(data)

        assert asset.symbol == 'DOT/USDT'
        assert asset.name == 'Polkadot'
        assert asset.weight == 1.0
        assert asset.required_history == 1000
        assert asset.timeframe == '1h'

    def test_str_representation(self):
        """Test string representation."""
        asset = ValidationAsset(
            symbol="LINK/USDT",
            name="Chainlink",
            weight=0.9
        )

        expected = "LINK/USDT (Chainlink) - weight: 0.9"
        assert str(asset) == expected


class TestAssetSelector:
    """Test AssetSelector class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'max_assets': 3,
            'asset_weights': 'equal',
            'correlation_filter': False,
            'max_correlation': 0.7,
            'validation_assets': [
                {
                    'symbol': 'ETH/USDT',
                    'name': 'Ethereum',
                    'weight': 1.0,
                    'required_history': 1000,
                    'timeframe': '1h'
                },
                {
                    'symbol': 'ADA/USDT',
                    'name': 'Cardano',
                    'weight': 0.8,
                    'required_history': 1000,
                    'timeframe': '1h'
                },
                {
                    'symbol': 'SOL/USDT',
                    'name': 'Solana',
                    'weight': 0.6,
                    'required_history': 1000,
                    'timeframe': '1h'
                }
            ],
            'market_cap_weights': {},
            'coinmarketcap_api_key': None
        }

    def test_initialization_with_config(self):
        """Test AssetSelector initialization with custom config."""
        selector = AssetSelector(self.config)

        assert selector.max_assets == 3
        assert selector.asset_weights == 'equal'
        assert selector.correlation_filter is False
        assert selector.max_correlation == 0.7
        assert len(selector.available_assets) == 3

    def test_initialization_default_config(self):
        """Test AssetSelector initialization with default config."""
        with patch('optimization.asset_selector.get_cross_asset_validation_config') as mock_config:
            mock_config.return_value.asset_selector.max_assets = 5
            mock_config.return_value.asset_selector.asset_weights = 'market_cap'
            mock_config.return_value.asset_selector.correlation_filter = True
            mock_config.return_value.asset_selector.max_correlation = 0.8
            mock_config.return_value.asset_selector.validation_assets = []
            mock_config.return_value.asset_selector.market_cap_weights = {}
            mock_config.return_value.asset_selector.coinmarketcap_api_key = 'test_key'

            selector = AssetSelector()

            assert selector.max_assets == 5
            assert selector.asset_weights == 'market_cap'
            assert selector.correlation_filter is True
            assert selector.max_correlation == 0.8

    def test_load_validation_assets_from_config(self):
        """Test loading validation assets from configuration."""
        selector = AssetSelector(self.config)

        assets = selector.available_assets
        assert len(assets) == 3

        # Check first asset
        assert assets[0].symbol == 'ETH/USDT'
        assert assets[0].name == 'Ethereum'
        assert assets[0].weight == 1.0

    def test_load_validation_assets_default(self):
        """Test loading default validation assets when none provided."""
        config = self.config.copy()
        config['validation_assets'] = []

        selector = AssetSelector(config)

        assets = selector.available_assets
        assert len(assets) == 3  # Default assets

        # Check default assets
        assert assets[0].symbol == 'ETH/USDT'
        assert assets[1].symbol == 'ADA/USDT'
        assert assets[2].symbol == 'SOL/USDT'

    def test_select_validation_assets_basic(self):
        """Test basic asset selection without correlation filtering."""
        selector = AssetSelector(self.config)

        selected = selector.select_validation_assets('BTC/USDT')

        assert len(selected) == 3
        assert all(asset.symbol != 'BTC/USDT' for asset in selected)
        assert selected[0].symbol == 'ETH/USDT'

    def test_select_validation_assets_limit(self):
        """Test asset selection with max_assets limit."""
        config = self.config.copy()
        config['max_assets'] = 2

        selector = AssetSelector(config)
        selected = selector.select_validation_assets('BTC/USDT')

        assert len(selected) == 2

    def test_select_validation_assets_no_candidates(self):
        """Test asset selection when no candidates available."""
        config = self.config.copy()
        config['validation_assets'] = [
            {'symbol': 'BTC/USDT', 'name': 'Bitcoin'}
        ]

        selector = AssetSelector(config)
        selected = selector.select_validation_assets('BTC/USDT')

        assert len(selected) == 0

    @pytest.mark.asyncio
    async def test_select_validation_assets_with_correlation_filter(self):
        """Test asset selection with correlation filtering enabled."""
        config = self.config.copy()
        config['correlation_filter'] = True

        selector = AssetSelector(config)

        # Mock data fetcher
        mock_fetcher = AsyncMock()
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.__getitem__.return_value.pct_change.return_value.dropna.return_value = MagicMock()
        mock_fetcher.get_historical_data.return_value = mock_data

        # Mock the entire correlation filtering to avoid async issues
        with patch.object(selector, '_filter_correlated_assets') as mock_filter:
            mock_filter.return_value = selector.available_assets[:2]  # Return first 2 assets

            selected = selector.select_validation_assets('BTC/USDT', mock_fetcher)

            assert len(selected) == 2
            mock_filter.assert_called_once()

    def test_apply_weighting_equal(self):
        """Test equal weighting scheme."""
        selector = AssetSelector(self.config)

        assets = [
            ValidationAsset('ETH/USDT', 'Ethereum'),
            ValidationAsset('ADA/USDT', 'Cardano'),
            ValidationAsset('SOL/USDT', 'Solana')
        ]

        weighted = selector._apply_weighting(assets)

        assert len(weighted) == 3
        for asset in weighted:
            assert asset.weight == pytest.approx(1.0 / 3.0)

    def test_apply_weighting_market_cap(self):
        """Test market cap weighting scheme."""
        config = self.config.copy()
        config['asset_weights'] = 'market_cap'

        selector = AssetSelector(config)

        assets = [
            ValidationAsset('ETH/USDT', 'Ethereum'),
            ValidationAsset('ADA/USDT', 'Cardano')
        ]

        # Mock market cap fetching
        with patch.object(selector, '_fetch_market_caps_dynamically') as mock_fetch:
            mock_fetch.return_value = {
                'ETH/USDT': 70000000000,  # 70B
                'ADA/USDT': 30000000000   # 30B
            }

            weighted = selector._apply_weighting(assets)

            # Should be proportional: 70/100 = 0.7, 30/100 = 0.3
            assert weighted[0].weight == 0.7
            assert weighted[1].weight == 0.3

    def test_get_market_cap_weights_from_config(self):
        """Test getting market cap weights from configuration."""
        config = self.config.copy()
        config['market_cap_weights'] = {
            'ETH/USDT': 0.6,
            'ADA/USDT': 0.4
        }

        selector = AssetSelector(config)

        assets = [
            ValidationAsset('ETH/USDT', 'Ethereum'),
            ValidationAsset('ADA/USDT', 'Cardano')
        ]

        # Mock dynamic fetching to fail so it falls back to config
        with patch.object(selector, '_fetch_market_caps_dynamically', return_value=None):
            weights = selector._get_market_cap_weights(assets)

            assert weights['ETH/USDT'] == 0.6
            assert weights['ADA/USDT'] == 0.4

    def test_get_market_cap_weights_fallback_to_equal(self):
        """Test market cap weights fallback to equal when no data available."""
        selector = AssetSelector(self.config)

        assets = [
            ValidationAsset('ETH/USDT', 'Ethereum'),
            ValidationAsset('ADA/USDT', 'Cardano')
        ]

        # Mock failed dynamic fetching
        with patch.object(selector, '_fetch_market_caps_dynamically', return_value=None):
            weights = selector._get_market_cap_weights(assets)

            assert weights['ETH/USDT'] == 0.5
            assert weights['ADA/USDT'] == 0.5

    def test_fetch_market_caps_dynamically_coingecko_success(self):
        """Test successful market cap fetching from CoinGecko."""
        selector = AssetSelector(self.config)

        with patch.object(selector, '_fetch_from_coingecko') as mock_coingecko:
            mock_coingecko.return_value = {'ETH/USDT': 1000000000}

            result = selector._fetch_market_caps_dynamically(['ETH/USDT'])

            assert result == {'ETH/USDT': 1000000000}
            mock_coingecko.assert_called_once_with(['ETH/USDT'])

    def test_fetch_market_caps_dynamically_coinmarketcap_fallback(self):
        """Test CoinMarketCap fallback when CoinGecko fails."""
        config = self.config.copy()
        config['coinmarketcap_api_key'] = 'test_key'

        selector = AssetSelector(config)

        with patch.object(selector, '_fetch_from_coingecko', return_value=None), \
             patch.object(selector, '_fetch_from_coinmarketcap') as mock_cmc:

            mock_cmc.return_value = {'ETH/USDT': 1000000000}

            result = selector._fetch_market_caps_dynamically(['ETH/USDT'])

            assert result == {'ETH/USDT': 1000000000}
            mock_cmc.assert_called_once_with(['ETH/USDT'])

    def test_fetch_market_caps_dynamically_local_cache_fallback(self):
        """Test local cache fallback when APIs fail."""
        selector = AssetSelector(self.config)

        with patch.object(selector, '_fetch_from_coingecko', return_value=None), \
             patch.object(selector, '_fetch_from_coinmarketcap', return_value=None), \
             patch.object(selector, '_fetch_from_local_cache') as mock_cache:

            mock_cache.return_value = {'ETH/USDT': 1000000000}

            result = selector._fetch_market_caps_dynamically(['ETH/USDT'])

            assert result == {'ETH/USDT': 1000000000}
            mock_cache.assert_called_once_with(['ETH/USDT'])

    @pytest.mark.asyncio
    async def test_fetch_from_coingecko_async_success(self):
        """Test successful async CoinGecko API fetch."""
        selector = AssetSelector(self.config)

        mock_response_data = [
            {
                'id': 'ethereum',
                'market_cap': 50000000000
            }
        ]

        # Mock the aiohttp ClientSession properly
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            result = await selector._fetch_from_coingecko_async(['ETH/USDT'])

            assert result == {'ETH/USDT': 50000000000}

    @pytest.mark.asyncio
    async def test_fetch_from_coingecko_async_failure(self):
        """Test CoinGecko API fetch failure."""
        selector = AssetSelector(self.config)

        mock_response = AsyncMock()
        mock_response.status = 500

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            result = await selector._fetch_from_coingecko_async(['ETH/USDT'])

            assert result is None

    def test_fetch_from_coinmarketcap_success(self):
        """Test successful CoinMarketCap API fetch."""
        config = self.config.copy()
        config['coinmarketcap_api_key'] = 'test_key'

        selector = AssetSelector(config)

        mock_response_data = {
            'data': {
                '1027': {
                    'quote': {
                        'USD': {
                            'market_cap': 30000000000
                        }
                    }
                }
            }
        }

        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_response_data
            mock_get.return_value = mock_response

            result = selector._fetch_from_coinmarketcap(['ETH/USDT'])

            assert result == {'ETH/USDT': 30000000000}

    def test_fetch_from_coinmarketcap_failure(self):
        """Test CoinMarketCap API fetch failure."""
        config = self.config.copy()
        config['coinmarketcap_api_key'] = 'test_key'

        selector = AssetSelector(config)

        with patch('requests.get', side_effect=Exception("API Error")):
            result = selector._fetch_from_coinmarketcap(['ETH/USDT'])

            assert result is None

    def test_fetch_from_local_cache_success(self):
        """Test successful local cache fetch."""
        selector = AssetSelector(self.config)

        cache_data = {
            'timestamp': time.time(),
            'market_caps': {
                'ETH/USDT': 40000000000,
                'ADA/USDT': 20000000000
            }
        }

        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(cache_data))):

            result = selector._fetch_from_local_cache(['ETH/USDT'])

            assert result == {'ETH/USDT': 40000000000}

    def test_fetch_from_local_cache_expired(self):
        """Test local cache fetch with expired data."""
        selector = AssetSelector(self.config)

        cache_data = {
            'timestamp': time.time() - 100000,  # Very old timestamp
            'market_caps': {'ETH/USDT': 40000000000}
        }

        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(cache_data))):

            result = selector._fetch_from_local_cache(['ETH/USDT'])

            assert result is None

    def test_calculate_market_cap_weights(self):
        """Test market cap weight calculation."""
        selector = AssetSelector(self.config)

        market_caps = {
            'ETH/USDT': 30000000000,
            'ADA/USDT': 10000000000,
            'SOL/USDT': 5000000000
        }

        weights = selector._calculate_market_cap_weights(market_caps)

        expected_total = 30000000000 + 10000000000 + 5000000000
        assert weights['ETH/USDT'] == pytest.approx(30000000000 / expected_total)
        assert weights['ADA/USDT'] == pytest.approx(10000000000 / expected_total)
        assert weights['SOL/USDT'] == pytest.approx(5000000000 / expected_total)

    def test_calculate_market_cap_weights_empty(self):
        """Test market cap weight calculation with empty data."""
        selector = AssetSelector(self.config)

        weights = selector._calculate_market_cap_weights({})

        assert weights == {}

    def test_is_cache_valid(self):
        """Test cache validity check."""
        selector = AssetSelector(self.config)

        # Initially invalid
        assert selector._is_cache_valid() is False

        # Set valid timestamp
        selector._cache_timestamp = time.time()
        assert selector._is_cache_valid() is True

        # Set expired timestamp
        selector._cache_timestamp = time.time() - 100000
        assert selector._is_cache_valid() is False

    def test_get_asset_summary(self):
        """Test getting asset summary."""
        selector = AssetSelector(self.config)

        summary = selector.get_asset_summary()

        assert summary['total_assets'] == 3
        assert len(summary['assets']) == 3
        assert summary['weighting_scheme'] == 'equal'
        assert summary['correlation_filter_enabled'] is False
        assert summary['max_correlation_threshold'] == 0.7

    def test_update_asset_weights(self):
        """Test updating asset weights."""
        selector = AssetSelector(self.config)

        new_weights = {
            'ETH/USDT': 0.8,
            'ADA/USDT': 0.2
        }

        selector.update_asset_weights(new_weights)

        eth_asset = next(asset for asset in selector.available_assets if asset.symbol == 'ETH/USDT')
        ada_asset = next(asset for asset in selector.available_assets if asset.symbol == 'ADA/USDT')

        assert eth_asset.weight == 0.8
        assert ada_asset.weight == 0.2

    def test_add_validation_asset(self):
        """Test adding a new validation asset."""
        selector = AssetSelector(self.config)

        new_asset = ValidationAsset('DOT/USDT', 'Polkadot', weight=0.7)

        selector.add_validation_asset(new_asset)

        assert len(selector.available_assets) == 4
        assert selector.available_assets[-1].symbol == 'DOT/USDT'

    def test_add_validation_asset_duplicate(self):
        """Test adding duplicate validation asset."""
        selector = AssetSelector(self.config)

        duplicate_asset = ValidationAsset('ETH/USDT', 'Ethereum Duplicate')

        selector.add_validation_asset(duplicate_asset)

        # Should still have only 3 assets
        assert len(selector.available_assets) == 3

    def test_remove_validation_asset(self):
        """Test removing a validation asset."""
        selector = AssetSelector(self.config)

        result = selector.remove_validation_asset('ADA/USDT')

        assert result is True
        assert len(selector.available_assets) == 2
        assert not any(asset.symbol == 'ADA/USDT' for asset in selector.available_assets)

    def test_remove_validation_asset_not_found(self):
        """Test removing non-existent validation asset."""
        selector = AssetSelector(self.config)

        result = selector.remove_validation_asset('NONEXISTENT/USDT')

        assert result is False
        assert len(selector.available_assets) == 3

    @pytest.mark.asyncio
    async def test_filter_correlated_assets_async(self):
        """Test async correlation filtering."""
        selector = AssetSelector(self.config)

        # Mock data fetcher
        mock_fetcher = AsyncMock()

        # Create mock dataframes
        primary_data = MagicMock()
        primary_data.empty = False
        primary_returns = MagicMock()
        primary_returns.corr.return_value = 0.5  # Low correlation
        primary_data.__getitem__.return_value.pct_change.return_value.dropna.return_value = primary_returns

        asset_data = MagicMock()
        asset_data.empty = False
        asset_returns = MagicMock()
        asset_data.__getitem__.return_value.pct_change.return_value.dropna.return_value = asset_returns

        mock_fetcher.get_historical_data.side_effect = [primary_data, asset_data]

        candidates = [ValidationAsset('ETH/USDT', 'Ethereum')]

        filtered = await selector._filter_correlated_assets_async(candidates, 'BTC/USDT', mock_fetcher)

        assert len(filtered) == 1
        assert filtered[0].symbol == 'ETH/USDT'

    @pytest.mark.asyncio
    async def test_filter_correlated_assets_async_high_correlation(self):
        """Test async correlation filtering with high correlation."""
        selector = AssetSelector(self.config)

        # Mock data fetcher
        mock_fetcher = AsyncMock()

        # Create mock dataframes
        primary_data = MagicMock()
        primary_data.empty = False
        primary_returns = MagicMock()
        primary_returns.corr.return_value = 0.9  # High correlation
        primary_data.__getitem__.return_value.pct_change.return_value.dropna.return_value = primary_returns

        asset_data = MagicMock()
        asset_data.empty = False
        asset_returns = MagicMock()
        asset_data.__getitem__.return_value.pct_change.return_value.dropna.return_value = asset_returns

        mock_fetcher.get_historical_data.side_effect = [primary_data, asset_data]

        candidates = [ValidationAsset('ETH/USDT', 'Ethereum')]

        filtered = await selector._filter_correlated_assets_async(candidates, 'BTC/USDT', mock_fetcher)

        assert len(filtered) == 0  # Should be filtered out due to high correlation

    def test_get_coingecko_id_mapping(self):
        """Test CoinGecko ID mapping."""
        selector = AssetSelector(self.config)

        mapping = selector._get_coingecko_id_mapping()

        assert mapping['BTC'] == 'bitcoin'
        assert mapping['ETH'] == 'ethereum'
        assert 'ADA' in mapping
        assert 'SOL' in mapping


if __name__ == "__main__":
    pytest.main([__file__])
