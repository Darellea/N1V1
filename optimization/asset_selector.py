"""
Asset Selector Module

This module handles the selection and configuration of validation assets
for cross-asset validation. It provides functionality to choose appropriate
validation assets based on various criteria including correlation filtering,
market capitalization weighting, and user-defined preferences.

Key Features:
- Dynamic asset selection based on correlation analysis
- Market capitalization-based weighting
- Configurable asset filtering and ranking
- Support for multiple data sources (CoinGecko, CoinMarketCap, local cache)
- Parallel asset data fetching
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from pathlib import Path
import warnings

import aiohttp
import numpy as np
import pandas as pd

from .config import get_cross_asset_validation_config
from .validation_criteria import ValidationCriteria


class ValidationAsset:
    """
    Represents a validation asset with its configuration.

    This dataclass encapsulates all the information needed to describe
    a validation asset including its symbol, metadata, and evaluation
    parameters.
    """

    def __init__(self, symbol: str, name: str, weight: float = 1.0,
                 required_history: int = 1000, timeframe: str = '1h',
                 market_cap: Optional[float] = None):
        """
        Initialize validation asset.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            name: Human-readable name
            weight: Relative weight in validation (higher = more important)
            required_history: Minimum historical data points required
            timeframe: Data timeframe for validation
            market_cap: Market capitalization (optional)
        """
        self.symbol = symbol
        self.name = name
        self.weight = weight
        self.required_history = required_history
        self.timeframe = timeframe
        self.market_cap = market_cap

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'weight': self.weight,
            'required_history': self.required_history,
            'timeframe': self.timeframe
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationAsset':
        """Create from dictionary."""
        return cls(
            symbol=data['symbol'],
            name=data['name'],
            weight=data.get('weight', 1.0),
            required_history=data.get('required_history', 1000),
            timeframe=data.get('timeframe', '1h')
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.symbol} ({self.name}) - weight: {self.weight}"


class AssetSelector:
    """
    Handles selection and configuration of validation assets.

    This class provides comprehensive functionality for selecting validation
    assets based on various criteria including correlation analysis, market
    capitalization, and user preferences. It supports multiple data sources
    and provides both synchronous and asynchronous operations.

    Key responsibilities:
    - Load and manage available validation assets
    - Apply correlation-based filtering
    - Calculate market cap-based weights
    - Fetch data from multiple sources with fallbacks
    - Provide asset ranking and selection logic
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize asset selector.

        Args:
            config: Configuration dictionary for asset selection.
                   If None, uses default configuration from config module.
        """
        if config is None:
            asset_config = get_cross_asset_validation_config()
            config = {
                'max_assets': asset_config.asset_selector.max_assets,
                'asset_weights': asset_config.asset_selector.asset_weights,
                'correlation_filter': asset_config.asset_selector.correlation_filter,
                'max_correlation': asset_config.asset_selector.max_correlation,
                'validation_assets': asset_config.asset_selector.validation_assets,
                'market_cap_weights': asset_config.asset_selector.market_cap_weights,
                'coinmarketcap_api_key': asset_config.asset_selector.coinmarketcap_api_key
            }

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Core configuration
        self.max_assets = config.get('max_assets', 3)
        self.asset_weights = config.get('asset_weights', 'equal')
        self.correlation_filter = config.get('correlation_filter', False)
        self.max_correlation = config.get('max_correlation', 0.7)

        # Asset data
        self.available_assets = self._load_validation_assets(config)
        self.market_cap_weights = config.get('market_cap_weights', {})
        self.coinmarketcap_api_key = config.get('coinmarketcap_api_key')

        # Caching
        self._market_caps_cache: Optional[Dict[str, float]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = 86400  # 24 hours

        self.logger.info(f"AssetSelector initialized with {len(self.available_assets)} available assets")

    def _load_validation_assets(self, config: Dict[str, Any]) -> List[ValidationAsset]:
        """
        Load validation assets from configuration.

        Args:
            config: Configuration containing asset definitions

        Returns:
            List of ValidationAsset objects
        """
        assets_config = config.get('validation_assets', [])

        if not assets_config:
            # Default validation assets if none provided
            assets_config = [
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
            ]

        assets = []
        for asset_config in assets_config:
            asset = ValidationAsset(
                symbol=asset_config['symbol'],
                name=asset_config['name'],
                weight=asset_config.get('weight', 1.0),
                required_history=asset_config.get('required_history', 1000),
                timeframe=asset_config.get('timeframe', '1h')
            )
            assets.append(asset)

        return assets

    def select_validation_assets(self, primary_asset: str,
                               data_fetcher: Optional[Any] = None) -> List[ValidationAsset]:
        """
        Select validation assets for cross-asset validation.

        This is the main entry point for asset selection. It applies various
        filtering and ranking criteria to choose the most appropriate validation
        assets for a given primary asset.

        Args:
            primary_asset: Primary asset symbol to validate against
            data_fetcher: Optional data fetcher for correlation analysis

        Returns:
            List of selected validation assets
        """
        # Start with all available assets except the primary
        candidates = [asset for asset in self.available_assets
                     if asset.symbol != primary_asset]

        if not candidates:
            self.logger.warning("No validation assets available")
            return []

        # Apply correlation filter if enabled and data fetcher available
        if self.correlation_filter and data_fetcher:
            candidates = self._filter_correlated_assets(
                candidates, primary_asset, data_fetcher
            )

        # Limit to maximum number of assets
        selected = candidates[:self.max_assets]

        # Apply weighting scheme
        selected = self._apply_weighting(selected)

        self.logger.info(f"Selected {len(selected)} validation assets: "
                        f"{[asset.symbol for asset in selected]}")

        return selected

    def _filter_correlated_assets(self, candidates: List[ValidationAsset],
                                primary_asset: str, data_fetcher: Any) -> List[ValidationAsset]:
        """
        Filter out highly correlated assets.

        This method analyzes the correlation between candidate assets and the
        primary asset, removing assets that are too highly correlated to provide
        meaningful validation diversity.

        Args:
            candidates: Candidate validation assets
            primary_asset: Primary asset symbol
            data_fetcher: Data fetcher for correlation analysis

        Returns:
            Filtered list of assets with low correlation to primary
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are in an event loop, create a task
            task = loop.create_task(self._filter_correlated_assets_async(
                candidates, primary_asset, data_fetcher
            ))
            return loop.run_until_complete(task)
        except RuntimeError:
            # No running event loop, use asyncio.run
            try:
                return asyncio.run(self._filter_correlated_assets_async(
                    candidates, primary_asset, data_fetcher
                ))
            except Exception as e:
                self.logger.warning(f"Async correlation analysis failed, using sync fallback: {e}")
                return candidates

    async def _filter_correlated_assets_async(self, candidates: List[ValidationAsset],
                                           primary_asset: str, data_fetcher: Any) -> List[ValidationAsset]:
        """
        Async implementation of correlation filtering.

        Args:
            candidates: Candidate validation assets
            primary_asset: Primary asset symbol
            data_fetcher: Data fetcher for correlation analysis

        Returns:
            Filtered list of assets
        """
        filtered = []

        try:
            # Get primary asset data for correlation analysis
            primary_data = await data_fetcher.get_historical_data(primary_asset, '1d', 100)

            if primary_data.empty:
                self.logger.warning("Could not fetch primary asset data for correlation analysis")
                return candidates

            primary_returns = primary_data['close'].pct_change().dropna()

            for asset in candidates:
                try:
                    # Get asset data
                    asset_data = await data_fetcher.get_historical_data(asset.symbol, '1d', 100)

                    if asset_data.empty:
                        continue

                    asset_returns = asset_data['close'].pct_change().dropna()

                    # Calculate correlation if data lengths match
                    if len(primary_returns) == len(asset_returns):
                        correlation = primary_returns.corr(asset_returns)

                        if abs(correlation) <= self.max_correlation:
                            filtered.append(asset)
                            self.logger.debug(f"Asset {asset.symbol} correlation: {correlation:.3f} (accepted)")
                        else:
                            self.logger.debug(f"Asset {asset.symbol} correlation: {correlation:.3f} (filtered out)")
                    else:
                        # If data lengths don't match, include the asset
                        filtered.append(asset)

                except Exception as e:
                    self.logger.warning(f"Could not analyze correlation for {asset.symbol}: {e}")
                    # Include asset if correlation analysis fails
                    filtered.append(asset)

        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {e}")
            return candidates

        return filtered

    def _apply_weighting(self, assets: List[ValidationAsset]) -> List[ValidationAsset]:
        """
        Apply weighting scheme to selected assets.

        This method assigns weights to assets based on the configured weighting
        scheme (equal, market_cap, or custom).

        Args:
            assets: List of assets to weight

        Returns:
            Weighted list of assets
        """
        if self.asset_weights == 'equal':
            # Equal weighting
            weight = 1.0 / len(assets) if assets else 1.0
            for asset in assets:
                asset.weight = weight

        elif self.asset_weights == 'market_cap':
            # Get market cap data
            market_caps = self._fetch_market_caps_dynamically([asset.symbol for asset in assets])
            if market_caps:
                for asset in assets:
                    asset.market_cap = market_caps.get(asset.symbol, 0)

                # Calculate weights
                total_cap = sum(asset.market_cap for asset in assets if asset.market_cap)
                if total_cap > 0:
                    for asset in assets:
                        asset.weight = asset.market_cap / total_cap
                else:
                    # fallback to equal weights
                    for asset in assets:
                        asset.weight = 1 / len(assets)
            else:
                # fallback to equal weights
                for asset in assets:
                    asset.weight = 1 / len(assets)

        # Ensure weights sum to 1 (final safety check)
        total_weight = sum(asset.weight for asset in assets)
        if total_weight > 0:
            for asset in assets:
                asset.weight /= total_weight

        return assets

    def _get_market_cap_weights(self, assets: List[ValidationAsset]) -> Dict[str, float]:
        """
        Get market capitalization weights for assets.

        This method attempts to fetch current market cap data from multiple
        sources with appropriate fallbacks and caching.

        Args:
            assets: List of validation assets

        Returns:
            Dictionary mapping asset symbols to market cap weights
        """
        asset_symbols = [asset.symbol for asset in assets]

        # Try dynamic fetching first
        market_caps = self._fetch_market_caps_dynamically(asset_symbols)

        if market_caps:
            self.logger.info(f"Successfully fetched market caps for {len(market_caps)} assets")
            return self._calculate_market_cap_weights(market_caps)

        # Fallback to configured values
        if self.market_cap_weights:
            self.logger.info("Using configured market cap weights as fallback")
            return self.market_cap_weights

        # Final fallback to equal weights
        self.logger.warning("No market cap data available, falling back to equal weights")
        equal_weight = 1.0 / len(assets) if assets else 1.0
        return {asset.symbol: equal_weight for asset in assets}

    def _fetch_market_caps_dynamically(self, asset_symbols: List[str]) -> Optional[Dict[str, float]]:
        """
        Fetch market capitalization data dynamically from external sources.

        This method tries multiple data sources in order of preference:
        1. CoinGecko API (free tier)
        2. CoinMarketCap API (if API key available)
        3. Local cache (if fresh enough)

        Args:
            asset_symbols: List of asset symbols to fetch market caps for

        Returns:
            Dictionary mapping asset symbols to market caps, or None if all sources fail
        """
        # Check cache first
        if self._is_cache_valid():
            cached_data = self._load_from_cache(asset_symbols)
            if cached_data:
                return cached_data

        # Try CoinGecko API first
        market_caps = self._fetch_from_coingecko(asset_symbols)
        if market_caps:
            self._save_to_cache(market_caps)
            return market_caps

        # Try CoinMarketCap API if configured
        if self.coinmarketcap_api_key:
            market_caps = self._fetch_from_coinmarketcap(asset_symbols)
            if market_caps:
                self._save_to_cache(market_caps)
                return market_caps

        # Try local database/cache
        market_caps = self._fetch_from_local_cache(asset_symbols)
        if market_caps:
            return market_caps

        return None

    def _fetch_from_coingecko(self, asset_symbols: List[str]) -> Optional[Dict[str, float]]:
        """
        Fetch market caps from CoinGecko API.

        Args:
            asset_symbols: List of asset symbols

        Returns:
            Dictionary of market caps or None if failed
        """
        try:
            # Check if we're in an event loop
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._fetch_from_coingecko_async(asset_symbols))
            return loop.run_until_complete(task)
        except RuntimeError:
            try:
                return asyncio.run(self._fetch_from_coingecko_async(asset_symbols))
            except Exception as e:
                self.logger.warning(f"Async CoinGecko fetch failed: {e}")
                return None

    async def _fetch_from_coingecko_async(self, symbols: List[str]) -> Optional[Dict[str, int]]:
        """
        Async implementation of CoinGecko API fetching.

        Args:
            symbols: List of asset symbols

        Returns:
            Dictionary mapping symbols to market caps or None if failed
        """
        symbol_to_id = self._get_coingecko_id_mapping()
        id_to_symbol = {v: k for k, v in symbol_to_id.items()}

        ids = []
        for s in symbols:
            base = s.split("/")[0].upper()
            if base in symbol_to_id:
                ids.append(symbol_to_id[base])

        if not ids:
            return None

        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "ids": ",".join(ids)}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            f"{id_to_symbol.get(item['id'], item['id'].upper())}/USDT": item["market_cap"]
                            for item in data if "market_cap" in item
                        }
                    else:
                        self.logger.warning(f"Failed to fetch from CoinGecko: HTTP {response.status}")
        except Exception as e:
            self.logger.warning(f"Failed to fetch from CoinGecko: {e}")
        return None

    def _fetch_from_coinmarketcap(self, asset_symbols: List[str]) -> Optional[Dict[str, float]]:
        """
        Fetch market caps from CoinMarketCap API.

        Args:
            asset_symbols: List of asset symbols

        Returns:
            Dictionary of market caps or None if failed
        """
        try:
            import requests

            # Map symbols to CMC IDs
            symbol_to_cmc_id = {
                'BTC': '1', 'ETH': '1027', 'ADA': '2010', 'SOL': '5426',
                'DOT': '6636', 'LINK': '1975', 'BNB': '1839', 'XRP': '52',
                'LTC': '2', 'DOGE': '74'
            }

            cmc_ids = []
            for symbol in asset_symbols:
                base_symbol = symbol.split('/')[0].upper()
                if base_symbol in symbol_to_cmc_id:
                    cmc_ids.append(symbol_to_cmc_id[base_symbol])

            if not cmc_ids:
                return None

            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key,
            }
            params = {
                'id': ','.join(cmc_ids),
                'convert': 'USD'
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            market_caps = {}

            for cmc_id, coin_data in data['data'].items():
                market_cap = coin_data['quote']['USD'].get('market_cap', 0)

                # Find original symbol
                for symbol in asset_symbols:
                    base_symbol = symbol.split('/')[0].upper()
                    if symbol_to_cmc_id.get(base_symbol) == cmc_id:
                        market_caps[symbol] = market_cap
                        break

            return market_caps if market_caps else None

        except Exception as e:
            self.logger.warning(f"Failed to fetch from CoinMarketCap: {str(e)}")
            return None

    def _fetch_from_local_cache(self, asset_symbols: List[str]) -> Optional[Dict[str, float]]:
        """
        Fetch market caps from local cache/database.

        Args:
            asset_symbols: List of asset symbols

        Returns:
            Dictionary of market caps or None if failed
        """
        try:
            cache_file = os.path.join(os.getcwd(), 'data', 'market_caps_cache.json')

            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Check if cache is fresh
                cache_timestamp = cache_data.get('timestamp', 0)
                if time.time() - cache_timestamp < self._cache_ttl:
                    market_caps = cache_data.get('market_caps', {})
                    # Filter for requested symbols
                    filtered_caps = {symbol: market_caps.get(symbol)
                                   for symbol in asset_symbols if symbol in market_caps}
                    if filtered_caps:
                        self.logger.info(f"Loaded market caps from local cache for {len(filtered_caps)} assets")
                        return filtered_caps

        except Exception as e:
            self.logger.warning(f"Failed to load from local cache: {str(e)}")

        return None

    def _get_coingecko_id_mapping(self) -> Dict[str, str]:
        """
        Get mapping from common symbols to CoinGecko IDs.

        Returns:
            Dictionary mapping symbols to CoinGecko IDs
        """
        return {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'ADA': 'cardano',
            'SOL': 'solana', 'DOT': 'polkadot', 'LINK': 'chainlink',
            'BNB': 'binancecoin', 'XRP': 'ripple', 'LTC': 'litecoin',
            'DOGE': 'dogecoin', 'MATIC': 'matic-network', 'AVAX': 'avalanche-2'
        }

    def _calculate_market_cap_weights(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate normalized weights from market capitalization data.

        Args:
            market_caps: Dictionary of market caps

        Returns:
            Dictionary of normalized weights
        """
        if not market_caps:
            return {}

        # Filter out zero or invalid market caps
        valid_caps = {symbol: cap for symbol, cap in market_caps.items()
                     if cap and cap > 0}

        if not valid_caps:
            return {}

        # Calculate total market cap
        total_cap = sum(valid_caps.values())

        # Calculate weights
        weights = {}
        for symbol, cap in valid_caps.items():
            weights[symbol] = cap / total_cap

        self.logger.info(f"Calculated market cap weights: {weights}")
        return weights

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_timestamp is None:
            return False
        return time.time() - self._cache_timestamp < self._cache_ttl

    def _load_from_cache(self, asset_symbols: List[str]) -> Optional[Dict[str, float]]:
        """Load market caps from memory cache."""
        if not self._market_caps_cache:
            return None

        filtered_caps = {symbol: self._market_caps_cache.get(symbol)
                        for symbol in asset_symbols if symbol in self._market_caps_cache}
        return filtered_caps if filtered_caps else None

    def _save_to_cache(self, market_caps: Dict[str, float]) -> None:
        """Save market caps to memory cache."""
        self._market_caps_cache = market_caps.copy()
        self._cache_timestamp = time.time()

    def get_asset_summary(self) -> Dict[str, Any]:
        """
        Get summary of available assets and their properties.

        Returns:
            Dictionary containing asset summary information
        """
        return {
            'total_assets': len(self.available_assets),
            'assets': [asset.to_dict() for asset in self.available_assets],
            'weighting_scheme': self.asset_weights,
            'correlation_filter_enabled': self.correlation_filter,
            'max_correlation_threshold': self.max_correlation,
            'cache_valid': self._is_cache_valid()
        }

    def update_asset_weights(self, weights: Dict[str, float]) -> None:
        """
        Update weights for specific assets.

        Args:
            weights: Dictionary mapping asset symbols to new weights
        """
        for asset in self.available_assets:
            if asset.symbol in weights:
                asset.weight = weights[asset.symbol]

        self.logger.info(f"Updated weights for {len(weights)} assets")

    def add_validation_asset(self, asset: ValidationAsset) -> None:
        """
        Add a new validation asset to the available assets list.

        Args:
            asset: ValidationAsset to add
        """
        # Check if asset already exists
        if any(existing.symbol == asset.symbol for existing in self.available_assets):
            self.logger.warning(f"Asset {asset.symbol} already exists, skipping")
            return

        self.available_assets.append(asset)
        self.logger.info(f"Added new validation asset: {asset.symbol}")

    def remove_validation_asset(self, symbol: str) -> bool:
        """
        Remove a validation asset from the available assets list.

        Args:
            symbol: Asset symbol to remove

        Returns:
            True if asset was removed, False if not found
        """
        for i, asset in enumerate(self.available_assets):
            if asset.symbol == symbol:
                removed_asset = self.available_assets.pop(i)
                self.logger.info(f"Removed validation asset: {removed_asset.symbol}")
                return True

        self.logger.warning(f"Asset {symbol} not found, could not remove")
        return False
