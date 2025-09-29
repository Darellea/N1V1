"""
Cross-Asset Validation Module

This module implements cross-asset validation for trading strategies.
It ensures that strategies optimized on one asset perform robustly across
multiple market conditions by testing them on additional validation assets.

Key Features:
- Configurable validation asset selection
- Automated performance evaluation across assets
- Pass/fail criteria based on statistical thresholds
- Comprehensive logging and reporting
- Integration with optimization and backtesting frameworks

Refactored Structure:
- asset_selector.py: Handles selection and configuration of validation assets
- validation_criteria.py: Defines pass/fail criteria for cross-asset validation
- validation_results.py: Data structures for validation results
- market_data_fetcher.py: Handles fetching market data for validation assets
- cross_asset_validator.py: Main validator class that orchestrates the process
- config.py: Centralized configuration management
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_optimizer import BaseOptimizer

# Import DataFetcher conditionally to avoid circular imports
try:
    from data.data_fetcher import DataFetcher
except ImportError:
    DataFetcher = None


@dataclass
class ValidationAsset:
    """Represents a validation asset with its configuration."""

    symbol: str
    name: str
    weight: float = 1.0
    required_history: int = 1000
    timeframe: str = "1h"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "weight": self.weight,
            "required_history": self.required_history,
            "timeframe": self.timeframe,
        }


@dataclass
class AssetValidationResult:
    """Result of validating a strategy on a single asset."""

    asset: ValidationAsset
    optimized_params: Dict[str, Any]
    primary_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    pass_criteria: Dict[str, bool]
    overall_pass: bool
    validation_time: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asset": self.asset.to_dict(),
            "optimized_params": self.optimized_params,
            "primary_metrics": self.primary_metrics,
            "validation_metrics": self.validation_metrics,
            "pass_criteria": self.pass_criteria,
            "overall_pass": self.overall_pass,
            "validation_time": self.validation_time,
            "error_message": self.error_message,
            "timestamp": datetime.now().isoformat(),
        }


@dataclass
class CrossAssetValidationResult:
    """Complete cross-asset validation results."""

    strategy_name: str
    primary_asset: str
    validation_assets: List[ValidationAsset]
    asset_results: List[AssetValidationResult]
    aggregate_metrics: Dict[str, Any]
    pass_rate: float
    overall_pass: bool
    robustness_score: float
    timestamp: datetime
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy_name": self.strategy_name,
            "primary_asset": self.primary_asset,
            "validation_assets": [asset.to_dict() for asset in self.validation_assets],
            "asset_results": [result.to_dict() for result in self.asset_results],
            "aggregate_metrics": self.aggregate_metrics,
            "pass_rate": self.pass_rate,
            "overall_pass": self.overall_pass,
            "robustness_score": self.robustness_score,
            "timestamp": self.timestamp.isoformat(),
            "total_time": self.total_time,
        }


class ValidationCriteria:
    """Defines pass/fail criteria for cross-asset validation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation criteria.

        Args:
            config: Configuration dictionary containing:
                - min_sharpe_ratio: Minimum Sharpe ratio threshold
                - max_drawdown_limit: Maximum drawdown limit
                - min_win_rate: Minimum win rate threshold
                - min_profit_factor: Minimum profit factor threshold
                - consistency_threshold: Consistency threshold for metrics
                - required_pass_rate: Minimum pass rate across assets
        """
        self.min_sharpe_ratio = config.get("min_sharpe_ratio", 0.5)
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.15)  # 15%
        self.min_win_rate = config.get("min_win_rate", 0.45)
        self.min_profit_factor = config.get("min_profit_factor", 1.2)
        self.consistency_threshold = config.get("consistency_threshold", 0.7)
        self.required_pass_rate = config.get("required_pass_rate", 0.6)  # 60%

    def evaluate_asset(
        self, primary_metrics: Dict[str, Any], validation_metrics: Dict[str, Any]
    ) -> Tuple[Dict[str, bool], bool]:
        """
        Evaluate if an asset passes validation criteria.

        Args:
            primary_metrics: Metrics from primary asset optimization
            validation_metrics: Metrics from validation asset

        Returns:
            Tuple of (pass_criteria_dict, overall_pass)
        """
        pass_criteria = {}

        # Sharpe ratio criterion
        val_sharpe = validation_metrics.get("sharpe_ratio", 0)
        pass_criteria["sharpe_ratio"] = val_sharpe >= self.min_sharpe_ratio

        # Maximum drawdown criterion
        val_max_dd = validation_metrics.get("max_drawdown", 1.0)
        pass_criteria["max_drawdown"] = val_max_dd <= self.max_drawdown_limit

        # Win rate criterion
        val_win_rate = validation_metrics.get("win_rate", 0)
        pass_criteria["win_rate"] = val_win_rate >= self.min_win_rate

        # Profit factor criterion
        val_profit_factor = validation_metrics.get("profit_factor", 0)
        pass_criteria["profit_factor"] = val_profit_factor >= self.min_profit_factor

        # Consistency check (compare to primary metrics)
        consistency_score = self._calculate_consistency_score(
            primary_metrics, validation_metrics
        )
        pass_criteria["consistency"] = bool(
            consistency_score >= self.consistency_threshold
        )

        # Overall pass (all criteria must pass)
        overall_pass = all(pass_criteria.values())

        return pass_criteria, overall_pass

    def evaluate_overall(
        self, asset_results: List[AssetValidationResult]
    ) -> Tuple[float, bool]:
        """
        Evaluate overall validation results across all assets.

        Args:
            asset_results: List of individual asset validation results

        Returns:
            Tuple of (pass_rate, overall_pass)
        """
        if not asset_results:
            return 0.0, False

        # Calculate pass rate
        passed_assets = sum(1 for result in asset_results if result.overall_pass)
        pass_rate = passed_assets / len(asset_results)

        # Overall pass based on required pass rate
        overall_pass = pass_rate >= self.required_pass_rate

        return pass_rate, overall_pass

    def _calculate_consistency_score(
        self, primary: Dict[str, Any], validation: Dict[str, Any]
    ) -> float:
        """
        Calculate consistency score between primary and validation metrics.

        This method safely handles division by zero errors by checking if the primary
        metric value is zero before performing division operations. When primary_val
        is zero, we consider the metrics perfectly consistent only if validation_val
        is also zero (or very close to zero), otherwise they are considered inconsistent.

        Args:
            primary: Primary asset metrics
            validation: Validation asset metrics

        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        metrics_to_compare = [
            "sharpe_ratio",
            "total_return",
            "win_rate",
            "profit_factor",
        ]
        consistency_scores = []

        for metric in metrics_to_compare:
            primary_val = primary.get(metric, 0)
            validation_val = validation.get(metric, 0)

            # Check for division by zero to prevent ZeroDivisionError
            if primary_val != 0:
                # Calculate relative difference safely
                relative_diff = abs(validation_val - primary_val) / abs(primary_val)
                # Convert to consistency score (lower difference = higher consistency)
                consistency = max(0, 1 - relative_diff)
            else:
                # Handle edge case when primary_val is zero
                # If both values are zero (or very close), they are perfectly consistent
                # If primary is zero but validation is not, they are inconsistent
                if (
                    abs(validation_val) < 1e-6
                ):  # Very small threshold for floating point comparison
                    consistency = (
                        1.0  # Perfect consistency when both are effectively zero
                    )
                else:
                    consistency = (
                        0.0  # No consistency when primary is zero but validation is not
                    )

            consistency_scores.append(consistency)

        # Return average consistency score
        return np.mean(consistency_scores) if consistency_scores else 0.0


class AssetSelector:
    """Handles selection and configuration of validation assets."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize asset selector.

        Args:
            config: Configuration dictionary containing:
                - validation_assets: List of asset configurations
                - max_assets: Maximum number of validation assets
                - asset_weights: Weighting scheme for assets
                - correlation_filter: Whether to filter correlated assets
                - max_correlation: Maximum allowed correlation
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        self.max_assets = config.get("max_assets", 3)
        self.asset_weights = config.get("asset_weights", "equal")
        self.correlation_filter = config.get("correlation_filter", False)
        self.max_correlation = config.get("max_correlation", 0.7)

        # Load validation assets from config
        self.available_assets = self._load_validation_assets()

    def _load_validation_assets(self) -> List[ValidationAsset]:
        """Load validation assets from configuration."""
        assets_config = self.config.get("validation_assets", [])

        if not assets_config:
            # Default validation assets
            assets_config = [
                {
                    "symbol": "ETH/USDT",
                    "name": "Ethereum",
                    "weight": 1.0,
                    "required_history": 1000,
                    "timeframe": "1h",
                },
                {
                    "symbol": "ADA/USDT",
                    "name": "Cardano",
                    "weight": 0.8,
                    "required_history": 1000,
                    "timeframe": "1h",
                },
                {
                    "symbol": "SOL/USDT",
                    "name": "Solana",
                    "weight": 0.6,
                    "required_history": 1000,
                    "timeframe": "1h",
                },
            ]

        assets = []
        for asset_config in assets_config:
            asset = ValidationAsset(
                symbol=asset_config["symbol"],
                name=asset_config["name"],
                weight=asset_config.get("weight", 1.0),
                required_history=asset_config.get("required_history", 1000),
                timeframe=asset_config.get("timeframe", "1h"),
            )
            assets.append(asset)

        return assets

    def _get_market_cap_weights(
        self, assets: List[ValidationAsset]
    ) -> Dict[str, float]:
        """
        Get market capitalization weights for assets.

        This method attempts to dynamically fetch market cap data from various sources:
        1. Third-party API (CoinGecko, CoinMarketCap)
        2. Local database/cache
        3. Configuration file
        4. Fallback to equal weights

        Args:
            assets: List of validation assets

        Returns:
            Dictionary mapping asset symbols to market cap weights
        """
        asset_symbols = [asset.symbol for asset in assets]

        # Try dynamic fetching first
        market_caps = self._fetch_market_caps_dynamically(asset_symbols)

        if market_caps:
            self.logger.info(
                f"Successfully fetched market caps for {len(market_caps)} assets"
            )
            return self._calculate_market_cap_weights(market_caps)

        # Fallback to configured values
        configured_weights = self.config.get("market_cap_weights", {})
        if configured_weights:
            self.logger.info("Using configured market cap weights as fallback")
            return configured_weights

        # Final fallback to equal weights
        self.logger.warning(
            "No market cap data available, falling back to equal weights"
        )
        equal_weight = 1.0 / len(assets) if assets else 1.0
        return {asset.symbol: equal_weight for asset in assets}

    def _fetch_market_caps_dynamically(
        self, asset_symbols: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Fetch market capitalization data dynamically from external sources.

        Args:
            asset_symbols: List of asset symbols to fetch market caps for

        Returns:
            Dictionary mapping asset symbols to market caps, or None if fetch fails
        """
        # Try CoinGecko API first (free tier available)
        market_caps = self._fetch_from_coingecko(asset_symbols)
        if market_caps:
            return market_caps

        # Try CoinMarketCap API if configured
        market_caps = self._fetch_from_coinmarketcap(asset_symbols)
        if market_caps:
            return market_caps

        # Try local database/cache
        market_caps = self._fetch_from_local_cache(asset_symbols)
        if market_caps:
            return market_caps

        return None

    async def _fetch_from_coingecko_async(
        self, asset_symbols: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Fetch market caps from CoinGecko API (async version).

        Args:
            asset_symbols: List of asset symbols

        Returns:
            Dictionary of market caps or None if failed
        """
        try:
            import aiohttp

            # Map common symbols to CoinGecko IDs
            symbol_to_id = self._get_coingecko_id_mapping()

            # Convert symbols to CoinGecko IDs
            coingecko_ids = []
            for symbol in asset_symbols:
                # Extract base currency (e.g., 'BTC/USDT' -> 'BTC')
                base_symbol = symbol.split("/")[0].upper()
                if base_symbol in symbol_to_id:
                    coingecko_ids.append(symbol_to_id[base_symbol])

            if not coingecko_ids:
                return None

            # Fetch market data from CoinGecko
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "ids": ",".join(coingecko_ids),
                "order": "market_cap_desc",
                "per_page": len(coingecko_ids),
                "page": 1,
                "sparkline": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        market_caps = {}

                        for coin in data:
                            coin_id = coin["id"]
                            market_cap = coin.get("market_cap", 0)

                            # Find original symbol
                            for symbol in asset_symbols:
                                base_symbol = symbol.split("/")[0].upper()
                                if symbol_to_id.get(base_symbol) == coin_id:
                                    market_caps[symbol] = market_cap
                                    break

                        return market_caps if market_caps else None

        except Exception as e:
            self.logger.warning(f"Failed to fetch from CoinGecko: {str(e)}")

        return None

    def _fetch_from_coingecko(
        self, asset_symbols: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Fetch market caps from CoinGecko API.

        Args:
            asset_symbols: List of asset symbols

        Returns:
            Dictionary of market caps or None if failed
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are in an event loop, create a task
            task = loop.create_task(self._fetch_from_coingecko_async(asset_symbols))
            return loop.run_until_complete(task)
        except RuntimeError:
            # No running event loop, use asyncio.run
            try:
                return asyncio.run(self._fetch_from_coingecko_async(asset_symbols))
            except Exception as e:
                self.logger.warning(f"Async CoinGecko fetch failed: {str(e)}")
                return None

    def _fetch_from_coinmarketcap(
        self, asset_symbols: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Fetch market caps from CoinMarketCap API.

        Args:
            asset_symbols: List of asset symbols

        Returns:
            Dictionary of market caps or None if failed
        """
        try:
            # Check if API key is configured
            api_key = self.config.get("coinmarketcap_api_key")
            if not api_key:
                return None

            import requests

            # Map symbols to CMC IDs (simplified mapping)
            symbol_to_cmc_id = {
                "BTC": "1",
                "ETH": "1027",
                "ADA": "2010",
                "SOL": "5426",
                "DOT": "6636",
                "LINK": "1975",
            }

            cmc_ids = []
            for symbol in asset_symbols:
                base_symbol = symbol.split("/")[0].upper()
                if base_symbol in symbol_to_cmc_id:
                    cmc_ids.append(symbol_to_cmc_id[base_symbol])

            if not cmc_ids:
                return None

            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": api_key,
            }
            params = {"id": ",".join(cmc_ids), "convert": "USD"}

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            market_caps = {}

            for cmc_id, coin_data in data["data"].items():
                market_cap = coin_data["quote"]["USD"].get("market_cap", 0)

                # Find original symbol
                for symbol in asset_symbols:
                    base_symbol = symbol.split("/")[0].upper()
                    if symbol_to_cmc_id.get(base_symbol) == cmc_id:
                        market_caps[symbol] = market_cap
                        break

            return market_caps if market_caps else None

        except Exception as e:
            self.logger.warning(f"Failed to fetch from CoinMarketCap: {str(e)}")
            return None

    def _fetch_from_local_cache(
        self, asset_symbols: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Fetch market caps from local cache/database.

        Args:
            asset_symbols: List of asset symbols

        Returns:
            Dictionary of market caps or None if failed
        """
        try:
            # Try to load from a local market cap cache file
            cache_file = os.path.join(os.getcwd(), "data", "market_caps_cache.json")

            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)

                # Check if cache is fresh (less than 24 hours old)
                cache_timestamp = cache_data.get("timestamp", 0)
                if time.time() - cache_timestamp < 86400:  # 24 hours
                    market_caps = cache_data.get("market_caps", {})
                    # Filter for requested symbols
                    filtered_caps = {
                        symbol: market_caps.get(symbol)
                        for symbol in asset_symbols
                        if symbol in market_caps
                    }
                    if filtered_caps:
                        self.logger.info(
                            f"Loaded market caps from local cache for {len(filtered_caps)} assets"
                        )
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
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "ADA": "cardano",
            "SOL": "solana",
            "DOT": "polkadot",
            "LINK": "chainlink",
            "BNB": "binancecoin",
            "XRP": "ripple",
            "LTC": "litecoin",
            "DOGE": "dogecoin",
        }

    def _calculate_market_cap_weights(
        self, market_caps: Dict[str, float]
    ) -> Dict[str, float]:
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
        valid_caps = {
            symbol: cap for symbol, cap in market_caps.items() if cap and cap > 0
        }

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

    def select_validation_assets(
        self, primary_asset: str, data_fetcher: Optional[DataFetcher] = None
    ) -> List[ValidationAsset]:
        """
        Select validation assets for cross-asset validation.

        Args:
            primary_asset: Primary asset symbol
            data_fetcher: Optional data fetcher for correlation analysis

        Returns:
            List of selected validation assets
        """
        # Start with all available assets
        candidates = [
            asset for asset in self.available_assets if asset.symbol != primary_asset
        ]

        if not candidates:
            self.logger.warning("No validation assets available")
            return []

        # Apply correlation filter if enabled
        if self.correlation_filter and data_fetcher:
            candidates = self._filter_correlated_assets(
                candidates, primary_asset, data_fetcher
            )

        # Limit to maximum number of assets
        selected = candidates[: self.max_assets]

        # Apply weighting
        selected = self._apply_weighting(selected)

        self.logger.info(
            f"Selected {len(selected)} validation assets: "
            f"{[asset.symbol for asset in selected]}"
        )

        return selected

    async def _filter_correlated_assets_async(
        self,
        candidates: List[ValidationAsset],
        primary_asset: str,
        data_fetcher: DataFetcher,
    ) -> List[ValidationAsset]:
        """
        Filter out highly correlated assets (async version).

        Args:
            candidates: Candidate validation assets
            primary_asset: Primary asset symbol
            data_fetcher: Data fetcher for correlation analysis

        Returns:
            Filtered list of assets
        """
        filtered = []

        try:
            # Get primary asset data
            primary_data = await data_fetcher.get_historical_data(
                primary_asset, "1d", 100
            )

            if primary_data.empty:
                self.logger.warning(
                    "Could not fetch primary asset data for correlation analysis"
                )
                return candidates

            primary_returns = primary_data["close"].pct_change().dropna()

            for asset in candidates:
                try:
                    # Get asset data
                    asset_data = await data_fetcher.get_historical_data(
                        asset.symbol, "1d", 100
                    )

                    if asset_data.empty:
                        continue

                    asset_returns = asset_data["close"].pct_change().dropna()

                    # Calculate correlation
                    if len(primary_returns) == len(asset_returns):
                        correlation = primary_returns.corr(asset_returns)

                        if abs(correlation) <= self.max_correlation:
                            filtered.append(asset)
                            self.logger.debug(
                                f"Asset {asset.symbol} correlation: {correlation:.3f} (accepted)"
                            )
                        else:
                            self.logger.debug(
                                f"Asset {asset.symbol} correlation: {correlation:.3f} (filtered out)"
                            )
                    else:
                        # If data lengths don't match, include the asset
                        filtered.append(asset)

                except Exception as e:
                    self.logger.warning(
                        f"Could not analyze correlation for {asset.symbol}: {e}"
                    )
                    # Include asset if correlation analysis fails
                    filtered.append(asset)

        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {e}")
            return candidates

        return filtered

    def _filter_correlated_assets(
        self,
        candidates: List[ValidationAsset],
        primary_asset: str,
        data_fetcher: DataFetcher,
    ) -> List[ValidationAsset]:
        """
        Filter out highly correlated assets.

        This method safely handles both synchronous and asynchronous contexts
        by using asyncio.create_task() when already in an event loop, or asyncio.run()
        when no event loop is running.

        Args:
            candidates: Candidate validation assets
            primary_asset: Primary asset symbol
            data_fetcher: Data fetcher for correlation analysis

        Returns:
            Filtered list of assets
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are in an event loop, create a task instead of using asyncio.run
            task = loop.create_task(
                self._filter_correlated_assets_async(
                    candidates, primary_asset, data_fetcher
                )
            )
            # Wait for the task to complete
            return loop.run_until_complete(task)
        except RuntimeError:
            # No running event loop, we can use asyncio.run
            try:
                return asyncio.run(
                    self._filter_correlated_assets_async(
                        candidates, primary_asset, data_fetcher
                    )
                )
            except Exception as e:
                self.logger.warning(
                    f"Async correlation analysis failed, falling back to sync: {e}"
                )
                return candidates

    def _apply_weighting(self, assets: List[ValidationAsset]) -> List[ValidationAsset]:
        """
        Apply weighting scheme to selected assets.

        Args:
            assets: List of assets to weight

        Returns:
            Weighted list of assets
        """
        if self.asset_weights == "equal":
            # Equal weighting
            weight = 1.0 / len(assets) if assets else 1.0
            for asset in assets:
                asset.weight = weight

        elif self.asset_weights == "market_cap":
            # Weight by market capitalization - dynamically fetch or use configured values
            market_cap_weights = self._get_market_cap_weights(assets)

            total_weight = 0
            for asset in assets:
                asset.weight = market_cap_weights.get(asset.symbol, 0.1)
                total_weight += asset.weight

            # Normalize weights to ensure they sum to 1
            if total_weight > 0:
                for asset in assets:
                    asset.weight /= total_weight
            else:
                # If no weights found, fall back to equal weighting
                self.logger.warning(
                    "No valid market cap weights found, falling back to equal weighting"
                )
                weight = 1.0 / len(assets) if assets else 1.0
                for asset in assets:
                    asset.weight = weight

        # Ensure weights sum to 1 (final safety check)
        total_weight = sum(asset.weight for asset in assets)
        if total_weight > 0:
            for asset in assets:
                asset.weight /= total_weight

        return assets


class CrossAssetValidator(BaseOptimizer):
    """
    Cross-Asset Validator for robust strategy validation.

    This validator:
    1. Takes an optimized strategy from single-asset optimization
    2. Tests it on multiple validation assets
    3. Evaluates performance against statistical thresholds
    4. Determines if the strategy is robust across different market conditions
    5. Provides comprehensive reporting and analysis
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Cross-Asset Validator.

        Args:
            config: Configuration dictionary containing:
                - asset_selector: Configuration for asset selection
                - validation_criteria: Configuration for pass/fail criteria
                - data_fetcher: Configuration for data fetching
                - output_dir: Directory for saving results
                - parallel_validation: Whether to validate assets in parallel
        """
        super().__init__(config)

        # Cross-asset validation specific configuration
        self.asset_selector_config = config.get("asset_selector", {})
        self.validation_criteria_config = config.get("validation_criteria", {})
        self.data_fetcher_config = config.get("data_fetcher", {})
        self.output_dir = config.get("output_dir", "results/cross_asset_validation")
        self.parallel_validation = config.get("parallel_validation", False)

        # Initialize components
        self.asset_selector = AssetSelector(self.asset_selector_config)
        self.validation_criteria = ValidationCriteria(self.validation_criteria_config)

        # Data fetcher for validation assets
        self.data_fetcher = None

        # Results storage
        self.asset_results: List[AssetValidationResult] = []
        self.aggregate_metrics: Dict[str, Any] = {}

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("Cross-Asset Validator initialized")

    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run cross-asset validation.

        Note: This method is implemented for compatibility with BaseOptimizer,
        but cross-asset validation requires pre-optimized parameters.

        Args:
            strategy_class: Strategy class to validate
            data: Primary asset data (used for reference)

        Returns:
            Empty dict (validation results stored separately)
        """
        self.logger.warning(
            "Cross-asset validation requires pre-optimized parameters. "
            "Use validate_strategy() method instead."
        )
        return {}

    def validate_strategy(
        self,
        strategy_class,
        optimized_params: Dict[str, Any],
        primary_asset: str,
        primary_data: pd.DataFrame,
    ) -> CrossAssetValidationResult:
        """
        Validate an optimized strategy across multiple assets.

        Args:
            strategy_class: Strategy class to validate
            optimized_params: Optimized parameters from primary asset
            primary_asset: Primary asset symbol
            primary_data: Primary asset historical data

        Returns:
            Complete cross-asset validation results
        """
        start_time = time.time()
        self.logger.info(
            f"Starting Cross-Asset Validation for {strategy_class.__name__}"
        )
        self.logger.info(f"Primary asset: {primary_asset}")
        self.logger.info(f"Optimized parameters: {optimized_params}")

        # Initialize data fetcher if needed
        if self.data_fetcher is None:
            self._initialize_data_fetcher()

        # Select validation assets
        validation_assets = self.asset_selector.select_validation_assets(
            primary_asset, self.data_fetcher
        )

        if not validation_assets:
            self.logger.error("No validation assets selected")
            return self._create_empty_result(strategy_class.__name__, primary_asset)

        # Evaluate strategy on primary asset
        primary_metrics = self._evaluate_strategy_on_asset(
            strategy_class, optimized_params, primary_asset, primary_data
        )

        # Validate on each validation asset
        if self.parallel_validation:
            self.asset_results = self._validate_assets_parallel(
                strategy_class, optimized_params, validation_assets, primary_metrics
            )
        else:
            self.asset_results = self._validate_assets_sequential(
                strategy_class, optimized_params, validation_assets, primary_metrics
            )

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()

        # Evaluate overall results
        pass_rate, overall_pass = self.validation_criteria.evaluate_overall(
            self.asset_results
        )
        robustness_score = self._calculate_robustness_score()

        # Create result
        total_time = time.time() - start_time
        result = CrossAssetValidationResult(
            strategy_name=strategy_class.__name__,
            primary_asset=primary_asset,
            validation_assets=validation_assets,
            asset_results=self.asset_results,
            aggregate_metrics=self.aggregate_metrics,
            pass_rate=pass_rate,
            overall_pass=overall_pass,
            robustness_score=robustness_score,
            timestamp=datetime.now(),
            total_time=total_time,
        )

        # Save results
        self._save_results(result)

        # Log final results
        self._log_validation_results(result)

        self.logger.info(f"Cross-Asset Validation completed in {total_time:.2f}s")
        self.logger.info(f"Pass rate: {pass_rate:.1%}, Overall pass: {overall_pass}")

        return result

    def _initialize_data_fetcher(self) -> None:
        """Initialize data fetcher for validation assets."""
        try:
            # Use the already imported DataFetcher if available, otherwise import it
            if DataFetcher is not None:
                self.data_fetcher = DataFetcher(self.data_fetcher_config)
            else:
                from data.data_fetcher import DataFetcher as DataFetcherClass

                self.data_fetcher = DataFetcherClass(self.data_fetcher_config)
            self.logger.info("Data fetcher initialized for cross-asset validation")
        except Exception as e:
            self.logger.error(f"Failed to initialize data fetcher: {e}")
            self.data_fetcher = None

    def _validate_assets_sequential(
        self,
        strategy_class,
        optimized_params: Dict[str, Any],
        validation_assets: List[ValidationAsset],
        primary_metrics: Dict[str, Any],
    ) -> List[AssetValidationResult]:
        """Validate strategy on assets sequentially."""
        results = []

        for asset in validation_assets:
            self.logger.info(f"Validating on {asset.symbol} ({asset.name})")
            result = self._validate_single_asset(
                strategy_class, optimized_params, asset, primary_metrics
            )
            results.append(result)

        return results

    def _validate_assets_parallel(
        self,
        strategy_class,
        optimized_params: Dict[str, Any],
        validation_assets: List[ValidationAsset],
        primary_metrics: Dict[str, Any],
    ) -> List[AssetValidationResult]:
        """Validate strategy on assets in parallel."""
        # For simplicity, using ThreadPoolExecutor
        # In production, consider using ProcessPoolExecutor for CPU-intensive tasks
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        with ThreadPoolExecutor(max_workers=min(4, len(validation_assets))) as executor:
            futures = [
                executor.submit(
                    self._validate_single_asset,
                    strategy_class,
                    optimized_params,
                    asset,
                    primary_metrics,
                )
                for asset in validation_assets
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Asset validation failed: {e}")

        return results

    async def _validate_single_asset_async(
        self,
        strategy_class,
        optimized_params: Dict[str, Any],
        asset: ValidationAsset,
        primary_metrics: Dict[str, Any],
    ) -> AssetValidationResult:
        """Validate strategy on a single asset (async version)."""
        start_time = time.time()

        try:
            # Fetch asset data
            asset_data = await self.data_fetcher.get_historical_data(
                asset.symbol, asset.timeframe, asset.required_history
            )

            if asset_data.empty:
                raise ValueError(f"No data available for {asset.symbol}")

            # Evaluate strategy on this asset
            validation_metrics = self._evaluate_strategy_on_asset(
                strategy_class, optimized_params, asset.symbol, asset_data
            )

            # Evaluate pass criteria
            pass_criteria, overall_pass = self.validation_criteria.evaluate_asset(
                primary_metrics, validation_metrics
            )

            validation_time = time.time() - start_time

            return AssetValidationResult(
                asset=asset,
                optimized_params=optimized_params,
                primary_metrics=primary_metrics,
                validation_metrics=validation_metrics,
                pass_criteria=pass_criteria,
                overall_pass=overall_pass,
                validation_time=validation_time,
            )

        except Exception as e:
            self.logger.error(f"Validation failed for {asset.symbol}: {e}")

            validation_time = time.time() - start_time
            return AssetValidationResult(
                asset=asset,
                optimized_params=optimized_params,
                primary_metrics=primary_metrics,
                validation_metrics={},
                pass_criteria={},
                overall_pass=False,
                validation_time=validation_time,
                error_message=str(e),
            )

    def _validate_single_asset(
        self,
        strategy_class,
        optimized_params: Dict[str, Any],
        asset: ValidationAsset,
        primary_metrics: Dict[str, Any],
    ) -> AssetValidationResult:
        """Validate strategy on a single asset."""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are in an event loop, create a task instead of using asyncio.run
            task = loop.create_task(
                self._validate_single_asset_async(
                    strategy_class, optimized_params, asset, primary_metrics
                )
            )
            # Wait for the task to complete
            return loop.run_until_complete(task)
        except RuntimeError:
            # No running event loop, we can use asyncio.run
            try:
                return asyncio.run(
                    self._validate_single_asset_async(
                        strategy_class, optimized_params, asset, primary_metrics
                    )
                )
            except Exception as e:
                self.logger.error(
                    f"Async validation failed for {asset.symbol}, falling back to sync: {e}"
                )
                # Return failed result
                return AssetValidationResult(
                    asset=asset,
                    optimized_params=optimized_params,
                    primary_metrics=primary_metrics,
                    validation_metrics={},
                    pass_criteria={},
                    overall_pass=False,
                    validation_time=0.0,
                    error_message=f"Async validation failed: {str(e)}",
                )

    def _evaluate_strategy_on_asset(
        self,
        strategy_class,
        params: Dict[str, Any],
        asset_symbol: str,
        asset_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Evaluate strategy with given parameters on asset data."""
        try:
            # Create strategy instance
            strategy_config = {
                "name": f'cross_asset_validation_{asset_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                "symbols": [asset_symbol],
                "timeframe": "1h",
                "required_history": 100,
                "params": params,
            }

            strategy_instance = strategy_class(strategy_config)

            # Run evaluation
            fitness = self.evaluate_fitness(strategy_instance, asset_data)

            # Get detailed metrics
            equity_progression = self._run_backtest(strategy_instance, asset_data)
            if equity_progression:
                metrics = compute_backtest_metrics(equity_progression)
                metrics["fitness"] = fitness
                return metrics
            else:
                return {"fitness": fitness, "error": "No equity progression"}

        except Exception as e:
            self.logger.error(f"Strategy evaluation failed for {asset_symbol}: {e}")
            return {"error": str(e), "fitness": float("-inf")}

    def _calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics across all validation assets."""
        if not self.asset_results:
            return

        # Collect all validation metrics
        validation_sharpes = []
        validation_returns = []
        validation_win_rates = []
        validation_max_drawdowns = []
        validation_profit_factors = []

        for result in self.asset_results:
            if not result.error_message:  # Only include successful validations
                metrics = result.validation_metrics
                validation_sharpes.append(metrics.get("sharpe_ratio", 0))
                validation_returns.append(metrics.get("total_return", 0))
                validation_win_rates.append(metrics.get("win_rate", 0))
                validation_max_drawdowns.append(metrics.get("max_drawdown", 0))
                validation_profit_factors.append(metrics.get("profit_factor", 0))

        # Calculate aggregate statistics
        self.aggregate_metrics = {
            "total_validation_assets": len(self.asset_results),
            "successful_validations": len(validation_sharpes),
            "avg_validation_sharpe": np.mean(validation_sharpes)
            if validation_sharpes
            else 0,
            "std_validation_sharpe": np.std(validation_sharpes)
            if validation_sharpes
            else 0,
            "avg_validation_return": np.mean(validation_returns)
            if validation_returns
            else 0,
            "avg_validation_win_rate": np.mean(validation_win_rates)
            if validation_win_rates
            else 0,
            "avg_validation_max_drawdown": np.mean(validation_max_drawdowns)
            if validation_max_drawdowns
            else 0,
            "avg_validation_profit_factor": np.mean(validation_profit_factors)
            if validation_profit_factors
            else 0,
            "sharpe_ratio_range": self._calculate_range(validation_sharpes),
            "return_consistency": self._calculate_consistency(validation_returns),
        }

    def _calculate_range(self, values: List[float]) -> Dict[str, float]:
        """Calculate range statistics for a list of values."""
        if not values:
            return {"min": 0, "max": 0, "range": 0}

        return {
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency metric for a list of values."""
        if len(values) < 2:
            return 0.0

        # Count values with same sign as mean
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0

        consistent_count = sum(1 for v in values if (v >= 0) == (mean_val >= 0))
        return consistent_count / len(values)

    def _calculate_robustness_score(self) -> float:
        """Calculate overall robustness score for the validation."""
        if not self.asset_results:
            return 0.0

        # Component scores
        pass_rate = sum(1 for r in self.asset_results if r.overall_pass) / len(
            self.asset_results
        )

        # Sharpe consistency (lower std is better)
        sharpes = [
            r.validation_metrics.get("sharpe_ratio", 0)
            for r in self.asset_results
            if not r.error_message
        ]
        sharpe_consistency = 1.0 / (1.0 + np.std(sharpes)) if sharpes else 0.0

        # Average Sharpe quality
        avg_sharpe = np.mean(sharpes) if sharpes else 0.0
        sharpe_quality = max(0, min(1, (avg_sharpe + 1) / 2))  # Scale -1 to +1 to 0-1

        # Weighted robustness score
        robustness = 0.4 * pass_rate + 0.3 * sharpe_consistency + 0.3 * sharpe_quality

        return robustness

    def _create_empty_result(
        self, strategy_name: str, primary_asset: str
    ) -> CrossAssetValidationResult:
        """Create empty result when validation fails."""
        return CrossAssetValidationResult(
            strategy_name=strategy_name,
            primary_asset=primary_asset,
            validation_assets=[],
            asset_results=[],
            aggregate_metrics={},
            pass_rate=0.0,
            overall_pass=False,
            robustness_score=0.0,
            timestamp=datetime.now(),
            total_time=0.0,
        )

    def _save_results(self, result: CrossAssetValidationResult) -> None:
        """Save validation results to files."""
        # Save detailed results
        detailed_path = os.path.join(
            self.output_dir, "cross_asset_validation_results.json"
        )
        with open(detailed_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Save summary report
        summary_path = os.path.join(
            self.output_dir, "cross_asset_validation_summary.json"
        )
        summary = {
            "strategy_name": result.strategy_name,
            "primary_asset": result.primary_asset,
            "total_validation_assets": len(result.validation_assets),
            "pass_rate": result.pass_rate,
            "overall_pass": result.overall_pass,
            "robustness_score": result.robustness_score,
            "aggregate_metrics": result.aggregate_metrics,
            "timestamp": result.timestamp.isoformat(),
            "total_time": result.total_time,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save CSV summary for easy analysis
        self._save_csv_summary(result)

        self.logger.info(f"Cross-asset validation results saved to {self.output_dir}")

    def _save_csv_summary(self, result: CrossAssetValidationResult) -> None:
        """Save CSV summary of validation results."""
        csv_path = os.path.join(self.output_dir, "cross_asset_validation_summary.csv")

        summary_records = []
        for asset_result in result.asset_results:
            summary_records.append(
                {
                    "strategy_name": result.strategy_name,
                    "asset_symbol": asset_result.asset.symbol,
                    "asset_name": asset_result.asset.name,
                    "sharpe_ratio": asset_result.validation_metrics.get("sharpe_ratio"),
                    "win_rate": asset_result.validation_metrics.get("win_rate"),
                    "overall_pass": bool(asset_result.overall_pass),
                    "validation_time": asset_result.validation_time,
                }
            )

        df = pd.DataFrame(summary_records)
        df.to_csv(csv_path, index=False)

    def _log_validation_results(self, result: CrossAssetValidationResult) -> None:
        """Log comprehensive validation results."""
        self.logger.info("=" * 60)
        self.logger.info("CROSS-ASSET VALIDATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Strategy: {result.strategy_name}")
        self.logger.info(f"Primary Asset: {result.primary_asset}")
        self.logger.info(f"Validation Assets: {len(result.validation_assets)}")
        self.logger.info(f"Pass Rate: {result.pass_rate:.1%}")
        self.logger.info(f"Overall Pass: {result.overall_pass}")
        self.logger.info(f"Robustness Score: {result.robustness_score:.3f}")
        self.logger.info(f"Total Time: {result.total_time:.2f}s")
        self.logger.info("")

        # Log individual asset results
        for asset_result in result.asset_results:
            status = "✅ PASS" if asset_result.overall_pass else "❌ FAIL"
            self.logger.info(
                f"{asset_result.asset.symbol} ({asset_result.asset.name}): {status}"
            )

            if asset_result.error_message:
                self.logger.info(f"  Error: {asset_result.error_message}")
            else:
                # Log key metrics
                val_metrics = asset_result.validation_metrics
                self.logger.info(f"  Sharpe: {val_metrics.get('sharpe_ratio', 0):.3f}")
                self.logger.info(f"  Return: {val_metrics.get('total_return', 0):.1%}")
                self.logger.info(f"  Win Rate: {val_metrics.get('win_rate', 0):.1%}")
                self.logger.info(f"  Max DD: {val_metrics.get('max_drawdown', 0):.1%}")

                # Log pass criteria
                criteria_status = []
                for criterion, passed in asset_result.pass_criteria.items():
                    status_icon = "✅" if passed else "❌"
                    criteria_status.append(f"{status_icon}{criterion}")
                self.logger.info(f"  Criteria: {' '.join(criteria_status)}")

            self.logger.info("")

        self.logger.info("=" * 60)


# Convenience functions for easy integration
def create_cross_asset_validator(
    config: Optional[Dict[str, Any]] = None
) -> CrossAssetValidator:
    """
    Create a cross-asset validator with default configuration.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured CrossAssetValidator instance
    """
    default_config = {
        "asset_selector": {
            "max_assets": 3,
            "asset_weights": "equal",
            "correlation_filter": False,
            "max_correlation": 0.7,
            "validation_assets": [
                {
                    "symbol": "ETH/USDT",
                    "name": "Ethereum",
                    "weight": 1.0,
                    "required_history": 1000,
                    "timeframe": "1h",
                },
                {
                    "symbol": "ADA/USDT",
                    "name": "Cardano",
                    "weight": 0.8,
                    "required_history": 1000,
                    "timeframe": "1h",
                },
                {
                    "symbol": "SOL/USDT",
                    "name": "Solana",
                    "weight": 0.6,
                    "required_history": 1000,
                    "timeframe": "1h",
                },
            ],
        },
        "validation_criteria": {
            "min_sharpe_ratio": 0.5,
            "max_drawdown_limit": 0.15,
            "min_win_rate": 0.45,
            "min_profit_factor": 1.2,
            "consistency_threshold": 0.7,
            "required_pass_rate": 0.6,
        },
        "data_fetcher": {"name": "binance", "cache_enabled": True},
        "output_dir": "results/cross_asset_validation",
        "parallel_validation": False,
    }

    if config:
        # Deep merge configurations
        def merge_dicts(
            base: Dict[str, Any], override: Dict[str, Any]
        ) -> Dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        default_config = merge_dicts(default_config, config)

    return CrossAssetValidator(default_config)


def run_cross_asset_validation(
    strategy_class,
    optimized_params: Dict[str, Any],
    primary_asset: str,
    primary_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> CrossAssetValidationResult:
    """
    Run complete cross-asset validation.

    Args:
        strategy_class: Strategy class to validate
        optimized_params: Optimized parameters from primary asset
        primary_asset: Primary asset symbol
        primary_data: Primary asset historical data
        config: Optional configuration

    Returns:
        Complete cross-asset validation results
    """
    validator = create_cross_asset_validator(config)
    return validator.validate_strategy(
        strategy_class, optimized_params, primary_asset, primary_data
    )
