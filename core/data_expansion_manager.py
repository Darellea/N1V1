"""
Data Expansion Manager - Multi-pair and Multi-timeframe Data Collection System

This module implements comprehensive data expansion capabilities for the N1V1 trading system,
including multi-pair data collection, multi-timeframe support, automated data pulling,
and high-volatility period data collection.
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class DataExpansionManager:
    """
    Comprehensive data expansion system for collecting multi-pair, multi-timeframe data.

    Features:
    - Multi-pair data collection (EUR/USD, GBP/USD, USD/JPY, etc.)
    - Multi-timeframe support (1H, 4H, 1D)
    - Automated daily data pulling
    - High-volatility period data collection
    - Data validation and cleansing
    - ETL pipeline management
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Data Expansion Manager.

        Args:
            config: Configuration dictionary containing:
                - data_sources: List of data source configurations
                - target_pairs: List of currency pairs to collect
                - timeframes: List of timeframes to collect
                - data_dir: Directory to store collected data
                - min_samples_per_pair: Minimum samples per pair
                - volatility_periods: High-volatility periods to collect
        """
        self.config = config
        self.data_sources = config.get("data_sources", [])
        self.target_pairs = config.get(
            "target_pairs", ["EUR/USD", "GBP/USD", "USD/JPY"]
        )
        self.timeframes = config.get("timeframes", ["1h", "4H"])
        self.data_dir = Path(config.get("data_dir", "historical_data"))
        self.min_samples_per_pair = config.get("min_samples_per_pair", 1000)

        # High-volatility periods (COVID-19, 2008 crisis, etc.)
        self.volatility_periods = config.get(
            "volatility_periods",
            [
                {"start": "2020-03-01", "end": "2020-06-30", "name": "COVID-19"},
                {"start": "2008-09-01", "end": "2009-03-31", "name": "2008_Crisis"},
                {"start": "2016-06-01", "end": "2016-12-31", "name": "Brexit"},
                {"start": "2022-02-01", "end": "2022-05-31", "name": "Ukraine_Crisis"},
            ],
        )

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data collection state
        self.collection_status = {}
        self.last_collection_time = {}

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("Data Expansion Manager initialized")

    async def collect_multi_pair_data(
        self, target_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Collect multi-pair data to reach target sample count.

        Args:
            target_samples: Target number of samples per pair

        Returns:
            Collection results summary
        """
        # Input validation
        if not isinstance(target_samples, int) or target_samples <= 0:
            raise ValueError("target_samples must be a positive integer")
        if target_samples > 100000:  # Reasonable upper bound
            raise ValueError("target_samples cannot exceed 100,000")

        logger.info(
            f"Starting multi-pair data collection for {len(self.target_pairs)} pairs"
        )

        results = {
            "total_samples_collected": 0,
            "pairs_processed": [],
            "timeframes_processed": [],
            "collection_duration": 0,
            "errors": [],
        }

        start_time = time.time()

        try:
            # Collect data for each pair in parallel
            tasks = []
            for pair in self.target_pairs:
                for timeframe in self.timeframes:
                    task = self._collect_pair_timeframe_data(
                        pair, timeframe, target_samples
                    )
                    tasks.append(task)

            # Execute all collection tasks
            collection_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(collection_results):
                pair_idx = i // len(self.timeframes)
                tf_idx = i % len(self.timeframes)
                pair = self.target_pairs[pair_idx]
                timeframe = self.timeframes[tf_idx]

                if isinstance(result, Exception):
                    error_msg = f"Failed to collect {pair} {timeframe}: {str(result)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                else:
                    results["pairs_processed"].append(f"{pair}_{timeframe}")
                    results["total_samples_collected"] += result.get(
                        "samples_collected", 0
                    )

            results["collection_duration"] = time.time() - start_time

            # Save collection summary
            await self._save_collection_summary(results)

            logger.info(
                f"Multi-pair data collection completed: {results['total_samples_collected']} total samples"
            )

        except Exception as e:
            logger.error(f"Multi-pair data collection failed: {e}")
            results["errors"].append(str(e))

        return results

    async def _collect_pair_timeframe_data(
        self, pair: str, timeframe: str, target_samples: int
    ) -> Dict[str, Any]:
        """Collect data for a specific pair and timeframe."""
        logger.info(f"Collecting {pair} {timeframe} data")

        result = {
            "pair": pair,
            "timeframe": timeframe,
            "samples_collected": 0,
            "data_quality_score": 0.0,
            "file_path": None,
        }

        try:
            # Collect data from all configured sources
            all_data = []
            for source in self.data_sources:
                source_data = await self._collect_from_source(
                    source, pair, timeframe, target_samples
                )
                if source_data is not None and not source_data.empty:
                    all_data.append(source_data)

            if not all_data:
                logger.warning(f"No data collected for {pair} {timeframe}")
                return result

            # Combine and clean data
            combined_data = self._combine_and_clean_data(all_data)

            if len(combined_data) < self.min_samples_per_pair:
                logger.warning(
                    f"Insufficient data for {pair} {timeframe}: {len(combined_data)} < {self.min_samples_per_pair}"
                )
                return result

            # Validate and enhance data
            validated_data = self._validate_and_enhance_data(
                combined_data, pair, timeframe
            )

            # Save data
            file_path = self._save_pair_data(validated_data, pair, timeframe)

            result.update(
                {
                    "samples_collected": len(validated_data),
                    "data_quality_score": self._calculate_data_quality_score(
                        validated_data
                    ),
                    "file_path": str(file_path),
                }
            )

            logger.info(
                f"Collected {len(validated_data)} samples for {pair} {timeframe}"
            )

        except Exception as e:
            logger.error(f"Error collecting {pair} {timeframe} data: {e}")
            raise

        return result

    async def _collect_from_source(
        self,
        source_config: Dict[str, Any],
        pair: str,
        timeframe: str,
        target_samples: int,
    ) -> Optional[pd.DataFrame]:
        """Collect data from a specific source."""
        source_type = source_config.get("type", "csv")

        try:
            if source_type == "csv":
                return await self._collect_from_csv(source_config, pair, timeframe)
            elif source_type == "api":
                return await self._collect_from_api(
                    source_config, pair, timeframe, target_samples
                )
            elif source_type == "database":
                return await self._collect_from_database(
                    source_config, pair, timeframe, target_samples
                )
            else:
                logger.warning(f"Unsupported data source type: {source_type}")
                return None

        except Exception as e:
            logger.error(f"Error collecting from {source_type} source: {e}")
            return None

    async def _collect_from_csv(
        self, source_config: Dict[str, Any], pair: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Collect data from CSV files."""
        file_pattern = source_config.get(
            "file_pattern", f"{pair.replace('/', '_')}_{timeframe}.csv"
        )
        search_dir = Path(source_config.get("directory", self.data_dir))

        # Find matching files
        matching_files = list(search_dir.glob(file_pattern))
        if not matching_files:
            return None

        # Load and combine data from all matching files
        all_data = []
        for file_path in matching_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        if not all_data:
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Ensure required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close"]
        if "volume" in combined_df.columns:
            required_cols.append("volume")

        missing_cols = [col for col in required_cols if col not in combined_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in CSV data: {missing_cols}")
            return None

        return combined_df

    async def _collect_from_api(
        self,
        source_config: Dict[str, Any],
        pair: str,
        timeframe: str,
        target_samples: int,
    ) -> Optional[pd.DataFrame]:
        """Collect data from REST API."""
        base_url = source_config.get("base_url")
        endpoint = source_config.get("endpoint", "/api/v1/klines")
        api_key = source_config.get("api_key")
        headers = source_config.get("headers", {})

        if api_key:
            headers["X-API-Key"] = api_key

        # Prepare API request parameters
        params = {
            "symbol": pair.replace("/", ""),
            "interval": timeframe.lower(),
            "limit": min(target_samples, 1000),  # API limits
        }

        # Add date range if specified
        if "start_time" in source_config:
            params["startTime"] = int(
                pd.Timestamp(source_config["start_time"]).timestamp() * 1000
            )
        if "end_time" in source_config:
            params["endTime"] = int(
                pd.Timestamp(source_config["end_time"]).timestamp() * 1000
            )

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{base_url}{endpoint}"
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Convert API response to DataFrame
                        df = self._convert_api_response_to_df(
                            data, source_config.get("response_format", "binance")
                        )
                        return df
                    else:
                        logger.error(
                            f"API request failed: {response.status} - {await response.text()}"
                        )
                        return None

        except Exception as e:
            logger.error(f"API collection error: {e}")
            return None

    async def _collect_from_database(
        self,
        source_config: Dict[str, Any],
        pair: str,
        timeframe: str,
        target_samples: int,
    ) -> Optional[pd.DataFrame]:
        """Collect data from database."""
        # Placeholder for database collection
        # This would implement database-specific collection logic
        logger.warning("Database collection not yet implemented")
        return None

    def _convert_api_response_to_df(
        self, data: List[Any], format_type: str
    ) -> pd.DataFrame:
        """Convert API response to standardized DataFrame format."""
        if format_type == "binance":
            # Binance API format: [timestamp, open, high, low, close, volume, ...]
            records = []
            for item in data:
                if len(item) >= 6:
                    record = {
                        "timestamp": pd.Timestamp.fromtimestamp(item[0] / 1000),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    }
                    records.append(record)

            return pd.DataFrame(records)

        elif format_type == "generic":
            # Generic OHLCV format
            return pd.DataFrame(data)

        else:
            logger.warning(f"Unsupported API response format: {format_type}")
            return pd.DataFrame()

    def _combine_and_clean_data(self, data_frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple data sources and clean the data."""
        if not data_frames:
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(data_frames, ignore_index=True)

        # Remove duplicates based on timestamp
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])

        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        # Clean numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

        # Remove rows with NaN in essential columns
        essential_cols = ["timestamp", "open", "high", "low", "close"]
        combined_df = combined_df.dropna(subset=essential_cols)

        # Remove invalid price data (negative or zero prices)
        combined_df = combined_df[
            (combined_df["open"] > 0)
            & (combined_df["high"] > 0)
            & (combined_df["low"] > 0)
            & (combined_df["close"] > 0)
        ]

        # Ensure high >= max(open, close) and low <= min(open, close)
        combined_df = combined_df[
            (combined_df["high"] >= combined_df[["open", "close"]].max(axis=1))
            & (combined_df["low"] <= combined_df[["open", "close"]].min(axis=1))
        ]

        return combined_df

    def _validate_and_enhance_data(
        self, df: pd.DataFrame, pair: str, timeframe: str
    ) -> pd.DataFrame:
        """Validate data quality and add enhancements."""
        if df.empty:
            return df

        # Add pair and timeframe columns
        df["pair"] = pair
        df["timeframe"] = timeframe

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Calculate additional validation metrics
        df["price_range"] = df["high"] - df["low"]
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Remove outliers based on price movements
        price_change = df["close"].pct_change().abs()
        outlier_threshold = price_change.quantile(0.99)  # 99th percentile
        df = df[price_change <= outlier_threshold * 2]  # Allow some tolerance

        return df

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if df.empty:
            return 0.0

        score = 1.0

        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_pct * 0.5

        # Check data completeness
        essential_cols = ["open", "high", "low", "close"]
        completeness = sum(1 for col in essential_cols if col in df.columns) / len(
            essential_cols
        )
        score *= completeness

        # Check for reasonable price movements
        if len(df) > 1:
            returns = df["close"].pct_change().dropna()
            extreme_returns = (returns.abs() > 0.1).sum() / len(returns)  # >10% moves
            score -= extreme_returns * 0.2

        return max(0.0, min(1.0, score))

    async def _save_pair_data(
        self, df: pd.DataFrame, pair: str, timeframe: str
    ) -> Path:
        """Save pair data to file asynchronously."""
        # Create filename
        pair_clean = pair.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{pair_clean}_{timeframe}_{timestamp}.csv"
        file_path = self.data_dir / filename

        # Save to CSV asynchronously
        csv_content = df.to_csv(index=False)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(csv_content)

        logger.info(f"Saved {len(df)} samples to {file_path}")

        return file_path

    async def _save_collection_summary(self, results: Dict[str, Any]):
        """Save collection summary to file asynchronously."""
        summary_file = self.data_dir / "collection_summary.json"

        # Add timestamp
        results["timestamp"] = datetime.now().isoformat()

        # Use orjson for faster JSON serialization if available
        try:
            import orjson

            json_content = orjson.dumps(results, option=orjson.OPT_INDENT_2).decode(
                "utf-8"
            )
        except ImportError:
            json_content = json.dumps(results, indent=2, default=str)

        async with aiofiles.open(summary_file, "w") as f:
            await f.write(json_content)

        logger.info(f"Collection summary saved to {summary_file}")

    async def collect_volatility_period_data(self) -> Dict[str, Any]:
        """Collect data from high-volatility periods."""
        logger.info("Starting high-volatility period data collection")

        results = {"periods_processed": [], "total_samples_collected": 0, "errors": []}

        for period in self.volatility_periods:
            try:
                period_data = await self._collect_single_volatility_period(period)
                results["periods_processed"].append(period["name"])
                results["total_samples_collected"] += period_data.get(
                    "samples_collected", 0
                )

                logger.info(
                    f"Collected {period_data.get('samples_collected', 0)} samples for {period['name']}"
                )

            except Exception as e:
                error_msg = f"Failed to collect {period['name']}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Save volatility data summary
        await self._save_volatility_summary(results)

        return results

    async def _collect_single_volatility_period(
        self, period: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect data for a single volatility period."""
        period_name = period["name"]
        start_date = period["start"]
        end_date = period["end"]

        logger.info(f"Collecting data for {period_name} ({start_date} to {end_date})")

        # Create source config for this period
        period_config = {"type": "api", "start_time": start_date, "end_time": end_date}

        # Collect data for all pairs and timeframes
        all_data = []
        for pair in self.target_pairs:
            for timeframe in self.timeframes:
                try:
                    data = await self._collect_from_source(
                        period_config, pair, timeframe, 5000
                    )
                    if data is not None and not data.empty:
                        data["volatility_period"] = period_name
                        all_data.append(data)
                except Exception as e:
                    logger.warning(
                        f"Error collecting {pair} {timeframe} for {period_name}: {e}"
                    )

        if not all_data:
            return {"samples_collected": 0}

        # Combine all period data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Clean and validate
        combined_data = self._combine_and_clean_data([combined_data])

        # Save period data asynchronously
        filename = f"volatility_{period_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
        file_path = self.data_dir / filename
        csv_content = combined_data.to_csv(index=False)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(csv_content)

        return {
            "samples_collected": len(combined_data),
            "file_path": str(file_path),
            "period_name": period_name,
        }

    async def _save_volatility_summary(self, results: Dict[str, Any]):
        """Save volatility data collection summary asynchronously."""
        summary_file = self.data_dir / "volatility_collection_summary.json"

        results["timestamp"] = datetime.now().isoformat()

        # Use orjson for faster JSON serialization if available
        try:
            import orjson

            json_content = orjson.dumps(results, option=orjson.OPT_INDENT_2).decode(
                "utf-8"
            )
        except ImportError:
            json_content = json.dumps(results, indent=2, default=str)

        async with aiofiles.open(summary_file, "w") as f:
            await f.write(json_content)

        logger.info(f"Volatility collection summary saved to {summary_file}")

    async def setup_automated_collection(self, schedule_config: Dict[str, Any]):
        """Set up automated daily data collection."""
        logger.info("Setting up automated data collection")

        # This would integrate with a scheduler (cron, APScheduler, etc.)
        # For now, we'll create a configuration file

        schedule_file = self.data_dir / "collection_schedule.json"

        schedule_config.update(
            {
                "created_at": datetime.now().isoformat(),
                "target_pairs": self.target_pairs,
                "timeframes": self.timeframes,
                "data_dir": str(self.data_dir),
            }
        )

        with open(schedule_file, "w") as f:
            json.dump(schedule_config, f, indent=2)

        logger.info(f"Automated collection schedule saved to {schedule_file}")

        # Create a simple cron job example
        self._create_cron_job_example(schedule_config)

    def _create_cron_job_example(self, schedule_config: Dict[str, Any]):
        """Create an example cron job for automated collection."""
        cron_file = self.data_dir / "cron_job_example.sh"

        cron_content = f"""#!/bin/bash
# Automated Data Collection Cron Job
# Run this script daily at 2 AM

export PYTHONPATH="/path/to/n1v1:$PYTHONPATH"

# Run data collection
python -c "
import asyncio
from core.data_expansion_manager import DataExpansionManager

async def main():
    config = {schedule_config}
    manager = DataExpansionManager(config)

    # Collect new data
    results = await manager.collect_multi_pair_data(target_samples=1000)
    print(f'Collected {{results[\"total_samples_collected\"]}} samples')

    # Update models if new data is available
    if results['total_samples_collected'] > 0:
        print('New data collected, triggering model updates...')

asyncio.run(main())
"

echo 'Data collection completed at $(date)' >> /var/log/n1v1_data_collection.log
"""

        with open(cron_file, "w") as f:
            f.write(cron_content)

        # Make executable
        os.chmod(cron_file, 0o755)

        logger.info(f"Cron job example created at {cron_file}")

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current data collection status."""
        status = {
            "data_directory": str(self.data_dir),
            "target_pairs": self.target_pairs,
            "timeframes": self.timeframes,
            "collection_status": self.collection_status,
            "last_collection_time": self.last_collection_time,
        }

        # Check existing data files
        data_files = list(self.data_dir.glob("*.csv"))
        status["existing_data_files"] = len(data_files)
        status["total_samples_estimated"] = self._estimate_total_samples(data_files)

        return status

    def _estimate_total_samples(self, data_files: List[Path]) -> int:
        """Estimate total samples from existing data files."""
        total_samples = 0

        for file_path in data_files[:10]:  # Sample first 10 files
            try:
                # Quick row count without loading full file
                with open(file_path, "r") as f:
                    total_samples += sum(1 for _ in f) - 1  # Subtract header
            except Exception:
                continue

        # Extrapolate for all files
        if len(data_files) > 10:
            total_samples = int(total_samples * len(data_files) / 10)

        return total_samples

    async def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate integrity of collected data."""
        logger.info("Starting data integrity validation")

        validation_results = {
            "total_files_checked": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "total_samples": 0,
            "issues_found": [],
        }

        # Check all CSV files in data directory
        data_files = list(self.data_dir.glob("*.csv"))

        for file_path in data_files:
            validation_results["total_files_checked"] += 1

            try:
                df = pd.read_csv(file_path)

                if df.empty:
                    validation_results["issues_found"].append(
                        f"{file_path.name}: Empty file"
                    )
                    validation_results["invalid_files"] += 1
                    continue

                # Check required columns
                required_cols = ["timestamp", "open", "high", "low", "close"]
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    validation_results["issues_found"].append(
                        f"{file_path.name}: Missing columns {missing_cols}"
                    )
                    validation_results["invalid_files"] += 1
                    continue

                # Check data quality
                quality_issues = self._check_data_quality(df)
                if quality_issues:
                    validation_results["issues_found"].extend(
                        [f"{file_path.name}: {issue}" for issue in quality_issues]
                    )
                    validation_results["invalid_files"] += 1
                else:
                    validation_results["valid_files"] += 1

                validation_results["total_samples"] += len(df)

            except Exception as e:
                validation_results["issues_found"].append(
                    f"{file_path.name}: Error reading file - {str(e)}"
                )
                validation_results["invalid_files"] += 1

        logger.info(
            f"Data integrity validation completed: {validation_results['valid_files']}/{validation_results['total_files_checked']} files valid"
        )

        return validation_results

    def _check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check data quality issues."""
        issues = []

        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            issues.append(f"Contains {nan_counts.sum()} NaN values")

        # Check for negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(f"{negative_count} negative/zero values in {col}")

        # Check OHLC relationships
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            invalid_ohlc = (
                (df["high"] < df[["open", "close"]].max(axis=1))
                | (df["low"] > df[["open", "close"]].min(axis=1))
            ).sum()
            if invalid_ohlc > 0:
                issues.append(f"{invalid_ohlc} invalid OHLC relationships")

        return issues


# Convenience functions
def create_data_expansion_manager(config: Dict[str, Any]) -> DataExpansionManager:
    """Create a DataExpansionManager instance."""
    return DataExpansionManager(config)


async def run_data_collection(
    config: Dict[str, Any], target_samples: int = 5000
) -> Dict[str, Any]:
    """Run data collection with the given configuration."""
    manager = DataExpansionManager(config)

    # Collect multi-pair data
    results = await manager.collect_multi_pair_data(target_samples)

    # Collect volatility period data
    volatility_results = await manager.collect_volatility_period_data()

    # Combine results
    combined_results = {
        "regular_collection": results,
        "volatility_collection": volatility_results,
        "total_samples": results["total_samples_collected"]
        + volatility_results["total_samples_collected"],
    }

    return combined_results


if __name__ == "__main__":
    # Example usage
    import asyncio

    config = {
        "data_sources": [
            {"type": "csv", "directory": "historical_data", "file_pattern": "*.csv"}
        ],
        "target_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"],
        "timeframes": ["1h", "4H"],
        "data_dir": "historical_data",
        "min_samples_per_pair": 1000,
    }

    async def main():
        results = await run_data_collection(config, target_samples=5000)
        print(f"Data collection completed: {results['total_samples']} total samples")

    asyncio.run(main())
