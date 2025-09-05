#!/usr/bin/env python3
"""
Training script for predictive models.

This script trains all predictive models using historical data and sliding window cross-validation.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys
from datetime import datetime

from predictive_models import PredictiveModelManager
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def load_historical_data(data_path: str, symbol: str = None) -> pd.DataFrame:
    """
    Load historical OHLCV data for training.

    Args:
        data_path: Path to historical data file
        symbol: Optional symbol filter

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading historical data from {data_path}")

    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            required_cols.append('volume')

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp if needed
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'object':
                # Try to parse timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9  # Convert to seconds

        # Filter by symbol if specified
        if symbol and 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].copy()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows of historical data")
        return df

    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        raise


def prepare_training_data(df: pd.DataFrame, min_samples: int = 1000) -> pd.DataFrame:
    """
    Prepare and validate training data.

    Args:
        df: Raw historical data
        min_samples: Minimum required samples

    Returns:
        Prepared DataFrame
    """
    logger.info("Preparing training data")

    # Basic validation
    if len(df) < min_samples:
        raise ValueError(f"Insufficient training data: {len(df)} < {min_samples}")

    # Ensure numeric columns
    numeric_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        numeric_cols.append('volume')

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with NaN values in essential columns
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # Remove rows with zero or negative prices
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    # Remove outliers (optional)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # Remove values that are more than 3 standard deviations from mean
            mean_val = df[col].mean()
            std_val = df[col].std()
            df = df[(df[col] >= mean_val - 3 * std_val) & (df[col] <= mean_val + 3 * std_val)]

    logger.info(f"Prepared {len(df)} samples for training")
    return df


def save_training_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save training results to file.

    Args:
        results: Training results dictionary
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Training results saved to {output_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train predictive models')
    parser.add_argument('--config', '-c', type=str, default='config.json',
                        help='Configuration file path')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Historical data file path')
    parser.add_argument('--symbol', '-s', type=str,
                        help='Symbol to train on (optional)')
    parser.add_argument('--output', '-o', type=str, default='training_results.json',
                        help='Output file for training results')
    parser.add_argument('--min-samples', type=int, default=1000,
                        help='Minimum number of samples required')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    try:
        args = parser.parse_args()

        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Check if predictive models are enabled before loading data
        predictive_config = config.get('predictive_models', {})
        if not predictive_config.get('enabled', False):
            logger.warning("Predictive models are disabled in config")
            return

        # Load historical data
        df = load_historical_data(args.data, args.symbol)

        # Prepare training data
        df = prepare_training_data(df, args.min_samples)

        manager = PredictiveModelManager(predictive_config)

        # Train models
        logger.info("Starting model training...")
        start_time = datetime.now()

        training_results = manager.train_models(df)

        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        # Add metadata to results
        training_results.update({
            "training_metadata": {
                "timestamp": end_time.isoformat(),
                "duration_seconds": training_duration,
                "data_samples": len(df),
                "data_file": args.data,
                "symbol": args.symbol,
                "config_file": args.config
            }
        })

        # Save results
        save_training_results(training_results, args.output)

        # Log summary
        logger.info("="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Duration: {training_duration:.2f} seconds")
        logger.info(f"Data samples: {len(df)}")
        logger.info(f"Status: {training_results.get('status', 'unknown')}")

        if training_results.get('status') == 'success':
            logger.info("Model Results:")
            for model_name, results in training_results.items():
                if model_name not in ['status', 'training_metadata']:
                    if isinstance(results, dict) and 'final_accuracy' in results:
                        logger.info(f"  {model_name}: {results['final_accuracy']:.3f}")
                    elif isinstance(results, dict) and 'final_r2' in results:
                        logger.info(f"  {model_name}: {results['final_r2']:.3f}")
                    elif isinstance(results, dict) and 'model_type' in results:
                        logger.info(f"  {model_name}: {results.get('model_type', 'unknown')} configured")

        logger.info(f"Results saved to: {args.output}")
        logger.info("="*50)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
