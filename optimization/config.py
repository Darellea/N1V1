"""
Centralized Configuration for Optimization Module

This module contains all configuration parameters for the optimization components,
replacing hard-coded values throughout the codebase. This improves maintainability,
flexibility, and makes parameters easy to tune without code changes.

Configuration Categories:
- Cross-asset validation parameters
- Strategy generation parameters
- Genetic algorithm parameters
- Bayesian optimization parameters
- Distributed evaluation parameters
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import os


@dataclass
class ValidationCriteriaConfig:
    """Configuration for cross-asset validation criteria."""

    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_limit: float = 0.15  # 15%
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2

    # Consistency and robustness
    consistency_threshold: float = 0.7
    required_pass_rate: float = 0.6  # 60%

    # Advanced metrics
    min_calmar_ratio: float = 0.3
    max_volatility: float = 0.25
    min_sortino_ratio: float = 0.4


@dataclass
class ValidationAssetsConfig:
    """Configuration for validation assets."""

    # Asset selection parameters
    max_assets: int = 3
    asset_weights: str = 'equal'  # 'equal', 'market_cap', 'custom'
    correlation_filter: bool = False
    max_correlation: float = 0.7

    # Default validation assets
    validation_assets: List[Dict[str, Any]] = field(default_factory=lambda: [
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
    ])

    # Market cap configuration
    market_cap_weights: Dict[str, float] = field(default_factory=dict)
    coinmarketcap_api_key: Optional[str] = None


@dataclass
class DataFetcherConfig:
    """Configuration for data fetching in validation."""

    # Data source configuration
    name: str = 'binance'
    cache_enabled: bool = True
    cache_dir: str = 'data/cache'

    # API rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

    # Data quality settings
    min_data_points: int = 100
    max_missing_data_pct: float = 0.05


@dataclass
class CrossAssetValidationConfig:
    """Complete configuration for cross-asset validation."""

    # Core components
    validation_criteria: ValidationCriteriaConfig = field(default_factory=ValidationCriteriaConfig)
    asset_selector: ValidationAssetsConfig = field(default_factory=ValidationAssetsConfig)
    data_fetcher: DataFetcherConfig = field(default_factory=DataFetcherConfig)

    # Execution parameters
    output_dir: str = 'results/cross_asset_validation'
    parallel_validation: bool = False
    max_parallel_workers: int = 4

    # Logging and monitoring
    log_level: str = 'INFO'
    save_detailed_results: bool = True
    save_csv_summary: bool = True


@dataclass
class StrategyGenerationConfig:
    """Configuration for strategy generation parameters."""

    # Population parameters
    population_size: int = 50
    generations: int = 20
    elitism_rate: float = 0.1

    # Genetic algorithm parameters
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    speciation_threshold: float = 0.3

    # Genome structure
    max_genes_per_genome: int = 10
    min_genes_per_genome: int = 3

    # Fitness evaluation
    fitness_metric: str = 'sharpe_ratio'
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 1.0,
        'total_return': 0.3,
        'win_rate': 0.2,
        'max_drawdown': -0.1
    })


@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""

    enabled: bool = True
    min_observations_for_training: int = 5
    acquisition_function: str = 'expected_improvement'  # 'expected_improvement', 'upper_confidence_bound'

    # Gaussian Process parameters
    gp_kernel: str = 'rbf'  # 'rbf', 'matern'
    gp_alpha: float = 1e-6
    gp_normalize_y: bool = True

    # Acquisition function parameters
    ei_kappa: float = 1.96  # for UCB
    ei_xi: float = 0.01     # for EI


@dataclass
class DistributedEvaluationConfig:
    """Configuration for distributed evaluation."""

    enabled: bool = True
    max_workers: int = 4
    use_processes: bool = False  # True for ProcessPool, False for ThreadPool

    # Resource management
    worker_timeout: int = 300  # seconds
    max_retries: int = 3

    # Caching
    cache_enabled: bool = True
    cache_max_size: int = 1000


@dataclass
class IndicatorParametersConfig:
    """Configuration for technical indicator parameters."""

    rsi: Dict[str, Any] = field(default_factory=lambda: {
        'period': {'min': 2, 'max': 50, 'default': 14},
        'overbought': {'min': 50, 'max': 95, 'default': 70},
        'oversold': {'min': 5, 'max': 50, 'default': 30}
    })

    macd: Dict[str, Any] = field(default_factory=lambda: {
        'fast_period': {'min': 5, 'max': 20, 'default': 12},
        'slow_period': {'min': 20, 'max': 50, 'default': 26},
        'signal_period': {'min': 5, 'max': 15, 'default': 9}
    })

    bollinger_bands: Dict[str, Any] = field(default_factory=lambda: {
        'period': {'min': 10, 'max': 50, 'default': 20},
        'std_dev': {'min': 1.0, 'max': 3.0, 'default': 2.0}
    })

    stochastic: Dict[str, Any] = field(default_factory=lambda: {
        'k_period': {'min': 5, 'max': 30, 'default': 14},
        'd_period': {'min': 3, 'max': 10, 'default': 3},
        'overbought': {'min': 70, 'max': 95, 'default': 80},
        'oversold': {'min': 5, 'max': 30, 'default': 20}
    })

    moving_average: Dict[str, Any] = field(default_factory=lambda: {
        'period': {'min': 5, 'max': 100, 'default': 20},
        'type': {'allowed_values': ['sma', 'ema', 'wma'], 'default': 'sma'}
    })

    atr: Dict[str, Any] = field(default_factory=lambda: {
        'period': {'min': 5, 'max': 30, 'default': 14}
    })

    volume: Dict[str, Any] = field(default_factory=lambda: {
        'period': {'min': 5, 'max': 50, 'default': 20}
    })

    price_action: Dict[str, Any] = field(default_factory=lambda: {
        'lookback': {'min': 3, 'max': 20, 'default': 5}
    })


@dataclass
class SignalLogicParametersConfig:
    """Configuration for signal logic parameters."""

    crossover: Dict[str, Any] = field(default_factory=lambda: {
        'fast_period': {'min': 5, 'max': 20, 'default': 9},
        'slow_period': {'min': 20, 'max': 50, 'default': 21}
    })

    threshold: Dict[str, Any] = field(default_factory=lambda: {
        'threshold': {'min': 0.1, 'max': 0.9, 'default': 0.5},
        'direction': {'allowed_values': ['above', 'below'], 'default': 'above'}
    })

    pattern: Dict[str, Any] = field(default_factory=lambda: {
        'pattern_type': {'allowed_values': ['double_bottom', 'double_top', 'head_shoulders'], 'default': 'double_bottom'},
        'tolerance': {'min': 0.01, 'max': 0.1, 'default': 0.02}
    })

    divergence: Dict[str, Any] = field(default_factory=lambda: {
        'lookback': {'min': 5, 'max': 20, 'default': 10},
        'threshold': {'min': 0.05, 'max': 0.5, 'default': 0.1}
    })

    momentum: Dict[str, Any] = field(default_factory=lambda: {
        'period': {'min': 5, 'max': 30, 'default': 10},
        'threshold': {'min': 0.01, 'max': 0.1, 'default': 0.02}
    })

    mean_reversion: Dict[str, Any] = field(default_factory=lambda: {
        'mean_period': {'min': 10, 'max': 50, 'default': 20},
        'std_threshold': {'min': 1.0, 'max': 3.0, 'default': 2.0}
    })


@dataclass
class RiskManagementParametersConfig:
    """Configuration for risk management parameters."""

    stop_loss: Dict[str, Any] = field(default_factory=lambda: {
        'min': 0.005, 'max': 0.1, 'default': 0.02
    })

    take_profit: Dict[str, Any] = field(default_factory=lambda: {
        'min': 0.01, 'max': 0.2, 'default': 0.05
    })

    position_sizing: Dict[str, Any] = field(default_factory=lambda: {
        'method': {'allowed_values': ['fixed', 'percentage', 'kelly'], 'default': 'percentage'},
        'percentage': {'min': 0.01, 'max': 0.1, 'default': 0.02}
    })


@dataclass
class OptimizationConfig:
    """Master configuration class for all optimization components."""

    # Component configurations
    cross_asset_validation: CrossAssetValidationConfig = field(default_factory=CrossAssetValidationConfig)
    strategy_generation: StrategyGenerationConfig = field(default_factory=StrategyGenerationConfig)
    bayesian_optimization: BayesianOptimizationConfig = field(default_factory=BayesianOptimizationConfig)
    distributed_evaluation: DistributedEvaluationConfig = field(default_factory=DistributedEvaluationConfig)

    # Parameter configurations
    indicators: IndicatorParametersConfig = field(default_factory=IndicatorParametersConfig)
    signal_logic: SignalLogicParametersConfig = field(default_factory=SignalLogicParametersConfig)
    risk_management: RiskManagementParametersConfig = field(default_factory=RiskManagementParametersConfig)

    # Global settings
    results_directory: str = 'results'
    log_directory: str = 'logs'
    cache_directory: str = 'cache'

    # Performance settings
    enable_caching: bool = True
    max_cache_size: int = 1000
    parallel_processing: bool = True

    @classmethod
    def from_json(cls, file_path: str) -> 'OptimizationConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create configuration from dictionary."""
        # This would need to be implemented to handle nested dataclasses
        # For now, return default config
        return cls()

    def to_json(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # This would need to be implemented to handle nested dataclasses
        # For now, return basic dict
        return {
            'results_directory': self.results_directory,
            'log_directory': self.log_directory,
            'cache_directory': self.cache_directory,
            'enable_caching': self.enable_caching,
            'max_cache_size': self.max_cache_size,
            'parallel_processing': self.parallel_processing
        }

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        # Validate cross-asset validation config
        if self.cross_asset_validation.validation_criteria.min_sharpe_ratio < 0:
            errors.append("min_sharpe_ratio must be non-negative")

        if not (0 < self.cross_asset_validation.validation_criteria.required_pass_rate <= 1):
            errors.append("required_pass_rate must be between 0 and 1")

        # Validate strategy generation config
        if self.strategy_generation.population_size < 1:
            errors.append("population_size must be at least 1")

        if not (0 <= self.strategy_generation.mutation_rate <= 1):
            errors.append("mutation_rate must be between 0 and 1")

        # Add more validation as needed

        return errors


# Global configuration instance
_default_config = None

def get_default_config() -> OptimizationConfig:
    """Get the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = OptimizationConfig()
    return _default_config

def load_config(file_path: str) -> OptimizationConfig:
    """Load configuration from file."""
    return OptimizationConfig.from_json(file_path)

def save_config(config: OptimizationConfig, file_path: str) -> None:
    """Save configuration to file."""
    config.to_json(file_path)

# Convenience functions for accessing specific configurations
def get_cross_asset_validation_config() -> CrossAssetValidationConfig:
    """Get cross-asset validation configuration."""
    return get_default_config().cross_asset_validation

def get_strategy_generation_config() -> StrategyGenerationConfig:
    """Get strategy generation configuration."""
    return get_default_config().strategy_generation

def get_indicator_params(indicator_type: str) -> Dict[str, Any]:
    """Get parameters for a specific indicator type."""
    config = get_default_config()
    if hasattr(config.indicators, indicator_type):
        return getattr(config.indicators, indicator_type)
    return {}

def get_signal_logic_params(signal_type: str) -> Dict[str, Any]:
    """Get parameters for a specific signal logic type."""
    config = get_default_config()
    if hasattr(config.signal_logic, signal_type):
        return getattr(config.signal_logic, signal_type)
    return {}

def get_distributed_evaluation_config() -> DistributedEvaluationConfig:
    """Get distributed evaluation configuration."""
    return get_default_config().distributed_evaluation
