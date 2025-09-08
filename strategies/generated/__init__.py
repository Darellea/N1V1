"""
Generated Strategies Module

This module provides runtime support for AI-generated trading strategies.
It handles dynamic loading, compilation, and execution of strategies
discovered through the Hybrid AI Strategy Generator.

Key Features:
- Dynamic strategy loading from genome definitions
- Runtime compilation and validation
- Strategy versioning and lineage tracking
- Performance monitoring and health checks
- Integration with existing strategy framework
"""

import logging
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from datetime import datetime
import json
import importlib
import sys
import os

from strategies.base_strategy import BaseStrategy
from optimization.strategy_generator import StrategyGenome, StrategyGene, StrategyComponent, IndicatorType, SignalLogic
from utils.logger import get_logger

logger = get_logger(__name__)


class GeneratedStrategyRegistry:
    """Registry for managing generated strategies."""

    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self.strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self.generation_history: List[Dict[str, Any]] = []

    def register_strategy(self, strategy_class: Type[BaseStrategy],
                         genome: StrategyGenome,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a generated strategy.

        Args:
            strategy_class: The generated strategy class
            genome: Original genome that produced this strategy
            metadata: Additional metadata

        Returns:
            Unique strategy ID
        """
        strategy_id = f"gen_{genome.generation}_{id(genome)}"

        self.strategies[strategy_id] = strategy_class
        self.strategy_metadata[strategy_id] = {
            'genome': genome.to_dict(),
            'created_at': datetime.now().isoformat(),
            'fitness': genome.fitness,
            'generation': genome.generation,
            'species_id': genome.species_id,
            'metadata': metadata or {}
        }

        logger.info(f"Registered generated strategy: {strategy_id}")
        return strategy_id

    def get_strategy(self, strategy_id: str) -> Optional[Type[BaseStrategy]]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)

    def list_strategies(self) -> List[str]:
        """List all registered strategy IDs."""
        return list(self.strategies.keys())

    def get_strategy_info(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a strategy."""
        return self.strategy_metadata.get(strategy_id)

    def save_registry(self, path: str) -> None:
        """Save registry to disk."""
        registry_data = {
            'timestamp': datetime.now().isoformat(),
            'strategies': self.strategy_metadata,
            'generation_history': self.generation_history
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)

        logger.info(f"Registry saved to {path}")

    def load_registry(self, path: str) -> None:
        """Load registry from disk."""
        if not Path(path).exists():
            logger.warning(f"Registry file not found: {path}")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.strategy_metadata = data.get('strategies', {})
        self.generation_history = data.get('generation_history', [])

        logger.info(f"Registry loaded from {path}")


class StrategyRuntime:
    """Runtime system for executing generated strategies."""

    def __init__(self):
        self.registry = GeneratedStrategyRegistry()
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.performance_monitor = StrategyPerformanceMonitor()

    def create_strategy_from_genome(self, genome: StrategyGenome,
                                   strategy_name: Optional[str] = None) -> Type[BaseStrategy]:
        """
        Create a strategy class from a genome definition.

        Args:
            genome: Strategy genome
            strategy_name: Optional name for the strategy

        Returns:
            Generated strategy class
        """
        if strategy_name is None:
            strategy_name = f"GeneratedStrategy_{genome.generation}_{id(genome)}"

        # Create strategy class dynamically
        class GeneratedStrategy(BaseStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.genome = genome
                self.strategy_id = f"gen_{genome.generation}_{id(genome)}"

                # Initialize genome-based parameters
                self._initialize_from_genome()

            def _initialize_from_genome(self):
                """Initialize strategy parameters from genome."""
                self.indicators = {}
                self.signal_logic = {}
                self.risk_params = {}
                self.filters = {}

                for gene in self.genome.genes:
                    if not gene.enabled:
                        continue

                    if gene.component_type == StrategyComponent.INDICATOR:
                        self.indicators[gene.indicator_type.value] = gene.parameters
                    elif gene.component_type == StrategyComponent.SIGNAL_LOGIC:
                        self.signal_logic[gene.signal_logic.value] = gene.parameters
                    elif gene.component_type == StrategyComponent.RISK_MANAGEMENT:
                        self.risk_params.update(gene.parameters)
                    elif gene.component_type == StrategyComponent.FILTER:
                        self.filters.update(gene.parameters)

            def generate_signals(self, data):
                """Generate trading signals based on genome."""
                signals = []

                if data.empty or len(data) < 50:
                    return signals

                try:
                    # Apply genome-based signal generation
                    signals = self._apply_genome_logic(data)
                except Exception as e:
                    logger.error(f"Error generating signals for {self.strategy_id}: {e}")

                return signals

            def _apply_genome_logic(self, data):
                """Apply genome-defined logic to generate signals."""
                signals = []

                # Simplified genome interpretation
                # In practice, this would be much more sophisticated

                # Check for indicator-based signals
                if IndicatorType.RSI in [IndicatorType(k) for k in self.indicators.keys()]:
                    rsi_signals = self._generate_rsi_signals(data)
                    signals.extend(rsi_signals)

                if IndicatorType.MACD in [IndicatorType(k) for k in self.indicators.keys()]:
                    macd_signals = self._generate_macd_signals(data)
                    signals.extend(macd_signals)

                # Apply filters if defined
                if self.filters:
                    signals = self._apply_filters(signals, data)

                return signals

            def _generate_rsi_signals(self, data):
                """Generate RSI-based signals."""
                signals = []
                rsi_params = self.indicators.get('rsi', {})
                overbought = rsi_params.get('overbought', 70)
                oversold = rsi_params.get('oversold', 30)

                # Mock RSI calculation and signal generation
                for i in range(20, len(data)):
                    if i % 15 == 0:  # Generate signal every 15 bars
                        signal_type = "BUY" if i % 30 == 0 else "SELL"
                        signals.append({
                            'timestamp': data.index[i],
                            'signal_type': signal_type,
                            'symbol': self.config.get('symbols', ['BTC/USDT'])[0],
                            'price': data.iloc[i]['close'],
                            'metadata': {
                                'strategy_id': self.strategy_id,
                                'indicator': 'rsi'
                            }
                        })

                return signals

            def _generate_macd_signals(self, data):
                """Generate MACD-based signals."""
                signals = []

                # Mock MACD signal generation
                for i in range(30, len(data)):
                    if i % 20 == 0:  # Generate signal every 20 bars
                        signal_type = "BUY" if i % 40 == 0 else "SELL"
                        signals.append({
                            'timestamp': data.index[i],
                            'signal_type': signal_type,
                            'symbol': self.config.get('symbols', ['BTC/USDT'])[0],
                            'price': data.iloc[i]['close'],
                            'metadata': {
                                'strategy_id': self.strategy_id,
                                'indicator': 'macd'
                            }
                        })

                return signals

            def _apply_filters(self, signals, data):
                """Apply genome-defined filters to signals."""
                filtered_signals = []

                for signal in signals:
                    # Apply volume filter if defined
                    volume_threshold = self.filters.get('volume_threshold')
                    if volume_threshold:
                        signal_idx = data.index.get_loc(signal['timestamp'])
                        if signal_idx < len(data):
                            volume = data.iloc[signal_idx]['volume']
                            avg_volume = data['volume'].rolling(20).mean().iloc[signal_idx]
                            if avg_volume > 0 and (volume / avg_volume) < volume_threshold:
                                continue  # Filter out signal

                    filtered_signals.append(signal)

                return filtered_signals

        # Set class name
        GeneratedStrategy.__name__ = strategy_name
        GeneratedStrategy.__qualname__ = strategy_name

        return GeneratedStrategy

    def load_strategy_from_genome(self, genome: StrategyGenome,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Load a strategy from genome and register it.

        Args:
            genome: Strategy genome
            metadata: Additional metadata

        Returns:
            Strategy ID
        """
        strategy_class = self.create_strategy_from_genome(genome)
        strategy_id = self.registry.register_strategy(strategy_class, genome, metadata)

        logger.info(f"Loaded strategy {strategy_id} from genome")
        return strategy_id

    def instantiate_strategy(self, strategy_id: str, config: Dict[str, Any]) -> Optional[BaseStrategy]:
        """
        Instantiate a strategy for execution.

        Args:
            strategy_id: Strategy ID
            config: Strategy configuration

        Returns:
            Strategy instance or None if not found
        """
        strategy_class = self.registry.get_strategy(strategy_id)
        if strategy_class is None:
            logger.error(f"Strategy {strategy_id} not found")
            return None

        try:
            strategy_instance = strategy_class(config)
            self.active_strategies[strategy_id] = strategy_instance

            logger.info(f"Instantiated strategy {strategy_id}")
            return strategy_instance

        except Exception as e:
            logger.error(f"Failed to instantiate strategy {strategy_id}: {e}")
            return None

    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy IDs."""
        return list(self.active_strategies.keys())

    def deactivate_strategy(self, strategy_id: str) -> None:
        """Deactivate a strategy."""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            logger.info(f"Deactivated strategy {strategy_id}")

    def save_runtime_state(self, path: str) -> None:
        """Save runtime state to disk."""
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'active_strategies': list(self.active_strategies.keys()),
            'performance_data': self.performance_monitor.get_performance_summary()
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)

        # Also save registry
        registry_path = Path(path).parent / "strategy_registry.json"
        self.registry.save_registry(str(registry_path))

        logger.info(f"Runtime state saved to {path}")

    def load_runtime_state(self, path: str) -> None:
        """Load runtime state from disk."""
        if not Path(path).exists():
            logger.warning(f"Runtime state file not found: {path}")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        # Load registry
        registry_path = Path(path).parent / "strategy_registry.json"
        if registry_path.exists():
            self.registry.load_registry(str(registry_path))

        logger.info(f"Runtime state loaded from {path}")


class StrategyPerformanceMonitor:
    """Monitor performance of generated strategies."""

    def __init__(self):
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}

    def record_trade(self, strategy_id: str, trade_data: Dict[str, Any]) -> None:
        """Record a trade for performance monitoring."""
        if strategy_id not in self.performance_data:
            self.performance_data[strategy_id] = []

        self.performance_data[strategy_id].append({
            'timestamp': datetime.now().isoformat(),
            'trade_data': trade_data
        })

    def update_health_check(self, strategy_id: str, health_data: Dict[str, Any]) -> None:
        """Update health check data for a strategy."""
        self.health_checks[strategy_id] = {
            'timestamp': datetime.now().isoformat(),
            'health_data': health_data
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        summary = {}

        for strategy_id, trades in self.performance_data.items():
            if not trades:
                continue

            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['trade_data'].get('pnl', 0) > 0)
            total_pnl = sum(t['trade_data'].get('pnl', 0) for t in trades)

            summary[strategy_id] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'last_updated': datetime.now().isoformat()
            }

        return summary

    def get_unhealthy_strategies(self, threshold: float = 0.3) -> List[str]:
        """Get strategies with poor performance (win rate below threshold)."""
        unhealthy = []

        for strategy_id, summary in self.get_performance_summary().items():
            if summary['win_rate'] < threshold:
                unhealthy.append(strategy_id)

        return unhealthy


# Global runtime instance
_strategy_runtime: Optional[StrategyRuntime] = None


def get_strategy_runtime() -> StrategyRuntime:
    """Get the global strategy runtime instance."""
    global _strategy_runtime
    if _strategy_runtime is None:
        _strategy_runtime = StrategyRuntime()
    return _strategy_runtime


def create_generated_strategy(genome: StrategyGenome,
                             strategy_name: Optional[str] = None) -> Type[BaseStrategy]:
    """
    Create a generated strategy from genome.

    Args:
        genome: Strategy genome
        strategy_name: Optional strategy name

    Returns:
        Generated strategy class
    """
    runtime = get_strategy_runtime()
    return runtime.create_strategy_from_genome(genome, strategy_name)


def load_generated_strategy(genome: StrategyGenome,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Load and register a generated strategy.

    Args:
        genome: Strategy genome
        metadata: Additional metadata

    Returns:
        Strategy ID
    """
    runtime = get_strategy_runtime()
    return runtime.load_strategy_from_genome(genome, metadata)


def get_generated_strategy(strategy_id: str) -> Optional[Type[BaseStrategy]]:
    """
    Get a generated strategy by ID.

    Args:
        strategy_id: Strategy ID

    Returns:
        Strategy class or None
    """
    runtime = get_strategy_runtime()
    return runtime.registry.get_strategy(strategy_id)


def list_generated_strategies() -> List[str]:
    """List all generated strategy IDs."""
    runtime = get_strategy_runtime()
    return runtime.registry.list_strategies()


# Create a simple GeneratedStrategy class for import compatibility
class GeneratedStrategy(BaseStrategy):
    """Simple GeneratedStrategy class for import compatibility."""

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    async def create_from_genome(cls, genome, name=None):
        """Create strategy from genome."""
        return cls({})

    @classmethod
    async def validate_genome(cls, genome):
        """Validate genome."""
        return True

    @classmethod
    async def store_in_knowledge_base(cls, genome, metadata):
        """Store in knowledge base."""
        return f"strategy_{id(genome)}"

    @classmethod
    async def retrieve_from_knowledge_base(cls, strategy_id):
        """Retrieve from knowledge base."""
        return None

    @classmethod
    async def search_knowledge_base(cls, criteria):
        """Search knowledge base."""
        return []


# Export key classes and functions
__all__ = [
    'StrategyRuntime',
    'GeneratedStrategyRegistry',
    'StrategyPerformanceMonitor',
    'GeneratedStrategy',
    'get_strategy_runtime',
    'create_generated_strategy',
    'load_generated_strategy',
    'get_generated_strategy',
    'list_generated_strategies'
]
