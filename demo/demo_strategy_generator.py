#!/usr/bin/env python3
"""
Hybrid AI Strategy Generator Demo

This script demonstrates the revolutionary Hybrid AI Strategy Generator that
autonomously discovers, optimizes, and validates new trading strategies beyond
human design capabilities.

The system combines:
- Genetic Programming for strategy evolution
- Bayesian Optimization for efficient exploration
- Distributed evaluation across multiple cores
- Dynamic strategy generation and deployment

Usage:
    python demo_strategy_generator.py
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

# Import N1V1 components
from optimization.strategy_generator import (
    get_strategy_generator, create_strategy_generator,
    StrategyGenome, StrategyGene, StrategyComponent,
    IndicatorType, SignalLogic
)
from strategies.generated import (
    get_strategy_runtime, create_generated_strategy,
    load_generated_strategy, list_generated_strategies
)
from strategies.base_strategy import BaseStrategy
from backtest.backtester import compute_backtest_metrics
from utils.logger import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)


class StrategyGeneratorDemo:
    """Comprehensive demonstration of the Hybrid AI Strategy Generator."""

    def __init__(self):
        self.generator = None
        self.runtime = None
        self.best_genome = None
        self.generated_strategies = []

    async def initialize(self):
        """Initialize all components."""
        logger.info("🚀 Initializing Hybrid AI Strategy Generator Demo")
        logger.info("=" * 70)

        # Initialize strategy generator
        config = {
            'population_size': 30,  # Smaller for demo
            'generations': 10,      # Fewer generations for demo
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elitism_rate': 0.1,
            'bayesian_enabled': True,
            'distributed_enabled': False,  # Disable for demo simplicity
            'max_workers': 2
        }

        self.generator = await create_strategy_generator(config)
        self.runtime = get_strategy_runtime()

        logger.info("✅ All components initialized successfully")

    async def generate_synthetic_data(self, symbol: str = "BTC/USDT",
                                    days: int = 90) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for strategy testing.

        Args:
            symbol: Trading symbol
            days: Number of days of data to generate

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"📊 Generating synthetic data for {symbol} ({days} days)")

        # Generate timestamps (1-hour intervals)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')

        np.random.seed(42)  # For reproducible results

        # Generate synthetic price data with trends and volatility
        n_points = len(timestamps)

        # Base price around $50,000
        base_price = 50000

        # Generate random walk with drift and volatility
        drift = 0.0002  # Small upward drift
        volatility = 0.025  # 2.5% daily volatility

        # Create price series
        price_changes = np.random.normal(drift, volatility/np.sqrt(24), n_points)
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Add regime-like behavior
        for i in range(0, n_points, 200):  # Every ~8 days
            if np.random.random() > 0.4:  # 60% chance of trend
                trend_length = np.random.randint(48, 120)  # 2-5 days
                trend_strength = np.random.normal(0.001, 0.0005)
                end_idx = min(i + trend_length, n_points)
                trend_changes = np.linspace(0, trend_strength * trend_length, end_idx - i)
                prices[i:end_idx] *= np.exp(trend_changes)

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            # Add intrabar volatility
            high_mult = 1 + np.random.uniform(0, 0.008)
            low_mult = 1 - np.random.uniform(0, 0.008)
            open_price = price * (1 + np.random.normal(0, 0.003))
            close_price = price
            high_price = max(open_price, close_price) * high_mult
            low_price = min(open_price, close_price) * low_mult

            # Volume (simulated)
            volume = np.random.uniform(200, 2000)

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        logger.info(f"✅ Generated {len(df)} data points")
        return df

    async def demonstrate_genome_creation(self):
        """Demonstrate manual genome creation and strategy generation."""
        logger.info("🧬 Demonstrating Genome Creation and Strategy Generation")
        logger.info("-" * 60)

        # Create a sample genome manually
        genome = StrategyGenome()

        # Add indicator gene (RSI)
        rsi_gene = StrategyGene(
            component_type=StrategyComponent.INDICATOR,
            indicator_type=IndicatorType.RSI,
            parameters={'period': 14, 'overbought': 70, 'oversold': 30}
        )
        genome.genes.append(rsi_gene)

        # Add signal logic gene (Threshold)
        signal_gene = StrategyGene(
            component_type=StrategyComponent.SIGNAL_LOGIC,
            signal_logic=SignalLogic.THRESHOLD,
            parameters={'threshold': 0.5, 'direction': 'below'}
        )
        genome.genes.append(signal_gene)

        # Add risk management gene
        risk_gene = StrategyGene(
            component_type=StrategyComponent.RISK_MANAGEMENT,
            parameters={'stop_loss': 0.02, 'take_profit': 0.04}
        )
        genome.genes.append(risk_gene)

        logger.info(f"Created genome with {len(genome.genes)} genes:")
        for i, gene in enumerate(genome.genes):
            logger.info(f"  Gene {i+1}: {gene.component_type.value}")
            if gene.indicator_type:
                logger.info(f"    Indicator: {gene.indicator_type.value}")
            if gene.signal_logic:
                logger.info(f"    Signal Logic: {gene.signal_logic.value}")
            if gene.parameters:
                logger.info(f"    Parameters: {gene.parameters}")

        # Create strategy from genome
        strategy_class = create_generated_strategy(genome, "ManualStrategy")
        logger.info(f"✅ Created strategy class: {strategy_class.__name__}")

        # Register strategy
        strategy_id = load_generated_strategy(genome, {"source": "manual_creation"})
        logger.info(f"✅ Registered strategy with ID: {strategy_id}")

        return genome, strategy_class

    async def run_evolutionary_optimization(self, data: pd.DataFrame):
        """Run the evolutionary optimization process."""
        logger.info("🔬 Running Evolutionary Strategy Optimization")
        logger.info("-" * 60)

        start_time = time.time()

        # Run optimization
        result = self.generator.optimize(BaseStrategy, data)

        optimization_time = time.time() - start_time

        if result and 'genome' in result:
            genome_data = result['genome']
            self.best_genome = StrategyGenome.from_dict(genome_data)

            logger.info("🎯 Optimization completed successfully!")
            logger.info(".2f")
            logger.info(f"Best fitness: {result.get('fitness', 'N/A'):.4f}")
            logger.info(f"Generation: {result.get('generation', 'N/A')}")
            logger.info(f"Species ID: {result.get('species_id', 'N/A')}")

            # Show generation statistics
            gen_stats = self.generator.get_generation_stats()
            if gen_stats:
                logger.info("📈 Generation Statistics:")
                for stat in gen_stats[-3:]:  # Last 3 generations
                    logger.info(f"  Gen {stat['generation']}: Best={stat['best_fitness']:.4f}, "
                              f"Avg={stat['avg_fitness']:.4f}")

            # Show species information
            species_info = self.generator.get_species_info()
            if species_info:
                logger.info("🧬 Species Information:")
                for species in species_info[:3]:  # Top 3 species
                    logger.info(f"  Species {species['species_id']}: "
                              f"{species['member_count']} members, "
                              f"Best fitness: {species['best_fitness']:.4f}")

        else:
            logger.warning("❌ Optimization did not return expected results")

        return result

    async def demonstrate_strategy_deployment(self):
        """Demonstrate strategy deployment and execution."""
        logger.info("🚀 Demonstrating Strategy Deployment")
        logger.info("-" * 60)

        if not self.best_genome:
            logger.warning("No best genome available for deployment")
            return

        # Create strategy from best genome
        strategy_class = create_generated_strategy(self.best_genome, "OptimizedStrategy")

        # Generate test data
        test_data = await self.generate_synthetic_data(days=7)

        # Create strategy instance
        strategy_config = {
            'name': 'optimized_strategy_demo',
            'symbols': ['BTC/USDT'],
            'timeframe': '1h',
            'required_history': 50
        }

        strategy_instance = strategy_class(strategy_config)

        # Generate signals
        signals = strategy_instance.generate_signals(test_data)

        logger.info(f"✅ Generated {len(signals)} signals from optimized strategy")

        if signals:
            logger.info("📊 Sample signals:")
            for i, signal in enumerate(signals[:5]):  # Show first 5 signals
                logger.info(f"  Signal {i+1}: {signal['signal_type']} at "
                          f"{signal['price']:.2f} on {signal['timestamp']}")

        # Register with runtime
        strategy_id = load_generated_strategy(self.best_genome, {"source": "optimization"})
        logger.info(f"✅ Strategy deployed with ID: {strategy_id}")

        # Show all registered strategies
        all_strategies = list_generated_strategies()
        logger.info(f"📋 Total registered strategies: {len(all_strategies)}")
        for strategy in all_strategies[-3:]:  # Show last 3
            logger.info(f"  - {strategy}")

    async def run_performance_analysis(self):
        """Run comprehensive performance analysis."""
        logger.info("📊 Running Performance Analysis")
        logger.info("-" * 60)

        # Get generator summary
        summary = self.generator.get_strategy_generator_summary()

        logger.info("🎯 Strategy Generator Summary:")
        logger.info(f"  Population Size: {summary.get('population_size', 'N/A')}")
        logger.info(f"  Generations Completed: {summary.get('generations_completed', 'N/A')}")
        logger.info(f"  Best Fitness: {summary.get('best_fitness', 'N/A')}")
        logger.info(f"  Species Count: {summary.get('species_count', 'N/A')}")
        logger.info(f"  Total Evaluations: {summary.get('total_evaluations', 'N/A')}")

        # Show generation stats
        gen_stats = summary.get('generation_stats', [])
        if gen_stats:
            logger.info("📈 Recent Generation Performance:")
            for stat in gen_stats:
                logger.info(f"  Gen {stat['generation']}: Best={stat['best_fitness']:.4f}")

        # Show species info
        species_info = summary.get('species_info', [])
        if species_info:
            logger.info("🧬 Top Species:")
            for species in species_info[:3]:
                logger.info(f"  Species {species['species_id']}: "
                          f"{species['member_count']} members")

    async def demonstrate_population_diversity(self):
        """Demonstrate population diversity analysis."""
        logger.info("🎭 Demonstrating Population Diversity")
        logger.info("-" * 60)

        # This would normally show diversity metrics
        # For demo purposes, we'll show basic population info

        population_size = len(self.generator.population) if hasattr(self.generator, 'population') else 0
        logger.info(f"Current population size: {population_size}")

        if population_size > 0:
            # Show sample genomes
            logger.info("🧬 Sample Genomes from Population:")
            for i, genome in enumerate(self.generator.population[:3]):
                logger.info(f"  Genome {i+1}: {len(genome.genes)} genes, "
                          f"fitness={genome.fitness:.4f}")

    async def save_results(self):
        """Save optimization results and strategies."""
        logger.info("💾 Saving Results")
        logger.info("-" * 60)

        # Save population
        self.generator.save_population("models/demo_strategy_generator/population.json")

        # Save runtime state
        self.runtime.save_runtime_state("models/demo_strategy_generator/runtime_state.json")

        # Save best genome separately
        if self.best_genome:
            best_genome_path = "models/demo_strategy_generator/best_genome.json"
            with open(best_genome_path, 'w') as f:
                json.dump(self.best_genome.to_dict(), f, indent=2)
            logger.info(f"✅ Best genome saved to {best_genome_path}")

        logger.info("✅ All results saved successfully")

    async def run_complete_demo(self):
        """Run the complete strategy generator demonstration."""
        try:
            # Initialize
            await self.initialize()

            # Generate data
            data = await self.generate_synthetic_data(days=60)

            # Demonstrate manual genome creation
            await self.demonstrate_genome_creation()

            # Run evolutionary optimization
            await self.run_evolutionary_optimization(data)

            # Demonstrate strategy deployment
            await self.demonstrate_strategy_deployment()

            # Run performance analysis
            await self.run_performance_analysis()

            # Show population diversity
            await self.demonstrate_population_diversity()

            # Save results
            await self.save_results()

            logger.info("🎉 Hybrid AI Strategy Generator Demo completed successfully!")
            logger.info("=" * 70)

            # Final summary
            logger.info("📋 Demo Summary:")
            logger.info("  ✅ Genome creation and strategy generation")
            logger.info("  ✅ Evolutionary optimization process")
            logger.info("  ✅ Strategy deployment and execution")
            logger.info("  ✅ Performance analysis and monitoring")
            logger.info("  ✅ Results persistence and retrieval")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        finally:
            # Cleanup
            if self.generator:
                await self.generator.shutdown()


async def main():
    """Main entry point for the demo."""
    demo = StrategyGeneratorDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    Path("models/demo_strategy_generator").mkdir(parents=True, exist_ok=True)

    # Run the demo
    asyncio.run(main())
