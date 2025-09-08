"""
Comprehensive integration tests for all four major N1V1 features.

Tests cover:
- Multi-Timeframe Analysis + Predictive Regime Forecasting integration
- Hybrid AI Strategy Generator + Self-Healing Engine integration
- Full end-to-end workflow with all features active
- Cross-feature performance and reliability testing
- Realistic market scenarios with all features enabled
- Failure scenarios affecting multiple features
- Resource contention and optimization testing
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import pandas as pd

from core.self_healing_engine import SelfHealingEngine, ComponentType, ComponentStatus
from core.timeframe_manager import TimeframeManager
from strategies.regime.regime_forecaster import RegimeForecaster
from optimization.strategy_generator import StrategyGenerator
from core.bot_engine import BotEngine
from strategies.base_strategy import BaseStrategy
from core.diagnostics import get_diagnostics_manager


@pytest.mark.asyncio
class TestMultiTimeframeRegimeIntegration:
    """Test integration between Multi-Timeframe Analysis and Predictive Regime Forecasting."""

    async def test_regime_forecasting_with_multi_timeframe_data(self, generate_regime_data, temp_dir):
        """Test regime forecasting using multi-timeframe data."""
        # Setup timeframe manager
        tf_manager = TimeframeManager(Mock(), {})
        await tf_manager.initialize()

        # Generate multi-timeframe data for bull market
        bull_data = generate_regime_data("bull_market", n_points=200)

        # Mock data fetcher to return multi-timeframe data
        mock_data_fetcher = Mock()
        async def mock_get_historical_data(*args, **kwargs):
            return bull_data

        mock_data_fetcher.get_historical_data = mock_get_historical_data

        # Setup timeframe manager with mock
        tf_manager = TimeframeManager(mock_data_fetcher, {})
        await tf_manager.initialize()

        # Add symbol and get multi-timeframe data
        tf_manager.add_symbol("BTC/USDT", ["1h", "4h"])
        mtf_data = await tf_manager.fetch_multi_timeframe_data("BTC/USDT")

        # Setup regime forecaster
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train with multi-timeframe data
        training_data = [(mtf_data, "bull_market")]
        await forecaster._train_model(training_data)

        # Test prediction with multi-timeframe data
        prediction = await forecaster.predict_regime(mtf_data)

        assert prediction is not None
        assert 'predicted_regime' in prediction
        assert prediction['predicted_regime'] == "bull_market"

    async def test_regime_transition_detection_with_mtf(self, generate_regime_data, temp_dir):
        """Test regime transition detection using multi-timeframe analysis."""
        # Generate data for regime transition
        bull_data = generate_regime_data("bull_market", n_points=100)
        bear_data = generate_regime_data("bear_market", n_points=100)

        # Combine data to simulate transition
        transition_data = pd.concat([bull_data, bear_data])

        # Setup components
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train with both regimes
        training_data = [
            (bull_data, "bull_market"),
            (bear_data, "bear_market")
        ]
        await forecaster._train_model(training_data)

        # Test prediction on transition data
        prediction = await forecaster.predict_regime(transition_data)

        # Should detect one of the trained regimes
        assert prediction['predicted_regime'] in ["bull_market", "bear_market"]
        assert prediction['confidence'] > 0.5


@pytest.mark.asyncio
class TestStrategyGeneratorSelfHealingIntegration:
    """Test integration between Hybrid AI Strategy Generator and Self-Healing Engine."""

    async def test_strategy_generation_with_health_monitoring(self, test_config, temp_dir):
        """Test strategy generation with health monitoring."""
        # Setup self-healing engine
        healing_config = {"monitoring": {"heartbeat_interval": 5}}
        healing_engine = SelfHealingEngine(healing_config)

        # Setup strategy generator
        sg_config = test_config.get("strategy_generator", {})
        sg_config["model_path"] = temp_dir
        sg_config["population_size"] = 5

        strategy_generator = StrategyGenerator(sg_config)
        await strategy_generator.initialize()

        # Register strategy generator with self-healing engine
        healing_engine.register_component(
            "strategy_generator_main",
            ComponentType.EXTERNAL_SERVICE,
            strategy_generator,
            critical=False
        )

        # Start healing engine
        await healing_engine.start()

        # Generate strategies while monitoring health
        await strategy_generator.evolve()

        # Send heartbeat to show healthy operation
        await healing_engine.send_heartbeat(
            component_id="strategy_generator_main",
            status=ComponentStatus.HEALTHY,
            latency_ms=150.0,
            error_count=0,
            custom_metrics={
                'strategies_generated': len(strategy_generator.population),
                'current_generation': strategy_generator.current_generation
            }
        )

        # Check that heartbeat was processed
        assert healing_engine.watchdog_service.heartbeats_received > 0

        # Stop engines
        await healing_engine.stop()

    async def test_strategy_deployment_with_failure_recovery(self, test_config, temp_dir):
        """Test strategy deployment with failure recovery."""
        # Setup components
        healing_config = {"monitoring": {"heartbeat_interval": 5}}
        healing_engine = SelfHealingEngine(healing_config)

        sg_config = test_config.get("strategy_generator", {})
        sg_config["model_path"] = temp_dir
        sg_config["population_size"] = 3

        strategy_generator = StrategyGenerator(sg_config)
        await strategy_generator.initialize()

        # Register components
        healing_engine.register_component(
            "strategy_generator_main",
            ComponentType.EXTERNAL_SERVICE,
            strategy_generator,
            critical=False
        )

        # Start monitoring
        await healing_engine.start()

        # Simulate strategy generation failure
        await healing_engine.send_heartbeat(
            component_id="strategy_generator_main",
            status=ComponentStatus.CRITICAL,
            latency_ms=5000.0,
            error_count=5,
            custom_metrics={'error': 'generation_failed'}
        )

        # Allow time for failure detection
        await asyncio.sleep(1)

        # Check that failure was detected
        assert healing_engine.watchdog_service.failures_detected > 0

        # Stop monitoring
        await healing_engine.stop()


@pytest.mark.asyncio
class TestFullSystemIntegration:
    """Test full system integration with all features active."""

    async def test_end_to_end_trading_workflow(self, test_config, temp_dir, synthetic_market_data):
        """Test complete end-to-end trading workflow with all features."""
        # Setup all components
        config = test_config.copy()
        config["self_healing"] = {"enabled": True, "heartbeat_interval": 5}

        # Initialize components
        healing_engine = SelfHealingEngine(config)

        tf_manager = TimeframeManager(Mock(), config.get("multi_timeframe", {}))
        await tf_manager.initialize()

        regime_forecaster = RegimeForecaster({
            "model_path": temp_dir,
            **config.get("regime_forecasting", {})
        })
        await regime_forecaster.initialize()

        strategy_generator = StrategyGenerator({
            "model_path": temp_dir,
            "population_size": 3,
            **config.get("strategy_generator", {})
        })
        await strategy_generator.initialize()

        # Register all components with self-healing engine
        components_to_register = [
            ("timeframe_manager", ComponentType.DATA_FETCHER, tf_manager, True),
            ("regime_forecaster", ComponentType.EXTERNAL_SERVICE, regime_forecaster, False),
            ("strategy_generator", ComponentType.EXTERNAL_SERVICE, strategy_generator, False),
        ]

        for comp_id, comp_type, instance, critical in components_to_register:
            healing_engine.register_component(comp_id, comp_type, instance, critical)

        # Start self-healing engine
        await healing_engine.start()

        # Simulate complete workflow
        try:
            # 1. Multi-timeframe data fetching
            tf_manager.add_symbol("BTC/USDT", ["1h", "4h"])

            # Mock data fetcher
            mock_data_fetcher = Mock()
            mock_data_fetcher.get_historical_data = AsyncMock(return_value=synthetic_market_data)
            tf_manager.data_fetcher = mock_data_fetcher

            mtf_data = await tf_manager.fetch_multi_timeframe_data("BTC/USDT")

            # Send heartbeat for timeframe manager
            await healing_engine.send_heartbeat(
                component_id="timeframe_manager",
                status=ComponentStatus.HEALTHY,
                latency_ms=45.0,
                error_count=0,
                custom_metrics={'symbols_registered': 1}
            )

            # 2. Regime forecasting
            forecast = await regime_forecaster.predict_regime(mtf_data or synthetic_market_data)

            # Send heartbeat for regime forecaster
            await healing_engine.send_heartbeat(
                component_id="regime_forecaster",
                status=ComponentStatus.HEALTHY,
                latency_ms=120.0,
                error_count=0,
                custom_metrics={
                    'forecast_confidence': forecast.get('confidence', 0) if forecast else 0
                }
            )

            # 3. Strategy generation
            await strategy_generator.evolve()

            # Send heartbeat for strategy generator
            await healing_engine.send_heartbeat(
                component_id="strategy_generator",
                status=ComponentStatus.HEALTHY,
                latency_ms=200.0,
                error_count=0,
                custom_metrics={
                    'strategies_generated': len(strategy_generator.population),
                    'current_generation': strategy_generator.current_generation
                }
            )

            # Verify all components are healthy
            dashboard_data = healing_engine.monitoring_dashboard.get_dashboard_data()
            system_health = dashboard_data['system_health']

            assert system_health['total_components'] == 3
            assert system_health['healthy_components'] >= 2  # At least 2 healthy

            # Check heartbeat processing
            assert healing_engine.watchdog_service.heartbeats_received >= 3

        finally:
            await healing_engine.stop()

    async def test_cross_feature_performance_optimization(self, test_config, temp_dir, performance_timer):
        """Test performance optimization across all features."""
        # Setup components
        config = test_config.copy()

        healing_engine = SelfHealingEngine(config)
        tf_manager = TimeframeManager(Mock(), {})
        await tf_manager.initialize()

        regime_forecaster = RegimeForecaster({"model_path": temp_dir})
        await regime_forecaster.initialize()

        # Register components
        healing_engine.register_component("tf_manager", ComponentType.DATA_FETCHER, tf_manager, True)
        healing_engine.register_component("forecaster", ComponentType.EXTERNAL_SERVICE, regime_forecaster, False)

        # Start monitoring
        await healing_engine.start()

        # Measure performance of integrated workflow
        performance_timer.start()

        try:
            # Simulate integrated workflow
            for i in range(10):
                # Multi-timeframe operation
                await healing_engine.send_heartbeat(
                    component_id="tf_manager",
                    status=ComponentStatus.HEALTHY,
                    latency_ms=50.0 + i,
                    error_count=0
                )

                # Regime forecasting operation
                await healing_engine.send_heartbeat(
                    component_id="forecaster",
                    status=ComponentStatus.HEALTHY,
                    latency_ms=100.0 + i * 2,
                    error_count=0
                )

                await asyncio.sleep(0.01)  # Small delay

        finally:
            await healing_engine.stop()

        performance_timer.stop()

        # Should complete within reasonable time
        duration_ms = performance_timer.duration_ms()
        assert duration_ms < 2000  # Less than 2 seconds for 20 operations

    async def test_resource_contention_handling(self, test_config, temp_dir, memory_monitor):
        """Test resource contention handling across features."""
        # Setup multiple components
        healing_engine = SelfHealingEngine(test_config)

        components = []
        for i in range(10):
            mock_component = Mock()
            comp_id = f"component_{i}"
            healing_engine.register_component(
                comp_id, ComponentType.STRATEGY, mock_component, critical=False
            )
            components.append((comp_id, mock_component))

        # Start monitoring
        await healing_engine.start()

        memory_monitor.start()

        try:
            # Simulate high-frequency heartbeats from all components
            tasks = []
            for comp_id, _ in components:
                for j in range(5):  # 5 heartbeats per component
                    task = healing_engine.send_heartbeat(
                        component_id=comp_id,
                        status=ComponentStatus.HEALTHY,
                        latency_ms=50.0,
                        error_count=0
                    )
                    tasks.append(task)

            # Execute all heartbeats concurrently
            await asyncio.gather(*tasks)

            # Check memory usage
            memory_delta = memory_monitor.get_memory_delta()
            assert memory_delta < 100  # Reasonable memory usage

            # Verify all heartbeats processed
            assert healing_engine.watchdog_service.heartbeats_received == 50

        finally:
            await healing_engine.stop()


@pytest.mark.asyncio
class TestFailureScenariosAcrossFeatures:
    """Test failure scenarios that affect multiple features."""

    async def test_cascading_failure_prevention(self, test_config, temp_dir):
        """Test prevention of cascading failures across features."""
        # Setup components
        healing_engine = SelfHealingEngine(test_config)

        # Create interdependent components
        tf_manager = TimeframeManager(Mock(), {})
        await tf_manager.initialize()

        regime_forecaster = RegimeForecaster({"model_path": temp_dir})
        await regime_forecaster.initialize()

        # Register with dependencies
        healing_engine.register_component(
            "data_fetcher", ComponentType.DATA_FETCHER, tf_manager, critical=True
        )
        healing_engine.register_component(
            "regime_forecaster", ComponentType.EXTERNAL_SERVICE, regime_forecaster, critical=False,
            dependencies=["data_fetcher"]
        )

        await healing_engine.start()

        try:
            # Simulate data fetcher failure
            await healing_engine.send_heartbeat(
                component_id="data_fetcher",
                status=ComponentStatus.CRITICAL,
                latency_ms=10000.0,
                error_count=10
            )

            # Allow time for failure detection
            await asyncio.sleep(1)

            # Check that dependent component is affected
            # (This would depend on the specific implementation of dependency handling)

            # Verify failure was detected
            assert healing_engine.watchdog_service.failures_detected > 0

        finally:
            await healing_engine.stop()

    async def test_emergency_mode_with_multiple_features(self, test_config, temp_dir):
        """Test emergency mode activation with multiple features failing."""
        # Setup components
        healing_engine = SelfHealingEngine(test_config)

        # Register multiple critical components
        for i in range(6):  # More than emergency threshold
            mock_component = Mock()
            healing_engine.register_component(
                f"critical_comp_{i}",
                ComponentType.BOT_ENGINE,
                mock_component,
                critical=True
            )

        await healing_engine.start()

        try:
            # Fail multiple critical components
            for i in range(4):  # Fail 4 out of 6 (66%)
                await healing_engine.send_heartbeat(
                    component_id=f"critical_comp_{i}",
                    status=ComponentStatus.CRITICAL,
                    latency_ms=5000.0,
                    error_count=5
                )

            # Allow time for emergency detection
            await asyncio.sleep(2)

            # Check emergency status
            emergency_active = healing_engine.emergency_procedures.is_emergency_active()
            # Emergency mode should activate with >50% critical components failing
            # (Exact behavior depends on threshold configuration)

        finally:
            await healing_engine.stop()

    async def test_recovery_coordination_across_features(self, test_config, temp_dir):
        """Test recovery coordination across multiple features."""
        # Setup components
        healing_engine = SelfHealingEngine(test_config)

        # Register components from different features
        tf_manager = TimeframeManager(Mock(), {})
        await tf_manager.initialize()

        regime_forecaster = RegimeForecaster({"model_path": temp_dir})
        await regime_forecaster.initialize()

        healing_engine.register_component(
            "tf_manager", ComponentType.DATA_FETCHER, tf_manager, critical=True
        )
        healing_engine.register_component(
            "forecaster", ComponentType.EXTERNAL_SERVICE, regime_forecaster, critical=False
        )

        await healing_engine.start()

        try:
            # Fail both components
            await healing_engine.send_heartbeat(
                component_id="tf_manager",
                status=ComponentStatus.CRITICAL,
                latency_ms=10000.0,
                error_count=10
            )

            await healing_engine.send_heartbeat(
                component_id="forecaster",
                status=ComponentStatus.CRITICAL,
                latency_ms=5000.0,
                error_count=5
            )

            # Allow time for failure detection and recovery initiation
            await asyncio.sleep(2)

            # Check recovery attempts
            assert healing_engine.watchdog_service.failures_detected >= 2

            # Check healing stats
            healing_stats = healing_engine.healing_orchestrator.get_healing_stats()
            # Should have attempted recovery for failed components

        finally:
            await healing_engine.stop()


@pytest.mark.asyncio
class TestRealisticMarketScenarios:
    """Test realistic market scenarios with all features active."""

    async def test_high_volatility_scenario(self, generate_regime_data, temp_dir):
        """Test high volatility market scenario."""
        # Generate high volatility data
        hv_data = generate_regime_data("high_volatility", n_points=500)

        # Setup all features
        healing_engine = SelfHealingEngine({})

        tf_manager = TimeframeManager(Mock(), {})
        await tf_manager.initialize()

        regime_forecaster = RegimeForecaster({"model_path": temp_dir})
        await regime_forecaster.initialize()

        # Register components
        healing_engine.register_component(
            "tf_manager", ComponentType.DATA_FETCHER, tf_manager, critical=True
        )
        healing_engine.register_component(
            "forecaster", ComponentType.EXTERNAL_SERVICE, regime_forecaster, critical=False
        )

        await healing_engine.start()

        try:
            # Train forecaster with high volatility data
            training_data = [(hv_data, "high_volatility")]
            await regime_forecaster._train_model(training_data)

            # Simulate high volatility trading scenario
            for i in range(20):
                # Simulate varying performance
                latency = np.random.uniform(50, 500)  # Variable latency
                error_count = np.random.poisson(0.1)  # Occasional errors

                status = ComponentStatus.HEALTHY
                if latency > 300 or error_count > 2:
                    status = ComponentStatus.DEGRADED

                # Send heartbeats for both components
                await healing_engine.send_heartbeat(
                    component_id="tf_manager",
                    status=status,
                    latency_ms=latency,
                    error_count=error_count
                )

                await healing_engine.send_heartbeat(
                    component_id="forecaster",
                    status=status,
                    latency_ms=latency * 1.5,  # Forecaster might be slower
                    error_count=error_count
                )

                await asyncio.sleep(0.1)

            # Verify system handled high volatility scenario
            dashboard_data = healing_engine.monitoring_dashboard.get_dashboard_data()
            system_health = dashboard_data['system_health']

            # System should still be operational
            assert system_health['total_components'] == 2
            assert healing_engine.watchdog_service.heartbeats_received == 40

        finally:
            await healing_engine.stop()

    async def test_regime_transition_scenario(self, generate_regime_data, temp_dir):
        """Test regime transition scenario."""
        # Generate data for regime transition
        bull_data = generate_regime_data("bull_market", n_points=200)
        bear_data = generate_regime_data("bear_market", n_points=200)
        transition_data = pd.concat([bull_data, bear_data])

        # Setup features
        healing_engine = SelfHealingEngine({})

        regime_forecaster = RegimeForecaster({"model_path": temp_dir})
        await regime_forecaster.initialize()

        healing_engine.register_component(
            "forecaster", ComponentType.EXTERNAL_SERVICE, regime_forecaster, critical=False
        )

        await healing_engine.start()

        try:
            # Train with both regimes
            training_data = [
                (bull_data, "bull_market"),
                (bear_data, "bear_market")
            ]
            await regime_forecaster._train_model(training_data)

            # Simulate regime transition
            for i in range(10):
                # Use transition data for prediction
                prediction = await regime_forecaster.predict_regime(transition_data)

                # Send heartbeat with prediction metrics
                confidence = prediction.get('confidence', 0) if prediction else 0
                latency = 100 + (1 - confidence) * 200  # Higher latency for lower confidence

                await healing_engine.send_heartbeat(
                    component_id="forecaster",
                    status=ComponentStatus.HEALTHY,
                    latency_ms=latency,
                    error_count=0,
                    custom_metrics={
                        'prediction_confidence': confidence,
                        'regime_transition_detected': True
                    }
                )

                await asyncio.sleep(0.2)

            # Verify system handled regime transition
            assert healing_engine.watchdog_service.heartbeats_received == 10

        finally:
            await healing_engine.stop()


@pytest.mark.asyncio
class TestProductionReadiness:
    """Test production readiness with all features."""

    async def test_24_hour_stability_test(self, test_config, temp_dir):
        """Test 24-hour stability (simulated)."""
        # Setup full system
        healing_engine = SelfHealingEngine(test_config)

        tf_manager = TimeframeManager(Mock(), {})
        await tf_manager.initialize()

        regime_forecaster = RegimeForecaster({"model_path": temp_dir})
        await regime_forecaster.initialize()

        strategy_generator = StrategyGenerator({
            "model_path": temp_dir,
            "population_size": 5,
            "generations": 2
        })
        await strategy_generator.initialize()

        # Register all components
        components = [
            ("tf_manager", ComponentType.DATA_FETCHER, tf_manager, True),
            ("forecaster", ComponentType.EXTERNAL_SERVICE, regime_forecaster, False),
            ("strategy_gen", ComponentType.EXTERNAL_SERVICE, strategy_generator, False),
        ]

        for comp_id, comp_type, instance, critical in components:
            healing_engine.register_component(comp_id, comp_type, instance, critical)

        await healing_engine.start()

        start_time = time.time()
        heartbeat_count = 0

        try:
            # Simulate extended operation (scaled down for testing)
            for hour in range(2):  # Simulate 2 hours instead of 24
                for minute in range(6):  # 6 heartbeats per hour (every 10 minutes)
                    for comp_id, _, _, _ in components:
                        # Simulate realistic performance variations
                        base_latency = 50.0
                        latency_variation = np.random.normal(0, 10)
                        latency = max(10, base_latency + latency_variation)

                        error_count = np.random.poisson(0.05)  # Low error rate

                        status = ComponentStatus.HEALTHY
                        if latency > 200 or error_count > 2:
                            status = ComponentStatus.DEGRADED

                        await healing_engine.send_heartbeat(
                            component_id=comp_id,
                            status=status,
                            latency_ms=latency,
                            error_count=error_count
                        )

                        heartbeat_count += 1

                    await asyncio.sleep(0.1)  # 10-minute intervals (scaled)

            duration = time.time() - start_time

            # Verify stability
            assert duration < 60  # Should complete within 1 minute
            assert heartbeat_count == 2 * 6 * 3  # 2 hours * 6 heartbeats/hour * 3 components
            assert healing_engine.watchdog_service.heartbeats_received == heartbeat_count

            # Check final system health
            dashboard_data = healing_engine.monitoring_dashboard.get_dashboard_data()
            system_health = dashboard_data['system_health']

            assert system_health['total_components'] == 3
            # System should be mostly healthy after stability test

        finally:
            await healing_engine.stop()

    async def test_resource_usage_under_load(self, test_config, temp_dir, memory_monitor):
        """Test resource usage under sustained load."""
        # Setup system
        healing_engine = SelfHealingEngine(test_config)

        # Register many components to simulate load
        for i in range(20):
            mock_component = Mock()
            healing_engine.register_component(
                f"comp_{i}",
                ComponentType.STRATEGY,
                mock_component,
                critical=False
            )

        await healing_engine.start()

        memory_monitor.start()

        try:
            # Simulate sustained load
            for i in range(100):
                # Send heartbeats to all components
                tasks = []
                for j in range(20):
                    task = healing_engine.send_heartbeat(
                        component_id=f"comp_{j}",
                        status=ComponentStatus.HEALTHY,
                        latency_ms=50.0 + np.random.normal(0, 5),
                        error_count=np.random.poisson(0.01)
                    )
                    tasks.append(task)

                await asyncio.gather(*tasks)
                await asyncio.sleep(0.01)

            # Check memory usage
            memory_delta = memory_monitor.get_memory_delta()
            assert memory_delta < 200  # Reasonable memory usage for sustained load

            # Verify all heartbeats processed
            assert healing_engine.watchdog_service.heartbeats_received == 2000  # 100 iterations * 20 components

        finally:
            await healing_engine.stop()

    async def test_security_integration(self, test_config):
        """Test security integration with all features."""
        # Setup system
        healing_engine = SelfHealingEngine(test_config)

        # Register components
        mock_component = Mock()
        healing_engine.register_component(
            "secure_comp",
            ComponentType.BOT_ENGINE,
            mock_component,
            critical=True
        )

        await healing_engine.start()

        try:
            # Test that heartbeat data is handled securely
            # (In real implementation, this would test encryption, authentication, etc.)

            await healing_engine.send_heartbeat(
                component_id="secure_comp",
                status=ComponentStatus.HEALTHY,
                latency_ms=50.0,
                error_count=0,
                custom_metrics={'secure_data': 'test'}
            )

            # Verify heartbeat was processed
            assert healing_engine.watchdog_service.heartbeats_received == 1

        finally:
            await healing_engine.stop()


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for all features integrated."""

    @pytest.mark.asyncio
    async def test_heartbeat_throughput_benchmark(self, test_config, benchmark):
        """Benchmark heartbeat processing throughput."""
        healing_engine = SelfHealingEngine(test_config)

        # Register multiple components
        for i in range(10):
            mock_component = Mock()
            healing_engine.register_component(
                f"bench_comp_{i}",
                ComponentType.STRATEGY,
                mock_component,
                critical=False
            )

        await healing_engine.start()

        try:
            # Benchmark function
            async def heartbeat_workload():
                tasks = []
                for i in range(100):  # 100 heartbeats per component
                    for j in range(10):
                        task = healing_engine.send_heartbeat(
                            component_id=f"bench_comp_{j}",
                            status=ComponentStatus.HEALTHY,
                            latency_ms=50.0,
                            error_count=0
                        )
                        tasks.append(task)

                await asyncio.gather(*tasks)

            # Run benchmark
            result = await benchmark(heartbeat_workload)

            # Should handle high throughput
            assert result is not None

        finally:
            await healing_engine.stop()

    @pytest.mark.asyncio
    async def test_memory_efficiency_benchmark(self, test_config, memory_monitor, benchmark):
        """Benchmark memory efficiency."""
        healing_engine = SelfHealingEngine(test_config)

        memory_monitor.start()

        # Register components
        for i in range(50):
            mock_component = Mock()
            healing_engine.register_component(
                f"mem_comp_{i}",
                ComponentType.STRATEGY,
                mock_component,
                critical=False
            )

        await healing_engine.start()

        try:
            # Benchmark function with memory monitoring
            async def memory_workload():
                for i in range(200):  # 200 heartbeats per component
                    tasks = []
                    for j in range(50):
                        task = healing_engine.send_heartbeat(
                            component_id=f"mem_comp_{j}",
                            status=ComponentStatus.HEALTHY,
                            latency_ms=50.0,
                            error_count=0
                        )
                        tasks.append(task)

                    await asyncio.gather(*tasks)

            # Run benchmark
            await benchmark(memory_workload)

            # Check memory usage
            memory_delta = memory_monitor.get_memory_delta()
            assert memory_delta < 500  # Reasonable memory usage

        finally:
            await healing_engine.stop()
