#!/usr/bin/env python3
"""
Simple test script to verify anomaly detector integration.
"""

import pandas as pd
from risk.anomaly_detector import AnomalyDetector, AnomalyResponse

def test_basic_anomaly_detection():
    """Test basic anomaly detection functionality."""
    print("Testing basic anomaly detection...")

    # Create test data with a price gap
    data = pd.DataFrame({
        'close': [100.0, 120.0],  # 20% gap
        'volume': [1000, 1000]
    })

    detector = AnomalyDetector()
    results = detector.detect_anomalies(data, 'TEST')

    print(f"Detected {len(results)} anomalies")
    for result in results:
        print(f"  - {result.anomaly_type.value}: severity={result.severity.value}, confidence={result.confidence_score:.3f}")

    # Test signal checking
    signal = {'symbol': 'TEST', 'amount': 1000}
    should_proceed, response, anomaly = detector.check_signal_anomaly(signal, data, 'TEST')

    print(f"Signal check: proceed={should_proceed}, response={response.value if response else None}")
    if anomaly:
        print(f"  Anomaly: {anomaly.anomaly_type.value} ({anomaly.severity.value})")

def test_risk_manager_integration():
    """Test integration with risk manager."""
    print("\nTesting risk manager integration...")

    from risk.risk_manager import RiskManager
    from core.contracts import TradingSignal

    # Create test signal
    signal = TradingSignal(
        symbol='TEST',
        signal_type='LONG',
        order_type='MARKET',
        amount=1000,
        current_price=100.0
    )

    # Create market data with anomaly
    market_data = {
        'close': [100.0, 120.0],  # 20% gap
        'volume': [1000, 1000]
    }

    # Test risk manager evaluation
    risk_config = {'anomaly_detector': {'enabled': True}}
    risk_manager = RiskManager(risk_config)

    result = risk_manager.evaluate_signal(signal, market_data)
    print(f"Risk manager evaluation: {result}")

if __name__ == "__main__":
    test_basic_anomaly_detection()
    test_risk_manager_integration()
    print("\nAnomaly detector integration test completed!")
