#!/usr/bin/env python3

from datetime import datetime, timedelta

from scripts.run_model_benchmarks import detect_regressions

# Create mock benchmark history
history_data = []

base_date = datetime.now() - timedelta(days=35)

# Generate 30 days of benchmark results with gradual degradation
for i in range(35):
    date = base_date + timedelta(days=i)
    # Simulate gradual performance degradation
    degradation_factor = max(0.65, 0.75 - (i * 0.002))  # Start at 0.75, degrade to 0.65

    benchmark_result = {
        "model_name": "test_model",
        "timestamp": date.isoformat(),
        "status": "success",
        "metrics": {
            "f1": degradation_factor,
            "precision": degradation_factor + 0.05,
            "recall": degradation_factor - 0.02,
            "accuracy": degradation_factor + 0.03,
        },
        "validation_samples": 1000,
    }
    history_data.append(benchmark_result)

# Create current benchmark result (simulating today)
current_result = [
    {
        "model_name": "test_model",
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {
            "f1": 0.675,  # Below threshold - more degraded
            "precision": 0.73,
            "recall": 0.66,
            "accuracy": 0.71,
        },
        "validation_samples": 1000,
    }
]

print("Historical F1 values:")
for i, h in enumerate(history_data[-10:]):  # Last 10 values
    print(f"  Day {35-10+i}: {h['metrics']['f1']:.3f}")

print(f"\nCurrent F1: {current_result[0]['metrics']['f1']:.3f}")

# Calculate average of last 30
recent_history = history_data[-30:]
f1_values = [r["metrics"]["f1"] for r in recent_history]
avg_f1 = sum(f1_values) / len(f1_values)
print(f"Average F1 of last 30 days: {avg_f1:.3f}")
print(f"Difference: {current_result[0]['metrics']['f1'] - avg_f1:.3f}")
print(f"Relative change: {(current_result[0]['metrics']['f1'] - avg_f1) / avg_f1:.3f}")

# Detect regressions
regressions = detect_regressions(current_result, history_data, threshold=0.05)

print(f"\nRegressions detected: {regressions}")

if "test_model" in regressions and "f1" in regressions["test_model"]:
    f1_reg = regressions["test_model"]["f1"]
    print(f"F1 regression details: {f1_reg}")
