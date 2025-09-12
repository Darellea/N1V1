import os
import json
import tempfile
import pandas as pd
import numpy as np

# Test adapted for binary trade/no-trade model.
# Legacy multi-class path removed.

from ml.trainer import train_model_binary, generate_features, create_binary_labels

def make_sample_df(rows=200):
    # Create synthetic OHLCV data with deterministic values
    idx = pd.date_range("2020-01-01", periods=rows, freq="H")
    price = np.linspace(100, 200, rows) + np.sin(np.linspace(0, 10, rows)) * 2
    df = pd.DataFrame({
        "Open": price,
        "High": price + 1.0,
        "Low": price - 1.0,
        "Close": price,
        "Volume": np.random.rand(rows) * 1000,
    }, index=idx)
    return df

def test_train_creates_model_card(tmp_path):
    # Create synthetic OHLCV DataFrame for binary classification
    rows = 600
    idx = pd.date_range("2020-01-01", periods=rows, freq="H")
    rng = np.random.RandomState(42)

    # Create realistic OHLCV data with some trend and volatility
    base_price = 100 + np.cumsum(rng.normal(0, 0.5, rows))
    noise = rng.normal(0, 1, rows)
    close_prices = base_price + noise

    df = pd.DataFrame({
        "Open": close_prices + rng.normal(0, 0.5, rows),
        "High": close_prices + abs(rng.normal(0, 1, rows)),
        "Low": close_prices - abs(rng.normal(0, 1, rows)),
        "Close": close_prices,
        "Volume": rng.uniform(1000, 10000, rows),
    }, index=idx)

    # Generate features using the standard pipeline
    df = generate_features(df)

    # Create binary labels for trade/no-trade decisions
    df = create_binary_labels(df, horizon=5, profit_threshold=0.005)
    df['Label'] = df['label_binary']

    # Ensure we have enough samples for TimeSeriesSplit with 3 splits
    assert len(df) >= 10, "Insufficient data for CV splits in test"

    feature_columns = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']

    model_path = tmp_path / "model.pkl"
    results_path = tmp_path / "results.json"

    train_model_binary(
        df,
        str(model_path),
        results_path=str(results_path),
        n_splits=3,
        horizon=5,
        profit_threshold=0.005,
        include_fees=True,
        fee_rate=0.001,
        feature_columns=feature_columns,
        tune=False,
        feature_selection=False,
        early_stopping_rounds=50,
        eval_economic=True,
    )

    card_path = str(model_path).replace(".pkl", ".model_card.json")
    assert os.path.exists(card_path), "Model card JSON should be created"
    with open(card_path, "r", encoding="utf-8") as fh:
        card = json.load(fh)
    assert "feature_list" in card
    assert "training_metadata" in card
    assert card["model_file"].endswith("model.pkl")
