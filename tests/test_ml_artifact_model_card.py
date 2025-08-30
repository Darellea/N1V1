import os
import json
import tempfile
import pandas as pd
import numpy as np

from ml.trainer import train_model, generate_features, create_labels

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
    # Create a deterministic feature-only DataFrame to avoid rolling-window NaN issues.
    rows = 600
    idx = pd.date_range("2020-01-01", periods=rows, freq="H")
    rng = np.random.RandomState(42)
    feature_columns = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']
    data = {c: rng.rand(rows) for c in feature_columns}
    df = pd.DataFrame(data, index=idx)
    # Inject deterministic labels cycling through -1,0,1
    labels = np.tile(np.array([-1, 0, 1]), int(np.ceil(rows / 3)))[: rows]
    df['Label'] = labels

    # Ensure we have enough samples for TimeSeriesSplit with 3 splits
    assert len(df) >= 4, "Insufficient data for CV splits in test"

    model_path = tmp_path / "model.pkl"
    results_path = tmp_path / "results.json"

    train_model(
        df,
        str(model_path),
        results_path=str(results_path),
        n_splits=3,
        horizon=5,
        up_thresh=0.001,
        down_thresh=-0.001,
        feature_columns=feature_columns,
        tune=False,
        feature_selection=False,
        early_stopping_rounds=0,
        eval_profit=False,
    )

    card_path = str(model_path).replace(".pkl", ".model_card.json")
    assert os.path.exists(card_path), "Model card JSON should be created"
    with open(card_path, "r", encoding="utf-8") as fh:
        card = json.load(fh)
    assert "feature_list" in card
    assert "training_metadata" in card
    assert card["model_file"].endswith("model.pkl")
