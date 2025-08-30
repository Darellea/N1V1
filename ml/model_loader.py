import os
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)


def load_model(path: str):
    """
    Load a model from disk. Supports joblib/pickle files.
    Returns the unpickled model instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info(f"Loaded model from {path}")
    return model


def load_model_with_card(path: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a model and its companion model card (if present).

    The model card is expected at the same path with extension '.model_card.json'.
    Returns (model, model_card_dict|None)
    """
    model = load_model(path)
    card_path = os.path.splitext(os.path.abspath(path))[0] + ".model_card.json"
    model_card = None
    try:
        if os.path.exists(card_path):
            with open(card_path, "r", encoding="utf-8") as fh:
                model_card = json.load(fh)
            logger.info(f"Loaded model card from {card_path}")
    except Exception:
        logger.exception(f"Failed to load model card at {card_path}")
    return model, model_card


def _align_features(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Align feature dataframe columns to the feature order expected by the model.
    For lightgbm sklearn wrapper, use model.booster_.feature_name() or model.feature_name_ / model.classes_ fallback.
    """
    if hasattr(model, "booster_"):
        try:
            feature_names = model.booster_.feature_name()
        except Exception:
            feature_names = None
    else:
        feature_names = None

    # Try sklearn-style attribute
    if not feature_names and hasattr(model, "feature_name_"):
        try:
            feature_names = list(model.feature_name_)
        except Exception:
            feature_names = None

    # If still not available, try model._Booster.feature_name()
    if not feature_names and hasattr(model, "_Booster"):
        try:
            feature_names = model._Booster.feature_name()
        except Exception:
            feature_names = None

    # If we have explicit feature names, select & reorder; otherwise keep input order
    if feature_names:
        missing = [f for f in feature_names if f not in features.columns]
        if missing:
            logger.warning(f"Model expects features not present in input: {missing}. Missing features will be filled with 0.")
            for m in missing:
                features[m] = 0.0
        # Reindex to model feature order (any extra columns will be dropped)
        features = features.reindex(columns=feature_names, fill_value=0.0)
    else:
        logger.debug("No feature ordering information found on model; using provided DataFrame column order.")

    return features


def predict(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using the loaded model.

    Returns a DataFrame with columns:
      - prediction: predicted class label
      - confidence: probability/confidence for predicted label (0..1)
      - proba_<label>: per-class probabilities (if available)

    If predict_proba is not available, confidence is set to 1.0 for the predicted class.
    """
    if not isinstance(features, pd.DataFrame):
        raise ValueError("features must be a pandas DataFrame")

    X = features.copy()
    X = _align_features(model, X)

    # If model supports predict_proba
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception as e:
            logger.warning(f"predict_proba failed: {e}")
            proba = None

    preds = None
    try:
        preds = model.predict(X)
    except Exception as e:
        # As a fallback, try model.predict on numpy array
        logger.error(f"model.predict failed: {e}")
        raise

    # Prepare output DataFrame
    out = pd.DataFrame(index=X.index)
    out["prediction"] = preds

    if proba is not None:
        # model.classes_ contains mapping from class index to label
        classes = getattr(model, "classes_", None)
        if classes is None:
            # attempt to infer classes from prediction output
            classes = np.arange(proba.shape[1])
        # Add per-class probability columns
        for i, cls in enumerate(classes):
            col = f"proba_{cls}"
            out[col] = proba[:, i]
        # Compute confidence as the max probability for the chosen class
        # For each row, find prob of predicted label
        confs = []
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        for idx, pred in enumerate(preds):
            i = cls_to_idx.get(pred, None)
            if i is None:
                confs.append(float(np.max(proba[idx])))
            else:
                confs.append(float(proba[idx, i]))
        out["confidence"] = confs
    else:
        # No probabilities available: set confidence to 1.0
        out["confidence"] = 1.0

    return out
