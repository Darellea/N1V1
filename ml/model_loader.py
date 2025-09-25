import os
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import requests
import time

# Optional MLflow import
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration for remote inference
REMOTE_INFERENCE_ENABLED = os.getenv("REMOTE_INFERENCE_ENABLED", "false").lower() == "true"
REMOTE_INFERENCE_URL = os.getenv("REMOTE_INFERENCE_URL", "http://localhost:8000")
REMOTE_INFERENCE_TIMEOUT = int(os.getenv("REMOTE_INFERENCE_TIMEOUT", "30"))  # seconds


def load_model(path: str):
    """
    Load a model from disk. Supports joblib/pickle files.
    Returns the unpickled model instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        model = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return model
    except (Exception, TypeError) as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise ValueError(f"Model file at {path} is corrupted or of wrong format: {e}") from e


def load_model_with_card(identifier: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a model and its companion model card (if present).

    Supports both raw model names and full paths.
    The model card is expected at the same path with extension '.model_card.json'.
    Returns (model, model_card_dict|None)
    """
    # Allow both "test_model" and "models/test_model.pkl"
    if not identifier.endswith((".pkl", ".joblib", ".model")):
        path = os.path.join("models", f"{identifier}.pkl")
    else:
        path = identifier

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
    if feature_names and isinstance(feature_names, (list, tuple)):
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


def remote_predict(model_name: str, features: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using remote inference server.

    Args:
        model_name: Name of the model to use
        features: Feature DataFrame

    Returns:
        DataFrame with predictions
    """
    try:
        # Convert features to dict format for API
        features_dict = {col: features[col].tolist() for col in features.columns}

        payload = {
            "model_name": model_name,
            "features": features_dict,
            "correlation_id": str(np.random.randint(1000000))  # Simple correlation ID
        }

        start_time = time.time()
        response = requests.post(
            f"{REMOTE_INFERENCE_URL}/predict",
            json=payload,
            timeout=REMOTE_INFERENCE_TIMEOUT
        )
        latency = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Remote prediction successful for {model_name}, latency: {latency:.3f}s")

            # Convert response back to DataFrame
            out = pd.DataFrame(index=features.index)
            out["prediction"] = result["prediction"]
            out["confidence"] = result["confidence"]

            if result.get("probabilities"):
                for prob_key, prob_values in result["probabilities"].items():
                    out[prob_key] = prob_values

            return out
        else:
            logger.warning(f"Remote prediction failed with status {response.status_code}: {response.text}")
            raise Exception(f"Remote inference failed: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.warning(f"Remote inference request failed: {e}")
        raise Exception(f"Remote inference unavailable: {e}")


def predict(model, features: pd.DataFrame, model_name: Optional[str] = None) -> pd.DataFrame:
    """
    Make predictions using the loaded model or remote inference.

    If REMOTE_INFERENCE_ENABLED is True and model_name is provided,
    uses remote inference with fallback to local inference.

    Returns a DataFrame with columns:
      - prediction: predicted class label
      - confidence: probability/confidence for predicted label (0..1)
      - proba_<label>: per-class probabilities (if available)

    If predict_proba is not available, confidence is set to 1.0 for the predicted class.
    """
    if not isinstance(features, pd.DataFrame):
        raise ValueError("features must be a pandas DataFrame")

    # Try remote inference first if enabled and model_name provided
    if REMOTE_INFERENCE_ENABLED and model_name:
        try:
            return remote_predict(model_name, features)
        except Exception as e:
            logger.warning(f"Remote inference failed, falling back to local: {e}")

    # Local inference fallback
    X = features.copy()
    X = _align_features(model, X)

    # Convert to numpy array for sklearn models
    X_array = X.values if hasattr(X, 'values') else X

    # If model supports predict_proba
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_array)
        except Exception as e:
            logger.warning(f"predict_proba failed: {e}")
            proba = None

    preds = None
    try:
        preds = model.predict(X_array)
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
        # Ensure confs has the same length as out; fallback if empty
        if not confs:
            confs = [1.0] * len(out)
        out["confidence"] = confs
    else:
        # No probabilities available: set confidence to 1.0
        out["confidence"] = 1.0

    return out


def load_model_from_registry(model_name: str, version: str = None, experiment_name: str = None) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a model from MLflow model registry with fallback to local files.

    Args:
        model_name: Name of the model in registry
        version: Specific version to load (latest if None)
        experiment_name: Experiment name to search in

    Returns:
        Tuple of (model, model_card_dict)
    """
    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow not available")

    try:
        # Try to load from MLflow model registry first
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"

        logger.info(f"Attempting to load model from MLflow registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)

        # Try to get model card from run artifacts
        model_card = None
        try:
            # Get the run that created this model version
            client = mlflow.tracking.MlflowClient()
            if version:
                mv = client.get_model_version(model_name, version)
                run_id = mv.run_id
            else:
                # Get latest version
                latest_version = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])[0]
                run_id = latest_version.run_id

            # Try to download environment snapshot artifact
            if run_id:
                artifacts = client.list_artifacts(run_id)
                for artifact in artifacts:
                    if artifact.path == "environment_snapshot.json":
                        client.download_artifacts(run_id, artifact.path, ".")
                        with open("environment_snapshot.json", "r") as f:
                            model_card = json.load(f)
                        os.remove("environment_snapshot.json")  # Clean up
                        break

        except Exception as e:
            logger.debug(f"Could not load model card from registry: {e}")

        logger.info(f"Successfully loaded model {model_name} from MLflow registry")
        return model, model_card

    except Exception as e:
        logger.warning(f"Failed to load model {model_name} from MLflow registry: {e}")

        # Fallback: try to load local model file
        logger.info(f"Attempting fallback to local model file for {model_name}")
        return load_model_with_card(model_name)


def load_model_with_fallback(model_path_or_name: str, use_registry: bool = True) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a model with intelligent fallback: registry first, then local files.

    Args:
        model_path_or_name: Either a file path or model name (for registry lookup)
        use_registry: Whether to try registry first

    Returns:
        Tuple of (model, model_card_dict)
    """
    # If it's a file path, try loading directly
    if os.path.exists(model_path_or_name):
        logger.info(f"Loading model from file path: {model_path_or_name}")
        return load_model_with_card(model_path_or_name)

    # If registry is enabled and available, try registry first
    if use_registry and MLFLOW_AVAILABLE:
        try:
            return load_model_from_registry(model_path_or_name)
        except Exception as e:
            logger.warning(f"Registry loading failed for {model_path_or_name}: {e}")

    # Fallback to local file search
    logger.info(f"Attempting local file loading for {model_path_or_name}")
    return load_model_with_card(model_path_or_name)
