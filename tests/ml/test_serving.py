import asyncio
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml.serving import app, load_model, process_single_prediction


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample feature data for testing."""
    return {
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [0.5, 1.5, 2.5],
        "feature3": [10, 20, 30],
    }


@pytest.fixture
def sample_prediction_request(sample_features):
    """Sample prediction request."""
    return {
        "model_name": "test_model",
        "features": sample_features,
        "correlation_id": "test-123",
    }


class TestServingAPI:
    """Test cases for ML serving API endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "inference_requests_total" in response.text

    @patch("ml.serving.load_model")
    @patch("ml.serving.local_predict")
    def test_predict_single_success(
        self, mock_predict, mock_load_model, client, sample_prediction_request
    ):
        """Test successful single prediction."""
        # Mock the model and prediction
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Mock prediction result
        mock_result = pd.DataFrame(
            {
                "prediction": [1, 0, 1],
                "confidence": [0.9, 0.8, 0.95],
                "proba_0": [0.1, 0.2, 0.05],
                "proba_1": [0.9, 0.8, 0.95],
            }
        )
        mock_predict.return_value = mock_result

        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "correlation_id" in data
        assert "model_version" in data
        assert "latency_ms" in data
        assert len(data["prediction"]) == 3
        assert data["correlation_id"] == "test-123"

    def test_predict_single_invalid_request(self, client):
        """Test prediction with invalid request data."""
        invalid_request = {
            "model_name": "",  # Empty model name
            "features": {"invalid": "data"},
        }

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error

    @patch("ml.serving.load_model")
    def test_predict_single_model_not_found(
        self, mock_load_model, client, sample_prediction_request
    ):
        """Test prediction when model is not found."""
        mock_load_model.side_effect = Exception("Model not found")

        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_predict_batch_success(self, client, sample_prediction_request):
        """Test successful batch prediction."""
        batch_request = {
            "requests": [sample_prediction_request, sample_prediction_request],
            "batch_size": 2,
        }

        with patch("ml.serving.process_single_prediction") as mock_process:
            mock_process.return_value = {
                "prediction": [1, 0],
                "confidence": [0.9, 0.8],
                "correlation_id": "test-123",
                "model_version": "1.0.0",
                "latency_ms": 45.2,
            }

            response = client.post("/predict/batch", json=batch_request)
            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2
            assert all("prediction" in item for item in data)

    def test_predict_batch_empty_requests(self, client):
        """Test batch prediction with empty request list."""
        batch_request = {"requests": [], "batch_size": 10}

        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200

        data = response.json()
        assert data == []

    @patch("ml.serving.load_model")
    def test_reload_model_success(self, mock_load_model, client):
        """Test successful model reload."""
        mock_load_model.return_value = Mock()

        response = client.post("/reload/test_model")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "reloading" in data["message"].lower()

    def test_reload_model_invalid_name(self, client):
        """Test model reload with invalid model name."""
        response = client.post("/reload/")
        assert response.status_code == 404  # FastAPI path parameter validation


class TestModelLoading:
    """Test cases for model loading functionality."""

    @patch("ml.serving.load_model_with_fallback")
    def test_load_model_success(self, mock_load_fallback):
        """Test successful model loading."""
        mock_model = Mock()
        mock_model.version = "1.0.0"
        mock_load_fallback.return_value = (mock_model, None)

        result = load_model("test_model")
        assert result == mock_model

        # Verify model is cached
        result2 = load_model("test_model")
        assert result2 == mock_model
        # Should only call load_model_with_fallback once due to caching
        assert mock_load_fallback.call_count == 1

    @patch("ml.serving.load_model_with_fallback")
    def test_load_model_failure(self, mock_load_fallback):
        """Test model loading failure."""
        mock_load_fallback.side_effect = Exception("Load failed")

        with pytest.raises(Exception):
            load_model("invalid_model")


class TestPredictionProcessing:
    """Test cases for prediction processing logic."""

    @patch("ml.serving.load_model")
    @patch("ml.serving.local_predict")
    def test_process_single_prediction_success(self, mock_predict, mock_load_model):
        """Test successful single prediction processing."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        mock_result = pd.DataFrame(
            {"prediction": [1], "confidence": [0.9], "proba_0": [0.1], "proba_1": [0.9]}
        )
        mock_predict.return_value = mock_result

        request = Mock()
        request.model_name = "test_model"
        request.correlation_id = "test-123"
        request.features = {"feature1": [1.0]}

        result = process_single_prediction(request)

        assert "prediction" in result
        assert "confidence" in result
        assert "correlation_id" in result
        assert result["correlation_id"] == "test-123"

    @patch("ml.serving.load_model")
    def test_process_single_prediction_model_error(self, mock_load_model):
        """Test prediction processing with model loading error."""
        mock_load_model.side_effect = Exception("Model error")

        request = Mock()
        request.model_name = "test_model"
        request.correlation_id = "test-123"
        request.features = {"feature1": [1.0]}

        with pytest.raises(Exception):
            process_single_prediction(request)


class TestPerformanceMetrics:
    """Test cases for performance metrics collection."""

    @patch("ml.serving.load_model")
    @patch("ml.serving.local_predict")
    def test_metrics_collection_success(
        self, mock_predict, mock_load_model, client, sample_prediction_request
    ):
        """Test that metrics are collected for successful predictions."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        mock_result = pd.DataFrame({"prediction": [1], "confidence": [0.9]})
        mock_predict.return_value = mock_result

        # Make request
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

        # Check metrics endpoint
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_text = metrics_response.text

        # Verify inference metrics are present
        assert "inference_requests_total" in metrics_text
        assert "inference_latency_seconds" in metrics_text


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON in request."""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test handling of requests missing required fields."""
        incomplete_request = {
            "model_name": "test_model"
            # Missing features
        }

        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422

    @patch("ml.serving.load_model")
    @patch("ml.serving.local_predict")
    def test_prediction_timeout_simulation(
        self, mock_predict, mock_load_model, client, sample_prediction_request
    ):
        """Test handling of prediction timeouts (simulated)."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Simulate slow prediction
        mock_predict.side_effect = lambda *args, **kwargs: asyncio.sleep(
            35
        )  # Exceeds timeout

        # Note: In real scenario, this would be handled by request timeout
        # Here we just verify the error handling works
        mock_predict.side_effect = Exception("Prediction timeout")

        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__])
