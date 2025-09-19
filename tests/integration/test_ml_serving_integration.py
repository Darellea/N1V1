import pytest
import asyncio
import time
import threading
import requests
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import os
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import the serving module
from ml.serving import app
from ml.model_loader import predict as local_predict
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI test client for integration tests."""
    return TestClient(app)


@pytest.fixture
def sample_model():
    """Create a sample trained model for testing."""
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def temp_model_file(sample_model):
    """Create a temporary model file."""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        joblib.dump(sample_model, f.name)
        return f.name


@pytest.fixture
def sample_features():
    """Generate sample feature data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_0': np.random.randn(100),
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
        'feature_4': np.random.randn(100),
        'feature_5': np.random.randn(100),
        'feature_6': np.random.randn(100),
        'feature_7': np.random.randn(100),
        'feature_8': np.random.randn(100),
        'feature_9': np.random.randn(100)
    })


class TestMLServingIntegration:
    """Integration tests for ML serving infrastructure."""

    @patch('ml.serving.load_model_with_fallback')
    def test_single_prediction_integration(self, mock_load_fallback, client, sample_model, sample_features):
        """Test end-to-end single prediction flow."""
        mock_load_fallback.return_value = (sample_model, None)

        # Prepare request data
        features_dict = {col: sample_features[col].tolist() for col in sample_features.columns}
        request_data = {
            "model_name": "test_model",
            "features": features_dict,
            "correlation_id": "integration-test-001"
        }

        # Make prediction request
        start_time = time.time()
        response = client.post("/predict", json=request_data)
        end_time = time.time()

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "prediction" in data
        assert "confidence" in data
        assert "correlation_id" in data
        assert "latency_ms" in data
        assert data["correlation_id"] == "integration-test-001"

        # Verify latency is reasonable
        assert data["latency_ms"] > 0
        assert data["latency_ms"] < 1000  # Should be under 1 second

        # Verify prediction results
        assert len(data["prediction"]) == len(sample_features)
        assert len(data["confidence"]) == len(sample_features)
        assert all(isinstance(pred, (int, float)) for pred in data["prediction"])
        assert all(0 <= conf <= 1 for conf in data["confidence"])

    @patch('ml.serving.load_model_with_fallback')
    def test_batch_prediction_integration(self, mock_load_fallback, client, sample_model, sample_features):
        """Test end-to-end batch prediction flow."""
        mock_load_fallback.return_value = (sample_model, None)

        # Prepare batch request
        features_dict = {col: sample_features[col][:10].tolist() for col in sample_features.columns}
        request_data = {
            "model_name": "test_model",
            "features": features_dict,
            "correlation_id": "batch-test-001"
        }

        batch_request = {
            "requests": [request_data, request_data],
            "batch_size": 2
        }

        # Make batch prediction request
        start_time = time.time()
        response = client.post("/predict/batch", json=batch_request)
        end_time = time.time()

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Check batch response structure
        assert isinstance(data, list)
        assert len(data) == 2

        for item in data:
            assert "prediction" in item
            assert "confidence" in item
            assert "correlation_id" in item
            assert "latency_ms" in item
            assert len(item["prediction"]) == 10
            assert len(item["confidence"]) == 10

    def test_health_and_metrics_integration(self, client):
        """Test health check and metrics endpoints."""
        # Test health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        assert "models_loaded" in health_data

        # Test metrics endpoint
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_text = metrics_response.text
        assert "inference_requests_total" in metrics_text
        assert "inference_latency_seconds" in metrics_text

    @patch('ml.serving.load_model_with_fallback')
    def test_model_reloading_integration(self, mock_load_fallback, client, sample_model):
        """Test model reloading functionality."""
        mock_load_fallback.return_value = (sample_model, None)

        # Reload model
        response = client.post("/reload/test_model")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "reloading" in data["message"].lower()


class TestConcurrentLoad:
    """Test concurrent load handling."""

    @patch('ml.serving.load_model_with_fallback')
    def test_concurrent_predictions(self, mock_load_fallback, sample_model, sample_features):
        """Test handling of concurrent prediction requests."""
        mock_load_fallback.return_value = (sample_model, None)

        def make_prediction_request(request_id):
            """Make a single prediction request."""
            features_dict = {col: sample_features[col][:5].tolist() for col in sample_features.columns}
            request_data = {
                "model_name": "test_model",
                "features": features_dict,
                "correlation_id": f"concurrent-test-{request_id}"
            }

            client = TestClient(app)
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()

            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "latency": end_time - start_time,
                "success": response.status_code == 200
            }

        # Run concurrent requests
        num_requests = 10
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]

        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        latencies = [r["latency"] for r in results]

        # Verify performance
        assert len(successful_requests) == num_requests, f"Expected {num_requests} successful requests, got {len(successful_requests)}"

        # Check latency statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)

        print(f"Average latency: {avg_latency:.3f}s")
        print(f"95th percentile latency: {p95_latency:.3f}s")
        print(f"Max latency: {max_latency:.3f}s")

        # Assert reasonable performance (adjust thresholds based on environment)
        assert avg_latency < 1.0, f"Average latency too high: {avg_latency:.3f}s"
        assert p95_latency < 2.0, f"95th percentile latency too high: {p95_latency:.3f}s"


class TestAccuracyValidation:
    """Test prediction accuracy consistency."""

    @patch('ml.serving.load_model_with_fallback')
    def test_prediction_consistency(self, mock_load_fallback, sample_model, sample_features):
        """Test that serving predictions match local predictions."""
        mock_load_fallback.return_value = (sample_model, None)

        # Get local predictions
        local_result = local_predict(sample_model, sample_features.iloc[:10])

        # Get serving predictions
        client = TestClient(app)
        features_dict = {col: sample_features[col][:10].tolist() for col in sample_features.columns}
        request_data = {
            "model_name": "test_model",
            "features": features_dict
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        serving_result = response.json()

        # Compare predictions
        local_predictions = local_result['prediction'].tolist()
        serving_predictions = serving_result['prediction']

        # Predictions should match exactly
        assert local_predictions == serving_predictions, "Local and serving predictions don't match"

        # Confidence values should be very close (allowing for minor numerical differences)
        local_confidence = local_result['confidence'].tolist()
        serving_confidence = serving_result['confidence']

        for local_conf, serving_conf in zip(local_confidence, serving_confidence):
            assert abs(local_conf - serving_conf) < 1e-6, f"Confidence mismatch: {local_conf} vs {serving_conf}"


class TestFaultTolerance:
    """Test fault tolerance and error handling."""

    def test_invalid_model_name(self, client):
        """Test handling of invalid model names."""
        request_data = {
            "model_name": "nonexistent_model",
            "features": {"feature1": [1.0, 2.0]}
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 500

    def test_malformed_request(self, client):
        """Test handling of malformed requests."""
        # Missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422

        # Invalid feature format
        response = client.post("/predict", json={
            "model_name": "test",
            "features": "invalid_format"
        })
        assert response.status_code == 422

    @patch('ml.serving.load_model_with_fallback')
    def test_model_loading_failure(self, mock_load_fallback, client):
        """Test handling of model loading failures."""
        mock_load_fallback.side_effect = Exception("Model loading failed")

        request_data = {
            "model_name": "failing_model",
            "features": {"feature1": [1.0]}
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    @patch('ml.serving.load_model_with_fallback')
    def test_latency_under_load(self, mock_load_fallback, sample_model):
        """Test latency performance under load."""
        mock_load_fallback.return_value = (sample_model, None)

        # Generate test data
        np.random.seed(42)
        test_features = pd.DataFrame({
            f'feature_{i}': np.random.randn(50) for i in range(10)
        })

        client = TestClient(app)

        # Warm up
        features_dict = {col: test_features[col][:5].tolist() for col in test_features.columns}
        request_data = {
            "model_name": "test_model",
            "features": features_dict
        }

        for _ in range(3):
            client.post("/predict", json=request_data)

        # Benchmark
        latencies = []
        num_iterations = 20

        for _ in range(num_iterations):
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()

            assert response.status_code == 200
            latencies.append(end_time - start_time)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"Benchmark Results:")
        print(f"Average latency: {avg_latency*1000:.2f}ms")
        print(f"50th percentile: {p50_latency*1000:.2f}ms")
        print(f"95th percentile: {p95_latency*1000:.2f}ms")
        print(f"99th percentile: {p99_latency*1000:.2f}ms")

        # Assert performance targets
        assert p50_latency < 0.05, f"Median latency too high: {p50_latency*1000:.2f}ms (target: <50ms)"
        assert p95_latency < 0.1, f"95th percentile latency too high: {p95_latency*1000:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
