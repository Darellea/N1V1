(venv) C:\Users\TU\Desktop\new project\N1V1>pytest -q
.....sss........................................................................................................ [  3%]
................................................................................................................ [  6%]
................................................................................................................ [  9%]
................................................................................................................ [ 12%]
................................................................................................................ [ 16%]
................................................................................................................ [ 19%]
................................................................................................................ [ 22%]
...............................................................................................................s [ 25%]
sssssssssss..................................................................................................... [ 28%]
................................................................................................................ [ 32%]
................................................................................................................ [ 35%]
................................................................................................................ [ 38%]
...........................s.sssss.........................ss................................................... [ 41%]
................................................................................................................ [ 44%]
.......................................................F......F..........................F...................... [ 48%]
................................................................................................................ [ 51%]
................................................................s........ssss................................... [ 54%]
................................................................................................................ [ 57%]
.....DEBUG: check_drift called, ref_features=True, buffer_size=1000
........................................................................................................... [ 61%]
............................................................................s................................... [ 64%]
.....................................................sssss...................................................... [ 67%]
................................................................................................................ [ 70%]
................................................................................................................ [ 73%]
..............................................................sss...............................ss.............. [ 77%]
.......sssssssssssss.ssss..............sssss....s.s....ss....................................................... [ 80%]
............................................................s.sssssssssssssssss....sssssssssssssssss............ [ 83%]
................................................................................................................ [ 86%]
................................................................................................................ [ 89%]
................................................................................................................ [ 93%]
................................................................................................................ [ 96%]
................................................................................................................ [ 99%]
..............                                                                                                   [100%]
====================================================== FAILURES =======================================================
___________________ TestMonitoringPerformanceIntegration.test_performance_metrics_monitoring_system ___________________

self = <test_cross_feature_integration.TestMonitoringPerformanceIntegration object at 0x0000019C4A2C7940>

    @pytest.mark.asyncio
    async def test_performance_metrics_monitoring_system(self):
        """Test that performance metrics are accurately captured in monitoring system."""
        await self.monitor.start_monitoring()

        # Start profiling session
        self.profiler.start_profiling("test_session")

        # Perform profiled operations with deterministic workloads
        operations = ["fast_operation", "medium_operation", "slow_operation"]
        # Increased expected times to accommodate profiler overhead and system variance
        expected_times = [0.01, 0.1, 0.5]

        def deterministic_workload(duration_target):
            """Deterministic CPU workload to replace time.sleep() for stable timing."""
            start = time.perf_counter()
            iterations = 0
            # Perform CPU work until target duration is reached
            while time.perf_counter() - start < duration_target:
                # Simple CPU-bound operation
                _ = sum(i * i for i in range(100))
                iterations += 1
            return iterations

        for op, expected_time in zip(operations, expected_times):
            with self.profiler.profile_function(op):
                deterministic_workload(expected_time)

        # Stop profiling session
        self.profiler.stop_profiling()

        # Check that monitoring captured performance data
        status = await self.monitor.get_performance_status()

        # Should have performance baselines
>       assert status["total_baselines"] > 0
E       assert 0 > 0

tests\integration\test_cross_feature_integration.py:382: AssertionError
------------------------------------------------ Captured stdout setup ------------------------------------------------
2025-10-06 17:21:37 - slowapi - INFO - Storage has been reset and all limits cleared
------------------------------------------------- Captured log setup --------------------------------------------------
INFO     slowapi:extension.py:360 Storage has been reset and all limits cleared
------------------------------------------------ Captured stdout call -------------------------------------------------
2025-10-06 17:21:37 - core.performance_monitor - WARNING - WARNING -  | {"message": "Baseline file is empty"}
___________________________ TestFullSystemIntegration.test_performance_regression_detection ___________________________

self = <test_cross_feature_integration.TestFullSystemIntegration object at 0x0000019C4A2E9420>

    @pytest.mark.asyncio
    async def test_performance_regression_detection(self):
        """Test detection of performance regressions across features."""
        await self.monitor.start_monitoring()

        # Establish performance baseline
        baseline_times = []
        for i in range(20):
            start = time.perf_counter()
            with self.profiler.profile_function("baseline_func"):
                data = np.random.random(1000)
                result = np.sum(data**2)
            baseline_times.append(time.perf_counter() - start)

        baseline_avg = np.mean(baseline_times)

        # Simulate performance regression
        regression_times = []
        for i in range(20):
            start = time.perf_counter()
            with self.profiler.profile_function("regression_func"):
                # Simulate slower operation (regression)
                data = np.random.random(1000)
                result = np.sum(data**2)
                time.sleep(0.005)  # Artificial delay
            regression_times.append(time.perf_counter() - start)

        regression_avg = np.mean(regression_times)

        # Verify regression is detected
        regression_ratio = regression_avg / baseline_avg
        assert (
            regression_ratio > 1.5
        ), f"Regression not significant enough: {regression_ratio:.2f}x"

        # Check that monitoring detects the change
        status = await self.monitor.get_performance_status()

        # Should have captured performance data
>       assert status["total_baselines"] > 0
E       assert 0 > 0

tests\integration\test_cross_feature_integration.py:726: AssertionError
------------------------------------------------ Captured stdout setup ------------------------------------------------
2025-10-06 17:21:41 - slowapi - INFO - Storage has been reset and all limits cleared
------------------------------------------------- Captured log setup --------------------------------------------------
INFO     slowapi:extension.py:360 Storage has been reset and all limits cleared
------------------------------------------------ Captured stdout call -------------------------------------------------
2025-10-06 17:21:41 - core.performance_monitor - WARNING - WARNING -  | {"message": "Baseline file is empty"}
__________________________________ TestPerformanceBenchmarks.test_latency_under_load __________________________________

self = <test_ml_serving_integration.TestPerformanceBenchmarks object at 0x0000019C4A3C8160>
mock_load_fallback = <MagicMock name='load_model_with_fallback' id='1770826669344'>
sample_model = RandomForestClassifier(n_estimators=10, random_state=42)

    @patch("ml.serving.load_model_with_fallback")
    def test_latency_under_load(self, mock_load_fallback, sample_model):
        """Test latency performance under load."""
        mock_load_fallback.return_value = (sample_model, None)

        # Generate test data
        np.random.seed(42)
        test_features = pd.DataFrame(
            {f"feature_{i}": np.random.randn(50) for i in range(10)}
        )

        client = TestClient(app)

        # Warm up
        features_dict = {
            col: test_features[col][:5].tolist() for col in test_features.columns
        }
        request_data = {"model_name": "test_model", "features": features_dict}

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

        print("Benchmark Results:")
        print(f"Average latency: {avg_latency*1000:.2f}ms")
        print(f"50th percentile: {p50_latency*1000:.2f}ms")
        print(f"95th percentile: {p95_latency*1000:.2f}ms")
        print(f"99th percentile: {p99_latency*1000:.2f}ms")

        # Assert performance targets
>       assert (
            p50_latency < 0.05
        ), f"Median latency too high: {p50_latency*1000:.2f}ms (target: <50ms)"
E       AssertionError: Median latency too high: 50.94ms (target: <50ms)
E       assert 0.050940871238708496 < 0.05

tests\integration\test_ml_serving_integration.py:386: AssertionError
------------------------------------------------ Captured stdout setup ------------------------------------------------
2025-10-06 17:21:46 - slowapi - INFO - Storage has been reset and all limits cleared
------------------------------------------------- Captured log setup --------------------------------------------------
INFO     slowapi:extension.py:360 Storage has been reset and all limits cleared
------------------------------------------------ Captured stdout call -------------------------------------------------
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 213f2c39-5b44-437a-8651-8af0f5ba07d4, latency: 7.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: d89020be-ef23-4613-b574-56fe9f64f098, latency: 8.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 46f9a5eb-d2d0-460e-8d7e-473a43f34ffe, latency: 8.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: a11f238a-c6b9-4595-b017-a835f4dfe3e4, latency: 7.16ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: d264e13a-c61c-49ac-8df7-1e86911d325b, latency: 8.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 68a83e08-14e3-4084-b240-8f4e8dea15dc, latency: 6.14ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 7c6686c2-8cee-4076-b466-a4288001fe18, latency: 7.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: de708a96-059b-4061-a223-5d0c1edfdb70, latency: 7.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: bd56f353-a198-4036-aba3-4f5e7d0bc80d, latency: 9.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 883ee282-534d-4b28-9d27-c1cefe70e577, latency: 6.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: ffaf1b8a-8859-4f3b-8301-8dd211fec233, latency: 6.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:46 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 624d40fb-7a44-45e4-9a55-e283cf979a1e, latency: 6.00ms
2025-10-06 17:21:46 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 7feec062-452d-4069-b10c-537b31d56d48, latency: 8.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 682dfe6a-c7a5-4250-acfa-1e3f7bf95e22, latency: 7.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 502f3e90-0d9a-40b8-bb02-fbbbe1197bc2, latency: 6.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: ab9e7779-852b-493a-a3fc-7edc41a72c98, latency: 9.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 23888891-d58e-4853-aa44-1a416d198541, latency: 8.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 606e750d-28b1-43bd-b47d-63cb6cd949cc, latency: 7.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: fbb598c1-b499-4818-a5ac-8e0018db1a72, latency: 11.99ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 9ca18117-0ba9-4425-8981-8783f4bf09dd, latency: 7.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: e1873f83-fbb2-4a17-b11f-021d372f44a6, latency: 7.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: ce22b982-a71d-4d93-a9eb-7440575ab60e, latency: 7.94ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
2025-10-06 17:21:47 - ml.serving - INFO - Prediction completed for test_model, correlation_id: 93436f5f-43a9-40a4-a520-884562cd89af, latency: 7.00ms
2025-10-06 17:21:47 - httpx - INFO - HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
Benchmark Results:
Average latency: 49.78ms
50th percentile: 50.94ms
95th percentile: 57.61ms
99th percentile: 67.38ms
-------------------------------------------------- Captured log call --------------------------------------------------
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 213f2c39-5b44-437a-8651-8af0f5ba07d4, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: d89020be-ef23-4613-b574-56fe9f64f098, latency: 8.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 46f9a5eb-d2d0-460e-8d7e-473a43f34ffe, latency: 8.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: a11f238a-c6b9-4595-b017-a835f4dfe3e4, latency: 7.16ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: d264e13a-c61c-49ac-8df7-1e86911d325b, latency: 8.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 68a83e08-14e3-4084-b240-8f4e8dea15dc, latency: 6.14ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 7c6686c2-8cee-4076-b466-a4288001fe18, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: de708a96-059b-4061-a223-5d0c1edfdb70, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: bd56f353-a198-4036-aba3-4f5e7d0bc80d, latency: 9.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 883ee282-534d-4b28-9d27-c1cefe70e577, latency: 6.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: ffaf1b8a-8859-4f3b-8301-8dd211fec233, latency: 6.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 624d40fb-7a44-45e4-9a55-e283cf979a1e, latency: 6.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 7feec062-452d-4069-b10c-537b31d56d48, latency: 8.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 682dfe6a-c7a5-4250-acfa-1e3f7bf95e22, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 502f3e90-0d9a-40b8-bb02-fbbbe1197bc2, latency: 6.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: ab9e7779-852b-493a-a3fc-7edc41a72c98, latency: 9.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 23888891-d58e-4853-aa44-1a416d198541, latency: 8.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 606e750d-28b1-43bd-b47d-63cb6cd949cc, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: fbb598c1-b499-4818-a5ac-8e0018db1a72, latency: 11.99ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 9ca18117-0ba9-4425-8981-8783f4bf09dd, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: e1873f83-fbb2-4a17-b11f-021d372f44a6, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: ce22b982-a71d-4d93-a9eb-7440575ab60e, latency: 7.94ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
INFO     ml.serving:serving.py:128 Prediction completed for test_model, correlation_id: 93436f5f-43a9-40a4-a520-884562cd89af, latency: 7.00ms
INFO     httpx:_client.py:1013 HTTP Request: POST http://testserver/predict "HTTP/1.1 200 OK"
=============================================== short test summary info ===============================================
FAILED tests/integration/test_cross_feature_integration.py::TestMonitoringPerformanceIntegration::test_performance_metrics_monitoring_system - assert 0 > 0
FAILED tests/integration/test_cross_feature_integration.py::TestFullSystemIntegration::test_performance_regression_detection - assert 0 > 0
FAILED tests/integration/test_ml_serving_integration.py::TestPerformanceBenchmarks::test_latency_under_load - AssertionError: Median latency too high: 50.94ms (target: <50ms)
3 failed, 3383 passed, 100 skipped, 1195 warnings in 1313.24s (0:21:53)