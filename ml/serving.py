import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import joblib
import os

# Import local modules
from .model_loader import load_model_with_fallback, predict as local_predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total inference requests', ['model_name', 'status'])
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency in seconds', ['model_name'])

# Global variables for model management
models = {}  # model_name -> model_instance
model_versions = {}  # model_name -> version
executor = ThreadPoolExecutor(max_workers=4)

class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    features: Dict[str, List[float]] = Field(..., description="Feature data as dict of lists")
    correlation_id: Optional[str] = Field(None, description="Optional correlation ID for tracing")

    model_config = {"protected_namespaces": ()}

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")
    batch_size: Optional[int] = Field(10, description="Batch size for processing")

class PredictionResponse(BaseModel):
    prediction: List[Any]
    confidence: List[float]
    probabilities: Optional[Dict[str, List[float]]] = None
    correlation_id: str
    model_version: Optional[str] = None
    latency_ms: float

app = FastAPI(title="N1V1 ML Serving API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model(model_name: str) -> Any:
    """Load model if not already loaded."""
    if model_name not in models:
        try:
            model, _ = load_model_with_fallback(model_name)
            models[model_name] = model
            model_versions[model_name] = getattr(model, 'version', '1.0.0')
            logger.info(f"Loaded model {model_name} version {model_versions[model_name]}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Model {model_name} not found")
    return models[model_name]

def process_single_prediction(request: PredictionRequest) -> Dict[str, Any]:
    """Process a single prediction request."""
    start_time = time.time()
    correlation_id = request.correlation_id or str(uuid.uuid4())

    try:
        model = load_model(request.model_name)
        features_df = pd.DataFrame(request.features)

        # Run prediction in thread pool to avoid blocking
        loop = asyncio.new_event_loop()
        result_df = loop.run_until_complete(
            asyncio.get_event_loop().run_in_executor(
                executor, local_predict, model, features_df
            )
        )

        latency_ms = (time.time() - start_time) * 1000

        response = PredictionResponse(
            prediction=result_df['prediction'].tolist(),
            confidence=result_df['confidence'].tolist(),
            probabilities={col: result_df[col].tolist() for col in result_df.columns if col.startswith('proba_')} if any(col.startswith('proba_') for col in result_df.columns) else None,
            correlation_id=correlation_id,
            model_version=model_versions.get(request.model_name),
            latency_ms=latency_ms
        )

        INFERENCE_REQUESTS.labels(model_name=request.model_name, status='success').inc()
        INFERENCE_LATENCY.labels(model_name=request.model_name).observe(latency_ms / 1000)

        logger.info(f"Prediction completed for {request.model_name}, correlation_id: {correlation_id}, latency: {latency_ms:.2f}ms")
        return response.dict()

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        INFERENCE_REQUESTS.labels(model_name=request.model_name, status='error').inc()
        logger.error(f"Prediction failed for {request.model_name}, correlation_id: {correlation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Single prediction endpoint."""
    return process_single_prediction(request)

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    tasks = []
    for req in request.requests:
        tasks.append(process_single_prediction(req))

    # Process in batches
    results = []
    batch_size = request.batch_size
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*[asyncio.to_thread(lambda r=r: process_single_prediction(r)) for r in batch])
        results.extend(batch_results)

    return results

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.post("/reload/{model_name}")
async def reload_model(model_name: str, background_tasks: BackgroundTasks):
    """Reload a specific model."""
    if model_name in models:
        del models[model_name]
        del model_versions[model_name]

    background_tasks.add_task(load_model, model_name)
    return {"message": f"Reloading model {model_name}"}

@app.on_event("startup")
async def startup_event():
    """Preload models on startup."""
    # Preload common models - customize as needed
    preload_models = os.getenv("PRELOAD_MODELS", "").split(",")
    for model_name in preload_models:
        if model_name.strip():
            try:
                load_model(model_name.strip())
            except Exception as e:
                logger.warning(f"Failed to preload model {model_name}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
