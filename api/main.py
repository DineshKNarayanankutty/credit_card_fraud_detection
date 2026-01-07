"""
FastAPI application layer.
Responsibility: HTTP endpoints ONLY.
Uses: InferencePipeline from pipelines.inference_pipeline
NO training imports, NO orchestration logic, NO ML logic.

Modern FastAPI patterns:
- Lifespan context manager (replaces @on_event)
- Configurable default threshold
- Model version tracking
"""

import logging
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.logger import setup_logging
from src.utils.config import config
from src.inference.schema import (
    TransactionInput,
    BatchTransactionInput,
    PredictionOutput,
    BatchPredictionOutput,
    HealthCheckResponse,
    ErrorResponse
)
from pipelines.inference_pipeline import InferencePipeline

# ------------------------------------------------------------
# Logging MUST be initialized early
# ------------------------------------------------------------
setup_logging()

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Global state
inference_pipeline: Optional[InferencePipeline] = None

# Model version (Azure ML injects this automatically if deployed there)
model_version: str = os.getenv("AZUREML_MODEL_VERSION", "local")

# Configurable threshold via environment or config
DEFAULT_THRESHOLD = float(os.getenv(
    "FRAUD_THRESHOLD",
    getattr(config.evaluation, "default_threshold", 0.5)
))

logger.info(f"Configurable threshold: {DEFAULT_THRESHOLD}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Lifespan (Modern Pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Startup: Initialize inference pipeline
    Shutdown: Clean up resources
    """
    global inference_pipeline, model_version

    # ─────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("API STARTUP")
    logger.info("=" * 60)

    try:
        inference_pipeline = InferencePipeline(
            model_path=config.model.model_path,
            scaler_path=config.model.preprocessor_path
        )

        health = inference_pipeline.health_check()
        if health["status"] != "healthy":
            raise RuntimeError("Pipeline health check failed on startup")

        logger.info("Inference pipeline initialized")
        logger.info("Model loaded and healthy")
        logger.info(f"Default threshold: {DEFAULT_THRESHOLD:.4f}")
        logger.info(f"Model version: {model_version}")
        logger.info("API ready to serve predictions")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise RuntimeError(f"API startup failed: {e}")

    yield

    # ─────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("API SHUTDOWN")
    logger.info("=" * 60)
    logger.info("Cleaning up resources...")
    inference_pipeline = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="Fraud Detection API",
    description="Cloud-native MLOps fraud detection service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health Check Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check() -> HealthCheckResponse:
    if inference_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )

    health = inference_pipeline.health_check()
    if health["status"] != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unhealthy: {health.get('error', 'Unknown error')}"
        )

    return HealthCheckResponse(
        status="healthy",
        model_loaded=True,
        version=model_version
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single Prediction Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Predictions"],
    summary="Predict fraud for single transaction",
)
async def predict_single(request: TransactionInput) -> PredictionOutput:
    if inference_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    result = inference_pipeline.predict_transaction(
        features=request.features,
        amount=request.amount,
        threshold=DEFAULT_THRESHOLD,
        timestamp=request.timestamp
    )

    return PredictionOutput(
        prediction=result["prediction"],
        probability=result["probability"],
        threshold=result["threshold"],
        inference_time_ms=result["inference_time_ms"]
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch Prediction Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Predictions"],
    summary="Predict fraud for batch of transactions",
)
async def predict_batch(request: BatchTransactionInput) -> BatchPredictionOutput:
    if inference_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    transactions = [
        {
            "features": t.features,
            "amount": t.amount,
            "timestamp": t.timestamp
        }
        for t in request.transactions
    ]

    result = inference_pipeline.predict_batch(
        transactions=transactions,
        threshold=request.threshold
    )

    return BatchPredictionOutput(
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        fraud_count=result["fraud_count"],
        fraud_rate=result["fraud_rate"],
        threshold=result["threshold"],
        inference_time_ms=result["inference_time_ms"],
        avg_time_per_sample_ms=result["avg_time_per_sample_ms"]
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Info Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/info", tags=["Info"])
async def get_info() -> Dict[str, Any]:
    return {
        "service": "Fraud Detection API",
        "api_version": "1.0.0",
        "model_version": model_version,
        "model_path": config.model.model_path,
        "scaler_path": config.model.preprocessor_path,
        "default_threshold": DEFAULT_THRESHOLD,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Root Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/", tags=["Info"])
async def root() -> Dict[str, str]:
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exception Handlers (FIXED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Fraud Detection API server...")
    logger.info(f"Default threshold: {DEFAULT_THRESHOLD}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
