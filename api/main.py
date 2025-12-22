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

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Global state
inference_pipeline: Optional[InferencePipeline] = None
model_version: str = "1.0.0"  # Updated from registry (Azure ML) later

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
    
    Modern pattern replacing @app.on_event("startup") and @app.on_event("shutdown").
    Recommended for FastAPI 0.93+.
    
    Startup: Initialize inference pipeline
    Shutdown: Clean up resources
    """
    global inference_pipeline, model_version
    
    # ─────────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("API STARTUP")
    logger.info("="*60)
    
    try:
        # Initialize inference pipeline
        inference_pipeline = InferencePipeline(
            model_path=config.model.model_path,
            scaler_path=config.model.preprocessor_path
        )
        
        # Verify health
        health = inference_pipeline.health_check()
        if health['status'] != 'healthy':
            raise RuntimeError("Pipeline health check failed on startup")
        
        logger.info("Inference pipeline initialized")
        logger.info("Model loaded and healthy")
        logger.info(f"Default threshold: {DEFAULT_THRESHOLD:.4f}")
        logger.info(f"Model version: {model_version}")
        logger.info("API ready to serve predictions")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise RuntimeError(f"API startup failed: {e}")
    
    yield  # Application runs here
    
    # ─────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("API SHUTDOWN")
    logger.info("="*60)
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
    lifespan=lifespan,  # Modern FastAPI pattern
)

# Add CORS middleware for cross-origin requests
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
    """
    Check API and model health.
    
    Returns:
        HealthCheckResponse with status and model_loaded flag
    
    Raises:
        HTTPException: If pipeline not initialized or unhealthy
    """
    if inference_pipeline is None:
        logger.error("Pipeline not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        health = inference_pipeline.health_check()
        
        if health['status'] != 'healthy':
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model unhealthy: {health.get('error', 'Unknown error')}"
            )
        
        return HealthCheckResponse(
            status="healthy",
            model_loaded=True,
            version=model_version
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single Prediction Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Predictions"],
    summary="Predict fraud for single transaction",
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid input"},
        503: {"description": "Model not available"},
    }
)
async def predict_single(request: TransactionInput) -> PredictionOutput:
    """
    Predict fraud probability for a single transaction.
    
    Uses configurable default threshold (from environment or config).
    
    Args:
        request: TransactionInput with features, amount, and optional timestamp
    
    Returns:
        PredictionOutput with prediction, probability, threshold, and inference_time_ms
    
    Raises:
        HTTPException: If prediction fails or model unavailable
    """
    if inference_pipeline is None:
        logger.error("Pipeline not initialized for prediction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        logger.info(f"Prediction request: amount={request.amount}, threshold={DEFAULT_THRESHOLD}")
        
        result = inference_pipeline.predict_transaction(
            features=request.features,
            amount=request.amount,
            threshold=DEFAULT_THRESHOLD,  # Configurable threshold
            timestamp=request.timestamp
        )
        
        logger.info(f"Prediction result: {result['prediction']}")
        
        return PredictionOutput(
            prediction=result['prediction'],
            probability=result['probability'],
            threshold=result['threshold'],
            inference_time_ms=result['inference_time_ms']
        )
    
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch Prediction Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Predictions"],
    summary="Predict fraud for batch of transactions",
    responses={
        200: {"description": "Batch prediction successful"},
        400: {"description": "Invalid batch"},
        503: {"description": "Model not available"},
    }
)
async def predict_batch(request: BatchTransactionInput) -> BatchPredictionOutput:
    """
    Predict fraud for multiple transactions in a batch.
    
    Allows custom threshold per batch request.
    
    Args:
        request: BatchTransactionInput with list of transactions and threshold
    
    Returns:
        BatchPredictionOutput with predictions, fraud_count, fraud_rate, etc.
    
    Raises:
        HTTPException: If batch prediction fails or model unavailable
    """
    if inference_pipeline is None:
        logger.error("Pipeline not initialized for batch prediction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        logger.info(f"Batch prediction request: {len(request.transactions)} transactions, threshold={request.threshold}")
        
        # Convert to dict format for pipeline
        transactions = [
            {
                'features': t.features,
                'amount': t.amount,
                'timestamp': t.timestamp
            }
            for t in request.transactions
        ]
        
        result = inference_pipeline.predict_batch(
            transactions=transactions,
            threshold=request.threshold
        )
        
        logger.info(f"Batch prediction complete: {result['fraud_count']} frauds detected")
        
        return BatchPredictionOutput(
            predictions=result['predictions'],
            probabilities=result['probabilities'],
            fraud_count=result['fraud_count'],
            fraud_rate=result['fraud_rate'],
            threshold=result['threshold'],
            inference_time_ms=result['inference_time_ms'],
            avg_time_per_sample_ms=result['avg_time_per_sample_ms']
        )
    
    except ValueError as e:
        logger.warning(f"Invalid batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Info Endpoint (with Model Version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get(
    "/info",
    tags=["Info"],
    summary="Get API and model information"
)
async def get_info() -> Dict[str, Any]:
    """
    Get information about the API and model.
    
    Includes:
    - API version
    - Model version (from registry/Azure ML)
    - Default threshold
    - Configured paths
    - Available endpoints
    
    Returns:
        Dict with comprehensive service information
    """
    return {
        "service": "Fraud Detection API",
        "api_version": "1.0.0",
        "model_version": model_version,  # From Azure ML registry later
        "model_path": config.model.model_path,
        "scaler_path": config.model.preprocessor_path,
        "default_threshold": DEFAULT_THRESHOLD,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "info": "/info",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Root Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/", tags=["Info"])
async def root() -> Dict[str, str]:
    """Root endpoint with links to documentation."""
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "info": "/info"
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Error Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {exc}")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    setup_logging()
    
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
