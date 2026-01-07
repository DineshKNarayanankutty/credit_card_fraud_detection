"""
Inference schema layer.
Responsibility: Validate input/output data.
No business logic.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np


# ─────────────────────────────────────────
# INPUT SCHEMAS
# ─────────────────────────────────────────

class TransactionInput(BaseModel):
    """Single transaction for fraud prediction"""

    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 PCA-transformed features"
    )
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if any(np.isnan(f) or np.isinf(f) for f in v):
            raise ValueError("Features contain NaN or Inf")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.1] * 30,
                "amount": 150.0,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class BatchTransactionInput(BaseModel):
    """Batch of transactions for fraud prediction"""

    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of transactions"
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "transactions": [
                    {"features": [0.1] * 30, "amount": 150.0},
                    {"features": [0.2] * 30, "amount": 200.0}
                ],
                "threshold": 0.5
            }
        }
    }


# ─────────────────────────────────────────
# OUTPUT SCHEMAS
# ─────────────────────────────────────────

class PredictionOutput(BaseModel):
    """Single prediction output"""

    prediction: int = Field(..., description="Fraud prediction (0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    inference_time_ms: float


class BatchPredictionOutput(BaseModel):
    """Batch prediction output"""

    predictions: List[int]
    probabilities: List[float]
    fraud_count: int
    fraud_rate: float
    threshold: float
    inference_time_ms: float
    avg_time_per_sample_ms: float


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

    model_config = {
        "protected_namespaces": ()
    }


class ErrorResponse(BaseModel):
    """Error response schema"""

    detail: str
