"""
Inference schema layer.
Responsibility: Validate input/output data.
No business logic.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
import numpy as np


class TransactionInput(BaseModel):
    """Single transaction for fraud prediction"""
    
    features: List[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="30 PCA-transformed features"
    )
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.1] * 30,
                "amount": 150.0,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    
    @validator('features')
    def validate_features(cls, v):
        """Ensure features are valid numbers"""
        if any(np.isnan(f) or np.isinf(f) for f in v):
            raise ValueError("Features contain NaN or Inf")
        return v


class BatchTransactionInput(BaseModel):
    """Batch of transactions for fraud prediction"""
    
    transactions: List[TransactionInput] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of transactions"
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {"features": [0.1] * 30, "amount": 150.0},
                    {"features": [0.2] * 30, "amount": 200.0}
                ],
                "threshold": 0.5
            }
        }


class PredictionOutput(BaseModel):
    """Single prediction output"""
    
    pre
