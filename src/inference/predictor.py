"""
Inference predictor layer.

Responsibility:
- Load model & scaler
- Perform predictions ONLY

NO training
NO preprocessing
NO MLflow
"""

import logging
import time
from typing import Dict, Any, List

import numpy as np

from src.models.registry import load_model
from src.utils.io import load_pickle

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Production-ready fraud prediction service"""

    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize predictor with pre-trained model and scaler.
        """
        logger.info("Initializing FraudPredictor")

        self.model = load_model(model_path)
        self.scaler = load_pickle(scaler_path)

        # IMPORTANT: feature count inferred dynamically
        self.n_features: int | None = None

        self.model_path = model_path
        self.scaler_path = scaler_path

        logger.info("Predictor initialized successfully")

    # -----------------------------
    # INTERNAL VALIDATION
    # -----------------------------
    def _validate_features(self, X: np.ndarray) -> None:
        """
        Validate and infer feature dimensions.
        """
        if self.n_features is None:
            self.n_features = X.shape[1]
            logger.info(f"Inferred feature count = {self.n_features}")

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )

    # -----------------------------
    # SINGLE PREDICTION
    # -----------------------------
    def predict_single(
        self,
        features: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        start_time = time.time()

        X = np.array([features], dtype=np.float32)
        self._validate_features(X)

        X_scaled = self.scaler.transform(X)

        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = int(proba >= threshold)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "prediction": prediction,
            "probability": proba,
            "threshold": threshold,
            "inference_time_ms": latency_ms,
        }

    # -----------------------------
    # BATCH PREDICTION
    # -----------------------------
    def predict_batch(
        self,
        features_list: List[List[float]],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Predict fraud for a batch of transactions.
        """
        if not features_list:
            raise ValueError("features_list cannot be empty")

        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        start_time = time.time()

        X = np.array(features_list, dtype=np.float32)
        self._validate_features(X)

        X_scaled = self.scaler.transform(X)

        probas = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probas >= threshold).astype(int)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "predictions": predictions.tolist(),
            "probabilities": probas.astype(float).tolist(),
            "fraud_count": int(predictions.sum()),
            "fraud_rate": float(predictions.mean()),
            "threshold": threshold,
            "inference_time_ms": latency_ms,
            "avg_time_per_sample_ms": latency_ms / len(predictions),
        }

    # -----------------------------
    # HEALTH CHECK
    # -----------------------------
    def health_check(self) -> Dict[str, Any]:
        """
        Lightweight health check.
        """
        try:
            dummy = np.zeros((1, self.n_features or 1), dtype=np.float32)
            self._validate_features(dummy)

            dummy_scaled = self.scaler.transform(dummy)
            _ = self.model.predict_proba(dummy_scaled)

            return {
                "status": "healthy",
                "model_loaded": True,
            }

        except Exception as e:
            logger.exception("Health check failed")
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e),
            }


def get_predictor(
    model_path: str = "artifacts/model.pkl",
    scaler_path: str = "artifacts/scaler.pkl",
) -> FraudPredictor:
    """
    Factory function for predictor.
    """
    return FraudPredictor(model_path, scaler_path)
