"""
Inference predictor layer.
Responsibility: Load model & scaler, predict only.
NO training, NO preprocessing, NO model creation.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import time
from xgboost import XGBClassifier
from src.data.scaler import FeatureScaler
from src.models.registry import load_model
from src.utils.io import load_pickle

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Production-ready fraud prediction service"""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize predictor with pre-trained model and scaler.
        
        Args:
            model_path: Path to trained XGBoost model
            scaler_path: Path to fitted FeatureScaler
        
        Raises:
            FileNotFoundError: If model or scaler not found
        """
        logger.info("Initializing FraudPredictor...")
        
        # Load model
        self.model = load_model(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        
        # Load scaler
        try:
            self.scaler = load_pickle(scaler_path)
            logger.info(f"✓ Scaler loaded from {scaler_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        self.model_path = model_path
        self.scaler_path = scaler_path
    
    def predict_single(
        self,
        features: List[float],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict fraud for single transaction.
        
        Args:
            features: List of 30 features
            threshold: Classification threshold (0-1)
        
        Returns:
            Dict with prediction, probability, threshold, inference_time_ms
        
        Raises:
            ValueError: If features invalid
        """
        if len(features) != 30:
            raise ValueError(f"Expected 30 features, got {len(features)}")
        
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be 0-1, got {threshold}")
        
        start_time = time.time()
        
        # Convert to array and scale
        X = np.array([features], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Get prediction
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[0, 1]
        
        # Apply threshold
        final_pred = 1 if y_proba >= threshold else 0
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return {
            'prediction': int(final_pred),
            'probability': float(y_proba),
            'threshold': float(threshold),
            'inference_time_ms': float(inference_time_ms),
        }
    
    def predict_batch(
        self,
        features_list: List[List[float]],
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Predict fraud for batch of transactions.
        
        Args:
            features_list: List of feature lists (each 30 features)
            threshold: Classification threshold (0-1)
            batch_size: Batch size for processing
        
        Returns:
            Dict with predictions, probabilities, fraud_count, fraud_rate, etc.
        
        Raises:
            ValueError: If features invalid
        """
        if not features_list:
            raise ValueError("features_list cannot be empty")
        
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be 0-1, got {threshold}")
        
        start_time = time.time()
        
        # Validate input
        for i, features in enumerate(features_list):
            if len(features) != 30:
                raise ValueError(f"Sample {i}: expected 30 features, got {len(features)}")
        
        # Convert to array and scale
        X = np.array(features_list, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Predict in batches
        all_preds = []
        all_probas = []
        
        for i in range(0, len(X_scaled), batch_size):
            batch_end = min(i + batch_size, len(X_scaled))
            batch = X_scaled[i:batch_end]
            
            batch_preds = self.model.predict(batch)
            batch_probas = self.model.predict_proba(batch)[:, 1]
            
            all_preds.extend(batch_preds)
            all_probas.extend(batch_probas)
        
        # Apply threshold
        final_preds = [1 if p >= threshold else 0 for p in all_probas]
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return {
            'predictions': final_preds,
            'probabilities': [float(p) for p in all_probas],
            'fraud_count': int(sum(final_preds)),
            'fraud_rate': float(sum(final_preds) / len(final_preds)),
            'threshold': float(threshold),
            'inference_time_ms': float(inference_time_ms),
            'avg_time_per_sample_ms': float(inference_time_ms / len(final_preds)),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if predictor is healthy.
        
        Returns:
            Dict with status and model_loaded
        """
        try:
            # Try a dummy prediction
            dummy_features = [[0.0] * 30]
            _ = self.predict_batch(dummy_features)
            
            return {
                'status': 'healthy',
                'model_loaded': True,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'model_loaded': False,
                'error': str(e),
            }


def get_predictor(
    model_path: str = "models/fraud_detector.pkl",
    scaler_path: str = "models/scaler.pkl"
) -> FraudPredictor:
    """
    Factory function to get predictor instance.
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to fitted scaler
    
    Returns:
        FraudPredictor instance
    """
    return FraudPredictor(model_path, scaler_path)
