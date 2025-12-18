"""
Model factory layer.
Responsibility: Create model instances ONLY.
NO training logic.
NO file I/O.
"""

import logging
from typing import Dict, Any
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def get_model(model_type: str, params: Dict[str, Any] = None) -> XGBClassifier:
    """
    Create model instance.
    
    Choice: XGBoost (optimized for fraud detection)
    Reason: Better recall performance, handles imbalanced data well.
    
    Args:
        model_type: "xgboost" (only option)
        params: Model hyperparameters
    
    Returns:
        Untrained XGBClassifier instance
    
    Raises:
        ValueError: If model_type is not "xgboost"
    """
    if model_type != "xgboost":
        raise ValueError(f"Unsupported model type: {model_type}. Only 'xgboost' is supported.")
    
    # Default parameters (recall-focused)
    default_params = {
        "n_estimators": 100,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,  # Handle imbalance (will be overridden by SMOTE)
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # Override with provided params
    if params:
        default_params.update(params)
    
    logger.info(f"Creating XGBoost model with parameters: {default_params}")
    model = XGBClassifier(**default_params)
    
    return model
