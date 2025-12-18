"""
Model training layer.
Responsibility: Train models ONLY.
NO file I/O.
NO data loading.
NO preprocessing.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True
) -> XGBClassifier:
    """
    Train model on data.
    
    Args:
        model: Untrained XGBClassifier instance
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        verbose: Print training progress
    
    Returns:
        Trained model (modified in-place)
    """
    if X_train.shape != len(y_train):
        raise ValueError("X_train and y_train must have same number of samples")
    
    logger.info(f"Training on {X_train.shape:,} samples × {X_train.shape} features")
    
    # Train
    model.fit(
        X_train, y_train,
        verbose=verbose
    )
    
    logger.info("✓ Training complete")
    return model


def cross_validate(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Cross-validate model.
    
    Args:
        model: Untrained XGBClassifier instance
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        cv: Number of folds
    
    Returns:
        Dict with metrics: {metric: {mean, std}}
    """
    if X.shape != len(y):
        raise ValueError("X and y must have same number of samples")
    
    logger.info(f"Starting {cv}-fold cross-validation...")
    
    results = {}
    
    # Metrics for fraud detection (recall-focused)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for metric in metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            results[metric] = {
                'mean': float(scores.mean()),
                'std': float(scores.std())
            }
            logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            logger.warning(f"Could not compute {metric}: {e}")
    
    return results


def evaluate_on_set(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    set_name: str = "test"
) -> Dict[str, Any]:
    """
    Evaluate model on a specific set.
    
    Args:
        model: Trained XGBClassifier instance
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        set_name: Name of set (e.g., "validation", "test")
    
    Returns:
        Dict with predictions and probabilities
    """
    if X.shape != len(y):
        raise ValueError("X and y must have same number of samples")
    
    logger.info(f"Evaluating on {set_name} set ({X.shape:,} samples)")
    
    # Get predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (fraud)
    
    results = {
        'predictions': y_pred,
        'probabilities': y_proba,
        'true_labels': y,
    }
    
    logger.info(f"✓ {set_name.capitalize()} evaluation complete")
    return results
