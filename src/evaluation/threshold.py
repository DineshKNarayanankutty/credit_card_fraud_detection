"""
Threshold optimization layer.
Responsibility: Find optimal threshold for predictions.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    step: float = 0.01
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels (n_samples,)
        y_proba: Predicted probabilities (n_samples,)
        metric: "f1", "f2" (2x recall weight), or "precision_recall_balance"
        step: Step size for threshold search (0.01 = search every 1%)
    
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have same length")
    
    thresholds = np.arange(0, 1 + step, step)
    best_threshold = 0.5
    best_score = 0.0
    
    logger.info(f"Searching optimal threshold using {metric} metric...")
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        
        elif metric == "f2":
            # F2 = 5 * (precision * recall) / (4 * precision + recall)
            # Weights recall 2x more than precision
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            score = (5 * precision * recall / (4 * precision + recall)) if (precision + recall) > 0 else 0
        
        elif metric == "precision_recall_balance":
            # Balance precision and recall equally
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"✓ Optimal threshold: {best_threshold:.4f} (score: {best_score:.4f})")
    return best_threshold, best_score


def apply_threshold(
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply threshold to probabilities.
    
    Args:
        y_proba: Predicted probabilities (n_samples,)
        threshold: Classification threshold
    
    Returns:
        Binary predictions (n_samples,)
    """
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    return (y_proba >= threshold).astype(int)


def get_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list = None
) -> Dict[float, Dict[str, float]]:
    """
    Analyze metrics at different thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: List of thresholds to evaluate (default: [0.3, 0.5, 0.7, 0.9])
    
    Returns:
        Dict mapping threshold → {precision, recall, f1}
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]
    
    analysis = {}
    
    for threshold in thresholds:
        y_pred = apply_threshold(y_proba, threshold)
        
        analysis[threshold] = {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }
    
    logger.info(f"Threshold analysis complete for {len(thresholds)} thresholds")
    return analysis
