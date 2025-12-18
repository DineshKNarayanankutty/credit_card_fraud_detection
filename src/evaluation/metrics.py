"""
Metrics computation layer.
Responsibility: Compute metrics ONLY.
Return dict only (no printing).
"""

import logging
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, roc_curve,
    confusion_matrix, matthews_corrcoef
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
    
    Returns:
        Dict with metrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
    }
    
    logger.info(f"Classification metrics computed")
    return metrics


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute probabilistic metrics (ROC-AUC, PR-AUC).
    
    Args:
        y_true: True labels (n_samples,)
        y_proba: Predicted probabilities (n_samples,) - probability of class 1
    
    Returns:
        Dict with ROC-AUC and PR-AUC
    """
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have same length")
    
    # ROC-AUC
    roc_auc = float(roc_auc_score(y_true, y_proba))
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = float(auc(recall, precision))
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
    }
    
    logger.info(f"Probabilistic metrics computed")
    return metrics


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute confusion matrix and derived metrics.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
    
    Returns:
        Dict with TP, TN, FP, FN and derived metrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate derived metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    metrics = {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'fpr': float(fpr),
        'fnr': float(fnr),
    }
    
    logger.info(f"Confusion matrix metrics computed")
    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute all metrics at once.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # Classification metrics
    metrics.update(compute_classification_metrics(y_true, y_pred))
    
    # Confusion matrix metrics
    metrics.update(compute_confusion_matrix_metrics(y_true, y_pred))
    
    # Probabilistic metrics
    if y_proba is not None:
        metrics.update(compute_probabilistic_metrics(y_true, y_proba))
    
    logger.info(f"All metrics computed")
    return metrics
