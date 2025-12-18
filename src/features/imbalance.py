"""
Imbalance handling layer.
Responsibility: Handle class imbalance during training ONLY.
Must NOT be imported by: inference, API
"""

import logging
import numpy as np
from typing import Tuple
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to handle class imbalance.
    
    Use ONLY during training.
    Never use on test/validation/inference data.
    
    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        k_neighbors: Number of nearest neighbors for SMOTE
        random_state: Random seed
    
    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    # Check if imbalance exists
    unique, counts = np.unique(y, return_counts=True)
    imbalance_ratio = counts.min() / counts.max()
    
    logger.info(f"Initial class distribution:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls}: {count:,} samples ({count/len(y):.2%})")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.4f}")
    
    # Apply SMOTE
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Log results
    unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
    logger.info(f"✓ After SMOTE:")
    for cls, count in zip(unique_balanced, counts_balanced):
        logger.info(f"  Class {cls}: {count:,} samples ({count/len(y_balanced):.2%})")
    
    logger.info(f"✓ Total samples increased: {len(X):,} → {len(X_balanced):,}")
    
    return X_balanced, y_balanced


def get_imbalance_ratio(y: np.ndarray) -> float:
    """
    Calculate current class imbalance ratio.
    
    Args:
        y: Labels (n_samples,)
    
    Returns:
        Ratio of minority to majority class
    """
    unique, counts = np.unique(y, return_counts=True)
    return counts.min() / counts.max()
