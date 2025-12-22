"""
Data splitting layer.
Responsibility: Split data with no leakage.
Enforces stratification and correct proportions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    Enforces stratification to preserve class distribution.
    
    Args:
        X: Features (n_samples, n_features)
        y: Target (n_samples,)
        train_ratio: Proportion for training (0-1)
        val_ratio: Proportion for validation (0-1)
        test_ratio: Proportion for testing (0-1)
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Validate inputs
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    
    logger.info(f"Splitting {len(X):,} samples: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    
    # First split: train+val vs test
    test_size_ratio = test_ratio / (1 - test_ratio)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size_ratio,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: train vs val
    val_size_ratio = val_ratio / (val_ratio + train_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Log split results
    logger.info(f"Train: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
    logger.info(f"Val:   {len(X_val):,} samples ({len(X_val)/len(X):.1%})")
    logger.info(f"Test:  {len(X_test):,} samples ({len(X_test)/len(X):.1%})")
    
    # Verify stratification
    train_pos = (y_train == 1).sum()
    val_pos = (y_val == 1).sum()
    test_pos = (y_test == 1).sum()
    
    logger.info(f"Class distribution preserved:")
    logger.info(f"  Train positive rate: {train_pos/len(y_train):.4%}")
    logger.info(f"  Val positive rate:   {val_pos/len(y_val):.4%}")
    logger.info(f"  Test positive rate:  {test_pos/len(y_test):.4%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
