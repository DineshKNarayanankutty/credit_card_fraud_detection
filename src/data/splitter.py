"""
Data splitting layer.

Responsibility:
- Split data into train / validation / test sets
- Enforce stratification
- Prevent data leakage
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split dataframe into train, validation, and test sets with stratification.

    Args:
        df: Input dataframe (features + target)
        target_column: Name of target column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    # -------------------------
    # Validate ratios
    # -------------------------
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}"
        )

    # -------------------------
    # Validate target column
    # -------------------------
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataframe"
        )

    # -------------------------
    # Separate features & target
    # -------------------------
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    logger.info(
        f"Splitting {len(df):,} samples "
        f"(train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%})"
    )

    # -------------------------
    # First split: train+val vs test
    # -------------------------
    test_size_ratio = test_ratio / (1.0 - test_ratio)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size_ratio,
        random_state=random_state,
        stratify=y,
    )

    # -------------------------
    # Second split: train vs val
    # -------------------------
    val_size_ratio = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    # -------------------------
    # Logging
    # -------------------------
    logger.info(f"Train samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Test samples: {len(X_test):,}")

    logger.info(
        f"Positive class rates - "
        f"train={np.mean(y_train):.4%}, "
        f"val={np.mean(y_val):.4%}, "
        f"test={np.mean(y_test):.4%}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
