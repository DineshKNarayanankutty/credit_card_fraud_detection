"""
Imbalance handling utilities.

Responsibility:
- Handle class imbalance on TRAINING DATA ONLY
- NO data splitting
- NO model logic
"""

import logging
from typing import Tuple

import numpy as np

from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply imbalance handling to training data.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        method: Imbalance handling method (currently supports 'smote')
        k_neighbors: Number of neighbors for SMOTE
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """

    method = method.lower()

    if method != "smote":
        raise ValueError(
            f"Unsupported imbalance method: {method}. "
            "Currently supported: 'smote'."
        )

    logger.info(
        f"Applying SMOTE (k_neighbors={k_neighbors}, "
        f"random_state={random_state})"
    )

    smote = SMOTE(
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    logger.info(
        f"Total samples increased from {len(X):,} to {len(X_resampled):,}"
    )

    return X_resampled, y_resampled
