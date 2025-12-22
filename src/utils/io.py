"""
IO utilities.

Responsibility:
- Persist and load intermediate artifacts
- NO ML logic
- NO pipeline logic
"""

import os
import pickle
from typing import Tuple

import numpy as np

# Base paths (keep centralized)
PROCESSED_DATA_DIR = "data/processed"
SCALER_PATH = os.path.join("artifacts", "scaler.pkl")


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# PROCESSED DATA
# ------------------------------------------------------------------
def save_processed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Persist processed train/val/test splits.

    Saved as NumPy binaries for speed and simplicity.
    """
    _ensure_dir(PROCESSED_DATA_DIR)

    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)

    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)


def load_processed_data() -> Tuple[np.ndarray, ...]:
    """
    Load processed train/val/test splits.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    return (
        np.load(os.path.join(PROCESSED_DATA_DIR, "X_train.npy")),
        np.load(os.path.join(PROCESSED_DATA_DIR, "X_val.npy")),
        np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy")),
        np.load(os.path.join(PROCESSED_DATA_DIR, "y_train.npy")),
        np.load(os.path.join(PROCESSED_DATA_DIR, "y_val.npy")),
        np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy")),
    )


# ------------------------------------------------------------------
# SCALER
# ------------------------------------------------------------------
def save_scaler(scaler) -> None:
    """
    Persist fitted scaler.

    Scaler is reused during inference.
    """
    _ensure_dir(os.path.dirname(SCALER_PATH))

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
