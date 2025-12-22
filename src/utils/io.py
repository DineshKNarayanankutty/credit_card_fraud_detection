"""
IO utilities.

Responsibility:
- Persist and load intermediate artifacts
- Generic serialization helpers
- NO ML logic
- NO pipeline logic
"""

import os
import pickle
from typing import Tuple, Any

import numpy as np

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
PROCESSED_DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")


# ------------------------------------------------------------------
# INTERNAL UTILS
# ------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path:
        os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# GENERIC PICKLE HELPERS
# ------------------------------------------------------------------
def save_pickle(obj: Any, path: str) -> None:
    """
    Save any Python object using pickle.
    """
    _ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """
    Load any pickled Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


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
    Persist fitted scaler for inference reuse.
    """
    save_pickle(scaler, SCALER_PATH)
