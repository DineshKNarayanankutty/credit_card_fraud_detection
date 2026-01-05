"""
IO utilities.

Responsibility:
- Persist and load intermediate artifacts
- Generic serialization helpers
- NO ML logic
- NO pipeline logic
"""

import os
import json
import pickle
from typing import Tuple, Any, Dict

import numpy as np

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
PROCESSED_DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"

SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
CV_SCORES_PATH = os.path.join(ARTIFACTS_DIR, "cv_scores.json")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")


# ------------------------------------------------------------------
# INTERNAL UTILS
# ------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# GENERIC PICKLE HELPERS
# ------------------------------------------------------------------
def save_pickle(obj: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
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
    _ensure_dir(PROCESSED_DATA_DIR)

    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)

    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)


def load_processed_data() -> Tuple[np.ndarray, ...]:
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
    save_pickle(scaler, SCALER_PATH)


# ------------------------------------------------------------------
# METRICS (DVC OUTPUTS)
# ------------------------------------------------------------------
def save_metrics(
    metrics: Dict[str, Any],
    cv_scores: Dict[str, Any],
    threshold: float,
) -> None:
    _ensure_dir(ARTIFACTS_DIR)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(CV_SCORES_PATH, "w") as f:
        json.dump(cv_scores, f, indent=2)

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
