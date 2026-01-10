"""
Model training layer.

Responsibility:
- Train models ONLY
- Perform cross-validation
- Produce predictions and probabilities

NO file I/O
NO MLflow
NO orchestration
NO data loading
NO preprocessing
"""

import logging
from typing import Dict, Any

import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> XGBClassifier:
    """
    Train model on provided data.

    Returns:
        Trained model
    """

    # -----------------------------
    # Safety checks
    # -----------------------------
    if X_train.shape[0] != len(y_train):
        raise ValueError(
            f"X_train and y_train size mismatch: "
            f"X={X_train.shape[0]}, y={len(y_train)}"
        )

    logger.info(
        f"Training on {X_train.shape[0]:,} samples "
        f"with {X_train.shape[1]} features"
    )

    # -----------------------------
    # Model training
    # -----------------------------
    model.fit(X_train, y_train, verbose=verbose)

    logger.info("Model training completed")

    return model


def cross_validate(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation.

    Returns:
        Dict with mean and std for each metric
    """

    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y size mismatch: X={X.shape[0]}, y={len(y)}"
        )

    logger.info(f"Starting {cv}-fold cross-validation")

    results: Dict[str, Dict[str, float]] = {}
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics:
        try:
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=metric,
            )

            results[metric] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }

            logger.info(
                f"{metric}: {scores.mean():.4f} "
                f"(+/- {scores.std():.4f})"
            )

        except Exception as e:
            logger.warning(f"Failed CV metric {metric}: {e}")

    return results


def evaluate_on_set(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    set_name: str = "test",
) -> Dict[str, Any]:
    """
    Run inference on a dataset.

    Returns:
        Raw predictions and probabilities
    """

    if X.shape[0] != len(y):
        raise ValueError(
            f"{set_name} set mismatch: X={X.shape[0]}, y={len(y)}"
        )

    logger.info(
        f"Evaluating on {set_name} set "
        f"({X.shape[0]:,} samples)"
    )

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "predictions": y_pred,
        "probabilities": y_proba,
        "true_labels": y,
    }
