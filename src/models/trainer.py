"""
Model training layer.
Responsibility: Train models ONLY.
NO file I/O.
NO data loading.
NO preprocessing.
"""

import logging
import numpy as np
from typing import Dict, Any
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
    Train model on data.
    """

    # ✅ CORRECT CHECK
    if X_train.shape[0] != len(y_train):
        raise ValueError(
            f"X_train and y_train size mismatch: "
            f"X={X_train.shape[0]}, y={len(y_train)}"
        )

    logger.info(
        f"Training on {X_train.shape[0]:,} samples "
        f"with {X_train.shape[1]} features"
    )

    model.fit(X_train, y_train, verbose=verbose)

    logger.info("Training complete")
    return model


def cross_validate(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Cross-validate model.
    """

    # ✅ CORRECT CHECK
    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y size mismatch: X={X.shape[0]}, y={len(y)}"
        )

    logger.info(f"Starting {cv}-fold cross-validation")

    results = {}
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            results[metric] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }
            logger.info(
                f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})"
            )
        except Exception as e:
            logger.warning(f"Could not compute {metric}: {e}")

    return results


def evaluate_on_set(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    set_name: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    """

    # ✅ CORRECT CHECK
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

    logger.info(f"{set_name.capitalize()} evaluation complete")

    return {
        "predictions": y_pred,
        "probabilities": y_proba,
        "true_labels": y,
    }
