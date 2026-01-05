"""
Model training layer.

Responsibility:
- Train models ONLY
- Track experiments via MLflow

NO file I/O
NO data loading
NO preprocessing
"""

import logging
from typing import Dict, Any

import numpy as np
import mlflow
import mlflow.sklearn
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
    Train model on data and log training details to MLflow.
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
    # MLflow parameter logging
    # -----------------------------
    try:
        mlflow.log_params(model.get_params())
    except Exception as e:
        logger.warning(f"Could not log model parameters to MLflow: {e}")

    # -----------------------------
    # Model training
    # -----------------------------
    model.fit(X_train, y_train, verbose=verbose)

    logger.info("Training complete")

    # -----------------------------
    # Log trained model
    # -----------------------------
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )
    except Exception as e:
        logger.warning(f"Could not log model artifact to MLflow: {e}")

    return model


def cross_validate(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Cross-validate model and log aggregated CV metrics to MLflow.
    """

    # -----------------------------
    # Safety checks
    # -----------------------------
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

            mean_score = float(scores.mean())
            std_score = float(scores.std())

            results[metric] = {
                "mean": mean_score,
                "std": std_score,
            }

            logger.info(
                f"{metric}: {mean_score:.4f} (+/- {std_score:.4f})"
            )

            # Log only MEAN to MLflow (best practice)
            mlflow.log_metric(
                f"cv_{metric}",
                mean_score,
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
    Evaluate model on a dataset and log high-level metrics to MLflow.
    """

    # -----------------------------
    # Safety checks
    # -----------------------------
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

    # NOTE:
    # Detailed metrics (precision, recall, etc.)
    # are computed and logged in evaluation layer.
    # This function only returns raw outputs.

    return {
        "predictions": y_pred,
        "probabilities": y_proba,
        "true_labels": y,
    }
