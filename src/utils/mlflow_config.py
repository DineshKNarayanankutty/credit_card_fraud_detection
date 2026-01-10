"""
MLflow configuration utilities.

Responsibility:
- Configure MLflow tracking
- Set experiment consistently
- Remain backend-agnostic (local / remote / Azure ML)
"""

import os
import logging
import mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "credit-card-fraud-detection"
MODEL_NAME = "credit_card_fraud_model"


def configure_mlflow() -> None:
    """
    Configure MLflow in a backend-agnostic way.

    Behavior:
    - Tracking URI is read from environment variable
    - Experiment is created if it does not exist
    - Safe to call multiple times
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        logger.info("MLflow tracking URI not set — using default local backend")

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(
        f"MLflow experiment active | name='{experiment.name}', id={experiment.experiment_id}"
    )
