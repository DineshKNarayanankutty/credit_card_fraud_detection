import os
import mlflow


def configure_mlflow():
    """
    Central MLflow configuration.
    Works locally, in CI, and in Azure ML.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("credit-card-fraud-detection")
