"""
Training pipeline orchestration.
"""

import argparse
import logging
import os
import shutil

from src.utils.logger import setup_logging
from src.utils.config import config

# Data layer
from src.data.loader import load_csv
from src.data.cleaner import clean_pipeline
from src.data.scaler import get_feature_scaler
from src.data.splitter import train_val_test_split

# Feature layer
from src.features.imbalance import apply_smote

# Model layer
from src.models.factory import get_model
from src.models.trainer import train, cross_validate, evaluate_on_set
from src.models.registry import save_model

# Evaluation layer
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.threshold import find_optimal_threshold, apply_threshold
from src.evaluation.reports import generate_full_report, log_report

# IO utilities
from src.utils.io import (
    save_processed_data,
    load_processed_data,
    save_scaler,
    load_pickle,
    save_metrics,
    SCALER_PATH,
)

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        logger.info(
            f"TrainingPipeline initialized | model={config.model.model_type}"
        )

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    def preprocess(self, data_path: str) -> None:
        logger.info("Starting preprocessing stage")

        df = load_csv(data_path)
        df = clean_pipeline(df)

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            df,
            target_column=self.config.data.target_column,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
            test_ratio=self.config.data.test_ratio,
            random_state=self.config.data.random_state,
        )

        scaler = get_feature_scaler(self.config.features.scaling_method)
        scaler.fit(X_train)

        save_processed_data(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )
        save_scaler(scaler)

        logger.info("Preprocessing stage completed")

    # -----------------------------
    # TRAIN
    # -----------------------------
    def train(self, raw_data_path: str) -> None:   # <<< CHANGED
        logger.info("Starting training stage")

        # <<< CHANGED: Azure ML safety
        if not os.path.exists("data/processed/X_train.npy"):
            logger.info("Processed data not found — running preprocess stage")
            self.preprocess(raw_data_path)

        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

        if self.config.features.handle_imbalance:
            X_train, y_train = apply_smote(
                X_train,
                y_train,
                method=self.config.features.imbalance_method,
                k_neighbors=self.config.features.smote_k_neighbors,
                random_state=self.config.data.random_state,
            )

        scaler = load_pickle(SCALER_PATH)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        model = get_model(
            model_type=self.config.model.model_type,
            model_params=self.config.model.model_params,
        )

        train(model, X_train, y_train)

        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=self.config.model.cv_folds
        )

        val_results = evaluate_on_set(model, X_val, y_val, "validation")
        optimal_threshold, _ = find_optimal_threshold(
            y_true=y_val,
            y_proba=val_results["probabilities"],
            metric=self.config.evaluation.threshold_method,
        )

        test_results = evaluate_on_set(model, X_test, y_test, "test")
        test_preds = apply_threshold(
            test_results["probabilities"],
            optimal_threshold
        )

        metrics = compute_all_metrics(
            y_true=y_test,
            y_pred=test_preds,
            y_proba=test_results["probabilities"],
        )

        report = generate_full_report(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics.get("mcc", 0.0),
            },
            {
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
            },
            prob_metrics={
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
            },
        )
        log_report(report)

        save_metrics(metrics, cv_scores, optimal_threshold)
        save_model(model, self.config.model.output_path)

        # <<< CHANGED: Azure ML outputs contract
        os.makedirs("outputs", exist_ok=True)
        shutil.copy(self.config.model.output_path, "outputs/model.pkl")
        shutil.copy(SCALER_PATH, "outputs/scaler.pkl")

        self.register_model_to_azureml()

        logger.info("Training stage completed successfully")

    def register_model_to_azureml(self):
        """
        Register model + scaler to Azure ML Model Registry.
        Runs only when executed inside Azure ML.
        """
        if "AZUREML_RUN_ID" not in os.environ:
            logger.info("Not running inside Azure ML — skipping model registration")
            return

        logger.info("Registering model to Azure ML Model Registry")

        credential = DefaultAzureCredential()

        ml_client = MLClient(
            credential=credential,
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            workspace_name=os.environ["AZURE_ML_WORKSPACE"],
        )

        model = Model(
            name=self.config.model.name,          # e.g. "fraud-model"
            path="outputs",                        # contains model.pkl + scaler.pkl
            type="custom_model",
            description="Credit Card Fraud Detection model",
            labels={"latest": "true"}
        )

        ml_client.models.create_or_update(model)

        logger.info("Model successfully registered in Azure ML")
        registered = ml_client.models.create_or_update(model)
        logger.info(f"Registered model version: {registered.version}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["preprocess", "train"])
    parser.add_argument("--raw-data", required=True)   # <<< CHANGED

    args = parser.parse_args()

    pipeline = TrainingPipeline(config)

    if args.stage == "preprocess":
        pipeline.preprocess(args.raw_data)
    else:
        pipeline.train(args.raw_data)
