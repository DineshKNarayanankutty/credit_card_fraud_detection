"""
Training pipeline orchestration.
"""

import argparse
import logging
import os
import shutil
import yaml

import mlflow

from src.utils.logger import setup_logging
from src.utils.config import config
from src.utils.mlflow_config import configure_mlflow

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

# IO utilities (SINGLE SOURCE OF TRUTH)
from src.utils.io import (
    save_processed_data,
    load_processed_data,
    save_scaler,
    load_pickle,
    save_metrics,
    SCALER_PATH,
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        logger.info(
            f"TrainingPipeline initialized | model={config.model.model_type}"
        )

    # -------------------------------------------------
    # PREPROCESS (NO MLFLOW, PURE DATA)
    # -------------------------------------------------
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

        # ✅ SCALER ALWAYS SAVED TO artifacts/
        save_scaler(scaler)

        logger.info("Preprocessing stage completed successfully")

    # -------------------------------------------------
    # TRAIN (MLFLOW STARTS HERE)
    # -------------------------------------------------
    def train(self, raw_data_path: str) -> None:
        logger.info("Starting training stage")

        # -----------------------------
        # HARD SAFETY CHECK
        # -----------------------------
        if not os.path.exists("data/processed/X_train.npy"):
            logger.info("Processed data missing — running preprocess")
            self.preprocess(raw_data_path)

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler missing at {SCALER_PATH}. "
                "Preprocess stage did not complete correctly."
            )

        configure_mlflow()

        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        with mlflow.start_run(run_name="fraud_model_training"):

            # Log model params only
            mlflow.log_params(params["model"])

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
                model,
                X_train,
                y_train,
                cv=self.config.model.cv_folds,
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
                optimal_threshold,
            )

            # -----------------------------
            # Compute metrics ONCE
            # -----------------------------
            metrics = compute_all_metrics(
                y_true=y_test,
                y_pred=test_preds,
                y_proba=test_results["probabilities"],
            )   

            # -----------------------------
            # Build report inputs explicitly
            # -----------------------------
            classification_metrics = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
            }

            cm_metrics = {
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
            }

            prob_metrics = {
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
            }

            report = generate_full_report(
                classification_metrics=classification_metrics,
                cm_metrics=cm_metrics,
                prob_metrics=prob_metrics,
            )

            log_report(report)


            mlflow.log_metrics(metrics)
            mlflow.log_metric("optimal_threshold", optimal_threshold)

        
            save_metrics(metrics, cv_scores, optimal_threshold)

            save_model(model, self.config.model.output_path)

            # -----------------------------
            # Azure ML outputs contract
            # -----------------------------
            os.makedirs("outputs", exist_ok=True)

            shutil.copy("artifacts/model.pkl", "outputs/model.pkl")
            shutil.copy("artifacts/scaler.pkl", "outputs/scaler.pkl")

            logger.info("Artifacts copied to Azure ML outputs/")


            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="credit_card_fraud_model",
            )

        logger.info("Training stage completed successfully")


# -------------------------------------------------
# CLI ENTRYPOINT
# -------------------------------------------------
def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["preprocess", "train"])
    parser.add_argument("--raw-data", required=True)

    args = parser.parse_args()

    pipeline = TrainingPipeline(config)

    if args.stage == "preprocess":
        pipeline.preprocess(args.raw_data)
    else:
        pipeline.train(args.raw_data)


if __name__ == "__main__":
    main()
