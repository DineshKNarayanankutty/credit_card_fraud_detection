"""
Training pipeline orchestration.

Responsibility:
- Orchestrate preprocessing and training stages ONLY
- NO ML logic
- NO data manipulation logic
- NO serialization logic outside registry / io helpers

Designed for:
- dvc repro
- Azure ML Jobs
"""

import argparse
import logging
from typing import Dict, Any

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
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline.
    Orchestration only.
    """

    def __init__(self, config):
        self.config = config
        logger.info(
            f"TrainingPipeline initialized | model={config.model.model_type}"
        )

    # -----------------------------
    # PREPROCESS STAGE
    # -----------------------------
    def preprocess(self, data_path: str) -> None:
        """
        Preprocessing stage.

        Produces:
        - UN-SCALED train/val/test splits
        - fitted scaler

        Steps:
        1. Load raw data
        2. Clean data
        3. Train/val/test split
        4. Fit scaler on TRAIN only
        5. Persist UN-SCALED data + scaler
        """
        logger.info("Starting preprocessing stage")

        # Load + clean
        df = load_csv(data_path)
        df = clean_pipeline(df)

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            df,
            target_column=self.config.data.target_column,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
            test_ratio=self.config.data.test_ratio,
            random_state=self.config.data.random_state,
        )

        # Fit scaler ONLY (do NOT transform here)
        scaler = get_feature_scaler(self.config.features.scaling_method)
        scaler.fit(X_train)

        # Persist raw splits + scaler
        save_processed_data(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )
        save_scaler(scaler)

        logger.info("Preprocessing stage completed successfully")

    # -----------------------------
    # TRAIN STAGE
    # -----------------------------
    def train(self) -> Dict[str, Any]:
        """
        Training + evaluation stage.

        Steps:
        1. Load UN-SCALED data
        2. Apply SMOTE to TRAIN only
        3. Scale ALL splits ONCE
        4. Train model
        5. Evaluate & persist artifacts
        """
        logger.info("Starting training stage")

        # Load unscaled data
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

        # Apply SMOTE (TRAIN ONLY)
        if self.config.features.handle_imbalance:
            X_train, y_train = apply_smote(
                X_train,
                y_train,
                method=self.config.features.imbalance_method,
                k_neighbors=self.config.features.smote_k_neighbors,
                random_state=self.config.data.random_state,
            )

        # HARD GUARANTEE
        assert X_train.shape[0] == y_train.shape[0], (
            f"SMOTE mismatch: X={X_train.shape[0]}, y={y_train.shape[0]}"
        )

        # Load scaler & scale ONCE
        scaler = load_pickle("artifacts/scaler.pkl")

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Model creation & training
        model = get_model(
            model_type=self.config.model.model_type,
            model_params=self.config.model.model_params,
        )

        train(model, X_train, y_train)

        # Cross-validation
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=self.config.model.cv_folds,
        )

        # Validation threshold tuning
        val_results = evaluate_on_set(
            model,
            X_val,
            y_val,
            set_name="validation",
        )
        val_probs = val_results["probabilities"]

        optimal_threshold, threshold_score = find_optimal_threshold(
            y_true=y_val,
            y_proba=val_probs,
            metric=self.config.evaluation.threshold_method,
        )

        logger.info(
            f"Optimal threshold={optimal_threshold:.4f}, "
            f"score={threshold_score:.4f}"
        )

        # Test evaluation
        test_results = evaluate_on_set(
            model,
            X_test,
            y_test,
            set_name="test",
        )
        test_probs = test_results["probabilities"]
        test_preds = apply_threshold(test_probs, optimal_threshold)

        # -----------------------------
        # Metrics computation
        # -----------------------------
        metrics = compute_all_metrics(
            y_true=y_test,
            y_pred=test_preds,
            y_proba=test_probs,
        )

        # Split metrics explicitly (NO guessing)
        classification_metrics = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mcc": metrics.get("mcc", 0.0),
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

        # -----------------------------
        # Generate + log report
        # -----------------------------
        report = generate_full_report(
            classification_metrics,
            cm_metrics,
            prob_metrics=prob_metrics,
        )

        log_report(report)




        save_model(model, self.config.model.output_path)

        logger.info("Training stage completed successfully")

        return {
            "metrics": metrics,
            "cv_scores": cv_scores,
            "threshold": optimal_threshold,
        }


# -----------------------------
# CLI ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess", "train"],
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/creditcard.csv",
    )

    args = parser.parse_args()

    pipeline = TrainingPipeline(config)

    if args.stage == "preprocess":
        pipeline.preprocess(args.data_path)
    else:
        pipeline.train()
