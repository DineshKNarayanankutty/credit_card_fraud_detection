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
from src.evaluation.reports import generate_full_report

# IO utilities
from src.utils.io import (
    save_processed_data,
    load_processed_data,
    save_scaler,
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
        - data/processed/{train,val,test}.pkl
        - fitted scaler

        Steps:
        1. Load raw data
        2. Clean data
        3. Train/val/test split
        4. Fit scaler on TRAIN only
        5. Transform all splits
        6. Persist processed artifacts
        """
        logger.info("Starting preprocessing stage")

        # Load + clean
        df = load_csv(data_path)
        df = clean_pipeline(df)

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            df,
            target_col=self.config.data.target_column,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
            random_state=self.config.data.random_state,
        )

        # Scale (fit ONLY on train)
        scaler = get_feature_scaler(self.config)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Persist artifacts
        save_processed_data(
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
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

        Produces:
        - trained model (via registry)
        - metrics
        - evaluation report

        Steps:
        1. Load processed data
        2. Apply SMOTE to TRAIN only
        3. Train model
        4. Cross-validation
        5. Validation threshold tuning
        6. Test evaluation
        7. Persist model + reports
        """
        logger.info("Starting training stage")

        # Load processed data
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = load_processed_data()

        # Handle imbalance (TRAIN only)
        X_train_bal, y_train_bal = apply_smote(
            X_train,
            y_train,
            random_state=self.config.data.random_state,
        )

        # Model creation + training
        model = get_model(self.config)
        train(model, X_train_bal, y_train_bal)

        # Cross-validation
        cv_scores = cross_validate(
            model,
            X_train_bal,
            y_train_bal,
            scoring=self.config.evaluation.cv_metric,
        )
        logger.info(f"Cross-validation scores: {cv_scores}")

        # Validation threshold tuning
        val_probs = evaluate_on_set(model, X_val)
        optimal_threshold = find_optimal_threshold(
            y_val,
            val_probs,
            metric=self.config.evaluation.threshold_metric,
        )
        logger.info(f"Optimal threshold selected: {optimal_threshold:.4f}")

        # Test evaluation
        test_probs = evaluate_on_set(model, X_test)
        test_preds = apply_threshold(test_probs, optimal_threshold)

        metrics = compute_all_metrics(
            y_true=y_test,
            y_pred=test_preds,
            y_proba=test_probs,
        )

        # Reporting
        generate_full_report(metrics)

        # Persist model (registry handles serialization)
        save_model(model, self.config.model.output_path)

        logger.info("Training stage completed successfully")

        return {
            "metrics": metrics,
            "cv_scores": cv_scores,
            "threshold": optimal_threshold,
        }


# -----------------------------
# CLI ENTRYPOINT (DVC / AML)
# -----------------------------
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess", "train"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/creditcard.csv",
        help="Path to raw dataset (DVC-managed)",
    )

    args = parser.parse_args()

    pipeline = TrainingPipeline(config)

    if args.stage == "preprocess":
        pipeline.preprocess(args.data_path)
    elif args.stage == "train":
        pipeline.train()
