"""
Evaluation pipeline orchestration.

Responsibility:
- Orchestrate evaluation workflow ONLY
- NO training
- NO preprocessing
- NO model fitting

Designed for:
- dvc repro
- Azure ML Jobs
"""

import os
import json
import logging
from typing import Dict, Any, Tuple

import numpy as np

from src.utils.logger import setup_logging
from src.utils.config import config
from src.utils.io import load_pickle, load_processed_data

from src.inference.predictor import get_predictor
from src.evaluation.threshold import get_threshold_analysis
from src.evaluation.reports import generate_threshold_report, log_report

logger = logging.getLogger(__name__)

# OUTPUT PATH (DVC CONTRACT)

EVALUATION_METRICS_PATH = "artifacts/evaluation_metrics.json"


class EvaluationPipeline:
    """Model evaluation and threshold analysis pipeline"""

    def __init__(self, config):
        self.config = config
        logger.info("Initialized evaluation pipeline")

    def analyze_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: list | None = None,
    ) -> Dict[float, Dict[str, float]]:
        """
        Analyze model performance across thresholds.
        """
        logger.info("Analyzing thresholds...")

        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        analysis = get_threshold_analysis(
            y_true=y_true,
            y_proba=y_proba,
            thresholds=thresholds,
        )

        report = generate_threshold_report(analysis)
        log_report(report)

        return analysis

    def run(
        self,
        model_path: str,
        scaler_path: str,
        test_data: Tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Execute evaluation pipeline.
        """
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("=" * 60)

        # Load predictor
        logger.info("[STEP 1/2] Loading model and scaler")
        predictor = get_predictor(model_path, scaler_path)

        # Load test data
        logger.info("[STEP 2/2] Running threshold analysis")
        X_test, y_test = test_data

        preds = predictor.predict_batch(X_test.tolist())
        y_proba = np.array(preds["probabilities"])

        analysis = self.analyze_thresholds(
            y_true=y_test,
            y_proba=y_proba,
        )

        results = {
            "success": True,
            "threshold_analysis": analysis,
        }

        # PERSIST OUTPUT (REQUIRED BY DVC)
        os.makedirs(os.path.dirname(EVALUATION_METRICS_PATH), exist_ok=True)
        with open(EVALUATION_METRICS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation metrics saved to {EVALUATION_METRICS_PATH}")

        logger.info("=" * 60)
        logger.info("EVALUATION PIPELINE COMPLETE")
        logger.info("=" * 60)

        return results


# CLI ENTRYPOINT (DVC / AZURE ML)
def main():
    setup_logging()

    pipeline = EvaluationPipeline(config)

    # Load artifacts
    model_path = "artifacts/model.pkl"
    scaler_path = "artifacts/scaler.pkl"

    # Load processed test data
    _, _, X_test, _, _, y_test = load_processed_data()

    pipeline.run(
        model_path=model_path,
        scaler_path=scaler_path,
        test_data=(X_test, y_test),
    )


if __name__ == "__main__":
    main()
