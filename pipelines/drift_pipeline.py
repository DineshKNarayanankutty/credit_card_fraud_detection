"""
DVC-friendly drift detection pipeline wrapper.
Calls monitoring.evidently but exposes a CLI interface for DVC.
"""

import logging
from src.utils.logger import setup_logging
from src.utils.config import config
from src.monitoring.evidently import run_drift_check

logger = logging.getLogger(__name__)


def main():
    """Main entry point for DVC"""
    setup_logging()

    logger.info("Running drift check pipeline...")

    results = run_drift_check(
        reference_data_path="data/reference",
        current_data_path="data/incoming",
        output_dir="reports/drift",
    )

    # IMPORTANT:
    # JSON output is already written by monitoring.evidently
    # to artifacts/drift_report.json (DVC output contract)

    if results.get("data_drift") or results.get("prediction_drift"):
        logger.warning("DRIFT DETECTED - Consider retraining")
    else:
        logger.info("No drift detected")


if __name__ == "__main__":
    main()
