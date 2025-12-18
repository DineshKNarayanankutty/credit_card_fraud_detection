"""
DVC-friendly drift detection pipeline wrapper.
Calls monitoring.evidently but exposes a CLI interface for DVC.
"""

import logging
import json
from pathlib import Path
from src.utils.logger import setup_logging
from src.utils.config import config
from monitoring.evidently import run_drift_check

logger = logging.getLogger(__name__)


def main():
    """Main entry point for DVC"""
    setup_logging()
    
    logger.info("Running drift check pipeline...")
    
    results = run_drift_check(
        reference_data_path=config.monitoring.reference_data_path,
        current_data_path=config.monitoring.current_data_path,
        output_dir="reports/drift"
    )
    
    # Save metrics for DVC tracking
    metrics_file = Path("reports/drift/drift_metrics.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Drift metrics saved to {metrics_file}")
    
    if results["data_drift"] or results["prediction_drift"]:
        logger.warning("DRIFT DETECTED - Consider retraining")
    else:
        logger.info("No drift detected")


if __name__ == "__main__":
    main()
