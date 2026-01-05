"""
Drift detection contract.

Purpose:
- Keep DVC pipeline reproducible
- Allow future integration with Evidently / Azure ML Monitor
- NEVER break training pipelines
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

DVC_OUTPUT_PATH = "artifacts/drift_report.json"


def run_drift_check(
    reference_data_path: str,
    current_data_path: str,
    drift_columns: List[str] = None,
    output_dir: str = "reports/drift",
) -> Dict[str, bool]:
    """
    Drift detection stub.

    This function intentionally does NOT implement Evidently logic.
    Monitoring tools are executed post-deployment.
    """

    reference_path = Path(reference_data_path)
    current_path = Path(current_data_path)

    if not reference_path.exists() or not any(reference_path.iterdir()):
        results = {
            "data_drift": False,
            "prediction_drift": False,
            "skipped": True,
            "reason": "Reference data not available",
        }

    elif not current_path.exists() or not any(current_path.iterdir()):
        results = {
            "data_drift": False,
            "prediction_drift": False,
            "skipped": True,
            "reason": "Incoming data not available",
        }

    else:
        # Placeholder — real monitoring runs outside DVC
        results = {
            "data_drift": False,
            "prediction_drift": False,
            "skipped": False,
        }

    os.makedirs("artifacts", exist_ok=True)
    with open(DVC_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Drift report written to {DVC_OUTPUT_PATH}")
    return results
