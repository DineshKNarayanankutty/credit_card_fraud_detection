"""
Monitoring contract for Evidently (0.7.x).

Purpose:
- Validate availability of monitoring data
- Produce stable artifacts for pipelines
- Keep Evidently usage UI-only (by design)

Evidently is NOT executed programmatically.
"""

import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

REFERENCE_DATA = Path("data/reference/reference.csv")
INCOMING_DATA = Path("data/incoming/incoming.csv")
ARTIFACT_PATH = Path("artifacts/drift_status.json")

ARTIFACT_PATH.parent.mkdir(exist_ok=True)


def run_monitoring() -> Dict:
    """
    Monitoring guard + status artifact.

    Actual drift inspection is done via `evidently ui`.
    """

    if not REFERENCE_DATA.exists():
        result = {
            "skipped": True,
            "reason": "Reference data not found",
        }
        _write(result)
        logger.warning(result["reason"])
        return result

    if not INCOMING_DATA.exists():
        result = {
            "skipped": True,
            "reason": "Incoming data not found",
        }
        _write(result)
        logger.warning(result["reason"])
        return result

    result = {
        "skipped": False,
        "reference_rows": _count_rows(REFERENCE_DATA),
        "incoming_rows": _count_rows(INCOMING_DATA),
        "message": "Data available for Evidently UI inspection",
    }

    _write(result)
    logger.info("Monitoring data validated successfully")
    return result


def _count_rows(path: Path) -> int:
    return sum(1 for _ in open(path)) - 1


def _write(data: Dict):
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    run_monitoring()
