import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

REFERENCE_DATA = Path("data/reference/reference.csv")
INCOMING_DATA = Path("data/incoming/incoming.csv")
ARTIFACT_PATH = Path("artifacts/drift_report.json")

ARTIFACT_PATH.parent.mkdir(exist_ok=True)


# -----------------------
# Core Drift Metrics
# -----------------------

def ks_drift(ref: pd.Series, curr: pd.Series, alpha: float = 0.05) -> Dict:
    stat, p_value = ks_2samp(ref, curr)
    return {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "drifted": p_value < alpha,
    }


def psi(ref: pd.Series, curr: pd.Series, bins: int = 10) -> float:
    ref_percents, _ = np.histogram(ref, bins=bins)
    curr_percents, _ = np.histogram(curr, bins=bins)

    ref_percents = ref_percents / len(ref)
    curr_percents = curr_percents / len(curr)

    psi_value = np.sum(
        (ref_percents - curr_percents)
        * np.log((ref_percents + 1e-6) / (curr_percents + 1e-6))
    )
    return float(psi_value)


# -----------------------
# Main Pipeline
# -----------------------

def run_drift_detection(
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.2,
) -> Dict:
    """
    Batch drift detection.
    Designed for Airflow & DVC.
    """

    if not REFERENCE_DATA.exists() or not INCOMING_DATA.exists():
        result = {
            "skipped": True,
            "reason": "Reference or incoming data missing",
        }
        _write(result)
        logger.warning(result["reason"])
        return result

    ref_df = pd.read_csv(REFERENCE_DATA)
    curr_df = pd.read_csv(INCOMING_DATA)

    common_cols = ref_df.columns.intersection(curr_df.columns)

    drift_results = {}
    drifted_features = []

    for col in common_cols:
        if not np.issubdtype(ref_df[col].dtype, np.number):
            continue

        ks_result = ks_drift(ref_df[col], curr_df[col], ks_alpha)
        psi_value = psi(ref_df[col], curr_df[col])

        feature_drifted = ks_result["drifted"] or psi_value > psi_threshold

        drift_results[col] = {
            "ks": ks_result,
            "psi": psi_value,
            "drifted": feature_drifted,
        }

        if feature_drifted:
            drifted_features.append(col)

    final_report = {
        "skipped": False,
        "summary": {
            "total_features": len(drift_results),
            "drifted_features": len(drifted_features),
            "drift_ratio": len(drifted_features) / max(len(drift_results), 1),
        },
        "drifted_features": drifted_features,
        "details": drift_results,
    }

    _write(final_report)
    logger.info("Drift detection completed")
    return final_report


def _write(report: Dict):
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    run_drift_detection()
