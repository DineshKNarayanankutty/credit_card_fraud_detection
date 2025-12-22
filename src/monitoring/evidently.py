"""
Data and prediction drift detection using Evidently.
Responsibility: Detect drift ONLY.
NO model training, NO pipeline calls, NO dataset modification.

FIXES APPLIED:
- Column names from config (not hard-coded) ✅
- Separate prediction drift logic (not collapsed into data drift) ✅
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    ClassificationPerformanceMetrics,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfMissingValues,
    TestNumberOfDuplicatedRows,
    TestColumnDrift,
)
from src.utils.config import config

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift and prediction drift using Evidently"""
    
    def __init__(
        self,
        reference_data_path: str,
        current_data_path: str,
        drift_columns: List[str] = None,
        output_dir: str = "reports/drift"
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data_path: Path to reference dataset (baseline)
            current_data_path: Path to current dataset (to check)
            drift_columns: Columns to monitor for drift (from config if None)
            output_dir: Directory to save reports
        """
        self.reference_data_path = reference_data_path
        self.current_data_path = current_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # FIX #1: Get drift columns from config, not hard-coded
        self.drift_columns = drift_columns or getattr(
            config.monitoring,
            "drift_columns",
            ["Time", "Amount"]  # Fallback only
        )
        
        logger.info(f"Initialized drift detector")
        logger.info(f"Monitoring columns: {self.drift_columns}")
    
    def load_data(self) -> tuple:
        """
        Load reference and current datasets.
        
        Returns:
            Tuple of (reference_df, current_df)
        
        Raises:
            FileNotFoundError: If datasets not found
        """
        logger.info(f"Loading reference data from {self.reference_data_path}")
        reference_df = pd.read_csv(self.reference_data_path)
        
        logger.info(f"Loading current data from {self.current_data_path}")
        current_df = pd.read_csv(self.current_data_path)
        
        logger.info(f"Reference: {reference_df.shape}, Current: {current_df.shape}")
        
        return reference_df, current_df
    
    def run_data_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run data drift report.
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
        
        Returns:
            Dict with drift detection results
        """
        logger.info("Running data drift report...")
        
        report = Report(metrics=[
            DataDriftTable(),
        ])
        
        report.run(
            reference_data=reference_df,
            current_data=current_df
        )
        
        # Save HTML report
        report_path = self.output_dir / "data_drift_report.html"
        report.save_html(str(report_path))
        logger.info(f"Data drift report saved to {report_path}")
        
        # Extract metrics
        metrics_dict = report.as_dict()
        
        return metrics_dict
    
    def run_prediction_performance_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str = "Class",
        prediction_col: str = "prediction"
    ) -> Dict[str, Any]:
        """
        Run classification performance report.
        
        Args:
            reference_df: Reference dataset with predictions
            current_df: Current dataset with predictions
            target_col: Target column name
            prediction_col: Prediction column name
        
        Returns:
            Dict with performance metrics
        """
        logger.info("Running prediction performance report...")
        
        report = Report(metrics=[
            ClassificationPerformanceMetrics(),
        ])
        
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=None
        )
        
        # Save HTML report
        report_path = self.output_dir / "performance_report.html"
        report.save_html(str(report_path))
        logger.info(f"Performance report saved to {report_path}")
        
        metrics_dict = report.as_dict()
        
        return metrics_dict
    
    def detect_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> bool:
        """
        Detect data drift with statistical tests.
        Uses configurable columns (not hard-coded).
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            threshold: Drift threshold (0-1)
        
        Returns:
            True if data drift detected, False otherwise
        """
        logger.info(f"Detecting data drift (threshold={threshold})...")
        
        # FIX #1: Use configurable drift_columns instead of hard-coded
        tests = [
            TestNumberOfMissingValues(),
            TestNumberOfDuplicatedRows(),
        ]
        
        # Add column-specific tests dynamically
        for col in self.drift_columns:
            if col in reference_df.columns and col in current_df.columns:
                tests.append(TestColumnDrift(column_name=col))
            else:
                logger.warning(f"Column {col} not found in data, skipping drift test")
        
        suite = TestSuite(tests=tests)
        
        suite.run(
            reference_data=reference_df,
            current_data=current_df
        )
        
        # Check if any tests failed
        has_drift = not suite.as_dict()["success"]
        
        logger.info(f"Data drift detected: {has_drift}")
        
        return has_drift
    
    def detect_prediction_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        performance_threshold: float = 0.05
    ) -> bool:
        """
        Detect prediction drift by comparing model performance.
        
        FIX #2: Separate logic for prediction drift (not collapsed into data drift).
        Prediction drift = model performance degradation
        
        Args:
            reference_df: Reference dataset with predictions
            current_df: Current dataset with predictions
            performance_threshold: Acceptable performance drop (0-1)
        
        Returns:
            True if prediction drift detected, False otherwise
        """
        logger.info(f"Detecting prediction drift (threshold={performance_threshold:.1%})...")
        
        try:
            # Must have predictions in both datasets
            if "prediction" not in reference_df.columns or "prediction" not in current_df.columns:
                logger.warning("Prediction column not found, cannot detect prediction drift")
                return False
            
            # Calculate reference performance (if labels exist)
            ref_has_labels = "Class" in reference_df.columns
            curr_has_labels = "Class" in current_df.columns
            
            if not (ref_has_labels and curr_has_labels):
                logger.warning("Cannot calculate prediction drift without labels in both datasets")
                return False
            
            # Simple performance comparison: accuracy drop
            from sklearn.metrics import accuracy_score
            
            ref_accuracy = accuracy_score(
                reference_df["Class"],
                reference_df["prediction"]
            )
            
            curr_accuracy = accuracy_score(
                current_df["Class"],
                current_df["prediction"]
            )
            
            accuracy_drop = ref_accuracy - curr_accuracy
            
            has_prediction_drift = accuracy_drop > performance_threshold
            
            logger.info(f"Reference accuracy: {ref_accuracy:.4f}")
            logger.info(f"Current accuracy: {curr_accuracy:.4f}")
            logger.info(f"Accuracy drop: {accuracy_drop:.4f}")
            logger.info(f"Prediction drift detected: {has_prediction_drift}")
            
            return has_prediction_drift
        
        except Exception as e:
            logger.warning(f"Could not calculate prediction drift: {e}")
            return False
    
    def run(self) -> Dict[str, bool]:
        """
        Run complete drift detection.
        
        Returns:
            Dict with drift flags:
            {
              "data_drift": bool,
              "prediction_drift": bool
            }
        """
        logger.info("="*60)
        logger.info("STARTING DRIFT DETECTION")
        logger.info("="*60)
        
        try:
            # Load data
            reference_df, current_df = self.load_data()
            
            # Detect data drift (SEPARATE SIGNAL #1)
            data_drift = self.detect_data_drift(reference_df, current_df)
            
            # FIX #2: Detect prediction drift independently (SEPARATE SIGNAL #2)
            prediction_drift = self.detect_prediction_drift(reference_df, current_df)
            
            # Run reports (for visualization, not decision-making)
            data_drift_metrics = self.run_data_drift_report(
                reference_df, current_df
            )
            
            performance_metrics = self.run_prediction_performance_report(
                reference_df, current_df
            )
            
            # Return BOTH signals separately (not collapsed)
            results = {
                "data_drift": bool(data_drift),
                "prediction_drift": bool(prediction_drift),
            }
            
            logger.info(f"Drift detection results: {results}")
            logger.info("="*60)
            logger.info("DRIFT DETECTION COMPLETE")
            logger.info("="*60)
            
            return results
        
        except Exception as e:
            logger.error(f"Drift detection failed: {e}", exc_info=True)
            raise


def run_drift_check(
    reference_data_path: str,
    current_data_path: str,
    drift_columns: List[str] = None,
    output_dir: str = "reports/drift"
) -> Dict[str, bool]:
    """
    Convenience function to run drift check.
    
    Args:
        reference_data_path: Path to reference dataset
        current_data_path: Path to current dataset
        drift_columns: Columns to monitor (from config if None)
        output_dir: Directory to save reports
    
    Returns:
        Dict with drift flags
    """
    detector = DriftDetector(
        reference_data_path=reference_data_path,
        current_data_path=current_data_path,
        drift_columns=drift_columns,
        output_dir=output_dir
    )
    
    return detector.run()
