"""
Evaluation pipeline orchestration.
Responsibility: Orchestrate evaluation workflow ONLY.
Uses: inference layer, evaluation layer.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from src.utils.logger import setup_logging
from src.inference.predictor import get_predictor
from src.evaluation.threshold import get_threshold_analysis
from src.evaluation.reports import generate_threshold_report, log_report
from src.utils.config import config

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Model evaluation and threshold analysis pipeline"""
    
    def __init__(self, config):
        """
        Initialize evaluation pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("Initialized evaluation pipeline")
    
    def analyze_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: list = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Analyze model performance across thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to analyze
        
        Returns:
            Dict with analysis results
        """
        logger.info("Analyzing thresholds...")
        
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        analysis = get_threshold_analysis(
            y_true, y_proba,
            thresholds=thresholds
        )
        
        report = generate_threshold_report(analysis)
        log_report(report)
        
        return analysis
    
    def run(self, model_path: str, scaler_path: str, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Execute evaluation pipeline.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            test_data: Tuple of (X_test, y_test)
        
        Returns:
            Dict with evaluation results
        """
        logger.info("="*60)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("="*60)
        
        try:
            # Load predictor
            logger.info("\n[STEP 1/2] Loading model and scaler...")
            predictor = get_predictor(model_path, scaler_path)
            
            # Analyze thresholds
            logger.info("\n[STEP 2/2] Analyzing thresholds...")
            X_test, y_test = test_data
            
            # Get predictions
            result = predictor.predict_batch(X_test.tolist())
            
            analysis = self.analyze_thresholds(
                y_test,
                np.array(result['probabilities']),
                thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            
            results = {
                'success': True,
                'threshold_analysis': analysis,
            }
            
            logger.info("\n" + "="*60)
            logger.info("EVALUATION PIPELINE COMPLETE")
            logger.info("="*60)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


def main():
    """Main entry point"""
    setup_logging()
    pipeline = EvaluationPipeline(config)
    logger.info("Evaluation pipeline ready. Call run() with model paths and test data.")


if __name__ == "__main__":
    main()
