"""
Retrain decision and execution pipeline.
Responsibility: Orchestrate retraining workflow ONLY.
Uses: monitoring layer + training layer
NO ML logic, NO data manipulation, NO inference.

FIX #3: TrainingPipeline now calls train_pipeline.run() correctly
"""

import logging
from typing import Dict, Any

from src.utils.logger import setup_logging
from src.utils.config import config
from monitoring.evidently import run_drift_check
from pipelines.train_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class RetrainDecisionMaker:
    """Automated retraining orchestration"""
    
    def __init__(self, config):
        """
        Initialize retrain orchestrator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("Initialized retrain decision maker")
    
    def should_retrain(
        self,
        reference_data_path: str,
        current_data_path: str
    ) -> tuple:
        """
        Decide whether to retrain based on drift detection.
        
        Args:
            reference_data_path: Path to reference dataset
            current_data_path: Path to current dataset
        
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        logger.info("="*60)
        logger.info("CHECKING IF RETRAIN NEEDED")
        logger.info("="*60)
        
        try:
            # Run drift detection
            drift_results = run_drift_check(
                reference_data_path=reference_data_path,
                current_data_path=current_data_path,
                output_dir=self.config.monitoring.output_dir
            )
            
            # Decision logic: BOTH signals considered separately
            data_drift = drift_results.get("data_drift", False)
            prediction_drift = drift_results.get("prediction_drift", False)
            
            # Conservative: retrain if EITHER signal detected
            should_retrain = data_drift or prediction_drift
            
            reason = ""
            if data_drift:
                reason += "Data drift detected. "
            if prediction_drift:
                reason += "Prediction drift detected. "
            
            if not should_retrain:
                reason = "No significant drift detected."
            
            logger.info(f"Retrain decision: {should_retrain}")
            logger.info(f"Reason: {reason}")
            
            return should_retrain, reason
        
        except Exception as e:
            logger.error(f"Drift check failed: {e}", exc_info=True)
            # Conservative: if drift check fails, don't retrain
            return False, f"Drift check failed: {e}"
    
    def run(
        self,
        data_path: str,
        reference_data_path: str,
        current_data_path: str
    ) -> Dict[str, Any]:
        """
        Execute retraining pipeline.
        
        Steps:
        1. Check if retrain needed
        2. If NO: return early
        3. If YES: run training pipeline
        4. Return results
        
        Args:
            data_path: Path to raw data for training
            reference_data_path: Path to reference dataset
            current_data_path: Path to current dataset
        
        Returns:
            Dict with retrain status and results
        """
        logger.info("="*60)
        logger.info("STARTING RETRAIN PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Check if retrain needed
            logger.info("\n[STEP 1/3] Checking if retrain needed...")
            should_retrain, reason = self.should_retrain(
                reference_data_path=reference_data_path,
                current_data_path=current_data_path
            )
            
            # Step 2: If not needed, return early
            if not should_retrain:
                logger.info("\n[STEP 2/3] Skipping retraining (no drift detected)")
                
                results = {
                    'retrained': False,
                    'reason': reason,
                    'decision': 'SKIP',
                }
                
                logger.info("="*60)
                logger.info("✓ RETRAIN PIPELINE COMPLETE (SKIPPED)")
                logger.info("="*60)
                
                return results
            
            # Step 2: Run training pipeline
            logger.info("\n[STEP 2/3] Running training pipeline...")
            training_pipeline = TrainingPipeline(self.config)
            # FIX #3: Call run() method (TrainingPipeline is a class)
            train_results = training_pipeline.run(data_path)
            
            if not train_results.get('success', False):
                raise Exception(f"Training failed: {train_results.get('error')}")
            
            # Step 3: Return results
            logger.info("\n[STEP 3/3] Finalizing retrain results...")
            
            results = {
                'retrained': True,
                'reason': reason,
                'decision': 'RETRAIN',
                'training_results': train_results,
            }
            
            logger.info("="*60)
            logger.info("✓ RETRAIN PIPELINE COMPLETE (RETRAINED)")
            logger.info("="*60)
            
            return results
        
        except Exception as e:
            logger.error(f"Retrain pipeline failed: {e}", exc_info=True)
            return {
                'retrained': False,
                'reason': str(e),
                'decision': 'ERROR',
                'error': str(e),
            }


def main():
    """Main entry point"""
    setup_logging()
    decision_maker = RetrainDecisionMaker(config)
    
    results = decision_maker.run(
        data_path=config.data.raw_path,
        reference_data_path=config.monitoring.reference_data_path,
        current_data_path=config.monitoring.current_data_path
    )
    
    logger.info(f"Final results: {results}")


if __name__ == "__main__":
    main()
