"""
Inference pipeline orchestration.
Responsibility: Orchestrate inference workflow ONLY.
Uses: inference layer (predictor + schema).
"""

import logging
from typing import Dict, Any, List
from src.utils.logger import setup_logging
from src.inference.predictor import get_predictor
from src.inference.schema import TransactionInput, BatchTransactionInput
from src.utils.config import config

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Production inference pipeline"""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
        """
        logger.info("Initializing inference pipeline...")
        self.predictor = get_predictor(model_path, scaler_path)
        logger.info("Inference pipeline ready")
    
    def predict_transaction(
        self,
        features: List[float],
        amount: float,
        threshold: float = 0.5,
        timestamp: str = None
    ) -> Dict[str, Any]:
        """
        Predict fraud for single transaction.
        
        Args:
            features: List of 30 features
            amount: Transaction amount
            threshold: Classification threshold
            timestamp: Optional timestamp
        
        Returns:
            Dict with prediction and metadata
        """
        logger.debug(f"Predicting single transaction...")
        
        # Validate input
        transaction = TransactionInput(
            features=features,
            amount=amount,
            timestamp=timestamp
        )
        
        # Get prediction
        result = self.predictor.predict_single(transaction.features, threshold)
        
        # Add metadata
        result['amount'] = amount
        result['timestamp'] = timestamp
        
        logger.debug(f"Prediction: {result['prediction']}, Probability: {result['probability']:.4f}")
        
        return result
    
    def predict_batch(
        self,
        transactions: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict fraud for batch of transactions.
        
        Args:
            transactions: List of transaction dicts with 'features' and 'amount'
            threshold: Classification threshold
        
        Returns:
            Dict with batch predictions
        """
        logger.info(f"Predicting batch of {len(transactions)} transactions...")
        
        # Validate batch
        batch_input = BatchTransactionInput(
            transactions=[
                TransactionInput(
                    features=t['features'],
                    amount=t['amount'],
                    timestamp=t.get('timestamp')
                )
                for t in transactions
            ],
            threshold=threshold
        )
        
        # Get predictions
        features_list = [t.features for t in batch_input.transactions]
        result = self.predictor.predict_batch(features_list, threshold)
        
        logger.info(f"Batch complete: {result['fraud_count']} frauds detected ({result['fraud_rate']:.2%})")
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check pipeline health.
        
        Returns:
            Dict with health status
        """
        health = self.predictor.health_check()
        logger.info(f"Health check: {health['status']}")
        return health


def main():
    """Main entry point"""
    setup_logging()
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        model_path=config.model.model_path,
        scaler_path=config.model.preprocessor_path
    )
    
    # Check health
    health = pipeline.health_check()
    if health['status'] != 'healthy':
        logger.error("Pipeline is not healthy!")
        exit(1)
    
    logger.info("Inference pipeline ready for predictions")


if __name__ == "__main__":
    main()
