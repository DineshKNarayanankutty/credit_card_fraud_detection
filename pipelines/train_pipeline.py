"""
Training pipeline orchestration.
Responsibility: Orchestrate training workflow ONLY.
NO ML logic, NO data manipulation, NO preprocessing.
Uses: data layer, features layer, models layer, evaluation layer.
"""

import logging
from typing import Dict, Any
from src.utils.logger import setup_logging
from src.data.loader import load_csv
from src.data.cleaner import clean_pipeline
from src.data.scaler import get_feature_scaler
from src.data.splitter import train_val_test_split
from src.features.imbalance import apply_smote
from src.models.factory import get_model
from src.models.trainer import train, cross_validate, evaluate_on_set
from src.models.registry import save_model
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.threshold import find_optimal_threshold, apply_threshold
from src.evaluation.reports import generate_full_report, log_report
from src.utils.io import save_pickle
from src.utils.config import config

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info(f"Initialized training pipeline for {config.model.model_type} model")
    
    def run(self, data_path: str) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        
        Order (mandatory):
        1. Load data
        2. Clean data
        3. Scale features
        4. Split data (no SMOTE yet)
        5. Apply SMOTE to training set only
        6. Create model
        7. Train model
        8. Cross-validate
        9. Evaluate on validation set
        10. Find optimal threshold
        11. Evaluate on test set with 
