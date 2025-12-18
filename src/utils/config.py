"""
Configuration management.
Supports dev/prod environments and YAML files.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data layer configuration"""
    raw_path: str = "data/raw/creditcard.csv"
    processed_path: str = "data/processed"
    remove_outliers: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42


@dataclass
class FeatureConfig:
    """Features layer configuration"""
    scaling_method: str = "robust"  # robust, standard, minmax
    handle_imbalance: bool = True
    imbalance_method: str = "smote"
    smote_k_neighbors: int = 5


@dataclass
class ModelConfig:
    """Model layer configuration"""
    model_type: str = "xgboost"  # xgboost or rf
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    })
    cv_folds: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation layer configuration"""
    threshold_method: str = "f1"  # f1 or f2
    min_recall: float = 0.80
    min_precision: float = 0.85


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@dataclass
class Config:
    """Main configuration"""
    env: str = "dev"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load from environment"""
        env = os.getenv("ENV", "dev")
        logger.info(f"Loaded configuration for environment: {env}")
        return cls(env=env)


# Global instance
config = Config.from_env()
