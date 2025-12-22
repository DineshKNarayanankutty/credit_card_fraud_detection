"""
Configuration management.
Supports dev/prod environments and optional YAML files.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Any

logger = logging.getLogger(__name__)


# -----------------------------
# DATA CONFIG
# -----------------------------
@dataclass
class DataConfig:
    raw_path: str = "data/raw/creditcard.csv"
    processed_path: str = "data/processed"
    remove_outliers: bool = True
    target_column: str = "Class"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42


# -----------------------------
# FEATURE CONFIG
# -----------------------------
@dataclass
class FeatureConfig:
    scaling_method: str = "robust"  # robust, standard, minmax
    handle_imbalance: bool = True
    imbalance_method: str = "smote"
    smote_k_neighbors: int = 5


# -----------------------------
# MODEL CONFIG
# -----------------------------
@dataclass
class ModelConfig:
    model_type: str = "xgboost"  # xgboost or rf
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    })
    cv_folds: int = 5
    output_path: str = "artifacts/model.pkl"


# -----------------------------
# EVALUATION CONFIG
# -----------------------------
@dataclass
class EvaluationConfig:
    threshold_method: str = "f1"  # f1 or f2
    min_recall: float = 0.80
    min_precision: float = 0.85
    cv_metric: str = "roc_auc"


# -----------------------------
# API CONFIG
# -----------------------------
@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


# -----------------------------
# ROOT CONFIG
# -----------------------------
@dataclass
class Config:
    env: str = "dev"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.
        """
        env = os.getenv("ENV", "dev")

        logger.info(f"Loading configuration for environment: {env}")

        return cls(
            env=env,
            data=DataConfig(
                raw_path=os.getenv("DATA_RAW_PATH", DataConfig.raw_path),
                target_column=os.getenv("TARGET_COLUMN", DataConfig.target_column),
            ),
            model=ModelConfig(
                model_type=os.getenv("MODEL_TYPE", ModelConfig.model_type),
            ),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file.
        """
        logger.info(f"Loading configuration from YAML: {path}")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls(
            env=cfg.get("env", "dev"),
            data=DataConfig(**cfg.get("data", {})),
            features=FeatureConfig(**cfg.get("features", {})),
            model=ModelConfig(**cfg.get("model", {})),
            evaluation=EvaluationConfig(**cfg.get("evaluation", {})),
            api=APIConfig(**cfg.get("api", {})),
        )


# -----------------------------
# GLOBAL CONFIG INSTANCE
# -----------------------------
config = Config.from_env()
