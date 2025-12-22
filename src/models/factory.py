"""
Model factory.

Responsibility:
- Instantiate ML models based on configuration
- NO training logic
- NO data handling
"""

from typing import Dict, Any
import logging

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

logger = logging.getLogger(__name__)


def get_model(
    model_type: str,
    model_params: Dict[str, Any],
):
    """
    Create and return a model instance.

    Args:
        model_type: Type of model (e.g., 'xgboost', 'rf')
        model_params: Dictionary of model hyperparameters

    Returns:
        Instantiated model object

    Raises:
        ValueError: If model type is unsupported
    """

    model_type = model_type.lower()

    logger.info(f"Initializing model: {model_type}")

    if model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost`."
            )

        return XGBClassifier(**model_params)

    elif model_type == "rf":
        return RandomForestClassifier(**model_params)

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Supported types are: 'xgboost', 'rf'."
        )
