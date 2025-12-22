"""
Model registry layer.
Responsibility: Save and load models ONLY.
NO training or prediction logic.
"""

import logging
from pathlib import Path
from typing import Optional
from xgboost import XGBClassifier
from src.utils.io import save_pickle, load_pickle

logger = logging.getLogger(__name__)


def save_model(model: XGBClassifier, filepath: str, overwrite: bool = False) -> str:
    """
    Save model to disk.
    
    Args:
        model: Trained XGBClassifier instance
        filepath: Path where model will be saved
        overwrite: If False, raise error if file exists
    
    Returns:
        Path where model was saved
    
    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    path = Path(filepath)
    
    # Check if file exists
    if path.exists() and not overwrite:
        raise FileExistsError(f"Model already exists at {filepath}. Use overwrite=True to replace.")
    
    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    save_pickle(model, filepath)
    
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filepath: str) -> XGBClassifier:
    """
    Load model from disk.
    
    Args:
        filepath: Path to saved model
    
    Returns:
        Loaded XGBClassifier instance
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    model = load_pickle(filepath)
    
    logger.info(f"Model loaded from {filepath}")
    return model


def model_exists(filepath: str) -> bool:
    """
    Check if model exists.
    
    Args:
        filepath: Path to model
    
    Returns:
        True if model exists, False otherwise
    """
    return Path(filepath).exists()


def delete_model(filepath: str) -> None:
    """
    Delete model from disk.
    
    Args:
        filepath: Path to model
    
    Raises:
        FileNotFoundError: If model doesn't exist
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    path.unlink()
    logger.info(f"Model deleted from {filepath}")


def list_models(directory: str = "models") -> list:
    """
    List all saved models in directory.
    
    Args:
        directory: Directory to search
    
    Returns:
        List of model filepaths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    models = list(dir_path.glob("*.pkl"))
    logger.info(f"Found {len(models)} models in {directory}")
    
    return [str(m) for m in models]
