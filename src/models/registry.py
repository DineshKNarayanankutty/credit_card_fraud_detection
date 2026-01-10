"""
Model registry layer.

Responsibility:
- Save and load models ONLY
- Local filesystem / DVC-compatible registry

NO training logic
NO inference logic
NO MLflow
"""

import logging
from pathlib import Path
from typing import Any, List

from src.utils.io import save_pickle, load_pickle

logger = logging.getLogger(__name__)


def save_model(
    model: Any,
    filepath: str,
    overwrite: bool = True,
) -> str:
    """
    Save model to disk.

    Args:
        model: Trained model object
        filepath: Path where model will be saved
        overwrite: If False, raise error if file exists

    Returns:
        Filepath where model was saved

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    path = Path(filepath)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Model already exists at {filepath}. "
            f"Use overwrite=True to replace."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    save_pickle(model, filepath)

    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filepath: str) -> Any:
    """
    Load model from disk.

    Args:
        filepath: Path to saved model

    Returns:
        Loaded model object

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


def list_models(directory: str = "artifacts") -> List[str]:
    """
    List all saved models in directory.

    Args:
        directory: Directory to search

    Returns:
        List of model file paths
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    models = list(dir_path.glob("*.pkl"))

    logger.info(f"Found {len(models)} models in {directory}")

    return [str(m) for m in models]
