"""
Data loading layer.
Responsibility: Load raw data only.
No cleaning, scaling, or preprocessing.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Raw dataframe, unmodified
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV is empty
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError(f"Data file is empty: {filepath}")
    
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def load_reference_data(filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load reference/validation data.
    Used for model comparisons or baseline metrics.
    
    Args:
        filepath: Path to reference data
    
    Returns:
        Reference dataframe or None if not provided
    """
    if filepath is None:
        return None
    
    return load_csv(filepath)
