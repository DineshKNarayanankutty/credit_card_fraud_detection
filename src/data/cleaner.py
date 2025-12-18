"""
Data cleaning layer.
Responsibility: Remove bad data only.
No feature engineering, scaling, or splitting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing values.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with missing rows removed
    """
    initial_count = len(df)
    df_clean = df.dropna()
    removed_count = initial_count - len(df_clean)
    
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} rows with missing values")
    else:
        logger.info("✓ No missing values found")
    
    return df_clean


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with duplicates removed
    """
    initial_count = len(df)
    df_unique = df.drop_duplicates()
    removed_count = initial_count - len(df_unique)
    
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} duplicate rows")
    else:
        logger.info("✓ No duplicates found")
    
    return df_unique


def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method.
    Applied only to numerical columns (not target).
    
    Args:
        df: Input dataframe
        threshold: Z-score threshold (default 3.0 = 99.7% coverage)
    
    Returns:
        Dataframe with outliers removed
    """
    initial_count = len(df)
    
    # Get numeric columns, excluding Class (target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Class' in numeric_cols:
        numeric_cols.remove('Class')
    
    if not numeric_cols:
        logger.info("✓ No numeric columns to check for outliers")
        return df
    
    # Calculate z-scores
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    
    # Keep rows where all z-scores are below threshold
    mask = (z_scores <= threshold).all(axis=1)
    df_clean = df[mask]
    
    removed_count = initial_count - len(df_clean)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} outliers (z-score > {threshold})")
    else:
        logger.info("✓ No outliers detected")
    
    return df_clean


def validate_target_column(df: pd.DataFrame, target_col: str = 'Class') -> Tuple[pd.DataFrame, bool]:
    """
    Validate target column exists and is binary.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
    
    Returns:
        Tuple of (dataframe, is_valid)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    unique_values = df[target_col].unique()
    if len(unique_values) != 2:
        raise ValueError(f"Target column must be binary, found {len(unique_values)} classes")
    
    logger.info(f"✓ Target column '{target_col}' is valid (binary classification)")
    return df, True


def clean_pipeline(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """
    Execute cleaning pipeline.
    Order matters: missing → duplicates → outliers
    
    Args:
        df: Raw dataframe
        remove_outliers: Whether to apply outlier removal
    
    Returns:
        Cleaned dataframe
    """
    logger.info("Starting cleaning pipeline...")
    
    df = drop_missing_values(df)
    df = drop_duplicates(df)
    
    if remove_outliers:
        df = remove_outliers_zscore(df)
    
    logger.info(f"✓ Cleaning complete: {len(df):,} rows remaining")
    return df
