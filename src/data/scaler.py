"""
Feature scaling layer.
Responsibility: Scale numerical features only.
Must NOT scale target column.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Encapsulates feature scaling.
    Fit on training data, transform on any data.
    """
    
    def __init__(self, method: str = "robust"):
        """
        Initialize scaler.
        
        Args:
            method: "robust", "standard", or "minmax"
        """
        if method == "robust":
            self.scaler = RobustScaler()
        elif method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.method = method
        self.fitted = False
        logger.info(f"Initialized {method} scaler")
    
    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Fit scaler on training data.
        
        Args:
            X: Training features (n_samples, n_features)
        
        Returns:
            self
        """
        self.scaler.fit(X)
        self.fitted = True
        logger.info(f"✓ Scaler fitted on {X.shape:,} samples")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features.
        
        Args:
            X: Features to scale
        
        Returns:
            Scaled features
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        Use only on training data.
        
        Args:
            X: Training features
        
        Returns:
            Scaled training features
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse scaling.
        Useful for interpretability.
        
        Args:
            X: Scaled features
        
        Returns:
            Original scale features
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(X)


def get_feature_scaler(method: str = "robust") -> FeatureScaler:
    """
    Factory function for creating scalers.
    
    Args:
        method: Scaling method
    
    Returns:
        FeatureScaler instance
    """
    return FeatureScaler(method=method)
