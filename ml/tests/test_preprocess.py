"""
Tests for preprocess.py module
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocess import preprocess_features


def test_preprocess_features_shape_and_no_nans():
    """
    Test that preprocess_features returns X with shape (n, m) and no NaNs
    """
    # Create sample data
    data = {
        'orbital_period': [3.52, 5.73, 2.41, 8.92, 4.15],
        'transit_duration': [2.1, 3.4, 1.8, 4.2, 2.6],
        'planet_radius': [1.2, 0.9, 1.5, 0.8, 1.3],
        'stellar_temp': [5800, 6200, 5400, 6100, 5700],
        'flux_mean': [0.998, 1.001, 0.995, 1.002, 0.997],
        'label': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Call preprocess_features
    X, y = preprocess_features(df, label_column='label', extract_flux_features=False)
    
    # Assert X has shape (n, m) where n is number of samples
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert X.ndim == 2, "X should be 2-dimensional"
    assert X.shape[0] == len(df), f"X should have {len(df)} rows"
    assert X.shape[1] > 0, "X should have at least one feature column"
    
    # Assert no NaNs in X
    assert not np.isnan(X).any(), "X should not contain any NaN values"
    
    # Additional checks
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert len(y) == len(df), "y should have same length as input data"

