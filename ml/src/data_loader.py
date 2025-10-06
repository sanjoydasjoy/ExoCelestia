"""
Data loading utilities for Kepler/K2/TESS exoplanet datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_kepler_data(filepath: str) -> pd.DataFrame:
    """
    Load Kepler mission data from CSV
    
    TODO: Adjust column names based on actual Kepler dataset structure
    Expected columns might include:
    - KepID: Kepler ID
    - FLUX.1, FLUX.2, ...: Light curve flux values
    - LABEL: Classification (1 = exoplanet, 0 = no exoplanet)
    
    Args:
        filepath: Path to Kepler CSV file
        
    Returns:
        DataFrame with Kepler data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded Kepler data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise ValueError(f"Error loading Kepler data: {str(e)}")


def load_k2_data(filepath: str) -> pd.DataFrame:
    """
    Load K2 mission data from CSV
    
    TODO: Adjust for K2-specific format (similar to Kepler but may have differences)
    
    Args:
        filepath: Path to K2 CSV file
        
    Returns:
        DataFrame with K2 data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded K2 data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise ValueError(f"Error loading K2 data: {str(e)}")


def load_tess_data(filepath: str) -> pd.DataFrame:
    """
    Load TESS mission data from CSV
    
    TODO: Adjust for TESS-specific format
    TESS data structure may differ from Kepler/K2
    
    Args:
        filepath: Path to TESS CSV file
        
    Returns:
        DataFrame with TESS data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded TESS data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise ValueError(f"Error loading TESS data: {str(e)}")


def split_features_labels(
    df: pd.DataFrame, 
    label_column: str = 'LABEL'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and labels
    
    Args:
        df: Input dataframe
        label_column: Name of the label column
        
    Returns:
        Tuple of (features, labels)
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe")
    
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    # TODO: Remove non-feature columns (e.g., IDs, metadata)
    # Example: X = X.drop(columns=['KepID', 'KeplerName'], errors='ignore')
    
    return X, y


def load_and_prepare_data(
    filepath: str,
    mission: str = 'kepler',
    label_column: str = 'LABEL'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load and prepare data in one step
    
    Args:
        filepath: Path to data file
        mission: Mission type ('kepler', 'k2', or 'tess')
        label_column: Name of label column
        
    Returns:
        Tuple of (features, labels)
    """
    loaders = {
        'kepler': load_kepler_data,
        'k2': load_k2_data,
        'tess': load_tess_data
    }
    
    if mission.lower() not in loaders:
        raise ValueError(f"Unknown mission: {mission}. Choose from {list(loaders.keys())}")
    
    df = loaders[mission.lower()](filepath)
    X, y = split_features_labels(df, label_column)
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Data loader utilities ready")
    print("TODO: Test with actual Kepler/K2/TESS data files")

