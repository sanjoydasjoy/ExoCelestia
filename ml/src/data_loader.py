"""
Data loading utilities for Kepler/K2/TESS exoplanet datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_exoplanet_csv(path: str) -> pd.DataFrame:
    """
    Load and normalize NASA Kepler/K2/TESS exoplanet CSV data.
    
    This function:
    - Reads CSV files with pandas
    - Normalizes column names (lowercase, underscores)
    - Ensures required columns exist
    - Handles missing values with sensible defaults
    - Logs dropped rows
    
    Required columns:
    - orbital_period: Orbital period of the planet (days)
    - transit_duration: Transit duration (hours)
    - planet_radius: Planet radius (Earth radii)
    - stellar_temp: Stellar effective temperature (K)
    - flux: Light curve flux values (can be array/list column)
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame with normalized columns and handled missing values
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    logger = logging.getLogger(__name__)
    
    # Read CSV file
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded CSV from {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Normalize column names: lowercase and replace spaces/hyphens with underscores
    def normalize_column_name(col: str) -> str:
        """Convert column name to lowercase with underscores"""
        # Replace spaces, hyphens, and dots with underscores
        col = re.sub(r'[\s\-\.]+', '_', str(col))
        # Remove special characters except underscores
        col = re.sub(r'[^\w_]', '', col)
        # Convert to lowercase
        col = col.lower()
        # Remove multiple consecutive underscores
        col = re.sub(r'_+', '_', col)
        # Remove leading/trailing underscores
        col = col.strip('_')
        return col
    
    original_columns = df.columns.tolist()
    df.columns = [normalize_column_name(col) for col in df.columns]
    logger.info(f"Normalized column names from {len(set(original_columns))} unique columns")
    
    # Define required columns and their common aliases
    required_columns = {
        'orbital_period': ['orbital_period', 'period', 'koi_period', 'pl_orbper', 'period_days'],
        'transit_duration': ['transit_duration', 'duration', 'koi_duration', 'pl_trandur', 'transit_dur'],
        'planet_radius': ['planet_radius', 'radius', 'koi_prad', 'pl_radj', 'pl_rade', 'prad'],
        'stellar_temp': ['stellar_temp', 'temperature', 'koi_steff', 'st_teff', 'teff', 'star_temp'],
        'flux': ['flux', 'light_curve', 'lc', 'flux_values', 'timeseries']
    }
    
    # Map existing columns to required columns
    column_mapping = {}
    for required, aliases in required_columns.items():
        found = False
        for alias in aliases:
            if alias in df.columns:
                if alias != required:
                    column_mapping[alias] = required
                found = True
                break
        
        if not found:
            # Check if any column contains the required name as substring
            for col in df.columns:
                if any(alias in col for alias in aliases):
                    column_mapping[col] = required
                    found = True
                    logger.info(f"Mapped '{col}' to '{required}' (partial match)")
                    break
        
        if not found:
            raise ValueError(
                f"Required column '{required}' not found. "
                f"Looked for aliases: {aliases}. "
                f"Available columns: {df.columns.tolist()}"
            )
    
    # Rename columns if needed
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info(f"Renamed columns: {column_mapping}")
    
    # Verify all required columns now exist
    missing_cols = set(required_columns.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns after mapping: {missing_cols}")
    
    # Handle missing values
    initial_rows = len(df)
    logger.info(f"Initial missing values per column:\n{df[list(required_columns.keys())].isnull().sum()}")
    
    # Define sensible defaults for each column
    defaults = {
        'orbital_period': df['orbital_period'].median(),
        'transit_duration': df['transit_duration'].median(),
        'planet_radius': df['planet_radius'].median(),
        'stellar_temp': 5778,  # Sun-like star temperature as default
    }
    
    # Fill missing values for numeric columns
    for col, default_value in defaults.items():
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(default_value)
                logger.info(f"Filled {missing_count} missing values in '{col}' with {default_value:.2f}")
    
    # For flux column, handle differently (it might be a list/array)
    if 'flux' in df.columns:
        flux_missing = df['flux'].isnull().sum()
        if flux_missing > 0:
            logger.warning(f"Found {flux_missing} rows with missing flux data")
            # Drop rows with missing flux as it's critical for time-series analysis
            df = df.dropna(subset=['flux'])
            logger.info(f"Dropped {flux_missing} rows with missing flux data")
    
    # Drop any remaining rows with missing values in required columns
    rows_before = len(df)
    df = df.dropna(subset=list(required_columns.keys()))
    rows_dropped = rows_before - len(df)
    
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows ({rows_dropped/initial_rows*100:.2f}%) with remaining missing values")
    
    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


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

