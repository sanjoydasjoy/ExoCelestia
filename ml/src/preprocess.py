"""
Data preprocessing utilities for exoplanet detection
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'mean'
) -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    TODO: Determine best strategy for light curve data
    - mean/median: Simple imputation
    - interpolate: Time-series interpolation for flux values
    - drop: Remove rows with missing values
    
    Args:
        df: Input dataframe
        strategy: Strategy for handling missing values
        
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median())
    elif strategy == 'interpolate':
        df_clean = df_clean.interpolate(method='linear', axis=0)
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values after {strategy}: {missing_after}")
    
    return df_clean


def normalize_features(
    X: pd.DataFrame, 
    method: str = 'standard',
    scaler: Optional[object] = None
) -> Tuple[np.ndarray, object]:
    """
    Normalize/scale features
    
    TODO: Choose appropriate normalization for light curve data
    - standard: StandardScaler (mean=0, std=1)
    - minmax: MinMaxScaler (range 0-1)
    
    Args:
        X: Feature matrix
        method: Normalization method
        scaler: Pre-fitted scaler (for test data)
        
    Returns:
        Tuple of (normalized features, scaler object)
    """
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        X_normalized = scaler.fit_transform(X)
    else:
        X_normalized = scaler.transform(X)
    
    return X_normalized, scaler


def remove_outliers(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_std: float = 3.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers using standard deviation method
    
    TODO: Consider if outlier removal is appropriate for exoplanet detection
    Some "outliers" might be actual exoplanet signals!
    
    Args:
        X: Feature matrix
        y: Labels
        n_std: Number of standard deviations for outlier threshold
        
    Returns:
        Tuple of (filtered features, filtered labels)
    """
    # Calculate z-scores
    z_scores = np.abs((X - X.mean()) / X.std())
    
    # Keep rows where all features are within n_std standard deviations
    mask = (z_scores < n_std).all(axis=1)
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    removed = len(X) - len(X_clean)
    print(f"Removed {removed} outliers ({removed/len(X)*100:.2f}%)")
    
    return X_clean, y_clean


def balance_dataset(
    X: pd.DataFrame, 
    y: pd.Series, 
    method: str = 'undersample'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance imbalanced dataset
    
    TODO: Exoplanet datasets are typically highly imbalanced
    Choose appropriate balancing strategy:
    - undersample: Reduce majority class
    - oversample: Increase minority class (consider SMOTE)
    - class_weights: Use weighted loss in model training
    
    Args:
        X: Feature matrix
        y: Labels
        method: Balancing method
        
    Returns:
        Tuple of (balanced features, balanced labels)
    """
    if method == 'undersample':
        # Simple random undersampling
        class_counts = y.value_counts()
        min_class = class_counts.min()
        
        indices = []
        for class_label in y.unique():
            class_indices = y[y == class_label].index
            sampled = np.random.choice(class_indices, size=min_class, replace=False)
            indices.extend(sampled)
        
        X_balanced = X.loc[indices]
        y_balanced = y.loc[indices]
        
        print(f"Balanced dataset from {len(X)} to {len(X_balanced)} samples")
        print(f"Class distribution: {y_balanced.value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    # TODO: Implement other balancing methods (SMOTE, etc.)
    raise NotImplementedError(f"Method '{method}' not implemented yet")


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def full_preprocessing_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    normalize_method: str = 'standard',
    handle_missing: str = 'mean',
    remove_outliers_flag: bool = False,
    balance_flag: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Complete preprocessing pipeline
    
    Args:
        X: Feature matrix
        y: Labels
        normalize_method: Normalization method
        handle_missing: Strategy for missing values
        remove_outliers_flag: Whether to remove outliers
        balance_flag: Whether to balance dataset
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        Dictionary with processed data and scaler
    """
    print("Starting preprocessing pipeline...")
    
    # Handle missing values
    X_clean = handle_missing_values(X, strategy=handle_missing)
    
    # Remove outliers (optional)
    if remove_outliers_flag:
        X_clean, y = remove_outliers(X_clean, y)
    
    # Balance dataset (optional)
    if balance_flag:
        X_clean, y = balance_dataset(X_clean, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X_clean, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize features
    X_train_norm, scaler = normalize_features(X_train, method=normalize_method)
    X_test_norm, _ = normalize_features(X_test, method=normalize_method, scaler=scaler)
    
    print("Preprocessing complete!")
    
    return {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'scaler': scaler
    }


if __name__ == "__main__":
    print("Preprocessing utilities ready")
    print("TODO: Test with actual data")

