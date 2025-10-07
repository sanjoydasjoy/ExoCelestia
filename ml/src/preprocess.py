"""
Data preprocessing utilities for exoplanet detection
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import scipy for periodicity analysis
try:
    from scipy import signal, stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Periodicity features will be skipped.")


def preprocess_features(
    df: pd.DataFrame,
    label_column: str = 'label',
    extract_flux_features: bool = True,
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess features for exoplanet detection.
    
    This function:
    - Encodes categorical values using LabelEncoder
    - Scales numeric features using StandardScaler
    - Extracts time-series features from 'flux' column (mean, std, periodicity)
    - Returns processed features and labels as numpy arrays
    
    Args:
        df: Input DataFrame (should have been processed by load_exoplanet_csv)
        label_column: Name of the label column (default: 'label')
        extract_flux_features: Whether to extract time-series features from flux (default: True)
        feature_columns: List of feature columns to use. If None, uses all except label
        
    Returns:
        Tuple of (X, y) where:
            X: numpy array of shape (n_samples, n_features) - preprocessed features
            y: numpy array of shape (n_samples,) - labels
            
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger = logging.getLogger(__name__)
    
    df_processed = df.copy()
    
    # Separate labels if present
    if label_column in df_processed.columns:
        y = df_processed[label_column].values
        df_features = df_processed.drop(columns=[label_column])
        logger.info(f"Extracted labels: {len(y)} samples, {len(np.unique(y))} unique classes")
    else:
        y = None
        df_features = df_processed
        logger.warning(f"Label column '{label_column}' not found. Returning None for labels.")
    
    # Determine feature columns
    if feature_columns is None:
        # Use standard exoplanet features
        base_features = ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp']
        feature_columns = [col for col in base_features if col in df_features.columns]
        
        # Add any other numeric columns
        for col in df_features.columns:
            if col not in feature_columns and col != 'flux':
                if pd.api.types.is_numeric_dtype(df_features[col]):
                    feature_columns.append(col)
    
    logger.info(f"Using feature columns: {feature_columns}")
    
    # Extract flux features if requested
    flux_features_df = None
    if extract_flux_features and 'flux' in df_features.columns:
        logger.info("Extracting time-series features from flux column...")
        flux_features_df = _extract_flux_features(df_features['flux'])
    
    # Select numeric features
    X_numeric = df_features[feature_columns].copy()
    
    # Identify and encode categorical columns
    categorical_cols = []
    label_encoders = {}
    
    for col in X_numeric.columns:
        if pd.api.types.is_object_dtype(X_numeric[col]) or pd.api.types.is_categorical_dtype(X_numeric[col]):
            categorical_cols.append(col)
            le = LabelEncoder()
            X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Encoded categorical column '{col}': {len(le.classes_)} unique values")
    
    if categorical_cols:
        logger.info(f"Encoded {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # Combine numeric features with flux features
    if flux_features_df is not None:
        X_combined = pd.concat([X_numeric, flux_features_df], axis=1)
        logger.info(f"Combined features: {X_numeric.shape[1]} base + {flux_features_df.shape[1]} flux = {X_combined.shape[1]} total")
    else:
        X_combined = X_numeric
    
    # Scale all features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    logger.info(f"Scaled features using StandardScaler: shape {X_scaled.shape}")
    logger.info(f"Feature statistics - Mean: {X_scaled.mean():.6f}, Std: {X_scaled.std():.6f}")
    
    # Return as numpy arrays
    if y is not None:
        return X_scaled, y
    else:
        return X_scaled, np.array([])


def _extract_flux_features(flux_series: pd.Series) -> pd.DataFrame:
    """
    Extract statistical and periodicity features from flux time-series data.
    
    Features extracted:
    - Mean flux
    - Standard deviation of flux
    - Median flux
    - Min/Max flux
    - Flux range (max - min)
    - Skewness
    - Kurtosis
    - Dominant frequency (periodicity)
    - Power at dominant frequency
    
    Args:
        flux_series: Series containing flux values (can be arrays, lists, or scalars)
        
    Returns:
        DataFrame with extracted features
    """
    logger = logging.getLogger(__name__)
    
    features = {
        'flux_mean': [],
        'flux_std': [],
        'flux_median': [],
        'flux_min': [],
        'flux_max': [],
        'flux_range': [],
        'flux_skew': [],
        'flux_kurtosis': []
    }
    
    if SCIPY_AVAILABLE:
        features['flux_dominant_freq'] = []
        features['flux_dominant_power'] = []
    
    for idx, flux_value in enumerate(flux_series):
        # Handle different flux formats
        if isinstance(flux_value, (list, np.ndarray)):
            flux_array = np.array(flux_value, dtype=float)
        elif isinstance(flux_value, str):
            # Try to parse string representation of list/array
            try:
                flux_array = np.fromstring(flux_value.strip('[]'), sep=',')
            except:
                # If parsing fails, treat as single value
                flux_array = np.array([float(flux_value)])
        else:
            # Single numeric value
            flux_array = np.array([float(flux_value)])
        
        # Remove NaN values
        flux_array = flux_array[~np.isnan(flux_array)]
        
        if len(flux_array) == 0:
            # No valid data, use default values
            features['flux_mean'].append(0.0)
            features['flux_std'].append(0.0)
            features['flux_median'].append(0.0)
            features['flux_min'].append(0.0)
            features['flux_max'].append(0.0)
            features['flux_range'].append(0.0)
            features['flux_skew'].append(0.0)
            features['flux_kurtosis'].append(0.0)
            if SCIPY_AVAILABLE:
                features['flux_dominant_freq'].append(0.0)
                features['flux_dominant_power'].append(0.0)
        else:
            # Basic statistics
            features['flux_mean'].append(np.mean(flux_array))
            features['flux_std'].append(np.std(flux_array))
            features['flux_median'].append(np.median(flux_array))
            features['flux_min'].append(np.min(flux_array))
            features['flux_max'].append(np.max(flux_array))
            features['flux_range'].append(np.max(flux_array) - np.min(flux_array))
            
            # Higher-order statistics
            if SCIPY_AVAILABLE and len(flux_array) > 3:
                features['flux_skew'].append(stats.skew(flux_array))
                features['flux_kurtosis'].append(stats.kurtosis(flux_array))
            else:
                features['flux_skew'].append(0.0)
                features['flux_kurtosis'].append(0.0)
            
            # Periodicity analysis using FFT
            if SCIPY_AVAILABLE and len(flux_array) > 10:
                try:
                    # Compute power spectral density
                    frequencies, power = signal.periodogram(flux_array)
                    
                    # Find dominant frequency (excluding DC component at freq=0)
                    if len(frequencies) > 1:
                        non_dc_idx = frequencies > 0
                        if np.any(non_dc_idx):
                            dominant_idx = np.argmax(power[non_dc_idx]) + 1
                            features['flux_dominant_freq'].append(frequencies[dominant_idx])
                            features['flux_dominant_power'].append(power[dominant_idx])
                        else:
                            features['flux_dominant_freq'].append(0.0)
                            features['flux_dominant_power'].append(0.0)
                    else:
                        features['flux_dominant_freq'].append(0.0)
                        features['flux_dominant_power'].append(0.0)
                except:
                    features['flux_dominant_freq'].append(0.0)
                    features['flux_dominant_power'].append(0.0)
            elif SCIPY_AVAILABLE:
                features['flux_dominant_freq'].append(0.0)
                features['flux_dominant_power'].append(0.0)
    
    features_df = pd.DataFrame(features)
    logger.info(f"Extracted {len(features)} time-series features from {len(flux_series)} flux observations")
    
    return features_df


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

