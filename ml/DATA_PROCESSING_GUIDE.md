# Exoplanet Data Processing Guide

This guide explains how to use the data loading and preprocessing functions for NASA Kepler/K2/TESS exoplanet datasets.

## Overview

Two main functions are provided:

1. **`load_exoplanet_csv(path: str) -> pd.DataFrame`** - Load and normalize CSV data
2. **`preprocess_features(df) -> (X, y)`** - Preprocess features for machine learning

## Installation

Make sure you have the required dependencies installed:

```bash
cd ml
pip install -r requirements.txt
```

Key dependencies:
- pandas
- numpy
- scikit-learn
- scipy (for periodicity analysis)

## Function 1: `load_exoplanet_csv`

### Purpose
Load NASA Kepler/K2/TESS CSV files with automatic column normalization and missing value handling.

### Location
`ml/src/data_loader.py`

### Signature
```python
def load_exoplanet_csv(path: str) -> pd.DataFrame
```

### Required Columns

The function expects these columns (or their aliases):

| Required Column | Common Aliases | Description |
|----------------|----------------|-------------|
| `orbital_period` | period, koi_period, pl_orbper | Orbital period (days) |
| `transit_duration` | duration, koi_duration, pl_trandur | Transit duration (hours) |
| `planet_radius` | radius, koi_prad, pl_rade | Planet radius (Earth radii) |
| `stellar_temp` | temperature, koi_steff, st_teff | Stellar temperature (K) |
| `flux` | light_curve, lc, flux_values | Light curve flux values |

### Features

1. **Column Name Normalization**
   - Converts to lowercase
   - Replaces spaces/hyphens with underscores
   - Removes special characters

2. **Column Alias Recognition**
   - Automatically recognizes common NASA column naming conventions
   - Renames columns to standard names

3. **Missing Value Handling**
   - Fills missing numeric values with median (for orbital_period, transit_duration, planet_radius)
   - Uses 5778K (Sun-like) as default for missing stellar_temp
   - Drops rows with missing flux data (critical for analysis)
   - Logs all operations

### Usage Example

```python
from data_loader import load_exoplanet_csv

# Load CSV file
df = load_exoplanet_csv('path/to/kepler_data.csv')

print(f"Loaded {len(df)} exoplanet candidates")
print(f"Columns: {df.columns.tolist()}")
```

### Error Handling

- Raises `FileNotFoundError` if CSV file doesn't exist
- Raises `ValueError` if required columns are missing
- Logs warnings for dropped rows

## Function 2: `preprocess_features`

### Purpose
Preprocess exoplanet features for machine learning, including categorical encoding, feature scaling, and time-series feature extraction.

### Location
`ml/src/preprocess.py`

### Signature
```python
def preprocess_features(
    df: pd.DataFrame,
    label_column: str = 'label',
    extract_flux_features: bool = True,
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]
```

### Parameters

- **`df`**: Input DataFrame (output from `load_exoplanet_csv`)
- **`label_column`**: Name of the label/target column (default: 'label')
- **`extract_flux_features`**: Extract time-series features from flux (default: True)
- **`feature_columns`**: Specific columns to use as features (default: all numeric columns)

### Returns

- **`X`**: numpy array of shape (n_samples, n_features) - scaled features
- **`y`**: numpy array of shape (n_samples,) - labels

### Features Extracted

#### Base Features (scaled with StandardScaler)
- orbital_period
- transit_duration
- planet_radius
- stellar_temp
- Any other numeric columns

#### Flux Time-Series Features (if enabled)
- **Statistical Features:**
  - Mean flux
  - Standard deviation
  - Median
  - Min/Max values
  - Range (max - min)
  - Skewness
  - Kurtosis

- **Periodicity Features (requires scipy):**
  - Dominant frequency
  - Power at dominant frequency

### Usage Example

```python
from data_loader import load_exoplanet_csv
from preprocess import preprocess_features

# Load data
df = load_exoplanet_csv('kepler_data.csv')

# Preprocess features
X, y = preprocess_features(
    df, 
    label_column='label',
    extract_flux_features=True
)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Feature statistics - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
```

### Advanced Usage

#### Without Flux Feature Extraction
```python
# Faster processing, uses only base features
X, y = preprocess_features(df, extract_flux_features=False)
```

#### Custom Feature Selection
```python
# Use only specific features
X, y = preprocess_features(
    df,
    feature_columns=['orbital_period', 'planet_radius', 'stellar_temp']
)
```

#### No Label Column
```python
# For prediction on unlabeled data
X, _ = preprocess_features(df, label_column='nonexistent')
# Returns X with features, empty array for y
```

## Complete Pipeline Example

```python
from data_loader import load_exoplanet_csv
from preprocess import preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load data
print("Loading data...")
df = load_exoplanet_csv('data/kepler_exoplanets.csv')

# Step 2: Preprocess features
print("Preprocessing features...")
X, y = preprocess_features(df, label_column='label', extract_flux_features=True)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Flux Data Formats

The flux column can contain data in various formats:

1. **Python list**: `[1.0, 2.0, 3.0, ...]`
2. **Numpy array**: `np.array([1.0, 2.0, 3.0, ...])`
3. **String representation**: `"1.0,2.0,3.0,..."`
4. **Single value**: `1.0` (treated as array of length 1)

The preprocessing function automatically handles all these formats.

## Logging

Both functions use Python's logging module to provide detailed information:

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Now you'll see detailed logs when calling the functions
df = load_exoplanet_csv('data.csv')
# INFO - Loaded CSV from data.csv: 1000 rows, 10 columns
# INFO - Normalized column names from 10 unique columns
# INFO - Filled 5 missing values in 'orbital_period' with 15.23
# ...
```

## Testing

Run the test suite to verify functionality:

```bash
cd ml
pytest tests/test_data_preprocessing.py -v
```

Run the example script:

```bash
cd ml
python examples/usage_example.py
```

## Common Issues

### Issue: "Required column 'X' not found"
**Solution**: Check that your CSV has the required columns or their aliases. The function will tell you which columns are available.

### Issue: scipy import warning
**Solution**: Install scipy with `pip install scipy==1.11.2`. Without scipy, periodicity features will be skipped.

### Issue: "All features are NaN after scaling"
**Solution**: Check that your data contains valid numeric values. Missing values should be handled by `load_exoplanet_csv`.

## Performance Considerations

- **Flux feature extraction** adds ~8-10 features per sample but can be slow for large datasets
- For faster processing on large datasets, set `extract_flux_features=False`
- Periodicity analysis (FFT) requires flux arrays with >10 values for meaningful results

## NASA Data Sources

Compatible with data from:

- **Kepler Mission**: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- **K2 Mission**: Extended Kepler mission data
- **TESS Mission**: Transiting Exoplanet Survey Satellite

Download CSV files from these sources and use directly with these functions.

## Support

For issues or questions:
1. Check the error messages and logs
2. Review the test files for usage examples
3. Ensure all dependencies are installed correctly

