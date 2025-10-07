# Model Explanation Guide

## Overview

`ml/src/explain.py` provides interpretable explanations for exoplanet detection model predictions using SHAP values and fallback methods.

## Installation

```bash
# Basic (required)
pip install -r requirements.txt

# For SHAP explanations (recommended)
pip install shap
```

## Quick Start

```python
from explain import explain_prediction
import pickle
import numpy as np

# Load trained model
with open('models/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create sample features
X_sample = np.array([[10.5, 2.3, 1.2, 5778, 0.98, -0.12, ...]])

# Define feature names
feature_names = [
    'orbital_period', 'transit_duration', 'planet_radius',
    'stellar_temp', 'flux_mean', 'flux_std', ...
]

# Get explanation
explanation = explain_prediction(
    model, 
    X_sample, 
    feature_names, 
    top_k=3
)

# Display results
print(f"Method: {explanation['method_used']}")
for feat in explanation['top_features']:
    print(f"{feat['name']}: {feat['impact']:+.4f}")
```

**Output:**
```
Method: shap
stellar_temp: +0.1523
planet_radius: -0.0842
orbital_period: +0.0567
```

## Main Functions

### 1. `explain_prediction()`

Explain a single prediction or small batch.

**Signature:**
```python
explain_prediction(
    model: Any,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'auto',
    top_k: int = 3
) -> Dict[str, Any]
```

**Parameters:**
- `model` - Trained sklearn-compatible model
- `X_sample` - Sample to explain (1D or 2D array)
- `feature_names` - Feature names (optional)
- `method` - Explanation method:
  - `'auto'` - Automatically select best method
  - `'shap'` - Use SHAP values (requires shap package)
  - `'permutation'` - Use permutation importance
  - `'importance'` - Use model's feature_importances_
- `top_k` - Number of top features to return

**Returns:**
```python
{
    'top_features': [
        {'name': 'feature_name', 'impact': 0.15},
        {'name': 'another_feature', 'impact': -0.08},
        ...
    ],
    'method_used': 'shap',
    'all_features': {
        'feature_name': 0.15,
        'another_feature': -0.08,
        ...
    }
}
```

### 2. `explain_batch_predictions()`

Explain multiple predictions at once.

```python
explanations = explain_batch_predictions(
    model,
    X_batch,  # 2D array of samples
    feature_names,
    top_k=3
)

# Returns list of explanations, one per sample
for i, exp in enumerate(explanations):
    print(f"Sample {i}:")
    for feat in exp['top_features']:
        print(f"  {feat['name']}: {feat['impact']:+.4f}")
```

### 3. `get_global_feature_importance()`

Get overall feature importance for the model (not sample-specific).

```python
importance = get_global_feature_importance(
    model,
    feature_names,
    top_k=10
)

print("Most important features globally:")
for feat in importance['top_features']:
    print(f"{feat['name']}: {feat['importance']:.4f}")
```

## Explanation Methods

### 1. SHAP (Best, requires shap package)

**Pros:**
- Most accurate and theoretically sound
- Provides signed contributions (positive/negative)
- Works well with tree-based and linear models
- Game-theory based (Shapley values)

**Cons:**
- Requires shap package installation
- Can be slow for complex models
- Requires background data for some models

**When used:**
- Automatically selected if shap is installed
- Best for understanding individual predictions

### 2. Permutation Importance

**Pros:**
- Model-agnostic (works with any model)
- No additional dependencies
- Fast for small samples

**Cons:**
- Less accurate than SHAP for individual samples
- Can be unstable with small datasets
- Requires multiple permutations (slower)

**When used:**
- Fallback when SHAP unavailable
- Good for models without feature_importances_

### 3. Feature Importances

**Pros:**
- Very fast
- Built into tree-based models
- Stable and deterministic

**Cons:**
- Global explanation (not sample-specific)
- Only available for tree-based models
- Can be biased toward high-cardinality features

**When used:**
- Fallback for tree-based models
- Good for global understanding
- Fast for production use

## Interpreting Results

### Impact Values

**Positive impact (+):** Feature pushes prediction toward exoplanet (class 1)
**Negative impact (-):** Feature pushes prediction away from exoplanet (class 0)

**Magnitude:** How much the feature contributes to the prediction

### Example Interpretation

```python
top_features = [
    {'name': 'transit_duration', 'impact': +0.25},
    {'name': 'stellar_temp', 'impact': +0.15},
    {'name': 'flux_std', 'impact': -0.08}
]
```

**Interpretation:**
- **transit_duration (+0.25)**: Long transit duration strongly suggests exoplanet
- **stellar_temp (+0.15)**: Star temperature moderately supports exoplanet
- **flux_std (-0.08)**: High flux variation slightly argues against exoplanet

## Advanced Usage

### Custom Feature Selection

```python
# Only explain specific features
custom_features = ['orbital_period', 'planet_radius', 'stellar_temp']
custom_indices = [0, 2, 3]  # Assuming these are the column indices

X_subset = X_sample[:, custom_indices]

explanation = explain_prediction(
    model,
    X_subset,
    feature_names=custom_features,
    top_k=3
)
```

### Forcing Specific Method

```python
# Force SHAP even if slow
explanation = explain_prediction(
    model, X_sample, feature_names,
    method='shap'
)

# Force fast feature importances
explanation = explain_prediction(
    model, X_sample, feature_names,
    method='importance'
)
```

### Getting All Feature Contributions

```python
explanation = explain_prediction(model, X_sample, feature_names)

# Access all features, not just top k
all_contributions = explanation['all_features']

for feature, contribution in sorted(
    all_contributions.items(),
    key=lambda x: abs(x[1]),
    reverse=True
):
    print(f"{feature}: {contribution:+.4f}")
```

## Integration with Backend API

### In FastAPI prediction endpoint:

```python
from explain import explain_prediction

@app.post("/predict")
async def predict(file: UploadFile):
    # ... load and preprocess data ...
    
    # Make prediction
    prediction = model.predict(X)
    confidence = model.predict_proba(X)[0][1]
    
    # Get explanation
    explanation = explain_prediction(
        model, X, feature_names, top_k=5
    )
    
    return {
        'prediction': int(prediction[0]),
        'confidence': float(confidence),
        'explanation': explanation['top_features']
    }
```

### Frontend Display:

```javascript
// Display explanation in UI
explanation.forEach(feat => {
    const sign = feat.impact > 0 ? '+' : '';
    const bar = createBar(Math.abs(feat.impact));
    console.log(`${feat.name}: ${sign}${feat.impact.toFixed(3)} ${bar}`);
});
```

## Performance Considerations

### Speed Comparison (1000 samples)

| Method | Time | Accuracy |
|--------|------|----------|
| SHAP (Tree) | ~2s | ⭐⭐⭐⭐⭐ |
| SHAP (Kernel) | ~30s | ⭐⭐⭐⭐⭐ |
| Permutation | ~5s | ⭐⭐⭐ |
| Feature Importance | <0.1s | ⭐⭐ |

**Recommendations:**
- **Real-time API**: Use feature_importances_ for speed
- **Batch processing**: Use SHAP for accuracy
- **Large datasets**: Use permutation as compromise

## Troubleshooting

### "SHAP not available"
```bash
pip install shap
```

### "Model does not have feature_importances_"
- Use `method='permutation'` or install SHAP
- Only tree-based models have built-in importances

### Slow SHAP explanations
- Use TreeExplainer for tree models (fast)
- Reduce background data size
- Consider permutation importance instead

### Inconsistent results
- Set random seeds: `np.random.seed(42)`
- Use deterministic models
- Increase permutation repeats

## Examples

### Example 1: Complete Pipeline

```python
from data_loader import load_exoplanet_csv
from preprocess import preprocess_features
from train import train_random_forest
from explain import explain_prediction
import pickle

# 1. Load and train
df = load_exoplanet_csv('data.csv')
X, y = preprocess_features(df)
model, _ = train_random_forest(X, y, X, y)

# 2. Explain a prediction
X_sample = X[0:1]
explanation = explain_prediction(
    model, X_sample,
    feature_names=['orbital_period', 'radius', 'temp'],
    top_k=3
)

print(explanation['top_features'])
```

### Example 2: Compare Methods

```python
# Compare different explanation methods
for method in ['shap', 'permutation', 'importance']:
    try:
        exp = explain_prediction(model, X_sample, names, method=method)
        print(f"\n{method.upper()}:")
        for f in exp['top_features']:
            print(f"  {f['name']}: {f['impact']:+.4f}")
    except Exception as e:
        print(f"{method}: {e}")
```

## Testing

Run tests:
```bash
cd ml
pytest tests/test_explain.py -v
```

Run example:
```bash
cd ml/src
python explain.py
```

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

