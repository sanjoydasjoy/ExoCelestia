# MLflow Experiment Tracking Guide

## Overview

MLflow integration for tracking experiments, parameters, metrics, and models in the exoplanet detection pipeline.

## Installation

```bash
cd ml
pip install mlflow==2.9.2
```

## Quick Start

### 1. Train with MLflow Tracking

```bash
cd ml/src

# Random Forest with MLflow
python train.py ../data/data.csv \
    --model random_forest \
    --use-mlflow \
    --experiment-name my-experiment

# XGBoost with MLflow
python train.py ../data/data.csv \
    --model xgboost \
    --use-mlflow

# Neural Network with MLflow
python train.py ../data/data.csv \
    --model nn \
    --epochs 100 \
    --use-mlflow
```

### 2. View Results in MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Open browser to: http://localhost:5000
```

## What Gets Tracked

### Parameters

**Common (all models)**:
- model_type
- test_size
- random_state
- n_features
- n_train_samples
- n_test_samples
- flux_features_extracted

**Random Forest**:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- class_weight

**XGBoost**:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- scale_pos_weight

**Neural Network**:
- epochs
- batch_size
- learning_rate
- hidden_layers
- optimizer
- loss_function

### Metrics

- accuracy
- precision
- recall
- f1
- roc_auc (when available)
- train_loss (per epoch for NN)

### Artifacts

- Trained model (sklearn/pytorch format)
- Model metadata
- Output path

## Usage Examples

### Basic Training

```bash
python train.py data.csv --model random_forest --use-mlflow
```

### Custom Experiment Name

```bash
python train.py data.csv \
    --model xgboost \
    --use-mlflow \
    --experiment-name kepler-xgboost-experiments
```

### Multiple Experiments

Using `run_mlflow.py`:

```bash
# Run all models
python run_mlflow.py \
    --dataset ../data/data.csv \
    --models random_forest xgboost nn

# Run specific models
python run_mlflow.py \
    --dataset ../data/data.csv \
    --models random_forest xgboost \
    --experiment-name model-comparison
```

## MLflow UI Features

### 1. Experiment List

View all experiments and their runs:
- Experiment name
- Number of runs
- Creation time

### 2. Run Comparison

Compare multiple runs side-by-side:
- Parameters (hyperparameters)
- Metrics (accuracy, F1, etc.)
- Artifacts (models)
- Tags and notes

### 3. Metric Visualization

Plot metrics over time:
- Train loss curves (for NN)
- Metric comparisons across runs
- Scatter plots
- Parallel coordinates

### 4. Model Registry

Register and version models:
- Production models
- Staging models
- Archived models
- Model lineage

## Directory Structure

```
ml/
├── mlruns/                    # MLflow tracking directory
│   ├── 0/                     # Default experiment
│   │   ├── meta.yaml
│   │   └── {run_id}/          # Individual runs
│   │       ├── artifacts/     # Model files
│   │       ├── metrics/       # Metric files
│   │       ├── params/        # Parameter files
│   │       └── tags/          # Tags
│   └── {experiment_id}/       # Named experiments
```

## Advanced Features

### Search Runs

```python
import mlflow

# Search by metrics
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

print(runs[['run_id', 'metrics.accuracy', 'params.model_type']])
```

### Load Best Model

```python
import mlflow

# Find best run
best_run = mlflow.search_runs(
    filter_string="metrics.f1 > 0",
    order_by=["metrics.f1 DESC"],
    max_results=1
)

run_id = best_run.iloc[0]['run_id']

# Load model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Make predictions
predictions = model.predict(X_test)
```

### Custom Logging

Add to your training code:

```python
import mlflow

with mlflow.start_run():
    # Log custom parameters
    mlflow.log_param("dataset_name", "kepler_20241007")
    mlflow.log_param("preprocessing", "standard_scaler")
    
    # Log custom metrics
    mlflow.log_metric("training_time", training_time)
    mlflow.log_metric("inference_time", inference_time)
    
    # Log files
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.csv")
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## Workflow Integration

### 1. Hyperparameter Tuning

```bash
# Test different hyperparameters
for depth in 10 15 20 25; do
    python train.py data.csv \
        --model random_forest \
        --use-mlflow \
        --experiment-name rf-depth-tuning
done

# View in MLflow UI to find best
mlflow ui --backend-store-uri ./mlruns
```

### 2. Cross-Validation

```bash
# Test different random seeds
for seed in 42 123 456 789; do
    python train.py data.csv \
        --model xgboost \
        --random-state $seed \
        --use-mlflow \
        --experiment-name xgb-stability-test
done
```

### 3. Dataset Comparison

```bash
# Compare different datasets
python train.py kepler_data.csv --model rf --use-mlflow --experiment-name kepler
python train.py tess_data.csv --model rf --use-mlflow --experiment-name tess
python train.py k2_data.csv --model rf --use-mlflow --experiment-name k2
```

## Best Practices

### 1. Naming Conventions

```bash
# Good experiment names
--experiment-name kepler-rf-baseline
--experiment-name tess-xgboost-tuning
--experiment-name combined-nn-production

# Include version/date for tracking
--experiment-name kepler-v2-20241007
```

### 2. Tagging Runs

```python
with mlflow.start_run():
    mlflow.set_tag("version", "2.0")
    mlflow.set_tag("dataset", "kepler_confirmed")
    mlflow.set_tag("purpose", "production")
```

### 3. Documenting Experiments

```python
with mlflow.start_run() as run:
    mlflow.set_tag("mlflow.note.content", """
    Experiment testing impact of flux features on accuracy.
    Dataset: Kepler confirmed planets (2000 samples)
    Goal: Achieve >0.95 F1 score
    """)
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Train and track with MLflow
  run: |
    cd ml/src
    python train.py data.csv --model rf --use-mlflow
    
- name: Upload MLflow artifacts
  uses: actions/upload-artifact@v3
  with:
    name: mlflow-runs
    path: ml/mlruns/
```

## Troubleshooting

### MLflow Not Found

```bash
pip install mlflow
```

### Port Already in Use

```bash
# Use different port
mlflow ui --backend-store-uri ./mlruns --port 5001
```

### Can't Find Runs

```bash
# Check tracking directory
ls -la mlruns/

# Verify experiment
mlflow experiments list
```

### Model Loading Fails

```python
# Ensure correct model flavor
# For sklearn models:
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# For pytorch models:
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
```

## MLflow UI Overview

```
┌─────────────────────────────────────────────────────────┐
│  Experiments  │  Runs  │  Models  │  Compare            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  experiment-name          Runs  Created                 │
│  ─────────────────────    ────  ────────                │
│  exoplanet-detection       15   2024-10-07              │
│  kepler-baseline            8   2024-10-05              │
│  model-comparison          12   2024-10-06              │
│                                                          │
│  [Click experiment to view runs]                        │
│                                                          │
│  Run Details:                                           │
│  ┌──────────────────────────────────────────────┐      │
│  │ Parameters    │ Metrics      │ Artifacts      │      │
│  ├──────────────────────────────────────────────┤      │
│  │ model_type: rf │ accuracy: 0.94 │ model/     │      │
│  │ n_estimators:  │ precision: 0.91│            │      │
│  │   200          │ recall: 0.89   │            │      │
│  │ max_depth: 20  │ f1: 0.90       │            │      │
│  └──────────────────────────────────────────────┘      │
│                                                          │
│  [Compare] [Download] [Register Model]                 │
└─────────────────────────────────────────────────────────┘
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)

