# MLflow Quick Start

## Installation

```bash
pip install mlflow==2.9.2
```

## 1. Train with MLflow Tracking

```bash
cd ml/src

# Train with MLflow enabled
python train.py ../data/data.csv \
    --model random_forest \
    --use-mlflow

# Train with custom experiment name
python train.py ../data/data.csv \
    --model xgboost \
    --use-mlflow \
    --experiment-name my-experiment
```

## 2. View Results in MLflow UI

```bash
# Start MLflow UI (from ml/src directory)
mlflow ui --backend-store-uri ./mlruns

# OR use the helper script
python run_mlflow.py --ui
```

Then open: **http://localhost:5000**

## What You'll See

- **Experiments**: All your training runs organized by experiment name
- **Parameters**: Hyperparameters used (n_estimators, learning_rate, etc.)
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC
- **Artifacts**: Saved models
- **Comparison**: Side-by-side comparison of different runs

## Run Multiple Experiments

```bash
# Compare all models
python run_mlflow.py \
    --dataset ../data/data.csv \
    --models random_forest xgboost nn
```

This will:
1. Train all three models
2. Log all parameters and metrics
3. Save results to MLflow
4. Print summary

Then view in MLflow UI to compare!

## Common Commands

```bash
# Train single model with MLflow
python train.py data.csv --model rf --use-mlflow

# Train with specific experiment name  
python train.py data.csv --model xgb --use-mlflow --experiment-name production

# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# List experiments
mlflow experiments list --tracking-uri ./mlruns

# Search runs
mlflow runs list --experiment-name exoplanet-detection
```

## Directory Structure After Training

```
ml/
├── src/
│   ├── train.py
│   ├── run_mlflow.py
│   └── mlruns/              ← MLflow tracking directory
│       ├── 0/               ← Default experiment
│       │   ├── meta.yaml
│       │   └── {run_id}/
│       │       ├── artifacts/   ← Model files
│       │       ├── metrics/     ← Metrics (accuracy, F1, etc.)
│       │       ├── params/      ← Parameters
│       │       └── tags/
│       └── {experiment_id}/     ← Named experiments
```

## Troubleshooting

**MLflow not found:**
```bash
pip install mlflow
```

**Port 5000 in use:**
```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

**Can't find runs:**
```bash
# Check if mlruns directory exists
ls -la mlruns/

# Verify you're in the correct directory (ml/src/)
pwd
```

## Next Steps

See `MLFLOW_GUIDE.md` for:
- Advanced features
- Model registry
- Custom logging
- CI/CD integration

