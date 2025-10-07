# MLflow Implementation Summary

## Overview

Integrated MLflow experiment tracking into the training pipeline for comprehensive tracking of parameters, metrics, and model artifacts.

## ✅ Files Created/Modified

### Modified
1. **`ml/src/train.py`** - Added MLflow integration
   - Import mlflow and flavors (sklearn, pytorch, xgboost)
   - Added `use_mlflow` parameter to all training functions
   - Track parameters in `train_random_forest()`, `train_xgboost()`, `train_neural_network()`
   - Track metrics (accuracy, precision, recall, F1, ROC-AUC)
   - Track epoch-by-epoch loss for neural networks
   - Log model artifacts
   - Added CLI arguments: `--use-mlflow`, `--experiment-name`

### Created
2. **`ml/src/run_mlflow.py`** - Standalone MLflow experiment runner
3. **`ml/MLFLOW_GUIDE.md`** - Comprehensive usage guide
4. **`ml/MLFLOW_QUICKSTART.md`** - Quick reference
5. **`ml/MLFLOW_IMPLEMENTATION.md`** - This file

### Updated
6. **`ml/requirements.txt`** - Documented mlflow dependency (commented)
7. **`.gitignore`** - Added mlruns/ directories

## MLflow Integration in train.py

### Imports Added

```python
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
```

### Tracking URI Configuration

```python
# MLflow tracking URI (default: ./mlruns directory)
# To view results: mlflow ui --backend-store-uri ./mlruns
# Then open http://localhost:5000 in your browser
MLFLOW_TRACKING_URI = "./mlruns"
```

**Instruction comment clearly visible at line 67-69**

### Parameters Tracked

#### Common (all models)
- `model_type` - random_forest, xgboost, or nn
- `test_size` - Train/test split ratio
- `random_state` - Random seed for reproducibility
- `n_features` - Number of input features
- `n_train_samples` - Training set size
- `n_test_samples` - Test set size
- `flux_features_extracted` - Whether flux features were used
- `output_path` - Where model was saved

#### Random Forest Specific
- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `n_jobs`: -1
- `class_weight`: 'balanced'

#### XGBoost Specific
- `n_estimators`: 200
- `max_depth`: 7
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: auto-calculated
- `eval_metric`: 'logloss'

#### Neural Network Specific
- `epochs`: configurable
- `batch_size`: configurable
- `learning_rate`: configurable
- `hidden_layers`: [128, 64, 32]
- `optimizer`: 'Adam'
- `loss_function`: 'CrossEntropyLoss'

### Metrics Tracked

All models:
- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc` (when applicable)

Neural networks additionally track:
- `train_loss` (per epoch with step parameter)

### Artifacts Logged

- Trained model (sklearn/pytorch format)
- Automatically logged by:
  - `mlflow.sklearn.log_model(model, "model")` for RF/XGBoost
  - `mlflow.pytorch.log_model(model, "model")` for NN

## Usage Examples

### Basic Training with MLflow

```bash
cd ml/src

python train.py ../data/data.csv \
    --model random_forest \
    --use-mlflow
```

### With Custom Experiment Name

```bash
python train.py ../data/data.csv \
    --model xgboost \
    --use-mlflow \
    --experiment-name production-models
```

### Neural Network with MLflow

```bash
python train.py ../data/data.csv \
    --model nn \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --use-mlflow \
    --experiment-name nn-hyperparameter-tuning
```

### View Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Open browser to http://localhost:5000
```

## run_mlflow.py Features

### Run Multiple Experiments

```bash
python run_mlflow.py \
    --dataset ../data/data.csv \
    --models random_forest xgboost nn
```

### Start MLflow UI

```bash
python run_mlflow.py --ui
```

### Custom Experiment

```bash
python run_mlflow.py \
    --dataset ../data/data.csv \
    --models random_forest \
    --experiment-name kepler-baseline \
    --test-size 0.3
```

## Code Flow

### Without MLflow

```python
model, metrics = train_random_forest(X_train, y_train, X_test, y_test)
```

### With MLflow

```python
if use_mlflow:
    with mlflow.start_run(run_name=f"{model}_{timestamp}"):
        mlflow.log_param('model_type', 'random_forest')
        mlflow.log_param('test_size', 0.2)
        # ... log all parameters
        
        model, metrics = train_random_forest(
            X_train, y_train, X_test, y_test,
            use_mlflow=True  # ← Enable tracking
        )
        
        mlflow.sklearn.log_model(model, "model")
```

### Inside Training Functions

```python
def train_random_forest(..., use_mlflow=False):
    params = {
        'n_estimators': 200,
        'max_depth': 20,
        ...
    }
    
    # Log parameters
    if use_mlflow and MLFLOW_AVAILABLE:
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    # Log metrics
    if use_mlflow and MLFLOW_AVAILABLE:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    return model, metrics
```

## MLflow UI Views

After training, MLflow UI shows:

### Experiments Page
```
┌─────────────────────────────────────────┐
│ Experiments                              │
├─────────────────────────────────────────┤
│ exoplanet-detection          5 runs     │
│ production-models            3 runs     │
│ hyperparameter-tuning       12 runs     │
└─────────────────────────────────────────┘
```

### Runs Table
```
┌──────────────┬───────────┬──────────┬──────────┬─────┐
│ Run Name     │ Model     │ Accuracy │ F1       │ ... │
├──────────────┼───────────┼──────────┼──────────┼─────┤
│ rf_20241007  │ rf        │ 0.9456   │ 0.9123   │     │
│ xgb_20241007 │ xgboost   │ 0.9512   │ 0.9234   │ ✓   │
│ nn_20241007  │ nn        │ 0.9389   │ 0.9045   │     │
└──────────────┴───────────┴──────────┴──────────┴─────┘
```

### Run Details
```
Parameters          Metrics            Artifacts
─────────────       ──────────         ──────────
model_type: rf      accuracy: 0.9456   model/
n_estimators: 200   precision: 0.9234     ├─ MLmodel
max_depth: 20       recall: 0.9012        ├─ conda.yaml
test_size: 0.2      f1: 0.9123            └─ model.pkl
random_state: 42    roc_auc: 0.9678
```

## Installation Note

MLflow is **optional** and commented in requirements.txt:

```txt
# Optional: for experiment tracking (recommended)
# mlflow==2.9.2  # For MLflow tracking
```

To enable:
```bash
pip install mlflow==2.9.2
```

Training works without MLflow if not installed (graceful fallback).

## Graceful Degradation

```python
if args.use_mlflow and not MLFLOW_AVAILABLE:
    logger.warning("MLflow requested but not available. Install with: pip install mlflow")
    use_mlflow = False
```

Script continues without MLflow if:
- `--use-mlflow` not specified
- mlflow not installed

## Directory Structure

```
ml/
├── src/
│   ├── train.py              ← MLflow integration
│   ├── run_mlflow.py         ← Experiment runner
│   └── mlruns/               ← Created on first run
│       └── 0/                ← Default experiment
│           └── {run_id}/
│               ├── artifacts/model/
│               ├── metrics/
│               ├── params/
│               └── tags/
└── MLFLOW_*.md              ← Documentation
```

## Key Features

✅ **Optional dependency** - Works without mlflow  
✅ **Graceful fallback** - Clear warnings if unavailable  
✅ **Comprehensive tracking** - All params/metrics logged  
✅ **Model artifacts** - Full model saved  
✅ **CLI integration** - Simple `--use-mlflow` flag  
✅ **Experiment organization** - Named experiments  
✅ **Run naming** - Timestamped run names  
✅ **UI instructions** - Comment at top of file (line 67-69)  
✅ **Helper script** - run_mlflow.py for batch experiments  
✅ **Documentation** - 3 guide files  

## Verification

```bash
# 1. Compile check
python -m py_compile ml/src/train.py ml/src/run_mlflow.py
# ✅ Both compile successfully

# 2. Test training (dry run without data)
python ml/src/train.py --help | grep mlflow
#   --use-mlflow          Enable MLflow experiment tracking
#   --experiment-name     MLflow experiment name

# 3. View instruction comment
head -n 70 ml/src/train.py | tail -n 5
# MLflow tracking URI (default: ./mlruns directory)
# To view results: mlflow ui --backend-store-uri ./mlruns
# Then open http://localhost:5000 in your browser
```

## Commit Message

```
feat(ml): integrate MLflow tracking for experiments with params/metrics/artifacts

• train.py modifications
  - Added MLflow imports (sklearn, pytorch, xgboost flavors)
  - Modified all training functions to accept use_mlflow parameter
  - Track all hyperparameters (RF: n_estimators/max_depth, XGB: learning_rate, NN: epochs/batch_size)
  - Track all metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Track epoch-by-epoch train_loss for neural networks
  - Log model artifacts via mlflow.sklearn/pytorch.log_model()
  - Added CLI args: --use-mlflow, --experiment-name
  - Instruction comment at line 67-69: "mlflow ui --backend-store-uri ./mlruns"

• run_mlflow.py
  - Standalone script to run multiple experiments
  - Start MLflow UI: python run_mlflow.py --ui
  - Batch experiments: --models rf xgb nn

• Documentation
  - MLFLOW_GUIDE.md: comprehensive usage
  - MLFLOW_QUICKSTART.md: quick reference
  - MLFLOW_IMPLEMENTATION.md: technical details

• Configuration
  - Updated .gitignore (mlruns/ directories)
  - Documented mlflow==2.9.2 in requirements.txt (optional)

Graceful fallback if mlflow not installed. View UI: mlflow ui --backend-store-uri ./mlruns
```

