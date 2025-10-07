# Model Training Guide

## Quick Start

```bash
cd ml/src
python train.py ../data/exoplanets.csv --model random_forest --output ../models/rf_model.pkl
```

## Usage

```bash
python train.py INPUT_CSV [OPTIONS]
```

### Required Arguments
- `INPUT_CSV` - Path to CSV file with exoplanet data

### Optional Arguments
- `--output PATH` - Output model path (default: `../models/model.pkl`)
- `--model {random_forest,xgboost,nn}` - Model type (default: `random_forest`)
- `--test-size FLOAT` - Test set fraction (default: 0.2)
- `--random-state INT` - Random seed (default: 42)
- `--epochs INT` - Epochs for NN (default: 50)
- `--batch-size INT` - Batch size for NN (default: 64)
- `--learning-rate FLOAT` - Learning rate for NN (default: 0.001)
- `--label-column STR` - Label column name (default: 'label')
- `--no-flux-features` - Disable flux feature extraction

## Examples

### Random Forest
```bash
python train.py data.csv --model random_forest --output models/rf.pkl
```

### XGBoost
```bash
python train.py data.csv --model xgboost --output models/xgb.pkl --test-size 0.3
```

### Neural Network
```bash
python train.py data.csv --model nn --output models/nn.pt --epochs 100 --batch-size 32
```

## Output Files

Training produces three files in the model directory:

1. **Model file** (`model.pkl` or `model.pt`)
2. **`metrics.json`** - Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
3. **`model_metadata.json`** - Training configuration and metadata

## Model Types

### Random Forest
- **File**: `.pkl`
- **Dependencies**: scikit-learn (included)
- **Best for**: Baseline, interpretability
- **Speed**: Fast training, fast inference

### XGBoost
- **File**: `.pkl`
- **Dependencies**: xgboost (included in requirements.txt)
- **Best for**: Best performance on tabular data
- **Speed**: Medium training, fast inference

### Neural Network (nn)
- **File**: `.pt` (PyTorch state dict)
- **Dependencies**: PyTorch (optional, uncomment in requirements.txt)
- **Best for**: Complex patterns, large datasets
- **Speed**: Slow training, fast inference
- **Architecture**: 3 hidden layers [128, 64, 32] with BatchNorm and Dropout

## Metrics Explained

- **Accuracy**: Overall correctness (true predictions / total)
- **Precision**: Of predicted exoplanets, how many are real (TP / (TP + FP))
- **Recall**: Of real exoplanets, how many we found (TP / (TP + FN))
- **F1**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

## Tips

1. **Start with Random Forest** - Fast baseline to verify data pipeline
2. **Use XGBoost for production** - Best accuracy/speed tradeoff
3. **Try Neural Network for large datasets** (>100k samples)
4. **Use `--random-state 42`** for reproducibility
5. **Adjust `--test-size`** based on dataset size (smaller for <1000 samples)

## Troubleshooting

**Import Error: XGBoost**
```bash
pip install xgboost==2.0.3
```

**Import Error: PyTorch**
```bash
# CPU version
pip install torch==2.1.0
# OR GPU version (see pytorch.org)
```

**ValueError: Label column not found**
- Use `--label-column YOUR_COLUMN_NAME`

**Class imbalance warning**
- Random Forest and XGBoost handle this automatically (`class_weight='balanced'`, `scale_pos_weight`)

