"""
Model training script for exoplanet detection

Usage:
    python train.py data/exoplanets.csv --model random_forest --output models/rf_model.pkl
    python train.py data/exoplanets.csv --model xgboost --output models/xgb_model.pkl
    python train.py data/exoplanets.csv --model nn --output models/nn_model.pt
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, Any

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Import our custom functions
from data_loader import load_exoplanet_csv
from preprocess import preprocess_features

# Optional: XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Optional: PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Neural Network Model (PyTorch)
class ExoplanetNN(nn.Module):
    """
    Simple feedforward neural network for tabular exoplanet data.
    """
    
    def __init__(self, input_size: int, hidden_sizes: list = [128, 64, 32]):
        """
        Initialize neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
        """
        super(ExoplanetNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       random_state: int = 42) -> Tuple[Any, Dict[str, float]]:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger = logging.getLogger(__name__)
    logger.info("Training Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    return model, metrics


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 random_state: int = 42) -> Tuple[Any, Dict[str, float]]:
    """
    Train XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")
    
    logger = logging.getLogger(__name__)
    logger.info("Training XGBoost Classifier...")
    
    # Calculate scale_pos_weight for imbalanced data
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    return model, metrics


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        random_state: int = 42,
                        epochs: int = 50,
                        batch_size: int = 64,
                        learning_rate: float = 0.001) -> Tuple[Any, Dict[str, float]]:
    """
    Train PyTorch neural network.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed for reproducibility
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")
    
    logger = logging.getLogger(__name__)
    logger.info("Training Neural Network...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = ExoplanetNN(input_size=input_size, hidden_sizes=[128, 64, 32])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        y_pred_proba = probabilities[:, 1].numpy()
        y_pred = outputs.argmax(dim=1).numpy()
    
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    return model, metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Print metrics to console."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 60)


def save_model(model: Any, model_path: str, model_type: str):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save model
        model_type: Type of model ('random_forest', 'xgboost', 'nn')
    """
    logger = logging.getLogger(__name__)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'nn':
        # Save PyTorch model
        torch.save(model.state_dict(), model_path)
        logger.info(f"Neural network saved to {model_path}")
    else:
        # Save sklearn/xgboost model with pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"{model_type} model saved to {model_path}")


def save_metrics(metrics: Dict[str, float], metrics_path: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        metrics_path: Path to save metrics JSON
    """
    logger = logging.getLogger(__name__)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    metrics_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        **metrics
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_timestamp, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")


def main():
    """Main training function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train exoplanet detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py data.csv --model random_forest --output models/rf.pkl
  python train.py data.csv --model xgboost --output models/xgb.pkl --test-size 0.3
  python train.py data.csv --model nn --output models/nn.pt --epochs 100
        """
    )
    
    parser.add_argument(
        'input_csv',
        type=str,
        help='Path to input CSV file with exoplanet data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../models/model.pkl',
        help='Path to save trained model (default: ../models/model.pkl)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'xgboost', 'nn'],
        default='random_forest',
        help='Model type to train (default: random_forest)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size as fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs for neural network (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for neural network (default: 64)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for neural network (default: 0.001)'
    )
    
    parser.add_argument(
        '--label-column',
        type=str,
        default='label',
        help='Name of label column in CSV (default: label)'
    )
    
    parser.add_argument(
        '--no-flux-features',
        action='store_true',
        help='Disable flux feature extraction (faster but less accurate)'
    )
    
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 80)
    print("EXOPLANET DETECTION MODEL TRAINING")
    print("=" * 80)
    print(f"Model type: {args.model}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output path: {args.output}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print("=" * 80 + "\n")
    
    # Step 1: Load data
    logger.info("Step 1: Loading CSV data...")
    df = load_exoplanet_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} samples")
    
    # Step 2: Preprocess features
    logger.info("Step 2: Preprocessing features...")
    extract_flux = not args.no_flux_features
    X, y = preprocess_features(
        df,
        label_column=args.label_column,
        extract_flux_features=extract_flux
    )
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Step 3: Train-test split
    logger.info("Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Step 4: Train model
    logger.info(f"Step 4: Training {args.model} model...")
    
    if args.model == 'random_forest':
        model, metrics = train_random_forest(
            X_train, y_train, X_test, y_test, args.random_state
        )
    elif args.model == 'xgboost':
        model, metrics = train_xgboost(
            X_train, y_train, X_test, y_test, args.random_state
        )
    elif args.model == 'nn':
        model, metrics = train_neural_network(
            X_train, y_train, X_test, y_test,
            random_state=args.random_state,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    # Step 5: Print metrics
    print_metrics(metrics)
    
    # Step 6: Save model
    logger.info("Step 5: Saving model...")
    save_model(model, args.output, args.model)
    
    # Step 7: Save metrics
    metrics_path = str(Path(args.output).parent / 'metrics.json')
    save_metrics(metrics, metrics_path)
    
    # Save model metadata
    metadata_path = str(Path(args.output).parent / 'model_metadata.json')
    metadata = {
        'model_type': args.model,
        'input_csv': args.input_csv,
        'output_path': args.output,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'n_features': X.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'flux_features_extracted': extract_flux,
        'trained_at': datetime.now().isoformat()
    }
    
    if args.model == 'nn':
        metadata['epochs'] = args.epochs
        metadata['batch_size'] = args.batch_size
        metadata['learning_rate'] = args.learning_rate
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {args.output}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

