"""
Model evaluation script for exoplanet detection
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_and_prepare_data
from preprocess import normalize_features


def load_model_and_scaler(model_path: str):
    """Load trained model and associated scaler"""
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load config
    config_path = model_path.replace('.pkl', '_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return model, scaler, config


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_pred_proba is not None:
        # For binary classification, use the positive class probability
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    # For binary classification, use the positive class probability
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(
    model_path: str,
    test_data_path: str = None,
    X_test=None,
    y_test=None,
    output_dir: str = "evaluation_results"
):
    """
    Evaluate trained model
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data CSV (optional)
        X_test: Test features (optional, if not using test_data_path)
        y_test: Test labels (optional, if not using test_data_path)
        output_dir: Directory to save evaluation results
    """
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Load model
    print("\n1. Loading model and scaler...")
    model, scaler, config = load_model_and_scaler(model_path)
    print(f"Model trained at: {config.get('trained_at', 'Unknown')}")
    
    # Load test data if path provided
    if test_data_path:
        print("\n2. Loading test data...")
        # TODO: Adjust based on actual data format
        X_test, y_test = load_and_prepare_data(test_data_path, mission='kepler')
        X_test, _ = normalize_features(X_test, scaler=scaler)
    elif X_test is None or y_test is None:
        print("ERROR: No test data provided")
        return
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    print("\n4. Calculating metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    print("\nEvaluation Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=['No Exoplanet', 'Exoplanet']))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    print("\n5. Saving results...")
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    # Plot confusion matrix
    cm_path = output_path / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    
    # Plot ROC curve if probabilities available
    if y_pred_proba is not None:
        roc_path = output_path / "roc_curve.png"
        plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to {output_dir}/")
    print("=" * 50)
    
    return metrics


if __name__ == "__main__":
    # TODO: Update with actual model and test data paths
    MODEL_PATH = "models/model.pkl"
    TEST_DATA_PATH = "data/test_data.csv"
    
    # For testing without data
    print("WARNING: Using dummy test data")
    X_test = np.random.randn(200, 100)
    y_test = np.random.randint(0, 2, 200)
    
    # Load model for dummy evaluation
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print("Metrics:", metrics)

