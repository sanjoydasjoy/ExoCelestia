"""
Unit tests for training functions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train import (
    train_random_forest,
    compute_metrics,
    XGBOOST_AVAILABLE,
    PYTORCH_AVAILABLE
)

if XGBOOST_AVAILABLE:
    from train import train_xgboost

if PYTORCH_AVAILABLE:
    from train import train_neural_network


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    X_test = np.random.randn(50, n_features)
    y_test = np.random.choice([0, 1], 50, p=[0.7, 0.3])
    
    return X_train, y_train, X_test, y_test


class TestComputeMetrics:
    """Tests for compute_metrics function."""
    
    def test_basic_metrics(self):
        """Test basic metric computation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_with_probabilities(self):
        """Test metrics with probability scores."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_perfect_predictions(self):
        """Test perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0


class TestRandomForest:
    """Tests for Random Forest training."""
    
    def test_train_random_forest(self, sample_data):
        """Test Random Forest training."""
        X_train, y_train, X_test, y_test = sample_data
        
        model, metrics = train_random_forest(
            X_train, y_train, X_test, y_test, random_state=42
        )
        
        # Check model exists
        assert model is not None
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check model can predict
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_deterministic_training(self, sample_data):
        """Test that training is deterministic with same random_state."""
        X_train, y_train, X_test, y_test = sample_data
        
        model1, metrics1 = train_random_forest(
            X_train, y_train, X_test, y_test, random_state=42
        )
        
        model2, metrics2 = train_random_forest(
            X_train, y_train, X_test, y_test, random_state=42
        )
        
        # Predictions should be identical
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestXGBoost:
    """Tests for XGBoost training."""
    
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_train_xgboost(self, sample_data):
        """Test XGBoost training."""
        X_train, y_train, X_test, y_test = sample_data
        
        model, metrics = train_xgboost(
            X_train, y_train, X_test, y_test, random_state=42
        )
        
        # Check model exists
        assert model is not None
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        
        # Check model can predict
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestNeuralNetwork:
    """Tests for Neural Network training."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_nn(self, sample_data):
        """Test neural network training."""
        X_train, y_train, X_test, y_test = sample_data
        
        model, metrics = train_neural_network(
            X_train, y_train, X_test, y_test,
            random_state=42,
            epochs=5,  # Quick test
            batch_size=32
        )
        
        # Check model exists
        assert model is not None
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'f1' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

