"""
Unit tests for model explanation functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from explain import (
    explain_prediction,
    explain_batch_predictions,
    get_global_feature_importance,
    SHAP_AVAILABLE
)


@pytest.fixture
def trained_model():
    """Create a trained Random Forest model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice([0, 1], 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model


@pytest.fixture
def sample_data():
    """Create sample data for explanations."""
    np.random.seed(42)
    X_sample = np.random.randn(1, 5)
    feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    
    return X_sample, feature_names


class TestExplainPrediction:
    """Tests for explain_prediction function."""
    
    def test_basic_explanation(self, trained_model, sample_data):
        """Test basic explanation generation."""
        X_sample, feature_names = sample_data
        
        explanation = explain_prediction(
            trained_model, X_sample, feature_names, top_k=3
        )
        
        assert 'top_features' in explanation
        assert 'method_used' in explanation
        assert len(explanation['top_features']) == 3
    
    def test_top_features_format(self, trained_model, sample_data):
        """Test that top features have correct format."""
        X_sample, feature_names = sample_data
        
        explanation = explain_prediction(
            trained_model, X_sample, feature_names, top_k=3
        )
        
        for feat in explanation['top_features']:
            assert 'name' in feat
            assert 'impact' in feat
            assert isinstance(feat['name'], str)
            assert isinstance(feat['impact'], float)
    
    def test_without_feature_names(self, trained_model, sample_data):
        """Test explanation without providing feature names."""
        X_sample, _ = sample_data
        
        explanation = explain_prediction(
            trained_model, X_sample, feature_names=None, top_k=3
        )
        
        # Should generate default feature names
        assert len(explanation['top_features']) == 3
        assert 'feature_' in explanation['top_features'][0]['name']
    
    def test_1d_sample_input(self, trained_model, sample_data):
        """Test that 1D input is properly reshaped."""
        X_sample, feature_names = sample_data
        X_1d = X_sample.flatten()
        
        explanation = explain_prediction(
            trained_model, X_1d, feature_names, top_k=3
        )
        
        assert len(explanation['top_features']) == 3
    
    def test_different_top_k_values(self, trained_model, sample_data):
        """Test different top_k values."""
        X_sample, feature_names = sample_data
        
        for k in [1, 3, 5]:
            explanation = explain_prediction(
                trained_model, X_sample, feature_names, top_k=k
            )
            assert len(explanation['top_features']) == k
    
    def test_feature_importance_method(self, trained_model, sample_data):
        """Test explicit feature importance method."""
        X_sample, feature_names = sample_data
        
        explanation = explain_prediction(
            trained_model, X_sample, feature_names, 
            method='importance', top_k=3
        )
        
        assert explanation['method_used'] == 'feature_importances'
        assert 'all_features' in explanation
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
    def test_shap_method(self, trained_model, sample_data):
        """Test SHAP explanation method."""
        X_sample, feature_names = sample_data
        
        explanation = explain_prediction(
            trained_model, X_sample, feature_names,
            method='shap', top_k=3
        )
        
        assert explanation['method_used'] == 'shap'
        assert len(explanation['top_features']) == 3


class TestExplainBatchPredictions:
    """Tests for explain_batch_predictions function."""
    
    def test_batch_explanation(self, trained_model):
        """Test explanation for batch of samples."""
        np.random.seed(42)
        X_batch = np.random.randn(5, 5)
        feature_names = [f'feature_{i}' for i in range(5)]
        
        explanations = explain_batch_predictions(
            trained_model, X_batch, feature_names, top_k=3
        )
        
        assert len(explanations) == 5
        for exp in explanations:
            assert 'top_features' in exp
            assert len(exp['top_features']) == 3
    
    def test_single_sample_batch(self, trained_model, sample_data):
        """Test batch explanation with single sample."""
        X_sample, feature_names = sample_data
        
        explanations = explain_batch_predictions(
            trained_model, X_sample, feature_names, top_k=2
        )
        
        assert len(explanations) == 1
        assert len(explanations[0]['top_features']) == 2


class TestGlobalFeatureImportance:
    """Tests for get_global_feature_importance function."""
    
    def test_global_importance(self, trained_model):
        """Test global feature importance extraction."""
        feature_names = [f'feature_{i}' for i in range(5)]
        
        result = get_global_feature_importance(
            trained_model, feature_names, top_k=5
        )
        
        assert 'top_features' in result
        assert 'method_used' in result
        assert result['method_used'] == 'global_feature_importances'
        assert len(result['top_features']) == 5
    
    def test_global_importance_without_names(self, trained_model):
        """Test global importance without feature names."""
        result = get_global_feature_importance(
            trained_model, feature_names=None, top_k=3
        )
        
        assert len(result['top_features']) == 3
    
    def test_top_k_limit(self, trained_model):
        """Test that top_k limits results correctly."""
        feature_names = [f'feature_{i}' for i in range(5)]
        
        result = get_global_feature_importance(
            trained_model, feature_names, top_k=3
        )
        
        assert len(result['top_features']) == 3
    
    def test_importance_values(self, trained_model):
        """Test that importance values are valid."""
        feature_names = [f'feature_{i}' for i in range(5)]
        
        result = get_global_feature_importance(
            trained_model, feature_names, top_k=5
        )
        
        for feat in result['top_features']:
            assert 'name' in feat
            assert 'importance' in feat
            assert feat['importance'] >= 0
            assert feat['importance'] <= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_model_without_feature_importances(self):
        """Test with model that doesn't have feature_importances_."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        model = LogisticRegression()
        model.fit(X, y)
        
        X_sample = np.random.randn(1, 5)
        
        # Should still work with fallback methods
        explanation = explain_prediction(model, X_sample, top_k=3)
        
        assert 'top_features' in explanation
        assert explanation['method_used'] != 'feature_importances'
    
    def test_mismatched_feature_names(self, trained_model, sample_data):
        """Test handling of mismatched feature names."""
        X_sample, _ = sample_data
        wrong_names = ['a', 'b']  # Wrong number of names
        
        explanation = explain_prediction(
            trained_model, X_sample, wrong_names, top_k=3
        )
        
        # Should fall back to default names
        assert len(explanation['top_features']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

