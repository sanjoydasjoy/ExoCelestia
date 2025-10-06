"""
Unit tests for Pydantic models
"""
import pytest
from pydantic import ValidationError

from app.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionClass,
    FeatureImportance,
    PredictionExplanation,
    BatchPredictionItem,
    BatchPredictionResponse
)


class TestPredictionRequest:
    """Test PredictionRequest model"""
    
    def test_valid_request(self):
        """Test valid prediction request"""
        request = PredictionRequest(
            features={
                "orbital_period": 3.5,
                "transit_depth": 0.02
            }
        )
        assert len(request.features) == 2
        assert request.features["orbital_period"] == 3.5
    
    def test_empty_features(self):
        """Test request with empty features"""
        with pytest.raises(ValidationError):
            PredictionRequest(features={})
    
    def test_non_numeric_features(self):
        """Test request with non-numeric features"""
        with pytest.raises(ValidationError):
            PredictionRequest(
                features={
                    "orbital_period": "not a number"
                }
            )


class TestPredictionResponse:
    """Test PredictionResponse model"""
    
    def test_valid_response(self):
        """Test valid prediction response"""
        response = PredictionResponse(
            prediction=PredictionClass.CONFIRMED,
            confidence=0.87,
            explain=PredictionExplanation(
                top_features=[
                    FeatureImportance(name="orbital_period", value=0.8)
                ]
            )
        )
        
        assert response.prediction == PredictionClass.CONFIRMED
        assert response.confidence == 0.87
        assert len(response.explain.top_features) == 1
    
    def test_invalid_confidence(self):
        """Test response with invalid confidence value"""
        with pytest.raises(ValidationError):
            PredictionResponse(
                prediction=PredictionClass.CONFIRMED,
                confidence=1.5,  # Out of range
                explain=PredictionExplanation(top_features=[])
            )
    
    def test_all_prediction_classes(self):
        """Test all prediction class values"""
        for pred_class in [
            PredictionClass.CONFIRMED,
            PredictionClass.CANDIDATE,
            PredictionClass.FALSE_POSITIVE
        ]:
            response = PredictionResponse(
                prediction=pred_class,
                confidence=0.7,
                explain=PredictionExplanation(top_features=[])
            )
            assert response.prediction == pred_class


class TestFeatureImportance:
    """Test FeatureImportance model"""
    
    def test_valid_feature_importance(self):
        """Test valid feature importance"""
        fi = FeatureImportance(name="orbital_period", value=0.8)
        assert fi.name == "orbital_period"
        assert fi.value == 0.8
    
    def test_feature_importance_validation(self):
        """Test feature importance validation"""
        # Should accept any float value
        fi = FeatureImportance(name="test", value=1.5)
        assert fi.value == 1.5


class TestBatchPredictionResponse:
    """Test BatchPredictionResponse model"""
    
    def test_valid_batch_response(self):
        """Test valid batch prediction response"""
        response = BatchPredictionResponse(
            message="Success",
            rows_processed=2,
            predictions=[
                BatchPredictionItem(
                    row_index=0,
                    prediction=PredictionClass.CONFIRMED,
                    confidence=0.9,
                    explain=PredictionExplanation(top_features=[])
                ),
                BatchPredictionItem(
                    row_index=1,
                    prediction=PredictionClass.FALSE_POSITIVE,
                    confidence=0.7,
                    explain=PredictionExplanation(top_features=[])
                )
            ]
        )
        
        assert response.rows_processed == 2
        assert len(response.predictions) == 2
        assert response.predictions[0].row_index == 0
        assert response.predictions[1].row_index == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

