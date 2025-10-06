"""
Unit tests for prediction endpoints
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import tempfile
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.main import app
from app.api.predict import get_model_path, validate_and_predict
from app.model_loader import ScikitLearnModelLoader, get_model_loader


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_model_file(tmp_path):
    """Create a mock model file for testing"""
    # Create a simple sklearn model
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = tmp_path / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save config
    config_path = tmp_path / "model_config.json"
    config = {
        "feature_names": ["orbital_period", "transit_depth", "transit_duration", "stellar_radius", "stellar_mass"],
        "model_type": "RandomForestClassifier",
        "trained_at": "2024-01-01T00:00:00"
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    return model_path


class TestModelLoader:
    """Test model loader functionality"""
    
    def test_sklearn_model_loader(self, mock_model_file):
        """Test loading scikit-learn model"""
        loader = ScikitLearnModelLoader(str(mock_model_file))
        loader.load_model()
        
        assert loader.model is not None
        assert len(loader.feature_names) == 5
    
    def test_get_model_loader_auto_detect(self, mock_model_file):
        """Test auto-detection of framework from file extension"""
        loader = get_model_loader(str(mock_model_file))
        assert isinstance(loader, ScikitLearnModelLoader)
    
    def test_model_loader_missing_file(self):
        """Test error when model file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            loader = ScikitLearnModelLoader("nonexistent.pkl")
    
    def test_prepare_features(self, mock_model_file):
        """Test feature preparation from dictionary"""
        loader = ScikitLearnModelLoader(str(mock_model_file))
        loader.load_model()
        
        features_dict = {
            "orbital_period": 3.5,
            "transit_depth": 0.02,
            "transit_duration": 0.15,
            "stellar_radius": 1.2,
            "stellar_mass": 1.0
        }
        
        features = loader.prepare_features(features_dict)
        assert features.shape == (1, 5)
        assert np.allclose(features[0], [3.5, 0.02, 0.15, 1.2, 1.0])
    
    def test_predict(self, mock_model_file):
        """Test model prediction"""
        loader = ScikitLearnModelLoader(str(mock_model_file))
        loader.load_model()
        
        features = np.random.randn(1, 5)
        predictions, probabilities = loader.predict(features)
        
        assert len(predictions) == 1
        assert probabilities.shape == (1, 2)
        assert np.isclose(probabilities.sum(), 1.0)
    
    def test_feature_importance(self, mock_model_file):
        """Test feature importance extraction"""
        loader = ScikitLearnModelLoader(str(mock_model_file))
        loader.load_model()
        
        features = np.random.randn(1, 5)
        importances = loader.get_feature_importance(features)
        
        assert len(importances) <= 5
        assert all('name' in imp and 'value' in imp for imp in importances)


class TestPredictionEndpoints:
    """Test prediction API endpoints"""
    
    def test_predict_json_success(self, client, mock_model_file, monkeypatch):
        """Test successful JSON prediction"""
        monkeypatch.setenv("MODEL_PATH", str(mock_model_file))
        
        # Reset global model loader
        from app.api import predict
        predict._model_loader = None
        
        response = client.post(
            "/api/predict",
            json={
                "features": {
                    "orbital_period": 3.5,
                    "transit_depth": 0.02,
                    "transit_duration": 0.15,
                    "stellar_radius": 1.2,
                    "stellar_mass": 1.0
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert data["prediction"] in ["confirmed", "candidate", "false_positive"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "explain" in data
        assert "top_features" in data["explain"]
    
    def test_predict_json_invalid_features(self, client, mock_model_file, monkeypatch):
        """Test prediction with invalid features"""
        monkeypatch.setenv("MODEL_PATH", str(mock_model_file))
        
        response = client.post(
            "/api/predict",
            json={
                "features": {}
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_json_missing_model(self, client, monkeypatch):
        """Test prediction when model file is missing"""
        monkeypatch.setenv("MODEL_PATH", "nonexistent.pkl")
        
        # Reset global model loader
        from app.api import predict
        predict._model_loader = None
        
        response = client.post(
            "/api/predict",
            json={
                "features": {
                    "orbital_period": 3.5,
                    "transit_depth": 0.02
                }
            }
        )
        
        assert response.status_code == 400
        assert "Model file not found" in response.json()["detail"]
    
    def test_predict_batch_csv_success(self, client, mock_model_file, monkeypatch):
        """Test successful CSV batch prediction"""
        monkeypatch.setenv("MODEL_PATH", str(mock_model_file))
        
        # Reset global model loader
        from app.api import predict
        predict._model_loader = None
        
        # Create test CSV
        csv_content = """orbital_period,transit_depth,transit_duration,stellar_radius,stellar_mass
3.5,0.02,0.15,1.2,1.0
5.2,0.01,0.18,0.9,0.8
"""
        
        response = client.post(
            "/api/predict/batch",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["rows_processed"] == 2
        assert len(data["predictions"]) == 2
        
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "confidence" in pred
            assert "explain" in pred
    
    def test_predict_batch_invalid_file_type(self, client, mock_model_file, monkeypatch):
        """Test batch prediction with non-CSV file"""
        monkeypatch.setenv("MODEL_PATH", str(mock_model_file))
        
        response = client.post(
            "/api/predict/batch",
            files={"file": ("test.txt", "not a csv", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Only CSV files are accepted" in response.json()["detail"]
    
    def test_model_info_endpoint(self, client, mock_model_file, monkeypatch):
        """Test model info endpoint"""
        monkeypatch.setenv("MODEL_PATH", str(mock_model_file))
        
        # Reset global model loader
        from app.api import predict
        predict._model_loader = None
        
        response = client.get("/api/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_path" in data
        assert "model_type" in data
        assert "feature_names" in data
        assert len(data["feature_names"]) == 5


class TestValidationFunctions:
    """Test validation and helper functions"""
    
    def test_validate_and_predict(self, mock_model_file):
        """Test validate_and_predict function"""
        loader = ScikitLearnModelLoader(str(mock_model_file))
        loader.load_model()
        
        features_dict = {
            "orbital_period": 3.5,
            "transit_depth": 0.02,
            "transit_duration": 0.15,
            "stellar_radius": 1.2,
            "stellar_mass": 1.0
        }
        
        result = validate_and_predict(features_dict, loader)
        
        assert "prediction" in result
        assert "confidence" in result
        assert "explain" in result
    
    def test_validate_and_predict_missing_features(self, mock_model_file):
        """Test validation with missing features"""
        loader = ScikitLearnModelLoader(str(mock_model_file))
        loader.load_model()
        
        features_dict = {
            "orbital_period": 3.5
            # Missing other required features
        }
        
        with pytest.raises(ValueError):
            validate_and_predict(features_dict, loader)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

