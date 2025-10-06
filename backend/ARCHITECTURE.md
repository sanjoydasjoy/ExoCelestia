# Backend Architecture

## Overview

The backend is built with FastAPI and supports multiple ML frameworks (PyTorch, TensorFlow, scikit-learn) through a modular model loading system.

## Key Components

### 1. API Layer (`app/main.py`, `app/api/predict.py`)

**Endpoints:**
- `POST /api/predict` - Single prediction from JSON
- `POST /api/predict/batch` - Batch predictions from CSV
- `GET /api/model/info` - Model metadata

**Features:**
- Automatic input validation with Pydantic
- Comprehensive error handling
- Structured logging
- CORS support for frontend integration

### 2. Data Models (`app/models.py`)

**Pydantic Models:**
- `PredictionRequest` - Validates JSON input features
- `PredictionResponse` - Standardized prediction response
- `BatchPredictionResponse` - CSV batch processing response
- `PredictionClass` - Enum (confirmed/candidate/false_positive)
- `FeatureImportance` - Feature explanation structure

**Validation:**
- Feature dictionary must have numeric values
- Confidence scores constrained to [0, 1]
- Automatic type coercion and validation

### 3. Model Loader (`app/model_loader.py`)

**Design Pattern:** Factory + Strategy

**Base Class:** `BaseModelLoader`
- Abstract interface for all model loaders
- Common functionality (config loading, feature preparation)
- Prediction interface standardization

**Implementations:**
- `PyTorchModelLoader` - For .pt/.pth files
- `TensorFlowModelLoader` - For .h5/.keras files  
- `ScikitLearnModelLoader` - For .pkl files

**Key Functions:**

```python
def get_model_loader(model_path, framework=None) -> BaseModelLoader
```
- Auto-detects framework from file extension
- Returns appropriate loader instance

```python
def load_model()
```
- Singleton pattern - loads model once
- Lazy loading on first request
- Caches model globally

```python
def validate_and_predict(feature_dict, loader) -> dict
```
- Validates and orders features
- Runs prediction
- Extracts feature importance
- Returns standardized response

## Data Flow

### Single Prediction (JSON)

```
Client Request (JSON)
    ↓
FastAPI Endpoint (/api/predict)
    ↓
Pydantic Validation (PredictionRequest)
    ↓
Load Model (if not loaded)
    ↓
validate_and_predict()
    ├→ prepare_features() - Order features correctly
    ├→ predict() - Run model inference
    └→ get_feature_importance() - Extract explanations
    ↓
create_prediction_response()
    ↓
PredictionResponse (validated)
    ↓
JSON Response to Client
```

### Batch Prediction (CSV)

```
Client Upload (CSV File)
    ↓
FastAPI Endpoint (/api/predict/batch)
    ↓
Validate File Type (.csv)
    ↓
Parse CSV → List of feature dicts
    ↓
For Each Row:
    ├→ validate_and_predict()
    └→ Collect results
    ↓
BatchPredictionResponse
    ↓
JSON Response with all predictions
```

## Model Configuration

Each model should have an accompanying `model_config.json`:

```json
{
  "feature_names": ["orbital_period", "transit_depth", ...],
  "model_type": "RandomForestClassifier",
  "trained_at": "2024-01-01T00:00:00"
}
```

This ensures:
1. Features are ordered correctly
2. Feature validation works properly
3. Model metadata is available

## Error Handling

### Model Not Found (400)
```python
raise HTTPException(
    status_code=400,
    detail="Model file not found. Please ensure..."
)
```

### Invalid Features (400)
```python
raise HTTPException(
    status_code=400,
    detail="Invalid features: {error}"
)
```

### Prediction Error (500)
```python
raise HTTPException(
    status_code=500,
    detail="Prediction failed: {error}"
)
```

## Feature Importance Extraction

### scikit-learn
- Uses `model.feature_importances_` (tree-based models)
- Uses `model.coef_` (linear models)
- Fallback: feature magnitudes

### PyTorch
- Gradient-based importance
- Computes gradients w.r.t. input features
- Normalizes and returns top features

### TensorFlow
- GradientTape for gradient computation
- Similar to PyTorch approach
- Returns normalized importance scores

## Testing Strategy

### Unit Tests (`tests/test_predict.py`)
- Model loader functionality
- Feature preparation
- Prediction logic
- API endpoints with mocked models

### Test Coverage
- Model loading/switching
- Feature validation
- Error handling
- Batch processing
- Response formatting

### Fixtures
- `mock_model_file` - Creates test model
- `client` - FastAPI test client
- `monkeypatch` - Environment variable injection

## Production Considerations

### Performance
- Model loaded once (singleton)
- Batch processing for multiple predictions
- Async endpoints for concurrency

### Security
- Input validation with Pydantic
- File type validation (CSV only)
- Error message sanitization
- CORS configuration

### Monitoring
- Structured logging
- Request/response logging
- Error tracking
- Model info endpoint for health checks

### Scalability
- Stateless design (model in memory)
- Docker containerization
- Easy horizontal scaling
- Environment-based configuration

## Extension Points

### Adding New ML Framework

1. Create new loader class:
```python
class NewFrameworkLoader(BaseModelLoader):
    def load_model(self):
        # Framework-specific loading
        
    def predict(self, features):
        # Framework-specific inference
        
    def get_feature_importance(self, features):
        # Framework-specific importance
```

2. Register in factory:
```python
loaders = {
    'new_framework': NewFrameworkLoader,
    ...
}
```

### Adding New Prediction Class

1. Update enum:
```python
class PredictionClass(str, Enum):
    NEW_CLASS = "new_class"
```

2. Update mapping logic:
```python
def map_prediction_to_class(self, prediction, confidence):
    # Add new mapping logic
```

### Adding New Features

1. Update Pydantic models
2. Implement validation logic
3. Add tests
4. Update documentation

## Dependencies

**Core:**
- `fastapi` - Web framework
- `pydantic` - Data validation
- `numpy` - Numerical operations

**Optional (based on model):**
- `torch` - PyTorch models
- `tensorflow` - TensorFlow models
- `scikit-learn` - sklearn models

**Testing:**
- `pytest` - Test framework
- `httpx` - Async test client

## File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app setup
│   ├── models.py            # Pydantic schemas
│   ├── model_loader.py      # Model loading abstraction
│   └── api/
│       ├── __init__.py
│       └── predict.py       # Prediction endpoints
├── tests/
│   ├── __init__.py
│   ├── test_predict.py      # Endpoint tests
│   └── test_models.py       # Model validation tests
├── requirements.txt         # Dependencies
├── Dockerfile              # Container config
├── pytest.ini              # Test configuration
├── README.md               # User guide
└── ARCHITECTURE.md         # This file
```

