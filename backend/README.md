# Exoplanet Detection API

FastAPI backend for exoplanet detection with modular ML framework support.

## Features

- ✅ JSON and CSV input support
- ✅ Modular model loading (PyTorch, TensorFlow, scikit-learn)
- ✅ Feature importance explanations
- ✅ Comprehensive error handling
- ✅ Full unit test coverage
- ✅ Production-ready logging
- ✅ Pydantic validation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

First, train a model using the ML pipeline:

```bash
cd ../ml/src
python train.py
```

This will create a model file in `../ml/models/` (e.g., `model.pkl`, `model.pt`, or `model.h5`)

### 3. Set Environment Variables (Optional)

```bash
cp .env.example .env
# Edit .env to set MODEL_PATH if needed
```

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /api/predict

Predict exoplanet classification from JSON features.

**Request:**
```json
{
  "features": {
    "orbital_period": 3.5,
    "transit_depth": 0.02,
    "transit_duration": 0.15,
    "stellar_radius": 1.2,
    "stellar_mass": 1.0
  }
}
```

**Response:**
```json
{
  "prediction": "confirmed",
  "confidence": 0.87,
  "explain": {
    "top_features": [
      {"name": "orbital_period", "value": 0.8},
      {"name": "transit_depth", "value": 0.65},
      {"name": "transit_duration", "value": 0.52}
    ]
  }
}
```

**Prediction Classes:**
- `confirmed` - High confidence exoplanet detection
- `candidate` - Possible exoplanet, needs verification
- `false_positive` - Not an exoplanet

### POST /api/predict/batch

Upload CSV file for batch predictions.

**Request:**
- Multipart form data with CSV file
- CSV must have header row with feature names

**Example CSV:**
```csv
orbital_period,transit_depth,transit_duration,stellar_radius,stellar_mass
3.5,0.02,0.15,1.2,1.0
5.2,0.01,0.18,0.9,0.8
```

**Response:**
```json
{
  "message": "Successfully processed 2 rows",
  "rows_processed": 2,
  "predictions": [
    {
      "row_index": 0,
      "prediction": "confirmed",
      "confidence": 0.87,
      "explain": {
        "top_features": [
          {"name": "orbital_period", "value": 0.8}
        ]
      }
    }
  ]
}
```

### GET /api/model/info

Get information about the loaded model.

**Response:**
```json
{
  "model_path": "/path/to/model.pkl",
  "model_type": "ScikitLearnModelLoader",
  "feature_names": ["orbital_period", "transit_depth", ...],
  "config": {
    "trained_at": "2024-01-01T00:00:00",
    "model_type": "RandomForestClassifier"
  }
}
```

## Model Switching

The API supports multiple ML frameworks. The framework is auto-detected from the file extension:

- `.pkl` or `.pickle` → scikit-learn
- `.pt` or `.pth` → PyTorch
- `.h5` or `.keras` → TensorFlow/Keras

### Using PyTorch Models

1. Install PyTorch:
   ```bash
   pip install torch
   ```

2. Save your model:
   ```python
   import torch
   torch.save(model, 'models/model.pt')
   ```

3. The API will auto-detect and load it

### Using TensorFlow Models

1. Install TensorFlow:
   ```bash
   pip install tensorflow
   ```

2. Save your model:
   ```python
   model.save('models/model.h5')
   ```

3. The API will auto-detect and load it

### Model Configuration

Create a `model_config.json` file alongside your model:

```json
{
  "feature_names": [
    "orbital_period",
    "transit_depth",
    "transit_duration",
    "stellar_radius",
    "stellar_mass"
  ],
  "model_type": "RandomForestClassifier",
  "trained_at": "2024-01-01T00:00:00"
}
```

This ensures features are ordered correctly when making predictions.

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app tests/
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid input, model not found)
- `422` - Validation error (invalid JSON schema)
- `500` - Internal server error

**Example Error Response:**
```json
{
  "detail": "Model file not found. Please ensure you have trained a model..."
}
```

## Architecture

```
backend/
├── app/
│   ├── main.py              # FastAPI app initialization
│   ├── models.py            # Pydantic models
│   ├── model_loader.py      # Modular model loading
│   └── api/
│       └── predict.py       # Prediction endpoints
├── tests/
│   ├── test_predict.py      # API tests
│   └── test_models.py       # Model tests
└── requirements.txt
```

## Key Functions

### `load_model()`
Loads the model from disk (singleton pattern for efficiency).

### `validate_and_predict(feature_dict, loader)`
Validates features and returns predictions with explanations.

### `get_model_loader(model_path, framework)`
Factory function that returns appropriate model loader based on framework.

## Production Deployment

### Docker

```bash
docker build -t exoplanet-api .
docker run -p 8000:8000 -v $(pwd)/../ml/models:/app/models exoplanet-api
```

### Environment Variables

```bash
export MODEL_PATH=/path/to/model.pkl
export MODEL_FRAMEWORK=sklearn  # Optional
export LOG_LEVEL=INFO
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Interactive Documentation

Visit these URLs when the server is running:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Write tests for new features
2. Follow PEP 8 style guide
3. Add docstrings to all functions
4. Update this README

## License

MIT

