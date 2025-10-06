# Development Guide

## Quick Start for Development

### 1. Train a Model

```bash
cd ml
pip install -r requirements.txt
cd src
python train.py
```

This creates:
- `ml/models/model.pkl` - Trained model
- `ml/models/model_config.json` - Model configuration
- `ml/models/model_scaler.pkl` - Feature scaler

### 2. Start Backend API

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to use the web interface.

## Testing the API

### Test with cURL (JSON)

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "orbital_period": 3.5,
      "transit_depth": 0.02,
      "transit_duration": 0.15,
      "stellar_radius": 1.2,
      "stellar_mass": 1.0
    }
  }'
```

### Test with cURL (CSV)

```bash
# Create test CSV
cat > test_data.csv << EOF
orbital_period,transit_depth,transit_duration,stellar_radius,stellar_mass
3.5,0.02,0.15,1.2,1.0
5.2,0.01,0.18,0.9,0.8
EOF

# Upload for batch prediction
curl -X POST http://localhost:8000/api/predict/batch \
  -F "file=@test_data.csv"
```

### Test with Python

```python
import requests

# JSON prediction
response = requests.post(
    "http://localhost:8000/api/predict",
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
print(response.json())

# CSV batch prediction
with open('test_data.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/predict/batch",
        files={"file": f}
    )
print(response.json())
```

## Running Tests

### Backend Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests (TODO)

```bash
cd frontend
npm test
```

## Model Framework Switching

### Using scikit-learn (Default)

```python
# ml/src/train.py
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# Train and save as .pkl
```

### Using PyTorch

```bash
pip install torch
```

```python
# ml/src/train.py
import torch
import torch.nn as nn

class ExoplanetNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 classes
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Save model
torch.save(model, 'models/model.pt')
```

### Using TensorFlow

```bash
pip install tensorflow
```

```python
# ml/src/train.py
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Save model
model.save('models/model.h5')
```

## Environment Variables

### Backend

Create `backend/.env`:

```bash
MODEL_PATH=../ml/models/model.pkl
MODEL_FRAMEWORK=sklearn  # or pytorch, tensorflow
LOG_LEVEL=INFO
```

### Frontend

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Docker Development

```bash
# Build and run full stack
docker-compose up --build

# Run individual services
docker-compose up backend
docker-compose up frontend
```

## API Documentation

Once the backend is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Code Structure

### Backend

```
backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── models.py            # Pydantic schemas
│   ├── model_loader.py      # ML framework abstraction
│   └── api/
│       └── predict.py       # Prediction endpoints
├── tests/                   # Unit tests
└── requirements.txt
```

### ML Pipeline

```
ml/
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocess.py        # Preprocessing functions
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── notebooks/               # Jupyter notebooks
├── models/                  # Saved models
└── data/                    # Datasets
```

### Frontend

```
frontend/
├── pages/
│   ├── index.tsx            # Main page
│   └── _app.tsx             # App wrapper
├── components/
│   └── UploadForm.tsx       # Upload component
└── styles/                  # CSS modules
```

## Common Issues

### Model Not Found Error

If you see: `Model file not found. Please ensure you have trained a model...`

**Solution:**
1. Train a model: `cd ml/src && python train.py`
2. Verify model exists: `ls ml/models/model.*`
3. Set MODEL_PATH environment variable if needed

### Feature Name Mismatch

If predictions fail with feature errors:

1. Check `ml/models/model_config.json` has correct feature names
2. Ensure JSON/CSV features match the config
3. Verify feature order in config

### CORS Errors in Frontend

If frontend can't connect to backend:

1. Check backend is running on port 8000
2. Verify CORS settings in `backend/app/main.py`
3. Update `NEXT_PUBLIC_API_URL` in frontend

## Performance Tips

1. **Model Loading**: Model is loaded once and cached globally
2. **Batch Predictions**: Use `/predict/batch` for multiple predictions
3. **Feature Validation**: Pydantic validates inputs automatically
4. **Error Logging**: All errors are logged with details

## Contributing

1. Create feature branch
2. Write tests for new features
3. Ensure all tests pass: `pytest`
4. Update documentation
5. Submit pull request

## Useful Commands

```bash
# Backend
uvicorn app.main:app --reload --port 8000
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html

# Frontend
npm run dev
npm run build
npm run lint

# ML
python train.py
python evaluate.py
jupyter notebook notebooks/

# Docker
docker-compose up
docker-compose down
docker-compose logs -f backend
```

