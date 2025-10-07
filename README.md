# Exoplanet Detection Platform

A full-stack application that downloads NASA light-curve data (Kepler / K2 / TESS), trains machine-learning models to detect exoplanets, serves predictions through a FastAPI backend, and offers a React/Next.js frontend for interactive uploads and results.

---

## 🌐 Datasets
| Mission | CSV Source | Notes |
|---------|------------|-------|
| Kepler  | https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv | Confirmed planets |
| K2      | https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2candidates&format=csv | Planet candidates |
| TESS    | https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI&format=csv | TESS Objects of Interest |

Fetch them automatically:
```bash
cd ml/src
python fetch_nasa.py                    # all datasets
python fetch_nasa.py --dataset kepler   # single dataset
```

---

## 🚀 Quick-Start (Local Dev)

### 1. Backend (FastAPI)
```bash
cd backend
python -m venv venv && source venv/bin/activate   # win: venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload            # http://localhost:8000/docs
```

### 2. Machine-Learning Pipeline
```bash
cd ml
pip install -r requirements.txt -r requirements-dev.txt
# Train baseline model
python src/train.py data/raw/kepler_20241007.csv --model random_forest --output models/rf.pkl
# Explain a prediction
python src/explain.py
```

### 3. Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev                              # http://localhost:3000
```

> **Docker**: run the whole stack with `docker-compose up`.

---

## 📁 Folder Layout
```
SpaceApps/
├── backend/         FastAPI service + Dockerfile
├── frontend/        Next.js UI + Dockerfile
├── ml/              Data, models, training, explainability
│   ├── data/        raw/   processed/
│   ├── src/         train.py fetch_nasa.py …
│   └── mlruns/      MLflow tracking artifacts
├── .github/         CI (GitHub Actions) & docs
└── docker-compose.yml  Dev stack (backend + frontend + redis)
```

---

## 🔧 How to Use the API
1. Start backend (`uvicorn` or `docker-compose`).
2. Send light-curve features to `/predict`.

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "orbital_period": 10.5,
           "transit_duration": 2.5,
           "planet_radius": 1.2,
           "stellar_temp": 5778,
           "flux": [1.01,0.99,1.02,0.98,...]
         }'
```
Response
```json
{
  "prediction": 1,
  "confidence": 0.93,
  "top_features": [
    {"name": "stellar_temp", "impact": 0.15},
    {"name": "planet_radius", "impact": -0.08},
    {"name": "orbital_period", "impact": 0.05}
  ]
}
```

---

## 🛠️ Development Notes
* **MLflow** tracking: `mlflow ui --backend-store-uri ./ml/mlruns`.
* **CI**: GitHub Actions – lint, tests, build.
* **Docker**: production-ready images (`backend/Dockerfile`, `frontend/Dockerfile`).
* **Data fetcher**: `ml/src/fetch_nasa.py` automates downloads.

---

## 🤝 Contributing
1. Fork & clone repository.
2. Create feature branch `git checkout -b feature/xyz`.
3. Follow code style: `black`, `flake8`, `eslint`.
4. Add/ update tests (`pytest`, `jest`).
5. Commit using conventional commits: `feat(scope): message`.
6. Push & open Pull Request; ensure CI passes.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full guidelines.

---

## Project Structure

```
.
├── backend/          # FastAPI backend
├── ml/              # Machine learning pipeline
└── frontend/        # Next.js frontend
```

## Getting Started

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/predict` - Single prediction (JSON with feature dict)
- `POST /api/predict/batch` - Batch predictions from CSV upload
- `GET /api/model/info` - Get loaded model information

**Response Format:**
```json
{
  "prediction": "confirmed|candidate|false_positive",
  "confidence": 0.87,
  "explain": {
    "top_features": [
      {"name": "orbital_period", "value": 0.8}
    ]
  }
}
```

**Docker:**
```bash
cd backend
docker build -t exoplanet-api .
docker run -p 8000:8000 exoplanet-api
```

### ML Pipeline

```bash
cd ml
pip install -r requirements.txt
```

**Training a model:**
```bash
cd ml/src
python train.py
```

**Evaluating a model:**
```bash
cd ml/src
python evaluate.py
```

**Exploration notebook:**
```bash
cd ml
jupyter notebook notebooks/0_exploration.ipynb
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

**Production build:**
```bash
npm run build
npm start
```

## TODO: Project Setup Checklist

### Data Preparation
- [ ] Download Kepler/K2/TESS datasets
- [ ] Place data in `ml/data/` directory
- [ ] Update data loader functions with actual column names
- [ ] Perform exploratory data analysis in notebook

### ML Development
- [ ] Choose ML framework (PyTorch or TensorFlow)
- [ ] Update requirements.txt with chosen framework
- [ ] Define model architecture in `train.py`
- [ ] Implement feature engineering
- [ ] Train and evaluate baseline model
- [ ] Optimize hyperparameters
- [ ] Save best model to `ml/models/`

### Backend Integration
- [ ] Update `predict.py` to load actual trained model
- [ ] Configure model path and loading logic
- [ ] Test API endpoints with sample data
- [ ] Add proper error handling and validation
- [ ] Configure CORS for production

### Frontend Enhancement
- [ ] Set `NEXT_PUBLIC_API_URL` environment variable
- [ ] Test file upload with backend
- [ ] Implement light curve visualization (Chart.js/D3.js)
- [ ] Add loading states and error handling
- [ ] Improve UI/UX design
- [ ] Add data validation

### Deployment
- [ ] Set up environment variables
- [ ] Configure Docker Compose for full stack
- [ ] Deploy backend API
- [ ] Deploy frontend
- [ ] Set up CI/CD pipeline

## Data Sources

### Kepler/K2/TESS Datasets
- **Kepler**: https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- **MAST Portal**: https://mast.stsci.edu/

## Model Architecture Ideas

### Option 1: 1D CNN
- Good for detecting local patterns in light curves
- Fast training and inference
- Works well with raw flux data

### Option 2: LSTM/GRU
- Captures temporal dependencies
- Effective for sequence data
- Can model long-range patterns

### Option 3: Transformer
- Attention mechanism for important features
- State-of-the-art for time series
- Requires more data and compute

### Option 4: Hybrid
- Combine CNN for feature extraction + LSTM for temporal modeling
- Best of both worlds

## Features to Consider

### Time-Domain Features
- Transit depth
- Transit duration
- Period
- Mean, std, skewness, kurtosis
- Min/max flux values

### Frequency-Domain Features
- Fourier transform coefficients
- Power spectral density
- Dominant frequencies

### Statistical Features
- Autocorrelation
- Moving averages
- Trend components

## Performance Metrics

Given the class imbalance (few exoplanets vs many non-exoplanets):
- **Precision**: Important to avoid false positives
- **Recall**: Don't miss actual exoplanets
- **F1-Score**: Balance of both
- **ROC-AUC**: Overall model performance
- **Confusion Matrix**: Detailed breakdown

## Resources

- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [TESS Mission](https://www.nasa.gov/tess-transiting-exoplanet-survey-satellite)
- [Exoplanet Detection Papers](https://arxiv.org/search/?query=exoplanet+detection+machine+learning)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

