# Exoplanet Detection Project

An end-to-end machine learning application for detecting exoplanets from light curve data (Kepler/K2/TESS missions).

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
- `POST /api/predict` - Single prediction
- `POST /api/predict/batch` - Batch predictions from CSV

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

