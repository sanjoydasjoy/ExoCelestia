from fastapi import APIRouter, HTTPException, UploadFile, File, status
from app.models import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionResponse,
    BatchPredictionItem
)
from app.model_loader import get_model_loader, create_prediction_response, BaseModelLoader
import numpy as np
from typing import Optional
import io
import csv
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Global model loader instance
_model_loader: Optional[BaseModelLoader] = None


def get_model_path() -> Path:
    """
    Get the path to the trained model
    
    Returns:
        Path to model file
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    # Check environment variable first
    model_path = os.getenv("MODEL_PATH")
    
    if model_path:
        path = Path(model_path)
    else:
        # Default paths to check
        possible_paths = [
            Path("../ml/models/model.pt"),
            Path("../ml/models/model.h5"),
            Path("../ml/models/model.pkl"),
            Path("models/model.pt"),
            Path("models/model.h5"),
            Path("models/model.pkl"),
        ]
        
        path = None
        for p in possible_paths:
            if p.exists():
                path = p
                break
    
    if not path or not path.exists():
        raise FileNotFoundError(
            "Model file not found. Please ensure you have trained a model using ml/src/train.py "
            "and the model is saved to one of these locations:\n"
            "- ../ml/models/model.pt (PyTorch)\n"
            "- ../ml/models/model.h5 (TensorFlow)\n"
            "- ../ml/models/model.pkl (scikit-learn)\n"
            "Or set the MODEL_PATH environment variable."
        )
    
    return path


def load_model() -> BaseModelLoader:
    """
    Load the model if not already loaded
    
    Returns:
        Model loader instance
    
    Raises:
        HTTPException: If model cannot be loaded
    """
    global _model_loader
    
    if _model_loader is None:
        try:
            model_path = get_model_path()
            framework = os.getenv("MODEL_FRAMEWORK")  # Optional: specify framework
            
            logger.info(f"Loading model from: {model_path}")
            _model_loader = get_model_loader(str(model_path), framework)
            _model_loader.load_model()
            logger.info(f"Model loaded successfully. Framework: {framework or 'auto-detected'}")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    return _model_loader


def validate_and_predict(
    feature_dict: dict,
    loader: BaseModelLoader
) -> dict:
    """
    Validate features and make prediction
    
    Args:
        feature_dict: Dictionary of features
        loader: Model loader instance
    
    Returns:
        Prediction response dictionary
    
    Raises:
        ValueError: If features are invalid
    """
    # Prepare features
    features = loader.prepare_features(feature_dict)
    
    # Make prediction
    predictions, probabilities = loader.predict(features)
    
    # Get feature importance
    feature_importances = loader.get_feature_importance(features)
    
    # Create response
    return create_prediction_response(
        predictions[0],
        probabilities[0],
        feature_importances,
        loader
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(request: PredictionRequest):
    """
    Predict exoplanet classification from light curve features (JSON input)
    
    Accepts a JSON body with feature names and values.
    Returns prediction class (confirmed/candidate/false_positive), 
    confidence score, and feature importance explanations.
    
    Args:
        request: PredictionRequest with features dictionary
    
    Returns:
        PredictionResponse with prediction, confidence, and explanations
    
    Raises:
        HTTPException: 400 if model not found or invalid features
        HTTPException: 500 if prediction fails
    
    Example:
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
    """
    try:
        loader = load_model()
        result = validate_and_predict(request.features, loader)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid features: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload CSV file and get predictions for all rows
    
    Accepts a CSV file with feature columns. The first row should be 
    the header with feature names. Each subsequent row will be processed
    and predictions returned for all rows.
    
    Expected CSV format:
    ```csv
    orbital_period,transit_depth,transit_duration,stellar_radius,stellar_mass
    3.5,0.02,0.15,1.2,1.0
    5.2,0.01,0.18,0.9,0.8
    ```
    
    Args:
        file: Uploaded CSV file
    
    Returns:
        BatchPredictionResponse with predictions for all rows
    
    Raises:
        HTTPException: 400 if model not found or invalid CSV format
        HTTPException: 500 if processing fails
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are accepted"
            )
        
        loader = load_model()
        
        # Read CSV content
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))
        reader = csv.DictReader(csv_file)
        
        predictions = []
        rows_processed = 0
        errors = []
        
        for row_idx, row in enumerate(reader):
            try:
                # Convert row to feature dictionary (with numeric values)
                feature_dict = {}
                for key, value in row.items():
                    try:
                        feature_dict[key] = float(value)
                    except ValueError:
                        logger.warning(f"Row {row_idx}: Could not convert '{key}' value '{value}' to float")
                        continue
                
                if not feature_dict:
                    errors.append(f"Row {row_idx}: No valid numeric features found")
                    continue
                
                # Make prediction
                result = validate_and_predict(feature_dict, loader)
                
                predictions.append(
                    BatchPredictionItem(
                        row_index=row_idx,
                        **result
                    )
                )
                rows_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing row {row_idx}: {str(e)}")
                errors.append(f"Row {row_idx}: {str(e)}")
        
        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No valid predictions made. Errors: {'; '.join(errors)}"
            )
        
        message = f"Successfully processed {rows_processed} rows"
        if errors:
            message += f" ({len(errors)} rows had errors)"
        
        return BatchPredictionResponse(
            message=message,
            rows_processed=rows_processed,
            predictions=predictions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV processing failed: {str(e)}"
        )


@router.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        Dictionary with model metadata
    """
    try:
        loader = load_model()
        
        return {
            "model_path": str(loader.model_path),
            "model_type": loader.__class__.__name__,
            "feature_names": loader.feature_names,
            "config": loader.config
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

