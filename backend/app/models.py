from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class PredictionRequest(BaseModel):
    """Request model for exoplanet prediction"""
    # TODO: Define specific features based on your dataset
    # Example features from Kepler/K2/TESS data
    features: List[float] = Field(
        ..., 
        description="List of normalized features from light curve data"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.7, 0.2, 0.8]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for exoplanet prediction"""
    prediction: int = Field(..., description="0 = No exoplanet, 1 = Exoplanet detected")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Probability distribution for each class"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "confidence": 0.85,
                "probabilities": {
                    "no_exoplanet": 0.15,
                    "exoplanet": 0.85
                }
            }
        }


class UploadResponse(BaseModel):
    """Response model for CSV upload"""
    message: str
    rows_processed: int
    predictions: List[Dict[str, Any]]

