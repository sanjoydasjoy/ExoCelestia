from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class PredictionClass(str, Enum):
    """Enum for prediction classes"""
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    FALSE_POSITIVE = "false_positive"


class FeatureImportance(BaseModel):
    """Model for feature importance/explanation"""
    name: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature importance value (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "orbital_period",
                "value": 0.8
            }
        }


class PredictionExplanation(BaseModel):
    """Model for prediction explanation"""
    top_features: List[FeatureImportance] = Field(
        ..., 
        description="Top contributing features for the prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "top_features": [
                    {"name": "orbital_period", "value": 0.8},
                    {"name": "transit_depth", "value": 0.65},
                    {"name": "transit_duration", "value": 0.52}
                ]
            }
        }


class PredictionRequest(BaseModel):
    """Request model for exoplanet prediction from JSON"""
    features: Dict[str, float] = Field(
        ..., 
        description="Dictionary of feature names and their values"
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        if not all(isinstance(val, (int, float)) for val in v.values()):
            raise ValueError("All feature values must be numeric")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "orbital_period": 3.5,
                    "transit_depth": 0.02,
                    "transit_duration": 0.15,
                    "stellar_radius": 1.2,
                    "stellar_mass": 1.0
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for exoplanet prediction"""
    prediction: PredictionClass = Field(
        ..., 
        description="Prediction class: confirmed, candidate, or false_positive"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Prediction confidence score (0-1)"
    )
    explain: PredictionExplanation = Field(
        ..., 
        description="Explanation of the prediction with feature importances"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "confirmed",
                "confidence": 0.87,
                "explain": {
                    "top_features": [
                        {"name": "orbital_period", "value": 0.8},
                        {"name": "transit_depth", "value": 0.65}
                    ]
                }
            }
        }


class BatchPredictionItem(BaseModel):
    """Single prediction item in batch response"""
    row_index: int
    prediction: PredictionClass
    confidence: float
    explain: PredictionExplanation


class BatchPredictionResponse(BaseModel):
    """Response model for CSV batch upload"""
    message: str
    rows_processed: int
    predictions: List[BatchPredictionItem]
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Successfully processed 10 rows",
                "rows_processed": 10,
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
        }

