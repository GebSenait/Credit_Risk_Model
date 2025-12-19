"""
Pydantic Models for Credit Risk API
Defines request and response schemas
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single prediction"""

    id: Optional[str] = Field(None, description="Optional request identifier")

    # Add your actual feature fields here
    # Example features (adjust based on your model):
    # age: float = Field(..., description="Age of the applicant")
    # income: float = Field(..., description="Annual income")
    # credit_score: float = Field(..., description="Credit score")
    # loan_amount: float = Field(..., description="Loan amount requested")

    class Config:
        schema_extra = {
            "example": {
                "id": "req_001",
                # Add example values for your features
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""

    id: Optional[str] = Field(None, description="Request identifier")
    prediction: int = Field(
        ..., description="Binary prediction (0=No Default, 1=Default)"
    )
    default_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of default"
    )
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")

    class Config:
        schema_extra = {
            "example": {
                "id": "req_001",
                "prediction": 0,
                "default_probability": 0.25,
                "risk_level": "Low",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    records: List[PredictionRequest] = Field(
        ..., description="List of prediction requests"
    )

    class Config:
        schema_extra = {
            "example": {
                "records": [
                    {
                        "id": "req_001",
                        # Add example features
                    },
                    {
                        "id": "req_002",
                        # Add example features
                    },
                ]
            }
        }
