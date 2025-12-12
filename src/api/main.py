"""
FastAPI Application for Credit Risk Model
Provides REST API endpoints for model predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from pathlib import Path

from src.predict import CreditRiskPredictor
from src.api.pydantic_models import PredictionRequest, PredictionResponse, BatchPredictionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Model API",
    description="API for credit risk prediction using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: CreditRiskPredictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global predictor
    model_path = Path("models/credit_risk_model.pkl")
    
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}. Please train the model first.")
        logger.info("API will start but predictions will fail until model is available.")
    else:
        try:
            predictor = CreditRiskPredictor(str(model_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Model API",
        "version": "1.0.0",
        "status": "operational" if predictor is not None else "model_not_loaded"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction
    
    Args:
        request: Prediction request with input features
        
    Returns:
        Prediction response with risk score and probability
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert request to dictionary
        input_data = request.dict(exclude={'id'})
        
        # Make prediction
        result = predictor.predict_risk_score(input_data)
        
        return PredictionResponse(
            id=request.id,
            prediction=result['prediction'],
            default_probability=result['default_probability'],
            risk_level=result['risk_level']
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions
    
    Args:
        request: Batch prediction request with multiple input records
        
    Returns:
        List of prediction responses
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        results = []
        for record in request.records:
            input_data = record.dict(exclude={'id'})
            result = predictor.predict_risk_score(input_data)
            
            results.append(PredictionResponse(
                id=record.id,
                prediction=result['prediction'],
                default_probability=result['default_probability'],
                risk_level=result['risk_level']
            ))
        
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": str(type(predictor.model).__name__),
        "feature_count": len(predictor.feature_names) if predictor.feature_names else None,
        "model_path": str(predictor.model_path)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

