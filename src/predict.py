"""
Prediction Module for Credit Risk Model
Handles model loading and making predictions
Supports loading from local files (joblib) or MLflow model registry
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.data_processing import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """Class for making credit risk predictions"""

    def __init__(self, model_path: str, use_mlflow: bool = False):
        """
        Initialize CreditRiskPredictor

        Args:
            model_path: Path to the saved model file or MLflow model URI
            use_mlflow: If True, load model from MLflow registry (model_path should be model URI)
        """
        self.model_path = model_path
        self.use_mlflow = use_mlflow
        self.model = None
        self.feature_names = None
        self.processor = DataProcessor()
        self._load_model()

    def _load_model(self):
        """Load the trained model and feature names"""
        if self.use_mlflow:
            # Load from MLflow model registry
            logger.info(f"Loading model from MLflow: {self.model_path}")
            try:
                self.model = mlflow.pyfunc.load_model(self.model_path)
                logger.info("Model loaded successfully from MLflow")
                # MLflow models may have feature names stored, try to extract
                # Note: For sklearn models in MLflow, feature names need to be loaded separately
            except Exception as e:
                logger.error(f"Failed to load model from MLflow: {e}")
                raise
        else:
            # Load from local file
            model_path_obj = Path(self.model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model file not found: {model_path_obj}")

            logger.info(f"Loading model from {model_path_obj}")
            self.model = joblib.load(model_path_obj)

            # Load feature names
            feature_path = model_path_obj.parent / f"{model_path_obj.stem}_features.pkl"
            if feature_path.exists():
                with open(feature_path, "rb") as f:
                    self.feature_names = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                logger.warning(
                    "Feature names file not found. Predictions may fail if feature mismatch occurs."
                )

    def preprocess_input(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """
        Preprocess input data for prediction

        Args:
            data: Input data as DataFrame or dictionary

        Returns:
            Preprocessed DataFrame
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Ensure all required features are present
        if self.feature_names is not None:
            # Add missing columns with default values
            for feature in self.feature_names:
                if feature not in data.columns:
                    data[feature] = 0

            # Select only the features used in training
            data = data[self.feature_names]

        return data

    def predict(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Make binary predictions (0 or 1)

        Args:
            data: Input data as DataFrame or dictionary

        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Initialize the predictor first.")

        X = self.preprocess_input(data)
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            data: Input data as DataFrame or dictionary

        Returns:
            Array of prediction probabilities [prob_class_0, prob_class_1]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Initialize the predictor first.")

        X = self.preprocess_input(data)
        probabilities = self.model.predict_proba(X)

        return probabilities

    def predict_risk_score(
        self, data: Union[pd.DataFrame, Dict]
    ) -> Dict[str, Union[float, int]]:
        """
        Get comprehensive risk prediction with score

        Args:
            data: Input data as DataFrame or dictionary

        Returns:
            Dictionary with prediction, probability, and risk level
        """
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)

        # Get probability of default (class 1)
        default_prob = (
            probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities[1]
        )

        # Determine risk level
        risk_levels = []
        for prob in default_prob:
            if prob < 0.3:
                risk_levels.append("Low")
            elif prob < 0.7:
                risk_levels.append("Medium")
            else:
                risk_levels.append("High")

        results = {
            "prediction": (
                int(predictions[0]) if len(predictions) == 1 else predictions.tolist()
            ),
            "default_probability": (
                float(default_prob[0])
                if len(default_prob) == 1
                else default_prob.tolist()
            ),
            "risk_level": risk_levels[0] if len(risk_levels) == 1 else risk_levels,
        }

        return results


def main():
    """Main prediction function for command-line usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <model_path> [data_file]")
        sys.exit(1)

    model_path = sys.argv[1]
    predictor = CreditRiskPredictor(model_path)

    if len(sys.argv) > 2:
        # Predict from file
        data_file = sys.argv[2]
        df = pd.read_csv(data_file)
        results = predictor.predict_risk_score(df)
        print(results)
    else:
        # Example prediction
        example_data = {
            "feature1": 0.5,
            "feature2": 1.0,
            # Add your actual features here
        }
        results = predictor.predict_risk_score(example_data)
        print(f"Prediction: {results}")


if __name__ == "__main__":
    main()
