"""
Training Module for Credit Risk Model
Handles model training, validation, and saving
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_processing import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training credit risk models"""

    def __init__(self, model_type: str = "random_forest", **model_params):
        """
        Initialize ModelTrainer

        Args:
            model_type: Type of model to train ('random_forest', 'xgboost', etc.)
            **model_params: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.feature_names = None

    def create_model(self):
        """Create and initialize the model"""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        logger.info(
            f"Created {self.model_type} model with parameters: {self.model_params}"
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility

        Returns:
            Dictionary containing training metrics
        """
        logger.info("Starting model training...")

        # Create model if not already created
        if self.model is None:
            self.create_model()

        # Store feature names
        self.feature_names = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Train model
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")

        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        test_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "train_auc": roc_auc_score(y_train, train_proba),
            "test_auc": roc_auc_score(y_test, test_proba),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
            "classification_report": classification_report(
                y_test, test_pred, output_dict=True
            ),
        }

        logger.info(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Training AUC: {metrics['train_auc']:.4f}")
        logger.info(f"Test AUC: {metrics['test_auc']:.4f}")

        return metrics

    def save_model(self, filepath: str):
        """
        Save the trained model to disk

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save feature names
        feature_path = model_path.parent / f"{model_path.stem}_features.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump(self.feature_names, f)
        logger.info(f"Feature names saved to {feature_path}")


def main():
    """Main training function"""
    # Initialize data processor
    processor = DataProcessor(data_path="data/raw")

    # Load and preprocess data
    # Assuming you have a dataset file - adjust filename as needed
    try:
        df = processor.load_data("credit_data.csv")
        X, y = processor.preprocess(df, target_column="target")

        # Initialize trainer
        trainer = ModelTrainer(
            model_type="random_forest", n_estimators=100, max_depth=10
        )

        # Train model
        metrics = trainer.train(X, y)

        # Save model
        trainer.save_model("models/credit_risk_model.pkl")

        logger.info("Training pipeline completed successfully")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info("Please ensure your data file is in data/raw/ directory")


if __name__ == "__main__":
    main()
