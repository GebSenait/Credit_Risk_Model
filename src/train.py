"""
Training Module for Credit Risk Model
Handles model training, validation, hyperparameter tuning, and MLflow tracking
Implements Task 5 - Model Training, Tracking, and Validation
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_processing import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed random state for reproducibility
RANDOM_STATE = 42


class ModelTrainer:
    """Class for training credit risk models with MLflow tracking"""

    def __init__(
        self,
        experiment_name: str = "credit_risk_modeling",
        random_state: int = RANDOM_STATE,
    ):
        """
        Initialize ModelTrainer

        Args:
            experiment_name: Name of the MLflow experiment
            random_state: Random state for reproducibility
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Set up MLflow experiment
        mlflow.set_experiment(self.experiment_name)

    def load_and_split_data(
        self,
        data_path: str = "data/processed/credit_data_with_proxy_target.csv",
        target_column: str = "is_high_risk",
        test_size: float = 0.2,
        use_woe: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load processed data and split into training and testing sets

        Args:
            data_path: Path to processed data file
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            use_woe: Whether to use WoE transformations

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Loading processed data from {data_path}")

        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {data_file}")

        # Load data with proxy target
        processor = DataProcessor(
            data_path=str(data_file.parent),
            target_column=target_column,
            use_woe=use_woe,
        )
        df = processor.load_data(data_file.name)

        # Preprocess data with feature engineering pipeline
        logger.info("Preprocessing data with feature engineering pipeline...")
        X, y = processor.preprocess(df, target_column=target_column, fit_pipeline=True)

        # Store feature names
        self.feature_names = list(X.columns)

        # Split data with stratification and fixed random state
        logger.info(
            f"Splitting data: test_size={test_size}, random_state={self.random_state}"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        logger.info(
            f"Target distribution - Train: {y_train.value_counts().to_dict()}, "
            f"Test: {y_test.value_counts().to_dict()}"
        )

        return X_train, X_test, y_train, y_test

    def evaluate_model(
        self, model: Any, X: pd.DataFrame, y: pd.Series, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Evaluate model and return comprehensive metrics

        Args:
            model: Trained model
            X: Features DataFrame
            y: Target Series
            prefix: Prefix for metric names (e.g., 'train_', 'test_')

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = {
            f"{prefix}accuracy": accuracy_score(y, y_pred),
            f"{prefix}precision": precision_score(y, y_pred, zero_division=0),
            f"{prefix}recall": recall_score(y, y_pred, zero_division=0),
            f"{prefix}f1_score": f1_score(y, y_pred, zero_division=0),
            f"{prefix}roc_auc": (
                roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
            ),
        }

        return metrics

    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train Logistic Regression model with hyperparameter tuning"""
        logger.info("Training Logistic Regression model...")

        # Define parameter grid
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "max_iter": [1000],
        }

        # Base model
        base_model = LogisticRegression(random_state=self.random_state)

        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, {
            "model_type": "logistic_regression",
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
        }

    def train_decision_tree(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train Decision Tree model with hyperparameter tuning"""
        logger.info("Training Decision Tree model...")

        # Define parameter grid
        param_grid = {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
        }

        # Base model
        base_model = DecisionTreeClassifier(random_state=self.random_state)

        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, {
            "model_type": "decision_tree",
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
        }

    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train Random Forest model with hyperparameter tuning"""
        logger.info("Training Random Forest model...")

        # Define parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

        # Base model
        base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

        # Randomized search (faster for Random Forest)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,  # Sample 20 combinations
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state,
        )

        random_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")

        return random_search.best_estimator_, {
            "model_type": "random_forest",
            "best_params": random_search.best_params_,
            "best_cv_score": random_search.best_score_,
        }

    def train_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train XGBoost model with hyperparameter tuning"""
        logger.info("Training XGBoost model...")

        # Define parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }

        # Base model
        base_model = XGBClassifier(
            random_state=self.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        # Randomized search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,  # Sample 20 combinations
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state,
        )

        random_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")

        return random_search.best_estimator_, {
            "model_type": "xgboost",
            "best_params": random_search.best_params_,
            "best_cv_score": random_search.best_score_,
        }

    def train_all_models(
        self,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
    ) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        Train all models and return results

        Args:
            X_train: Training features (uses self.X_train if None)
            y_train: Training target (uses self.y_train if None)

        Returns:
            Dictionary mapping model names to (model, metadata) tuples
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            raise ValueError(
                "Training data not available. Call load_and_split_data() first."
            )

        results = {}

        # Train Logistic Regression
        try:
            model, metadata = self.train_logistic_regression(X_train, y_train)
            results["logistic_regression"] = (model, metadata)
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")

        # Train Decision Tree
        try:
            model, metadata = self.train_decision_tree(X_train, y_train)
            results["decision_tree"] = (model, metadata)
        except Exception as e:
            logger.error(f"Error training Decision Tree: {e}")

        # Train Random Forest
        try:
            model, metadata = self.train_random_forest(X_train, y_train)
            results["random_forest"] = (model, metadata)
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")

        # Train XGBoost
        try:
            model, metadata = self.train_xgboost(X_train, y_train)
            results["xgboost"] = (model, metadata)
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")

        self.models = results
        return results

    def log_model_to_mlflow(
        self,
        model: Any,
        model_name: str,
        model_metadata: Dict[str, Any],
        metrics: Dict[str, float],
    ):
        """
        Log model, parameters, and metrics to MLflow

        Args:
            model: Trained model
            model_name: Name of the model
            model_metadata: Dictionary containing model metadata
            metrics: Dictionary of evaluation metrics
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if "best_params" in model_metadata:
                mlflow.log_params(model_metadata["best_params"])
            mlflow.log_param("model_type", model_metadata["model_type"])
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("cv_score", model_metadata.get("best_cv_score", 0))

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            if model_metadata["model_type"] == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            # Log feature names if available
            if self.feature_names:
                mlflow.log_param("n_features", len(self.feature_names))

            logger.info(f"Logged {model_name} to MLflow")

    def compare_models_and_select_best(
        self, metric: str = "test_roc_auc"
    ) -> Tuple[str, Any, Dict[str, float]]:
        """
        Compare all trained models and select the best one

        Args:
            metric: Metric to use for comparison (default: test_roc_auc)

        Returns:
            Tuple of (best_model_name, best_model, best_metrics)
        """
        if not self.models:
            raise ValueError("No models trained. Call train_all_models() first.")

        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not available. Call load_and_split_data() first."
            )

        best_score = -np.inf
        best_model_name = None
        best_model = None
        best_metrics = None

        all_metrics = {}

        # Evaluate all models
        for model_name, (model, metadata) in self.models.items():
            logger.info(f"Evaluating {model_name}...")

            # Evaluate on test set
            test_metrics = self.evaluate_model(
                model, self.X_test, self.y_test, prefix="test_"
            )

            # Evaluate on train set
            train_metrics = self.evaluate_model(
                model, self.X_train, self.y_train, prefix="train_"
            )

            # Combine metrics
            combined_metrics = {**train_metrics, **test_metrics}
            all_metrics[model_name] = combined_metrics

            # Log to MLflow
            self.log_model_to_mlflow(model, model_name, metadata, combined_metrics)

            # Update best model
            if metric in combined_metrics:
                score = combined_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = model
                    best_metrics = combined_metrics

        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_metrics = best_metrics

        logger.info(f"\n{'='*70}")
        logger.info("MODEL COMPARISON RESULTS")
        logger.info(f"{'='*70}")
        for model_name, metrics in all_metrics.items():
            marker = " <-- BEST" if model_name == best_model_name else ""
            logger.info(f"\n{model_name}{marker}:")
            for metric_name, value in sorted(metrics.items()):
                logger.info(f"  {metric_name}: {value:.4f}")

        return best_model_name, best_model, best_metrics

    def register_best_model(self, model_name: str = "credit_risk_model") -> str:
        """
        Register the best model in MLflow Model Registry

        Args:
            model_name: Name for the registered model

        Returns:
            Model version URI
        """
        if self.best_model is None:
            raise ValueError(
                "No best model selected. Call compare_models_and_select_best() first."
            )

        # Get the latest run for the best model
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")

        # Find the run for the best model
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f'tags.mlflow.runName = "{self.best_model_name}"',
            order_by=["metrics.test_roc_auc DESC"],
            max_results=1,
        )

        if runs.empty:
            raise ValueError(f"Run for {self.best_model_name} not found in MLflow")

        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/model"

        # Register model
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(
                f"Successfully registered model '{model_name}' version {model_version.version}"
            )
            logger.info(f"Model URI: {model_uri}")
            return model_uri
        except Exception as e:
            logger.warning(
                f"Model may already be registered. Attempting to log as new version: {e}"
            )
            # Try creating a new version
            try:
                result = mlflow.register_model(model_uri, model_name)
                return result
            except Exception as e2:
                logger.error(f"Failed to register model: {e2}")
                # Still return the model URI even if registration fails
                return model_uri

    def save_model(self, filepath: str, model: Optional[Any] = None):
        """
        Save the trained model to disk

        Args:
            filepath: Path to save the model
            model: Model to save (uses best_model if None)
        """
        model_to_save = model if model is not None else self.best_model

        if model_to_save is None:
            raise ValueError("No model to save. Train a model first.")

        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model_to_save, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save feature names
        if self.feature_names:
            feature_path = model_path.parent / f"{model_path.stem}_features.pkl"
            with open(feature_path, "wb") as f:
                pickle.dump(self.feature_names, f)
            logger.info(f"Feature names saved to {feature_path}")


def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("CREDIT RISK MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    # Initialize trainer
    trainer = ModelTrainer(
        experiment_name="credit_risk_modeling",
        random_state=RANDOM_STATE,
    )

    try:
        # Load and split data
        trainer.load_and_split_data(
            data_path="data/processed/credit_data_with_proxy_target.csv",
            target_column="is_high_risk",
            test_size=0.2,
            use_woe=True,
        )

        # Train all models
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 70)
        trainer.train_all_models()

        # Compare models and select best
        logger.info("\n" + "=" * 70)
        logger.info("COMPARING MODELS AND SELECTING BEST")
        logger.info("=" * 70)
        best_model_name, best_model, best_metrics = (
            trainer.compare_models_and_select_best()
        )

        # Register best model in MLflow
        logger.info("\n" + "=" * 70)
        logger.info("REGISTERING BEST MODEL IN MLFLOW REGISTRY")
        logger.info("=" * 70)
        model_uri = trainer.register_best_model(model_name="credit_risk_model")

        # Save best model locally
        logger.info("\n" + "=" * 70)
        logger.info("SAVING BEST MODEL LOCALLY")
        logger.info("=" * 70)
        trainer.save_model("models/credit_risk_model.pkl")

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best test ROC-AUC: {best_metrics.get('test_roc_auc', 0):.4f}")
        logger.info(f"Model registered at: {model_uri}")
        logger.info("\nTo view experiments, run: mlflow ui")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info("Please ensure processed data file exists in data/processed/")
        logger.info("Run Task 4 first to create proxy target variable.")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
