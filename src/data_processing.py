"""
Data Processing Module for Credit Risk Model
Handles data loading, cleaning, feature engineering, and preprocessing
Implements a comprehensive feature engineering pipeline using sklearn Pipeline
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to create customer-level aggregate features
    """

    def __init__(
        self, customer_id_col: str = "CustomerId", amount_col: str = "Amount"
    ):
        """
        Initialize CustomerAggregateFeatures

        Args:
            customer_id_col: Name of the customer ID column
            amount_col: Name of the transaction amount column
        """
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.agg_features_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by computing aggregate statistics

        Args:
            X: Input DataFrame
            y: Target (optional, not used)
        """
        if self.customer_id_col not in X.columns:
            logger.warning(
                f"Customer ID column '{self.customer_id_col}' not found. "
                "Skipping aggregate features."
            )
            return self

        if self.amount_col not in X.columns:
            logger.warning(
                f"Amount column '{self.amount_col}' not found. "
                "Skipping aggregate features."
            )
            return self

        # Compute aggregate features per customer
        self.agg_features_ = (
            X.groupby(self.customer_id_col)[self.amount_col]
            .agg(["sum", "mean", "count", "std"])
            .rename(
                columns={
                    "sum": f"{self.customer_id_col}_total_amount",
                    "mean": f"{self.customer_id_col}_avg_amount",
                    "count": f"{self.customer_id_col}_txn_count",
                    "std": f"{self.customer_id_col}_amount_std",
                }
            )
        )

        # Fill NaN std values (for customers with single transaction) with 0
        self.agg_features_.fillna(0, inplace=True)

        logger.info(
            f"Computed aggregate features for {len(self.agg_features_)} "
            "unique customers"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform by adding aggregate features

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with added aggregate features
        """
        X_transformed = X.copy()

        if self.agg_features_ is None or self.customer_id_col not in X.columns:
            return X_transformed

        # Merge aggregate features back to original data
        for col in self.agg_features_.columns:
            X_transformed[col] = X_transformed[self.customer_id_col].map(
                self.agg_features_[col]
            )

        # Fill any remaining NaN (for new customers not seen in training) with 0
        agg_cols = [
            col
            for col in X_transformed.columns
            if col.startswith(self.customer_id_col + "_")
        ]
        for col in agg_cols:
            X_transformed[col] = X_transformed[col].fillna(0)

        return X_transformed


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer to extract temporal features from timestamp column
    """

    def __init__(self, timestamp_col: str = "TransactionStartTime"):
        """
        Initialize TemporalFeatureExtractor

        Args:
            timestamp_col: Name of the timestamp column
        """
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer (no-op for temporal extraction)

        Args:
            X: Input DataFrame
            y: Target (optional, not used)
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with temporal features added
        """
        X_transformed = X.copy()

        if self.timestamp_col not in X.columns:
            logger.warning(
                f"Timestamp column '{self.timestamp_col}' not found. "
                "Skipping temporal features."
            )
            return X_transformed

        # Convert to datetime if not already
        try:
            X_transformed[self.timestamp_col] = pd.to_datetime(
                X_transformed[self.timestamp_col], errors="coerce"
            )

            # Extract temporal features
            X_transformed[f"{self.timestamp_col}_hour"] = X_transformed[
                self.timestamp_col
            ].dt.hour
            X_transformed[f"{self.timestamp_col}_day"] = X_transformed[
                self.timestamp_col
            ].dt.day
            X_transformed[f"{self.timestamp_col}_month"] = X_transformed[
                self.timestamp_col
            ].dt.month
            X_transformed[f"{self.timestamp_col}_year"] = X_transformed[
                self.timestamp_col
            ].dt.year

            # Drop original timestamp column (optional, but keeping for now)
            # X_transformed.drop(columns=[self.timestamp_col], inplace=True)

            logger.info("Extracted temporal features from timestamp")
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")

        return X_transformed


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to apply Weight of Evidence (WoE) encoding to categorical features
    WoE = ln((% of non-events in category) / (% of events in category))
    """

    def __init__(
        self, categorical_features: List[str], target_col: Optional[str] = None
    ):
        """
        Initialize WoETransformer

        Args:
            categorical_features: List of categorical feature names to transform
            target_col: Name of target column (required for WoE calculation)
        """
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.woe_dicts_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit WoE transformer by calculating WoE values for each category

        Args:
            X: Input DataFrame
            y: Target Series (required for WoE calculation)
        """
        if y is None and self.target_col is None:
            logger.warning(
                "No target provided. WoE transformation requires target variable."
            )
            return self

        # Get target from y parameter or X DataFrame
        if y is not None:
            target = y
        elif self.target_col and self.target_col in X.columns:
            target = X[self.target_col]
        else:
            logger.warning("Cannot find target variable for WoE calculation.")
            return self

        # Calculate WoE for each categorical feature
        for feature in self.categorical_features:
            if feature not in X.columns:
                logger.warning(
                    f"Feature '{feature}' not found in DataFrame. Skipping."
                )
                continue

            if feature == self.target_col:
                continue

            # Calculate WoE for each category
            woe_dict = {}
            for category in X[feature].unique():
                if pd.isna(category):
                    continue

                # Get events (target=1) and non-events (target=0) for this category
                category_mask = X[feature] == category
                category_events = target[category_mask].sum()
                category_non_events = (1 - target[category_mask]).sum()

                # Calculate overall event and non-event rates
                total_events = target.sum()
                total_non_events = (1 - target).sum()

                # Avoid division by zero
                if category_events == 0:
                    category_events = 0.5  # Smoothing
                if category_non_events == 0:
                    category_non_events = 0.5  # Smoothing
                if total_events == 0:
                    total_events = 0.5
                if total_non_events == 0:
                    total_non_events = 0.5

                # Calculate WoE
                pct_non_events = category_non_events / total_non_events
                pct_events = category_events / total_events

                if pct_events > 0 and pct_non_events > 0:
                    woe = np.log(pct_non_events / pct_events)
                else:
                    woe = 0.0

                woe_dict[category] = woe

            self.woe_dicts_[feature] = woe_dict
            logger.info(f"Calculated WoE for {len(woe_dict)} categories in '{feature}'")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using WoE values

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with WoE-transformed features
        """
        X_transformed = X.copy()

        for feature in self.categorical_features:
            if feature not in X.columns or feature not in self.woe_dicts_:
                continue

            if feature == self.target_col:
                continue

            woe_dict = self.woe_dicts_[feature]
            # Replace categories with WoE values, default to 0 for unseen categories
            X_transformed[feature + "_woe"] = (
                X_transformed[feature].map(woe_dict).fillna(0.0)
            )

        return X_transformed

    def calculate_iv(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate Information Value (IV) for each transformed feature

        Args:
            X: Input DataFrame
            y: Target Series

        Returns:
            Dictionary mapping feature names to IV values
        """
        if y is None and self.target_col is None:
            return {}

        target = y if y is not None else X[self.target_col]

        iv_dict = {}
        for feature in self.categorical_features:
            if feature not in X.columns or feature not in self.woe_dicts_:
                continue

            iv = 0.0
            woe_dict = self.woe_dicts_[feature]

            for category in X[feature].unique():
                if pd.isna(category) or category not in woe_dict:
                    continue

                category_mask = X[feature] == category
                category_events = target[category_mask].sum()
                category_non_events = (1 - target[category_mask]).sum()

                total_events = target.sum()
                total_non_events = (1 - target).sum()

                if total_events > 0 and total_non_events > 0:
                    pct_non_events = category_non_events / total_non_events
                    pct_events = category_events / total_events

                    if pct_events > 0 and pct_non_events > 0:
                        woe = woe_dict[category]
                        iv += (pct_non_events - pct_events) * woe

            iv_dict[feature] = iv

        return iv_dict


class DataProcessor:
    """Class for processing credit risk data with comprehensive feature engineering"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        target_column: str = "FraudResult",
        use_woe: bool = True,
        use_label_encoding: bool = False,
    ):
        """
        Initialize DataProcessor

        Args:
            data_path: Path to the data directory
            target_column: Name of the target column
            use_woe: Whether to apply WoE transformations
            use_label_encoding: Whether to use label encoding for
                high-cardinality categoricals
        """
        self.data_path = Path(data_path) if data_path else Path("data/raw")
        self.processed_data = None
        self.target_column = target_column
        self.use_woe = use_woe
        self.use_label_encoding = use_label_encoding
        self.feature_pipeline_ = None
        self.woe_transformer_ = None
        self.feature_names_ = None

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            filename: Name of the CSV file

        Returns:
            DataFrame with loaded data
        """
        file_path = self.data_path / filename
        logger.info(f"Loading data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        return df

    def _identify_feature_types(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify numerical, categorical, and ID columns

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (numerical_cols, categorical_cols, id_cols)
        """
        # ID columns (to be excluded from transformations)
        id_cols = [
            col
            for col in df.columns
            if col.endswith("Id") or col == "TransactionId" or col == "BatchId"
        ]

        # Numerical columns (exclude target and IDs)
        numerical_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col != self.target_column and col not in id_cols
        ]

        # Categorical columns (object type, exclude IDs and target)
        categorical_cols = [
            col
            for col in df.select_dtypes(include=["object"]).columns
            if col not in id_cols
            and col != self.target_column
            and col != "TransactionStartTime"  # Handled separately
        ]

        # Add timestamp column if present
        timestamp_cols = []
        if "TransactionStartTime" in df.columns:
            timestamp_cols = ["TransactionStartTime"]

        logger.info(
            f"Identified {len(numerical_cols)} numerical, "
            f"{len(categorical_cols)} categorical features"
        )
        return numerical_cols, categorical_cols, id_cols

    def _create_feature_pipeline(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        y: Optional[pd.Series] = None,
    ) -> Pipeline:
        """
        Create sklearn Pipeline for feature engineering

        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            y: Target Series (for WoE calculation)

        Returns:
            sklearn Pipeline
        """
        transformers = []

        # Numerical feature preprocessing: imputation + scaling
        if numerical_cols:
            numerical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numerical", numerical_pipeline, numerical_cols))

        # Categorical feature preprocessing
        if categorical_cols:
            # Identify high-cardinality categoricals for label encoding
            high_cardinality = []
            low_cardinality = []

            # This will be determined during fit based on unique values
            # For now, use one-hot encoding for all
            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            drop="first",
                            sparse_output=False,
                            handle_unknown="ignore",
                        ),
                    ),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",  # Pass through other columns (IDs, etc.)
            verbose_feature_names_out=False,
        )

        return Pipeline([("preprocessor", preprocessor)])

    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        fit_pipeline: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for model training using comprehensive feature engineering
        pipeline

        Args:
            df: Input DataFrame
            target_column: Name of target column (optional, uses self.target_column
                if not provided for backward compatibility)
            fit_pipeline: Whether to fit the pipeline (True for training,
                False for inference)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Starting comprehensive feature engineering pipeline...")

        # Use provided target_column or fall back to instance variable
        target_col = (
            target_column if target_column is not None else self.target_column
        )

        # Separate target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Step 1: Create customer-level aggregate features
        logger.info("Step 1: Creating customer-level aggregate features...")
        aggregate_transformer = CustomerAggregateFeatures(
            customer_id_col="CustomerId", amount_col="Amount"
        )
        aggregate_transformer.fit(X)
        X = aggregate_transformer.transform(X)

        # Step 2: Extract temporal features
        logger.info("Step 2: Extracting temporal features...")
        temporal_transformer = TemporalFeatureExtractor(
            timestamp_col="TransactionStartTime"
        )
        X = temporal_transformer.transform(X)

        # Step 3: Apply WoE transformations if enabled
        if self.use_woe:
            logger.info("Step 3: Applying WoE transformations...")
            numerical_cols, categorical_cols, id_cols = self._identify_feature_types(X)

            # Select suitable categorical features for WoE
            # (exclude high-cardinality IDs)
            woe_categoricals = [
                col
                for col in categorical_cols
                if col not in id_cols
                and X[col].nunique() < 50  # Limit to reasonable cardinality
            ]

            if woe_categoricals:
                if fit_pipeline:
                    # Create and fit new transformer during training
                    self.woe_transformer_ = WoETransformer(
                        categorical_features=woe_categoricals, target_col=None
                    )
                    self.woe_transformer_.fit(X, y)
                    X = self.woe_transformer_.transform(X)
                    # Calculate and log IV values
                    iv_dict = self.woe_transformer_.calculate_iv(X, y)
                    logger.info("Information Value (IV) for WoE features:")
                    for feature, iv in iv_dict.items():
                        logger.info(f"  {feature}: {iv:.4f}")
                else:
                    # Use existing fitted transformer for inference
                    if self.woe_transformer_ is None:
                        logger.warning(
                            "WoE transformer not fitted. Skipping WoE transformation."
                        )
                    else:
                        logger.info(
                            "Using pre-fitted WoE transformer for transformation"
                        )
                        X = self.woe_transformer_.transform(X)

        # Step 4: Identify final feature sets after feature engineering
        numerical_cols, categorical_cols, id_cols = self._identify_feature_types(X)

        # Get list of WoE-transformed features (to exclude from categorical encoding)
        woe_transformed_features = []
        if self.use_woe and self.woe_transformer_ is not None:
            woe_transformed_features = self.woe_transformer_.categorical_features

        # Remove ID columns, WoE-transformed columns (they're now numerical
        # via WoE), and original categoricals that were WoE-transformed
        categorical_cols = [
            col
            for col in categorical_cols
            if not col.endswith("_woe") 
            and col not in id_cols
            and col not in woe_transformed_features
            # Exclude original categoricals that were WoE-transformed
        ]

        # Add WoE columns and temporal features to numerical
        woe_cols = [col for col in X.columns if col.endswith("_woe")]
        temporal_cols = [
            col for col in X.columns if col.startswith("TransactionStartTime_")
        ]
        numerical_cols = numerical_cols + woe_cols + temporal_cols

        # Step 5: Create and fit/transform using sklearn pipeline
        logger.info("Step 4: Applying sklearn preprocessing pipeline...")
        if fit_pipeline or self.feature_pipeline_ is None:
            self.feature_pipeline_ = self._create_feature_pipeline(
                numerical_cols, categorical_cols, y
            )
            self.feature_pipeline_.fit(X, y)

        # Transform data
        X_transformed = self.feature_pipeline_.transform(X)

        # Convert to DataFrame with feature names
        feature_names = (
            self.feature_pipeline_.named_steps["preprocessor"].get_feature_names_out()
        )
        X_transformed_df = pd.DataFrame(
            X_transformed, columns=feature_names, index=X.index
        )

        # Store feature names
        self.feature_names_ = list(X_transformed_df.columns)

        logger.info(
            f"Feature engineering completed: {X_transformed_df.shape[0]} samples, "
            f"{X_transformed_df.shape[1]} features"
        )

        return X_transformed_df, y

    def save_processed_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str,
        output_path: Optional[str] = None,
    ):
        """
        Save processed data to CSV

        Args:
            X: Features DataFrame
            y: Target Series
            filename: Output filename
            output_path: Path to save the file (default: data/processed)
        """
        if output_path is None:
            output_path = Path("data/processed")
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Combine features and target
        processed_df = X.copy()
        processed_df[self.target_column] = y

        file_path = output_path / filename
        processed_df.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to {file_path}")

    # Maintain backward compatibility
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset (maintained for backward compatibility)
        Note: Missing value handling is now part of the pipeline

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.warning(
            "clean_data() is deprecated. Missing value handling is now part "
            "of the preprocessing pipeline."
        )
        df_clean = df.copy()

        # Handle missing values for backward compatibility
        # For numerical columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_value = df_clean[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else "Unknown"
                df_clean[col].fillna(fill_value, inplace=True)

        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features (maintained for backward compatibility)
        Note: Feature engineering is now part of the preprocessing pipeline

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.warning(
            "engineer_features() is deprecated. Feature engineering is now "
            "part of the preprocessing pipeline."
        )
        return df.copy()
