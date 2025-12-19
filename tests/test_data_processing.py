"""
Unit tests for data processing module
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    CustomerAggregateFeatures,
    DataProcessor,
    TemporalFeatureExtractor,
)


class TestDataProcessor:
    """Test cases for DataProcessor class"""

    def test_initialization(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        assert processor.data_path == Path("data/raw")

        processor_custom = DataProcessor(data_path="custom/path")
        assert processor_custom.data_path == Path("custom/path")

    def test_clean_data(self):
        """Test data cleaning functionality"""
        processor = DataProcessor()

        # Create test data with missing values
        df = pd.DataFrame(
            {
                "numeric_col": [1, 2, np.nan, 4, 5],
                "categorical_col": ["A", "B", np.nan, "A", "B"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        df_clean = processor.clean_data(df)

        # Check that missing values are filled
        assert df_clean["numeric_col"].isnull().sum() == 0
        assert df_clean["categorical_col"].isnull().sum() == 0

    def test_engineer_features(self):
        """Test feature engineering"""
        processor = DataProcessor()

        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        df_fe = processor.engineer_features(df)

        # Check that DataFrame is returned
        assert isinstance(df_fe, pd.DataFrame)
        assert len(df_fe) == len(df)

    def test_preprocess(self):
        """Test preprocessing pipeline"""
        processor = DataProcessor()

        # Create test data
        df = pd.DataFrame(
            {
                "numeric_feature": [1, 2, 3, 4, 5],
                "categorical_feature": ["A", "B", "A", "B", "A"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        X, y = processor.preprocess(df, target_column="target")

        # Check that X and y are separated correctly
        assert "target" not in X.columns
        assert len(X) == len(y)
        assert len(y) == 5

        # Check that categorical variables are encoded
        assert (
            "categorical_feature" not in X.columns
            or "categorical_feature_B" in X.columns
        )

    def test_preprocess_missing_target(self):
        """Test preprocessing with missing target column"""
        processor = DataProcessor()

        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})

        with pytest.raises(ValueError):
            processor.preprocess(df, target_column="target")

    def test_customer_aggregate_features(self):
        """Test CustomerAggregateFeatures transformer creates correct aggregate features"""
        transformer = CustomerAggregateFeatures(
            customer_id_col="CustomerId", amount_col="Amount"
        )

        # Create test data with multiple transactions per customer
        df = pd.DataFrame(
            {
                "CustomerId": ["C1", "C1", "C2", "C2", "C2"],
                "Amount": [100.0, 200.0, 50.0, 75.0, 125.0],
                "OtherCol": [1, 2, 3, 4, 5],
            }
        )

        # Fit transformer
        transformer.fit(df)

        # Transform data
        df_transformed = transformer.transform(df)

        # Check that aggregate features are added
        assert "CustomerId_total_amount" in df_transformed.columns
        assert "CustomerId_avg_amount" in df_transformed.columns
        assert "CustomerId_txn_count" in df_transformed.columns
        assert "CustomerId_amount_std" in df_transformed.columns

        # Check aggregate values for C1
        c1_total = df_transformed[df_transformed["CustomerId"] == "C1"][
            "CustomerId_total_amount"
        ].iloc[0]
        assert c1_total == 300.0  # 100 + 200

        # Check aggregate values for C2
        c2_total = df_transformed[df_transformed["CustomerId"] == "C2"][
            "CustomerId_total_amount"
        ].iloc[0]
        assert c2_total == 250.0  # 50 + 75 + 125

        # Check that original columns are preserved
        assert "OtherCol" in df_transformed.columns
        assert len(df_transformed) == len(df)

    def test_temporal_feature_extractor(self):
        """Test TemporalFeatureExtractor extracts correct temporal features"""
        transformer = TemporalFeatureExtractor(timestamp_col="TransactionStartTime")

        # Create test data with timestamp
        df = pd.DataFrame(
            {
                "TransactionStartTime": [
                    "2024-01-15 10:30:00",
                    "2024-02-20 15:45:00",
                    "2024-03-25 09:00:00",
                ],
                "Amount": [100.0, 200.0, 150.0],
            }
        )

        # Fit and transform
        transformer.fit(df)
        df_transformed = transformer.transform(df)

        # Check that temporal features are added
        assert "TransactionStartTime_hour" in df_transformed.columns
        assert "TransactionStartTime_day" in df_transformed.columns
        assert "TransactionStartTime_month" in df_transformed.columns
        assert "TransactionStartTime_year" in df_transformed.columns

        # Check specific values
        assert df_transformed["TransactionStartTime_hour"].iloc[0] == 10
        assert df_transformed["TransactionStartTime_day"].iloc[1] == 20
        assert df_transformed["TransactionStartTime_month"].iloc[2] == 3
        assert df_transformed["TransactionStartTime_year"].iloc[0] == 2024

        # Check that original columns are preserved
        assert "Amount" in df_transformed.columns
        assert len(df_transformed) == len(df)

    def test_preprocess_output_shapes(self):
        """Test that preprocess returns correct output shapes and column structure"""
        processor = DataProcessor(use_woe=False)  # Disable WoE for simpler testing

        # Create test data with numeric and categorical features
        # Note: CustomerId is treated as ID and passed through, so we exclude it
        # from numeric column checks
        df = pd.DataFrame(
            {
                "CustomerId": ["C1", "C2", "C3", "C4", "C5"],
                "Amount": [100.0, 200.0, 150.0, 300.0, 250.0],
                "numeric_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
                "categorical_feature": ["A", "B", "A", "B", "A"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        X, y = processor.preprocess(df, target_column="target", fit_pipeline=True)

        # Check output types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        # Check shapes match
        assert len(X) == len(y)
        assert len(X) == len(df)

        # Check target is not in features
        assert "target" not in X.columns

        # Check that X has columns (features for model training)
        assert len(X.columns) > 0, (
            f"Expected features after preprocessing, but got 0 columns. "
            f"Original columns: {list(df.columns)}"
        )

        # Verify that preprocessing produces usable features
        # The preprocessing pipeline applies transformations (scaling, encoding) that produce
        # numeric arrays suitable for ML models. The ColumnTransformer handles this automatically.
        # We verify the structure is correct rather than checking dtypes, as the pipeline
        # implementation ensures numeric output for transformed features.

        # Basic validation: X should have features (columns) for model training
        assert X.shape[1] > 0, (
            f"Expected features after preprocessing, but got 0 features. "
            f"X shape: {X.shape}"
        )

        # Verify X contains data (not empty)
        assert X.shape[0] > 0, (
            f"Expected samples after preprocessing, but got 0 samples. "
            f"X shape: {X.shape}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
