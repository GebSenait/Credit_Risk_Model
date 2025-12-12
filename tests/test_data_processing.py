"""
Unit tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import DataProcessor


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
