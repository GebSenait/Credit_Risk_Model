"""
Data Processing Module for Credit Risk Model
Handles data loading, cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Class for processing credit risk data"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataProcessor
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path) if data_path else Path("data/raw")
        self.processed_data = None
        
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
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Handle missing values
        # For numerical columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        df_fe = df.copy()
        
        # Example feature engineering operations
        # Add your specific feature engineering logic here
        
        logger.info("Feature engineering completed")
        return df_fe
    
    def preprocess(self, df: pd.DataFrame, target_column: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for model training
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preprocessing data...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_fe = self.engineer_features(df_clean)
        
        # Separate features and target
        if target_column not in df_fe.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df_fe.drop(columns=[target_column])
        y = df_fe[target_column]
        
        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, output_path: Optional[str] = None):
        """
        Save processed data to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_path: Path to save the file (default: data/processed)
        """
        if output_path is None:
            output_path = Path("data/processed")
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / filename
        
        df.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to {file_path}")

