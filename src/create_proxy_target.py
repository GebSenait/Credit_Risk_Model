"""
Script to create proxy target variable using RFM analysis and clustering.
This script implements Task 4 - Proxy Target Variable Engineering.
"""

import logging
from pathlib import Path

from src.data_processing import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to create proxy target variable and save processed dataset"""

    # Initialize data processor
    processor = DataProcessor(
        data_path="data/raw",
        target_column="is_high_risk",  # Will be created by RFM analysis
        use_woe=False,  # Disable WoE for now since we don't have target yet
    )

    # Load raw data
    logger.info("Loading raw data...")
    df = processor.load_data("credit_data.csv")

    # Create proxy target variable using RFM analysis
    logger.info("\nCreating proxy target variable...")
    df_with_target = processor.create_proxy_target_variable(
        df=df,
        snapshot_date=None,  # Will use max transaction date
        customer_id_col="CustomerId",
        transaction_date_col="TransactionStartTime",
        amount_col="Amount",
        n_clusters=3,
        random_state=42,
    )

    # Save the dataset with proxy target variable
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "credit_data_with_proxy_target.csv"
    df_with_target.to_csv(output_file, index=False)

    logger.info(f"\n✓ Processed dataset with proxy target saved to: {output_file}")
    logger.info(f"  Dataset shape: {df_with_target.shape}")
    logger.info(
        f"  High-risk customers: {df_with_target['is_high_risk'].sum()} "
        f"({df_with_target['is_high_risk'].mean() * 100:.2f}%)"
    )

    return df_with_target


if __name__ == "__main__":
    main()
