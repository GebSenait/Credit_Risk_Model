# Credit Risk Model

A production-ready machine learning project for credit risk prediction, featuring comprehensive data analysis, model training, prediction APIs, and automated testing infrastructure.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Project Tasks](#project-tasks)
  - [Task 1: Project Setup & Infrastructure](#task-1-project-setup--infrastructure)
  - [Task 2: Exploratory Data Analysis (EDA)](#task-2-exploratory-data-analysis-eda)
  - [Task 3: Feature Engineering](#task-3-feature-engineering)
  - [Task 4: Proxy Target Variable Engineering](#task-4-proxy-target-variable-engineering)
  - [Task 5: Model Training, Tracking, and Validation](#task-5-model-training-tracking-and-validation)
- [Usage Guide](#usage-guide)
- [Testing](#testing)
- [Deployment](#deployment)
- [Technical Stack](#technical-stack)

---

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for credit risk assessment, focusing on fraud detection in financial transactions. The solution includes standardized project architecture, comprehensive data exploration, and production-ready components for model training, prediction, and API deployment.

**Key Objectives:**
- Establish a standardized, scalable project structure
- Perform thorough exploratory data analysis
- Build and deploy a production-ready credit risk prediction model
- Create RESTful API for real-time predictions

---

## 📁 Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline configuration
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # Data loading, cleaning, and preprocessing
│   ├── train.py                  # Model training module
│   ├── predict.py                # Prediction module
│   └── api/
│       ├── main.py               # FastAPI application
│       └── pydantic_models.py    # API request/response models
├── tests/
│   └── test_data_processing.py   # Unit tests
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker and Docker Compose for containerized deployment

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd credit-risk-model
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place your raw data files in `data/raw/`
   - The dataset should include a target column (`FraudResult` for fraud detection)

---

## 📋 Project Tasks

### Task 1: Project Setup & Infrastructure

#### Problem Statement

Establish a production-ready, standardized project structure that supports the complete machine learning lifecycle—from data processing to model deployment—following industry best practices for maintainability, scalability, and reproducibility.

#### Solution Approach

Implemented a comprehensive project architecture with:
- **Modular Code Structure**: Separated concerns into distinct modules (data processing, training, prediction, API)
- **Standardized Folder Organization**: Clear separation of raw/processed data, source code, tests, and notebooks
- **CI/CD Pipeline**: Automated testing and deployment workflows using GitHub Actions
- **Containerization**: Docker support for consistent deployment across environments
- **API Framework**: FastAPI-based RESTful API for real-time predictions
- **Testing Infrastructure**: Unit tests with pytest for code quality assurance

#### Implementation Details

**Core Modules:**
- `src/data_processing.py`: Handles data loading, cleaning, feature engineering, and preprocessing
- `src/train.py`: Model training pipeline with configurable algorithms and hyperparameters
- `src/predict.py`: Prediction module with risk scoring and probability estimation
- `src/api/main.py`: FastAPI application with health checks and prediction endpoints

**Infrastructure Components:**
- **CI/CD**: Automated testing, linting, and Docker image building via GitHub Actions
- **Docker**: Multi-stage Dockerfile for optimized container images
- **API Endpoints**: 
  - `GET /` - API information
  - `GET /health` - Health check
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions
  - `GET /model/info` - Model metadata

**Testing:**
- Unit tests for data processing functions
- Test coverage reporting
- Automated test execution in CI pipeline

#### Results & Achievements

✅ **Completed Deliverables:**
- Standardized project folder structure following best practices
- Modular, reusable code architecture
- Production-ready API with FastAPI framework
- Docker containerization for deployment
- CI/CD pipeline for automated testing and building
- Comprehensive documentation and code comments

✅ **Key Benefits:**
- **Maintainability**: Clear separation of concerns enables easy updates
- **Scalability**: Modular design supports feature additions without refactoring
- **Reproducibility**: Docker and requirements.txt ensure consistent environments
- **Quality Assurance**: Automated testing prevents regressions
- **Deployment Ready**: Containerized solution for easy production deployment

---

### Task 2: Exploratory Data Analysis (EDA)

#### Problem Statement

Conduct comprehensive exploratory data analysis to understand the credit risk dataset's characteristics, identify data quality issues, discover feature relationships, and generate actionable insights to inform model development and feature engineering strategies.

#### Solution Approach

Developed an automated, comprehensive EDA notebook that systematically analyzes:
1. **Data Overview**: Structure, types, and basic statistics
2. **Distribution Analysis**: Numerical and categorical feature distributions
3. **Correlation Analysis**: Feature relationships and target correlations
4. **Data Quality Assessment**: Missing values and outlier detection
5. **Statistical Insights**: Skewness, kurtosis, and distribution patterns

#### Implementation Details

**EDA Notebook Sections (`notebooks/eda.ipynb`):**

1. **Overview of the Data**
   - Dataset dimensions and structure
   - Column names and data types
   - Sample data preview

2. **Summary Statistics**
   - Descriptive statistics (mean, median, std, quartiles)
   - Skewness and kurtosis for numerical features
   - Categorical feature frequency analysis

3. **Distribution of Numerical Features**
   - Histograms with Kernel Density Estimation (KDE)
   - Skewness calculation and interpretation
   - Distribution pattern identification

4. **Distribution of Categorical Features**
   - Bar charts for category frequencies
   - Top category analysis
   - Category-level variation assessment

5. **Correlation Analysis**
   - Correlation matrix heatmap
   - Feature-to-target correlation analysis
   - Highly correlated feature pair identification (|r| > 0.7)

6. **Missing Value Identification**
   - Comprehensive missing value counts and percentages
   - Missing data pattern analysis
   - Data completeness assessment

7. **Outlier Detection**
   - Box plots for all numerical features
   - IQR (Interquartile Range) method for outlier identification
   - Outlier percentage and bounds calculation

8. **Final Summary - Key Insights**
   - Top 5 most important insights
   - Dataset characteristics summary
   - Actionable recommendations

**Key Features:**
- **Automated Analysis**: Auto-detects numerical/categorical features and target variable
- **Comprehensive Visualizations**: Multiple chart types for better understanding
- **Statistical Rigor**: Advanced statistical measures (skewness, kurtosis, correlation)
- **Dynamic Insights**: Generates context-aware insights based on actual findings

#### Results & Insights

✅ **Analysis Completed:**
- Comprehensive examination of all dataset features
- Identification of data quality issues (missing values, outliers)
- Statistical characterization of feature distributions
- Correlation analysis revealing feature relationships
- Target variable distribution analysis for class imbalance assessment

✅ **Key Insights Generated:**

1. **Dataset Characteristics**
   - Transaction-level data with multiple identifiers for tracking
   - Mix of numerical (amounts, codes) and categorical (categories, channels) features
   - Dataset size and structure suitable for machine learning

2. **Target Variable Distribution**
   - Class imbalance analysis (fraud vs. non-fraud cases)
   - Recommendations for handling imbalanced data (SMOTE, class weights, stratified sampling)
   - Appropriate evaluation metrics identified (Precision, Recall, F1-score, ROC-AUC)

3. **Data Quality Assessment**
   - Missing value patterns identified and quantified
   - Data completeness percentage calculated
   - Recommendations for imputation strategies

4. **Feature Characteristics**
   - Skewness analysis identifying features requiring transformation
   - Distribution patterns revealing normal vs. skewed features
   - Recommendations for normalization and scaling approaches

5. **Outlier Analysis**
   - Outlier presence quantified per feature
   - IQR-based outlier bounds calculated
   - Recommendations for outlier treatment (capping, winsorization, robust scaling)

6. **Feature-Target Relationships**
   - Correlation analysis identifying predictive features
   - Highly correlated feature pairs identified for potential feature engineering
   - Non-linear relationship considerations noted

✅ **Actionable Recommendations:**
- Data preprocessing strategies (missing value imputation, feature transformation)
- Feature engineering opportunities (interaction features, temporal features)
- Model development considerations (class imbalance handling, evaluation metrics)
- Next steps for model training and optimization

**Deliverable:** Complete EDA notebook with visualizations, statistical analysis, and comprehensive insights summary ready for model development phase.

---

### Task 3: Feature Engineering

#### Problem Statement

Transform raw transaction data into a model-ready dataset through systematic feature engineering, ensuring reproducibility, automation, and compatibility with downstream machine learning pipelines.

#### Solution Approach

Implemented a comprehensive, automated feature engineering pipeline using scikit-learn's Pipeline and ColumnTransformer that systematically transforms raw data through multiple stages: aggregate feature creation, temporal extraction, categorical encoding, missing value handling, numerical scaling, and Weight of Evidence (WoE) transformations. The pipeline is designed to be deterministic, reusable between training and inference, and fully integrated with the scikit-learn ecosystem.

#### Implementation Details

**Feature Engineering Pipeline (`src/data_processing.py`):**

The pipeline implements a multi-stage transformation process:

1. **Customer-Level Aggregate Features**
   - Creates customer-level statistical aggregations using `CustomerAggregateFeatures` transformer
   - Generates: total transaction amount, average transaction amount, transaction count, and standard deviation of amounts per customer
   - Provides behavioral insights that capture customer transaction patterns and spending variability
   - Handles new customers during inference by defaulting to zero values

2. **Temporal Feature Extraction**
   - Extracts time-based features from transaction timestamps using `TemporalFeatureExtractor`
   - Derives: transaction hour, day of month, month, and year
   - Enables the model to capture time-dependent patterns and seasonal behaviors
   - Handles datetime parsing with error tolerance for robust processing

3. **Categorical Feature Encoding**
   - Implements dual encoding strategy through sklearn's ColumnTransformer
   - One-Hot Encoding: Applied to categorical features with moderate cardinality, creating binary indicator variables
   - Label Encoding: Available for high-cardinality categoricals (configurable)
   - Maintains encoding consistency between training and inference through fitted transformers
   - Handles unknown categories gracefully during inference

4. **Missing Value Handling**
   - Systematic imputation integrated into the preprocessing pipeline
   - Numerical features: Median imputation using `SimpleImputer`
   - Categorical features: Mode imputation (most frequent category)
   - Ensures data completeness without manual intervention

5. **Numerical Feature Scaling**
   - StandardScaler (Z-score normalization) applied to all numerical features
   - Centers features around zero and scales to unit variance
   - Prevents features with larger magnitudes from dominating model training
   - Ensures balanced contribution from all numerical features

6. **Weight of Evidence (WoE) and Information Value (IV)**
   - Custom `WoETransformer` implements WoE encoding for categorical features
   - WoE transformation: Replaces categorical values with log odds ratio of events vs. non-events
   - IV calculation: Quantifies predictive power of each categorical feature
   - Applied selectively to categoricals with reasonable cardinality (< 50 unique values)
   - Provides interpretable, risk-based encoding particularly valuable for credit risk modeling

**Pipeline Architecture:**

- **Modular Design**: Each transformation stage is implemented as a scikit-learn compatible transformer (BaseEstimator, TransformerMixin)
- **Reproducibility**: All transformations are deterministic and preserve state between training and inference
- **Pipeline Integration**: Uses sklearn Pipeline and ColumnTransformer for seamless chaining
- **Separation of Concerns**: Numerical, categorical, and WoE transformations are clearly separated
- **State Management**: Fitted transformers are stored in the DataProcessor instance for reuse during inference

**Key Features:**
- Fully automated: Single method call (`preprocess()`) handles all transformations
- Deterministic: Same input always produces same output
- Inference-ready: Pipeline state persists for consistent transformation of new data
- Scalable: Efficient handling of large datasets through vectorized operations

#### Results & Insights

✅ **Feature Engineering Completed:**
- Customer-level behavioral features capturing spending patterns and transaction frequency
- Temporal features enabling time-based pattern recognition
- Categorically encoded features in multiple formats (One-Hot, WoE) optimized for different modeling needs
- Numerically scaled features ensuring balanced model learning
- Comprehensive preprocessing pipeline ready for model training

✅ **Key Findings:**

1. **Customer Behavior Insights**
   - Aggregate features reveal spending patterns and transaction frequency distributions
   - Standard deviation of amounts captures customer spending variability, potentially indicative of risk patterns

2. **Temporal Patterns**
   - Extracted time features enable discovery of time-of-day, day-of-month, and seasonal patterns
   - Transaction timing features can capture behavioral anomalies related to fraud or risk

3. **WoE Transformation Value**
   - WoE encoding provides interpretable, risk-calibrated representation of categorical features
   - Information Value metrics help identify the most predictive categorical features
   - WoE features often improve model performance in credit risk applications

4. **Pipeline Efficiency**
   - Single-pass transformation reduces computational overhead
   - Pipeline state management ensures consistency between training and production inference
   - Modular design allows easy addition or modification of transformation steps

5. **Data Quality Enhancement**
   - Systematic missing value handling ensures complete datasets
   - Scaling normalizes feature distributions, improving model convergence
   - Encoding strategies handle categorical variables appropriately for different algorithm types

✅ **Technical Achievements:**
- Production-ready pipeline compatible with sklearn ecosystem
- Fully documented and modular transformer implementations
- Robust error handling and logging throughout the transformation process
- Backward compatibility maintained with existing DataProcessor interface

**Deliverable:** Complete feature engineering pipeline in `src/data_processing.py` with automated transformation capabilities, processed datasets saved to `data/processed/`, and comprehensive documentation of strategies and insights.

---

### Task 4: Proxy Target Variable Engineering

#### Problem Statement

In the absence of an explicit default label, develop a data-driven, reproducible proxy for credit risk using customer engagement behavior. The proxy must be defensible, explainable, and suitable for downstream model training.

#### Solution Approach

Implemented a comprehensive RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering to identify disengaged customers as high-risk proxies. The approach systematically segments customers based on their transaction behavior and assigns binary risk labels through unsupervised learning, ensuring reproducibility and clear interpretability.

#### Implementation Details

**RFM Analysis (`src/data_processing.py` - `create_proxy_target_variable` method):**

The implementation follows a three-stage process:

1. **RFM Metrics Calculation**
   - **Recency**: Days since the customer's most recent transaction, calculated relative to a fixed snapshot date (defaults to maximum transaction date in dataset)
   - **Frequency**: Total number of transactions per customer within the observation window
   - **Monetary**: Aggregate absolute monetary value of all transactions per customer
   - All metrics are computed at the customer level, ensuring one record per customer
   - Snapshot date is explicitly documented and fixed to prevent data leakage

2. **Customer Segmentation via K-Means Clustering**
   - Preprocessing: Log transformation applied to Monetary values to handle skewness, followed by StandardScaler normalization
   - K-Means clustering with 3 clusters (configurable) and fixed `random_state=42` for reproducibility
   - Clustering performed on standardized RFM features to ensure balanced contribution from all dimensions
   - Cluster characteristics analyzed to understand behavioral patterns

3. **High-Risk Cluster Identification**
   - Composite risk score calculated for each cluster based on normalized RFM metrics
   - Risk score formula: `normalized(Recency) + normalized(1-Frequency) + normalized(1-Monetary)`
   - Higher scores indicate more disengaged customers (high recency, low frequency, low monetary value)
   - Cluster with highest risk score identified as high-risk segment
   - Binary target variable `is_high_risk` created: 1 for high-risk cluster, 0 for all others

**Key Design Decisions:**

- **Fixed Snapshot Date**: Ensures deterministic Recency calculations and prevents temporal data leakage
- **Log Transformation**: Applied to Monetary values to handle highly skewed distributions common in transaction data
- **Standardization**: RFM features standardized before clustering to prevent scale bias
- **Reproducibility**: All random operations use fixed seeds (`random_state=42`)
- **Interpretability**: Clear cluster characteristics logged for business understanding

#### Results & Insights

✅ **Proxy Target Variable Created:**

The implementation successfully identifies customer segments with distinct engagement patterns:

- **High-Risk Segment**: Characterized by high recency (long time since last transaction), low frequency (few transactions), and low monetary value (minimal spending)
- **Target Distribution**: Binary classification with clear separation between engaged and disengaged customers
- **Business Logic**: Disengaged customers represent higher credit risk as they demonstrate reduced platform interaction and lower transaction volumes

✅ **Key Findings:**

1. **Customer Segmentation Patterns**
   - Three distinct behavioral clusters emerge from RFM analysis
   - Clear separation between highly engaged, moderately engaged, and disengaged customers
   - Cluster characteristics reveal meaningful business insights about customer lifecycle stages

2. **Risk Proxy Validity**
   - High-risk cluster demonstrates expected characteristics: infrequent transactions, low spending, and long time since last activity
   - Proxy label aligns with business intuition: disengaged customers are more likely to default or churn
   - Distribution provides sufficient positive class examples for model training

3. **Reproducibility & Determinism**
   - Fixed snapshot date ensures consistent Recency calculations across runs
   - Random state seeding guarantees identical cluster assignments
   - All transformations are deterministic and documented

4. **Integration Readiness**
   - Target variable seamlessly integrates with existing feature engineering pipeline
   - Compatible with downstream model training workflows
   - Maintains row-level alignment with original transaction data

✅ **Business Implications:**

- **Risk Identification**: Enables proactive identification of customers at risk of default or churn
- **Resource Allocation**: Supports targeted intervention strategies for high-risk segments
- **Model Training**: Provides labeled data for supervised learning in absence of explicit default labels
- **Monitoring**: RFM metrics can be recalculated periodically to track customer engagement changes

✅ **Known Limitations:**

- **Proxy Nature**: The target is a proxy for credit risk, not actual default labels. Model performance should be validated against real outcomes when available.
- **Temporal Assumptions**: Assumes that disengagement patterns are predictive of credit risk, which may vary by business context.
- **Cluster Stability**: Cluster assignments may shift with new data; periodic recalibration recommended.
- **Feature Dependency**: Proxy quality depends on transaction data quality and completeness.

**Deliverable:** Complete proxy target engineering implementation in `src/data_processing.py` with `create_proxy_target_variable()` method, processed dataset with `is_high_risk` column saved to `data/processed/`, and comprehensive documentation of methodology, results, and limitations.

---

### Task 5: Model Training, Tracking, and Validation

#### Problem Statement

Develop a structured, reproducible machine learning workflow that supports multiple model training, hyperparameter optimization, experiment tracking, and model comparison in a regulated analytics environment. The solution must enable systematic model evaluation, versioning, and selection of the best-performing model for deployment.

#### Solution Approach

Implemented a comprehensive model training pipeline using MLflow for experiment tracking and model registry, supporting multiple algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost) with automated hyperparameter tuning. The pipeline ensures full reproducibility through fixed random states, systematic evaluation using multiple metrics, and automatic selection of the best-performing model based on test set performance.

#### Implementation Details

**Model Training Pipeline (`src/train.py`):**

The training workflow follows a structured, multi-stage process:

1. **Data Loading and Splitting**
   - Loads processed dataset from `data/processed/credit_data_with_proxy_target.csv`
   - Applies feature engineering pipeline with WoE transformations
   - Stratified train-test split (80/20) with fixed `random_state=42` for reproducibility
   - Ensures balanced target distribution across splits

2. **Multiple Model Training**
   - **Logistic Regression**: Grid search over regularization strength (C), penalty types (L1/L2), and solvers
   - **Decision Tree**: Grid search over max depth, min samples split, min samples leaf, and splitting criteria
   - **Random Forest**: Randomized search over ensemble size, tree depth, and feature sampling strategies
   - **XGBoost**: Randomized search over boosting parameters (learning rate, depth, subsample, colsample_bytree)
   - All models use 5-fold stratified cross-validation for hyperparameter selection

3. **Hyperparameter Tuning**
   - Grid Search CV for Logistic Regression and Decision Tree (comprehensive search)
   - Randomized Search CV for Random Forest and XGBoost (efficient search over larger parameter spaces)
   - ROC-AUC used as primary optimization metric for all models
   - Cross-validation ensures robust parameter selection

4. **Comprehensive Model Evaluation**
   - All models evaluated on both training and test sets using:
     - **Accuracy**: Overall prediction correctness
     - **Precision**: Ability to avoid false positives (critical for risk assessment)
     - **Recall**: Ability to identify all high-risk cases
     - **F1 Score**: Harmonic mean of precision and recall
     - **ROC-AUC**: Discriminative power across classification thresholds
   - Metrics calculated consistently across all models for fair comparison

5. **MLflow Experiment Tracking**
   - All experiments logged to MLflow with:
     - Model parameters (including best hyperparameters from tuning)
     - All evaluation metrics (training and test performance)
     - Model artifacts (serialized models for reproducibility)
     - Feature metadata (number of features, feature names)
   - Experiments organized under "credit_risk_modeling" experiment
   - Each model run uniquely identified for easy comparison

6. **Model Comparison and Selection**
   - Automatic comparison of all trained models
   - Best model selected based on test set ROC-AUC (primary metric)
   - Comprehensive performance summary logged for all models
   - Clear identification of best-performing model with visual markers

7. **Model Registry and Versioning**
   - Best model automatically registered in MLflow Model Registry
   - Model versioning enables tracking of model evolution
   - Model URI stored for easy deployment and inference
   - Local model backup saved to `models/credit_risk_model.pkl`

**Key Features:**
- **Reproducibility**: Fixed random states at all levels (data splitting, model training, hyperparameter search)
- **Comprehensive Evaluation**: Multiple metrics provide holistic view of model performance
- **Automated Workflow**: Single command trains all models, compares them, and registers the best
- **Experiment Tracking**: Complete audit trail of all model experiments
- **Model Registry**: Versioned model storage for production deployment

#### Results & Insights

✅ **Model Training Completed:**

Successfully trained and evaluated four model types:
- Logistic Regression: Baseline linear model with regularization
- Decision Tree: Interpretable tree-based model
- Random Forest: Ensemble method combining multiple trees
- XGBoost: Gradient boosting model with advanced regularization

✅ **Key Findings:**

1. **Hyperparameter Impact**
   - Optimal hyperparameters vary significantly across model types
   - Cross-validation critical for robust parameter selection
   - Randomized search efficiently explores large parameter spaces for ensemble methods

2. **Model Performance Comparison**
   - Tree-based models (Random Forest, XGBoost) typically outperform linear models for complex patterns
   - Ensemble methods show better generalization and robustness
   - Trade-offs exist between interpretability (Decision Tree) and performance (XGBoost/Random Forest)

3. **Evaluation Metrics Insights**
   - ROC-AUC provides threshold-independent performance assessment
   - Precision-recall balance critical for risk models (avoiding false positives while capturing true risks)
   - F1 score provides balanced view when class distribution is imbalanced
   - All metrics logged for comprehensive performance assessment

4. **Reproducibility Verification**
   - Fixed random states ensure identical results across runs
   - MLflow tracking enables complete experiment reproducibility
   - All parameters and metrics logged for audit trail

5. **Best Model Selection**
   - Best model identified through systematic comparison
   - Selection criteria: Test set ROC-AUC (primary), balanced with other metrics
   - Model registered in MLflow for production deployment

✅ **Performance Characteristics:**

- **Best Model**: Selected based on test set ROC-AUC performance
- **Model Comparison**: All models evaluated on identical train-test split
- **Cross-Validation**: Hyperparameters optimized using 5-fold stratified CV
- **Generalization**: Test set performance validated model robustness

✅ **Technical Achievements:**

- Production-ready training pipeline with MLflow integration
- Automated hyperparameter tuning for all model types
- Comprehensive evaluation framework with multiple metrics
- Model registry integration for versioned model storage
- Full reproducibility through fixed random states and logged parameters

✅ **Deployment Readiness:**

- Best model registered in MLflow Model Registry
- Model artifacts saved for inference compatibility
- Feature names persisted for preprocessing consistency
- Predict module updated to support both local and MLflow model loading
- Complete experiment tracking enables model comparison and selection

✅ **Known Limitations:**

- **Proxy Target**: Model performance based on proxy target variable (disengagement-based), not actual defaults. Performance should be validated against real outcomes when available.
- **Computational Cost**: Full hyperparameter search can be time-intensive for large datasets; consider reducing search space or using early stopping for production workflows.
- **Class Imbalance**: Current implementation uses stratified sampling; additional techniques (SMOTE, class weights) may improve performance if severe imbalance exists.
- **Feature Dependency**: Model performance depends on quality of feature engineering pipeline; periodic feature review recommended.

**Deliverable:** Complete model training pipeline in `src/train.py` with MLflow experiment tracking, hyperparameter tuning for multiple models, comprehensive evaluation metrics, best model selection and registry, and updated unit tests validating data processing components.

---

## 📊 Usage Guide

### Running the EDA

```bash
jupyter notebook notebooks/eda.ipynb
```

Execute all cells sequentially to perform the complete exploratory data analysis.

### Data Processing

**Creating Proxy Target Variable (Task 4):**

```python
from src.data_processing import DataProcessor

# Initialize processor
processor = DataProcessor(data_path="data/raw")

# Load raw data
df = processor.load_data("credit_data.csv")

# Create proxy target variable using RFM analysis
df_with_target = processor.create_proxy_target_variable(
    df=df,
    snapshot_date=None,  # Uses max transaction date if None
    customer_id_col="CustomerId",
    transaction_date_col="TransactionStartTime",
    amount_col="Amount",
    n_clusters=3,
    random_state=42,
)

# Save dataset with proxy target
df_with_target.to_csv("data/processed/credit_data_with_proxy_target.csv", index=False)
```

**Feature Engineering Pipeline:**

The feature engineering pipeline automatically handles all transformations:

```python
from src.data_processing import DataProcessor

# Initialize processor
processor = DataProcessor(
    data_path="data/raw",
    target_column="is_high_risk",  # Use proxy target from Task 4
    use_woe=True,  # Enable WoE transformations
)

# Load data with proxy target
df = processor.load_data("credit_data_with_proxy_target.csv")

# Preprocess with full feature engineering pipeline
# fit_pipeline=True for training, False for inference
X, y = processor.preprocess(df, fit_pipeline=True)

# Save processed data
processor.save_processed_data(
    X, y, 
    filename="processed_credit_data.csv",
    output_path="data/processed"
)
```

The pipeline automatically:
- Creates customer-level aggregate features
- Extracts temporal features from timestamps
- Applies WoE transformations to categorical features
- Handles missing values
- Encodes categorical variables
- Scales numerical features

### Model Training

**Training with MLflow Tracking (Task 5):**

```bash
python src/train.py
```

This will:
- Load processed data with proxy target variable
- Train multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Perform hyperparameter tuning for each model
- Log all experiments to MLflow
- Compare models and select the best one
- Register best model in MLflow Model Registry
- Save best model locally to `models/credit_risk_model.pkl`

**View MLflow Experiments:**

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiments, compare runs, and access the model registry.

**Programmatic Training:**

```python
from src.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    experiment_name="credit_risk_modeling",
    random_state=42
)

# Load and split data
trainer.load_and_split_data(
    data_path="data/processed/credit_data_with_proxy_target.csv",
    target_column="is_high_risk",
    test_size=0.2,
    use_woe=True
)

# Train all models
trainer.train_all_models()

# Compare and select best model
best_model_name, best_model, best_metrics = trainer.compare_models_and_select_best()

# Register best model
model_uri = trainer.register_best_model(model_name="credit_risk_model")

# Save model locally
trainer.save_model("models/credit_risk_model.pkl")
```

### Making Predictions

**Command Line:**
```bash
python src/predict.py models/credit_risk_model.pkl data/test_data.csv
```

**Python API:**
```python
from src.predict import CreditRiskPredictor

predictor = CreditRiskPredictor("models/credit_risk_model.pkl")
result = predictor.predict_risk_score({
    'feature1': 0.5,
    'feature2': 1.0,
    # ... your features
})
```

### API Server

**Start the FastAPI server:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Or using Docker:**
```bash
docker-compose up
```

**API Endpoints:**
- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /predict` - Single prediction endpoint
- `POST /predict/batch` - Batch prediction endpoint
- `GET /model/info` - Model information endpoint

**Example API Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "req_001",
       "feature1": 0.5,
       "feature2": 1.0
     }'
```

---

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 🐳 Deployment

### Docker

**Build and run:**
```bash
docker build -t credit-risk-model .
docker run -p 8000:8000 credit-risk-model
```

**Using Docker Compose:**
```bash
docker-compose up -d
```

---

## 🛠️ Technical Stack

- **Language**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **API Framework**: FastAPI
- **Visualization**: Matplotlib, Seaborn
- **Testing**: Pytest
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Code Quality**: Black, isort, flake8

---

## 📈 Model Performance

After training, the model provides:
- Accuracy (Training and Test)
- AUC-ROC Score
- Confusion Matrix
- Classification Report

---

## 🔧 Development

### Code Style

This project follows PEP 8 style guidelines:
```bash
black src/ tests/      # Format code
isort src/ tests/      # Sort imports
flake8 src/ tests/     # Lint code
```

### CI/CD

GitHub Actions workflows include:
- Automated testing
- Code linting
- Docker image building

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- Your Name - Initial work

---

## 🙏 Acknowledgments

- Scikit-learn for machine learning tools
- FastAPI for the API framework
- All contributors and maintainers
