# Credit Risk Model

A production-ready machine learning project for credit risk prediction, featuring comprehensive data analysis, model training, prediction APIs, and automated testing infrastructure.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Project Tasks](#project-tasks)
  - [Task 1: Project Setup & Infrastructure](#task-1-project-setup--infrastructure)
  - [Task 2: Exploratory Data Analysis (EDA)](#task-2-exploratory-data-analysis-eda)
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

## 📊 Usage Guide

### Running the EDA

```bash
jupyter notebook notebooks/eda.ipynb
```

Execute all cells sequentially to perform the complete exploratory data analysis.

### Data Processing

```python
from src.data_processing import DataProcessor

processor = DataProcessor(data_path="data/raw")
df = processor.load_data("credit_data.csv")
X, y = processor.preprocess(df, target_column='FraudResult')
```

### Model Training

```bash
python src/train.py
```

Or programmatically:
```python
from src.train import ModelTrainer
from src.data_processing import DataProcessor

processor = DataProcessor()
df = processor.load_data("credit_data.csv")
X, y = processor.preprocess(df, target_column='FraudResult')

trainer = ModelTrainer(model_type='random_forest', n_estimators=100)
metrics = trainer.train(X, y)
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

---

**Note**: Update feature names in `src/api/pydantic_models.py` and `src/data_processing.py` to match your actual dataset features.
