# Credit Risk Model

A production-ready machine learning project for credit risk prediction, featuring data processing, model training, prediction APIs, and comprehensive testing.

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
   - Ensure your data has a target column (default: 'target')

## 📊 Usage

### 1. Exploratory Data Analysis

Open and run the Jupyter notebook for EDA:
```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Data Processing

Process your raw data:
```python
from src.data_processing import DataProcessor

processor = DataProcessor(data_path="data/raw")
df = processor.load_data("your_data.csv")
X, y = processor.preprocess(df, target_column='target')
```

### 3. Model Training

Train the credit risk model:
```bash
python src/train.py
```

Or use programmatically:
```python
from src.train import ModelTrainer
from src.data_processing import DataProcessor

# Load and preprocess data
processor = DataProcessor()
df = processor.load_data("credit_data.csv")
X, y = processor.preprocess(df, target_column='target')

# Train model
trainer = ModelTrainer(model_type='random_forest', n_estimators=100)
metrics = trainer.train(X, y)

# Save model
trainer.save_model("models/credit_risk_model.pkl")
```

### 4. Making Predictions

#### Command Line
```bash
python src/predict.py models/credit_risk_model.pkl data/test_data.csv
```

#### Python API
```python
from src.predict import CreditRiskPredictor

predictor = CreditRiskPredictor("models/credit_risk_model.pkl")
result = predictor.predict_risk_score({
    'feature1': 0.5,
    'feature2': 1.0,
    # ... your features
})

print(f"Prediction: {result['prediction']}")
print(f"Default Probability: {result['default_probability']}")
print(f"Risk Level: {result['risk_level']}")
```

### 5. API Server

Start the FastAPI server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Or use Docker:
```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /predict` - Single prediction endpoint
- `POST /predict/batch` - Batch prediction endpoint
- `GET /model/info` - Model information endpoint

#### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "req_001",
       "feature1": 0.5,
       "feature2": 1.0
     }'
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t credit-risk-model .

# Run the container
docker run -p 8000:8000 credit-risk-model
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 📝 Configuration

### Model Parameters

Edit `src/train.py` to customize model parameters:
- Model type (Random Forest, XGBoost, etc.)
- Hyperparameters (n_estimators, max_depth, etc.)

### API Configuration

Edit `src/api/main.py` to:
- Configure CORS settings
- Adjust model loading path
- Customize API endpoints

### Data Processing

Edit `src/data_processing.py` to:
- Customize data cleaning logic
- Add feature engineering steps
- Modify preprocessing pipeline

## 🔧 Development

### Code Style

This project follows PEP 8 style guidelines. Use the following tools:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/
```

### CI/CD

The project includes GitHub Actions workflows for:
- Automated testing
- Code linting
- Docker image building

## 📈 Model Performance

After training, the model provides the following metrics:
- Accuracy (Training and Test)
- AUC-ROC Score
- Confusion Matrix
- Classification Report

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Scikit-learn for machine learning tools
- FastAPI for the API framework
- All contributors and maintainers

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Note**: Remember to update the feature names in `src/api/pydantic_models.py` and `src/data_processing.py` to match your actual dataset features.

