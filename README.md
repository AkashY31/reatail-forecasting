# Retail Demand Forecasting & Inventory Prediction

Production-grade, end-to-end ML system for retail supply chain demand forecasting.

## Project Structure

```
retail_forecasting/
├── configs/
│   └── config.py              # Central config — all paths & params
├── src/
│   ├── data/
│   │   ├── data_generator.py  # Phase 1: Synthetic data generation
│   │   └── eda.py             # Phase 1: Full EDA pipeline
│   ├── features/
│   │   └── preprocessing.py   # Phase 2: Preprocessing + feature engineering
│   ├── models/
│   │   ├── statistical_models.py  # Phase 3: ARIMA / SARIMA / SARIMAX
│   │   └── ml_models.py           # Phase 4: LR / RF / XGBoost / LSTM
│   ├── evaluation/
│   │   ├── metrics.py         # RMSE / MAE / MAPE
│   │   └── ensemble.py        # Phase 5: Ensemble + comparison table
│   ├── monitoring/
│   │   └── drift_detection.py # Phase 6: PSI / KS / drift simulation
│   └── serving/
│       └── dashboard.py       # Phase 7: Streamlit dashboard
├── run_pipeline.py            # Master runner — all phases
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline
```bash
python run_pipeline.py
```

### 3. Run specific phase
```bash
python run_pipeline.py --phase 1   # EDA only
python run_pipeline.py --phase 4   # ML models only
python run_pipeline.py --phase 4 --n_trials 30   # More HPT trials
```

### 4. Launch dashboard
```bash
streamlit run src/serving/dashboard.py
```

### 5. Run in Docker
```bash
docker build -t retail-forecast .
docker run -p 8501:8501 retail-forecast
```

## Cloud Deployment

| Component | AWS | Azure | GCP |
|---|---|---|---|
| Data storage | S3 + Glue | ADLS Gen2 | GCS + BigQuery |
| Processing | EMR / Glue | Databricks | Dataproc |
| Training | SageMaker | Azure ML | Vertex AI |
| Serving | SageMaker Endpoints | Azure ML Endpoints | Vertex AI Prediction |
| Dashboard | ECS Fargate | Container Apps | Cloud Run |
| Monitoring | SageMaker Monitor | Azure ML Drift | Vertex AI Monitoring |

Set environment variables to switch between local and cloud:
```bash
export DATA_PATH=s3://my-bucket/data        # AWS
export MLFLOW_URI=https://my-mlflow-server  # MLflow tracking
```

## Models Implemented

| Model | Type | Library |
|---|---|---|
| Linear Regression (Ridge) | Baseline | scikit-learn |
| Random Forest | Ensemble | scikit-learn + Optuna |
| XGBoost | Gradient Boosting | xgboost + Optuna |
| LSTM | Deep Learning | TensorFlow/Keras |
| ARIMA | Statistical | statsmodels |
| SARIMA | Statistical + Seasonal | statsmodels |
| SARIMAX | Statistical + Exogenous | statsmodels |
| Weighted Ensemble | Ensemble | Custom |
| Stacking Ensemble | Meta-learner | scikit-learn |

## Evaluation Metrics
- **RMSE** — Root Mean Square Error
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error

## Key Features
- Temporal train/val/test split (no data leakage)
- Optuna HPT (TPE sampler — smarter than grid search)
- MLflow experiment tracking
- PSI + KS drift detection
- Drift simulation (sudden / gradual / seasonal shift)
- Cloud-ready config via environment variables
- Docker containerisation
- Streamlit dashboard with filters, KPIs, drift monitor
