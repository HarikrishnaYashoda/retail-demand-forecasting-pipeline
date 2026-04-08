# retail-demand-forecasting-pipeline
A comprehensive end-to-end machine learning pipeline for forecasting retail product demand, incorporating time series analysis, deep learning, pricing optimization, NLP-based sentiment analysis, and recommendation systems.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Data](#data)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Phases](#pipeline-phases)
- [Models & Techniques](#models--techniques)
- [Key Results](#key-results)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

This project builds a full retail demand forecasting system that helps businesses predict future product demand, optimize pricing strategies, and personalize customer recommendations. The pipeline spans data ingestion, cleaning, feature engineering, model training, evaluation, and experiment tracking using MLflow.

---

## Business Problem

Retailers face significant challenges in inventory management. Overstocking leads to waste and increased holding costs, while understocking results in lost sales and poor customer experience. This project addresses these challenges by:

- Forecasting demand at the product level using historical sales data
- Analyzing the impact of pricing and promotions on demand
- Leveraging customer review sentiment as a leading indicator of demand shifts
- Building a recommendation engine to support cross-selling strategies
- Validating business decisions through rigorous A/B test evaluation

---

## Data

This project uses **13 CSV files** containing retail transaction data, product information, customer demographics, pricing history, and customer reviews.

**Download the raw datasets from [Google Drive link / Kaggle link — update this] and place them inside the `data/raw/` folder.**

The raw data files include:

| File | Description |
|------|-------------|
| `sales_transactions.csv` | Historical sales records with timestamps |
| `product_catalog.csv` | Product details, categories, and attributes |
| `customer_demographics.csv` | Customer age, location, and segments |
| `pricing_history.csv` | Historical pricing and discount data |
| `inventory_levels.csv` | Stock levels over time |
| `promotions.csv` | Promotional campaign details |
| `customer_reviews.csv` | Text reviews and star ratings |
| *(update with your actual file names)* | |

> **Note:** Raw data is excluded from this repository via `.gitignore`. See instructions above to obtain the datasets.

---

## Project Structure

```
retail-demand-forecasting-pipeline/
│
├── README.md                    ← Project documentation (you are here)
├── requirements.txt             ← Python dependencies
├── .gitignore                   ← Files excluded from version control
│
├── data/
│   ├── raw/                     ← Original 13 CSVs (gitignored)
│   ├── processed/               ← Cleaned and merged datasets
│   └── features/                ← Engineered feature store for modeling
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_modeling_arima_baseline.ipynb
│   ├── 04_modeling_lstm.ipynb
│   ├── 05_modeling_xgboost_pricing.ipynb
│   ├── 06_nlp_sentiment.ipynb
│   ├── 07_recommendation_system.ipynb
│   └── 08_ab_test_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← Data ingestion and validation
│   ├── feature_engineering.py   ← Feature transformations and creation
│   ├── model_training.py        ← Training routines for all models
│   └── evaluation.py            ← Metrics computation and comparison
│
├── mlflow_experiments/          ← MLflow experiment tracking logs
└── reports/
    └── figures/                 ← Saved visualizations for documentation
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/HarikrishnaYashoda/retail-demand-forecasting-pipeline.git
cd retail-demand-forecasting-pipeline

# Install dependencies
pip install -r requirements.txt

# Place your raw data files in data/raw/
# Then open notebooks in order starting from 01_data_understanding.ipynb
```

### Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, tensorflow/keras
- statsmodels (ARIMA)
- nltk (sentiment analysis)
- mlflow (experiment tracking)
- matplotlib, seaborn (visualization)

---

## Pipeline Phases

### Phase 1 — Data Understanding
Exploratory data analysis across all 13 datasets. Identifies distributions, missing values, correlations, seasonal patterns, and initial hypotheses about demand drivers.

### Phase 2 — Data Preparation
Cleans, merges, and transforms raw data. Handles missing values, corrects data types, removes duplicates, and produces analysis-ready datasets saved to `data/processed/`.

### Phase 3 — Feature Engineering
Creates predictive features including lag variables, rolling averages, day-of-week/month indicators, price elasticity metrics, promotional flags, and sentiment scores. Final feature matrix saved to `data/features/`.

### Phase 4 — Modeling
Trains and evaluates multiple forecasting approaches:

| Model | Purpose |
|-------|---------|
| ARIMA | Statistical baseline for time series forecasting |
| LSTM | Deep learning model to capture nonlinear temporal patterns |
| XGBoost | Gradient boosting with pricing and promotional features |

### Phase 5 — Advanced Analytics
- **NLP Sentiment Analysis:** Extracts sentiment from customer reviews to use as a demand signal
- **Recommendation System:** Collaborative/content-based filtering for cross-sell opportunities
- **A/B Test Evaluation:** Statistical validation of pricing or recommendation strategies

### Phase 6 — Production Code
Refactors notebook logic into modular, reusable Python code in the `src/` directory.

---

## Models & Techniques

| Technique | Library | Notebook |
|-----------|---------|----------|
| ARIMA / SARIMA | statsmodels | `03_modeling_arima_baseline.ipynb` |
| LSTM Neural Network | TensorFlow / Keras | `04_modeling_lstm.ipynb` |
| XGBoost Regressor | xgboost | `05_modeling_xgboost_pricing.ipynb` |
| Sentiment Analysis | NLTK / VADER | `06_nlp_sentiment.ipynb` |
| Collaborative Filtering | scikit-learn | `07_recommendation_system.ipynb` |
| Hypothesis Testing | scipy | `08_ab_test_evaluation.ipynb` |

---

## Key Results

*(Update this section as you complete each phase)*

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| ARIMA Baseline | — | — | — |
| LSTM | — | — | — |
| XGBoost | — | — | — |

### Sample Visualizations

*(Add your saved charts from `reports/figures/` here)*

```
![Forecast vs Actual](reports/figures/forecast_vs_actual.png)
![Feature Importance](reports/figures/feature_importance.png)
```

---

## Usage

```python
# Example: Load data and generate features
from src.data_loader import load_raw_data
from src.feature_engineering import build_feature_matrix
from src.model_training import train_xgboost
from src.evaluation import evaluate_model

# Load and prepare data
data = load_raw_data("data/raw/")
features = build_feature_matrix(data)

# Train and evaluate
model = train_xgboost(features)
metrics = evaluate_model(model, features)
print(metrics)
```

---

## Future Improvements

- Deploy the model as a REST API using FastAPI or Flask
- Add real-time demand monitoring dashboard with Streamlit
- Implement automated retraining pipeline with Airflow
- Incorporate external data sources (weather, holidays, economic indicators)
- Expand A/B testing framework for multi-variant experiments
- Add Docker containerization for reproducible environments

---

## Author

**Harikrishna Yashoda**
GitHub: [@HarikrishnaYashoda](https://github.com/HarikrishnaYashoda)

---

*This project demonstrates end-to-end machine learning engineering skills including data wrangling, feature engineering, time series forecasting, deep learning, NLP, recommendation systems, statistical testing, and MLOps practices.*

