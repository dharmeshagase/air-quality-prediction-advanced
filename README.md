# Air Quality Prediction with Ensemble Modeling

## Overview

This project predicts air quality using an ensemble approach that combines the strengths of multiple machine learning models. Our chosen strategy is **Option B: Ensemble Modeling**. By integrating several models, we improve prediction robustness and accuracy, capturing a wide range of patterns in the data.

---

## Ensemble Modeling Strategy

### Individual Models

We train and optimize the following models:

- **Linear Regression:**  
  Provides a simple baseline using a linear relationship between predictors and the target variable. Its straightforward nature helps in understanding fundamental trends.

- **Random Forest (with Hyperparameter Tuning):**  
  An ensemble of decision trees, Random Forest captures complex non-linear relationships and interactions. We apply hyperparameter tuning using RandomizedSearchCV to refine its performance by finding the best combination of parameters (such as the number of trees, max depth, etc.).

- **XGBoost:**  
  A gradient boosting algorithm that efficiently captures complex interactions in tabular data. XGBoost often yields high performance due to its boosting framework.

- **LSTM Model:**  
  A deep learning model tailored for time-series data. LSTMs capture temporal dependencies, enabling the model to learn from historical data points for future prediction.

### Weighted Average Ensemble

Instead of selecting a single model, our ensemble aggregates the predictions of all four models using a weighted average. The weight for each model is based on its Mean Absolute Error (MAE) on validation data. The calculation follows this process:

1. **Calculate Inverse Error:**  
   For each model, compute the reciprocal of its MAE.

2. **Normalize Weights:**  
   Sum the inverse errors and divide each model’s inverse error by this total so that the weights add up to 1.

3. **Aggregate Predictions:**  
   Final prediction is given by:

  **Ensemble Prediction** = Σ (Weightᵢ × Predictionᵢ)

where  
  **Weightᵢ = (1 / MAEᵢ) / ∑ (1 / MAEⱼ)**

#### Example:
If the MAEs for three models are:  
- Linear Regression: MAE = 2.0  
- Random Forest: MAE = 1.0  
- XGBoost: MAE = 1.5  

Then their inverse errors are:  
- LR: 1/2.0 = 0.5  
- RF: 1/1.0 = 1.0  
- XGBoost: 1/1.5 ≈ 0.67  

Total = 0.5 + 1.0 + 0.67 = 2.17

Normalized weights:  
- LR: 0.5 / 2.17 ≈ 0.23  
- RF: 1.0 / 2.17 ≈ 0.46  
- XGBoost: 0.67 / 2.17 ≈ 0.31  

The ensemble prediction for a new input is then computed as:  
  0.23 × (LR prediction) + 0.46 × (RF prediction) + 0.31 × (XGBoost prediction)

This approach allows the ensemble to emphasize models with superior performance (lower MAE).

---

## Model Experimentation and MLflow Experiments

### Experimentation Process
- **Individual Training:**  
  Each model (Linear Regression, Random Forest, XGBoost, LSTM) is trained on the engineered air quality dataset.
  
- **Hyperparameter Tuning:**  
  The Random Forest model undergoes hyperparameter tuning via RandomizedSearchCV using a reduced parameter grid to optimize its performance.
  
- **MLflow Tracking:**  
  We use MLflow to log parameters, metrics, and artifacts for each model.  
  - **Parent Run:** Contains overall experiment parameters (target column, dataset size, etc.).
  - **Nested Runs:** Capture individual model experiments (including tuning for Random Forest, as well as runs for Linear Regression, XGBoost, and LSTM).
  - **Metrics Logged:** For each model, we log MAE and RMSE. The ensemble’s performance is also logged.
  
- **Ensemble Creation:**  
  Model predictions are aggregated using the weighted average approach described above. The final ensemble metrics are compared to those of the individual models to evaluate the benefit of the ensemble.

## Setup Instructions

### Clone the Repository:
Clone the repo by using git clone https://github.com/dharmeshagase/air-quality-prediction-advanced

### Create and Activate Virtual Environment:
python -m venv venv

Activate on Linux/Mac:
source venv/bin/activate

Activate on Windows:
venv\Scripts\activate

### Viewing Results in MLflow
- Start MLflow UI using:
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  Open http://127.0.0.1:5000 in your browser
  
- Run the python code
- 
  python airquality_prediction_mlflow.py
  
  This runs the full workflow: feature engineering, individual model training (with hyperparameter tuning for Random Forest), ensemble formation, and MLflow logging of all experiments
