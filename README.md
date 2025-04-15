# air-quality-prediction-advanced
A project to implement a comprehensive MLOps pipeline

# Air Quality Prediction Using Ensemble Modeling
Overview
This project predicts air quality measurements using an ensemble of multiple models:

Linear Regression

Random Forest (with hyperparameter tuning)

XGBoost

LSTM Neural Network

The ensemble aggregates the predictions using a weighted average strategy based on each model’s performance.

Repository Structure
air_quality_prediction.py: Main code for the integrated air quality prediction pipeline.

mlflow.db: SQLite database used for MLflow tracking (automatically generated).

saved_models/: Folder containing serialized models and scalers.

README.md: This file.

Prerequisites
Python 3.8 or above

Required packages (see requirements.txt):

pandas

numpy

scikit-learn

xgboost

tensorflow

mlflow

Other standard libraries (matplotlib, pickle, etc.)

You can install the required packages with:

bash
Copy
pip install -r requirements.txt
Setting Up MLflow
For Local Tracking:
Ensure your MLflow tracking URI is set to use the local SQLite database:

python
Copy
mlflow.set_tracking_uri("sqlite:///mlflow.db")
Start the MLflow UI with:

bash
Copy
mlflow ui --backend-store-uri sqlite:///mlflow.db
Open your browser and navigate to http://127.0.0.1:5000 to view your experiments.

Running the Pipeline
Place the input file (air_quality_streamed.csv) in the root directory or update the file path in the script.

Run the integrated script:

bash
Copy
python air_quality_prediction.py
The script performs feature engineering, trains individual models, tunes hyperparameters for the Random Forest model, and finally creates an ensemble model.

After execution, check the MLflow UI for detailed experiment logs.

Experimentation and Results
Model Experimentation:
Our experiments compared several model types with different hyperparameters (e.g., tuned Random Forest parameters).

MLflow Tracking:
Each model’s performance (MAE, RMSE) is logged, and detailed nested runs provide insight into how hyperparameter tuning affected outcomes.

Ensemble Strategy:
The ensemble uses a weighted average of all model predictions. The final ensemble metrics are compared against individual model metrics to validate improvements.

Feature Engineering:
The code demonstrates extensive feature engineering including time-based features, lag features, rolling window statistics, interaction features, and collinearity reduction
