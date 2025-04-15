import inspect
if not hasattr(inspect, 'formatargspec'):
    # Monkey patch inspect.formatargspec to point to formatargvalues.
    inspect.formatargspec = inspect.formatargvalues

import os
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import mlflow.tensorflow  # for logging tensorflow models
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import json

# For LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =============================================================================
# 1. Feature Engineering with Collinear Feature Removal
# =============================================================================
def remove_collinear_features(X, threshold=0.95):
    """Remove features that are highly collinear."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    if to_drop:
        print(f"Removing {len(to_drop)} collinear features: {to_drop}")
    return X.drop(columns=to_drop)

class AirQualityFeatureEngineering:
    def __init__(self, df):
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                        format='%d/%m/%Y %H.%M.%S')
        df.set_index('Datetime', inplace=True)
        df.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16', 'received_at', 'DateTime'],
                errors='ignore', inplace=True)
        self.df = df.select_dtypes(include=[np.number])
    
    def _generate_interaction_features(self, df, columns):
        interaction_features = {}
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                interaction_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        return pd.DataFrame(interaction_features, index=df.index)
    
    def engineer_features(self, target_column, remove_collinear=True, corr_threshold=0.95):
        df = self.df.copy()
        feature_columns = [col for col in df.columns if col != target_column]
        feature_list = []
        # Adding time features
        time_features = {
            'hour': df.index.hour,
            'day_of_week': df.index.dayofweek,
            'month': df.index.month,
            'is_weekend': (df.index.dayofweek.isin([5, 6])).astype(int),
            'is_morning': ((df.index.hour >= 6) & (df.index.hour < 12)).astype(int),
            'is_afternoon': ((df.index.hour >= 12) & (df.index.hour < 18)).astype(int),
            'is_evening': ((df.index.hour >= 18) & (df.index.hour < 22)).astype(int),
            'is_night': ((df.index.hour >= 22) | (df.index.hour < 6)).astype(int)
        }
        for name, values in time_features.items():
            df[name] = values
            feature_list.append(name)
        
        # Adding lagged, rolling and EWM features
        rolling_windows = [3, 6, 12, 24]
        lag_steps = [1, 2, 3, 6]
        lagged_features, rolling_features, ewm_features = {}, {}, {}
        for col in feature_columns:
            for lag in lag_steps:
                lagged_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            for window in rolling_windows:
                rolling_features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                rolling_features[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            ewm_features[f'{col}_ewm_alpha_0.2'] = df[col].ewm(alpha=0.2, adjust=False).mean()
        
        df = pd.concat([df,
                        pd.DataFrame(lagged_features, index=df.index),
                        pd.DataFrame(rolling_features, index=df.index),
                        pd.DataFrame(ewm_features, index=df.index)], axis=1)
        feature_list.extend(list(lagged_features.keys()) + list(rolling_features.keys()) + list(ewm_features.keys()))
        
        # Adding interaction features
        interaction_df = self._generate_interaction_features(df, feature_columns)
        df = pd.concat([df, interaction_df], axis=1)
        feature_list.extend(interaction_df.columns)
        
        df.dropna(subset=[target_column], inplace=True)
        missing_features = [f for f in feature_list if f not in df.columns]
        if missing_features:
            print(f"Warning: The following features are missing: {missing_features}")
            feature_list = [f for f in feature_list if f in df.columns]
        all_features = feature_list.copy()
        if target_column in all_features:
            all_features.remove(target_column)
        X = df[all_features].fillna(df.mean())
        if remove_collinear:
            X = remove_collinear_features(X, threshold=corr_threshold)
        y = df[target_column]
        print("Feature Engineering Debug:")
        print("Original DataFrame shape:", self.df.shape)
        print("Engineered DataFrame shape:", df.shape)
        print("Target column:", target_column)
        print("Features shape:", X.shape)
        print("Target shape:", y.shape)
        print("Number of features after collinearity removal:", X.shape[1])
        return X, y

# =============================================================================
# 2. AirQualityModelTrainer (Linear, Random Forrest, XGBoost & LSTM models)
# =============================================================================
class AirQualityModelTrainer:
    def __init__(self, X, y):
        split_index = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_index]
        self.X_test = X.iloc[split_index:]
        self.y_train = y.iloc[:split_index]
        self.y_test = y.iloc[split_index:]
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.lr_model = None
        self.rf_model = None
        self.xgb_model = None
        self.lstm_model = None

    def evaluate_model(self, y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{model_name} Performance:")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {'mae': mae, 'rmse': rmse}
    
    def linear_regression(self):
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train_scaled, self.y_train)
        y_pred = self.lr_model.predict(self.X_test_scaled)
        return self.evaluate_model(self.y_test, y_pred, "Linear Regression")

    def random_forest(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        y_pred = self.rf_model.predict(self.X_test_scaled)
        return self.evaluate_model(self.y_test, y_pred, "Random Forest")
    
    def xgboost(self):
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.xgb_model.fit(self.X_train_scaled, self.y_train)
        y_pred = self.xgb_model.predict(self.X_test_scaled)
        return self.evaluate_model(self.y_test, y_pred, "XGBoost")
    
    def train_lstm_model(self, epochs=10, batch_size=32):
        X_train_lstm = self.X_train_scaled.reshape((self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1]))
        X_test_lstm = self.X_test_scaled.reshape((self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1]))
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        self.lstm_model = Sequential([
            LSTM(50, input_shape=input_shape),
            Dense(1)
        ])
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        print("Training LSTM model...")
        self.lstm_model.fit(X_train_lstm, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = self.lstm_model.predict(X_test_lstm).flatten()
        return self.evaluate_model(self.y_test, y_pred, "LSTM Model"), y_pred

# =============================================================================
# 3. Hyperparameter Tuning using GridSearchCV 
# =============================================================================
# Only random forrest is used for tunning with reduced parameter grid because it 
# was taking a lot of time to run the model and the laptop was reaching its max capacity 
def train_rf_with_tuning(X_train, y_train, X_test, y_test, cv=2):
    # Reduced parameter grid for faster search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }
    
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    
    # Using RandomizedSearchCV for quicker tuning
    from sklearn.model_selection import RandomizedSearchCV
    grid_search = RandomizedSearchCV(
        pipeline,
        param_distributions={f'rf__{k}': v for k, v in param_grid.items()},
        n_iter=20,  # Trying 20 random combinations
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("Training Random Forest with hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {'mae': mae, 'rmse': rmse}
    
    return best_pipeline, grid_search.best_params_, metrics, grid_search.cv_results_

# =============================================================================
# 4. Ensemble Function (Weighted Average)
# =============================================================================
def weighted_ensemble_prediction(model_preds, model_errors):
    inv_errors = {name: 1/error for name, error in model_errors.items()}
    total = sum(inv_errors.values())
    weights = {name: w/total for name, w in inv_errors.items()}
    ensemble_pred = np.zeros_like(list(model_preds.values())[0])
    for name, pred in model_preds.items():
        ensemble_pred += weights[name] * pred
    return ensemble_pred

# =============================================================================
# 5. Model Saving Functions
# =============================================================================
MODEL_DIR = "saved_models"
SCALER_DIR = "saved_scalers"

def save_trained_model(model, scaler, target, model_type="model"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    model_filename = os.path.join(MODEL_DIR, f"{target}_{model_type}_model.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    scaler_filename = os.path.join(SCALER_DIR, f"{target}_scaler.pkl")
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved {model_type} model and scaler for {target} prediction")

def save_models_from_trainer(model_trainer, target):
    if model_trainer.lr_model:
        save_trained_model(model_trainer.lr_model, model_trainer.scaler, target, "linear")
    if model_trainer.rf_model:
        save_trained_model(model_trainer.rf_model, model_trainer.scaler, target, "random_forest")
    if model_trainer.xgb_model:
        save_trained_model(model_trainer.xgb_model, model_trainer.scaler, target, "xgboost")
    if model_trainer.lstm_model:
        save_trained_model(model_trainer.lstm_model, model_trainer.scaler, target, "lstm")

# =============================================================================
# 6. Integrated Air Quality Prediction Function with Hyperparameter Experimentation
# =============================================================================
def integrated_air_quality_prediction(filepath, target_column='CO(GT)'):
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print("Starting feature engineering...")
    fe = AirQualityFeatureEngineering(df)
    X, y = fe.engineer_features(target_column, remove_collinear=True, corr_threshold=0.95)
    
    # Split data for hyperparameter tuning and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start a parent MLflow run for this experiment by running a sqlite db for saving 
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Air Quality Prediction Experiment")
    with mlflow.start_run(run_name=f"AirQuality_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("num_features", X.shape[1])
        
        # Train baseline model
        baseline_metrics = None
        if hasattr(y_test, "shift"):
            y_pred_baseline = y_test.shift(1).dropna()
            y_true_baseline = y_test.iloc[1:]
            baseline_metrics = {
                "mae": mean_absolute_error(y_true_baseline, y_pred_baseline),
                "rmse": np.sqrt(mean_squared_error(y_true_baseline, y_pred_baseline))
            }
            mlflow.log_metric("baseline_mae", baseline_metrics["mae"])
            mlflow.log_metric("baseline_rmse", baseline_metrics["rmse"])
        
        # Train models using the AirQualityModelTrainer
        model_trainer = AirQualityModelTrainer(X_train, y_train)
        
        # Nested run for tuning Random Forest
        with mlflow.start_run(nested=True, run_name="Random Forest Tuning"):
            # Use hyperparameter tuning for Random Forest
            best_rf_pipeline, best_params, rf_metrics, cv_results = train_rf_with_tuning(X_train, y_train, X_test, y_test, cv=3)
            mlflow.log_params({k.replace('rf__', ''): v for k, v in best_params.items()})
            mlflow.log_metric("rf_mae", rf_metrics["mae"])
            mlflow.log_metric("rf_rmse", rf_metrics["rmse"])
            rf_model = best_rf_pipeline.named_steps['rf']
            model_trainer.rf_model = rf_model  # assign tuned model for ensemble later
            # log the cv_results.csv artifact
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv("rf_cv_results.csv", index=False)
            mlflow.log_artifact("rf_cv_results.csv")
        
        # Train remaining models without tuning
        with mlflow.start_run(nested=True, run_name="Linear Regression"):
            lr_metrics = model_trainer.linear_regression()
            mlflow.log_metric("linear_regression_mae", lr_metrics["mae"])
            mlflow.log_metric("linear_regression_rmse", lr_metrics["rmse"])
        
        with mlflow.start_run(nested=True, run_name="XGBoost"):
            xgb_metrics = model_trainer.xgboost()
            mlflow.log_metric("xgboost_mae", xgb_metrics["mae"])
            mlflow.log_metric("xgboost_rmse", xgb_metrics["rmse"])
        
        with mlflow.start_run(nested=True, run_name="LSTM Model"):
            lstm_result, lstm_pred = model_trainer.train_lstm_model(epochs=10, batch_size=32)
            lstm_metrics = lstm_result
            mlflow.log_metric("lstm_mae", lstm_metrics["mae"])
            mlflow.log_metric("lstm_rmse", lstm_metrics["rmse"])
        
        # Gather predictions for ensemble integration
        model_preds = {
            "Random Forest": model_trainer.rf_model.predict(model_trainer.X_test_scaled),
            "Linear Regression": model_trainer.lr_model.predict(model_trainer.X_test_scaled),
            "XGBoost": model_trainer.xgb_model.predict(model_trainer.X_test_scaled),
            "LSTM Model": lstm_pred
        }
        model_errors = {
            "Random Forest": rf_metrics["mae"],
            "Linear Regression": lr_metrics["mae"],
            "XGBoost": xgb_metrics["mae"],
            "LSTM Model": lstm_metrics["mae"]
        }
        ensemble_pred = weighted_ensemble_prediction(model_preds, model_errors)
        ensemble_mae = mean_absolute_error(model_trainer.y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(model_trainer.y_test, ensemble_pred))
        mlflow.log_metric("ensemble_mae", ensemble_mae)
        mlflow.log_metric("ensemble_rmse", ensemble_rmse)
        
        # Log model artifacts for each sub-run and logging all models here for simplicity
        mlflow.sklearn.log_model(model_trainer.lr_model, "linear_regression_model")
        mlflow.sklearn.log_model(model_trainer.rf_model, "random_forest_model")
        mlflow.sklearn.log_model(model_trainer.xgb_model, "xgboost_model")
        mlflow.tensorflow.log_model(model_trainer.lstm_model, "lstm_model")
        
        # End parent run
    try:
        save_models_from_trainer(model_trainer, target_column)
        print(f"Models saved for {target_column}")
    except Exception as e:
        print(f"Warning: Could not save models: {e}")
    
    print(f"\nIntegrated analysis for {target_column} completed successfully!")
    return ensemble_mae, ensemble_rmse

# =============================================================================
# 7. Main Execution
# =============================================================================
if __name__ == "__main__":
    filepath = "air_quality_streamed.csv" 
    if not os.path.exists(filepath):
        print(f"Error: Data file {filepath} not found.")
        exit(1)
    integrated_air_quality_prediction(filepath, target_column='CO(GT)')