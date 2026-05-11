"""
train.py

Purpose:
--------
Train forecasting models on prepared dataset.

Responsibilities:
-----------------
- Perform time-based train/test split
- Train models (e.g., SARIMA, XGBoost)
- Save trained models to disk

Input:
------
- Feature dataset

Output:
-------
- Serialized trained models (.pkl files)

Notes:
------
- No evaluation logic here (handled separately)
- Must ensure reproducibility
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from models.sarima import SARIMAModel
from models.xgboost import XGBoostModel

def train_models(data_path, model_dir):
    # Load prepared dataset
    data = pd.read_csv(data_path)
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Time-based train/test split (e.g., last 20% for testing)
    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    # Train SARIMA model
    sarima_model = SARIMAModel()
    sarima_model.fit(train_data)
    
    # Save SARIMA model
    sarima_path = os.path.join(model_dir, 'sarima_model.pkl')
    joblib.dump(sarima_model, sarima_path)
    
    # Train XGBoost model
    xgboost_model = XGBoostModel()
    xgboost_model.fit(train_data.drop(columns=['target']), train_data['target'])
    
    # Save XGBoost model
    xgboost_path = os.path.join(model_dir, 'xgboost_model.pkl')
    joblib.dump(xgboost_model, xgboost_path)

if __name__ == "__main__":
    data_path = 'data/prepared/feature_dataset.csv'
    model_dir = 'models/trained'
    train_models(data_path, model_dir)

