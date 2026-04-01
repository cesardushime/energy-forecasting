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