"""
features.py

Purpose:
--------
Create predictive features from time series data.

Responsibilities:
-----------------
- Generate time-based features (hour, day, month)
- Create lag features for temporal dependencies
- Compute rolling statistics (mean, std)
- Derive additional variables (e.g., unmetered energy)

Input:
------
- Cleaned DataFrame

Output:
-------
- Feature-enhanced dataset ready for modeling

Notes:
------
- Feature quality directly impacts model performance
- Must avoid data leakage (only past data used for features)
"""