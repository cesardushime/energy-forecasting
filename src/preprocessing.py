"""
preprocessing.py

Purpose:
--------
Transforms raw data into a clean and consistent time series.

Responsibilities:
-----------------
- Handle missing values (interpolation or imputation)
- Resample data (minute → hourly)
- Ensure temporal consistency

Input:
------
- Raw DataFrame from data_loader

Output:
-------
- Cleaned and resampled DataFrame

Notes:
------
- This step defines the temporal granularity of the project
- Tradeoff: smoothing vs loss of high-frequency detail
"""