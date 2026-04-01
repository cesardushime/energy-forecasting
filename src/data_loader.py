"""
data_loader.py

Purpose:
--------
Handles ingestion of the raw dataset.

Responsibilities:
-----------------
- Load dataset from raw text file
- Parse date and time into a unified datetime index
- Convert all features to numeric types
- Return a clean, structured DataFrame ready for preprocessing

Input:
------
- File path to raw dataset

Output:
-------
- Pandas DataFrame with datetime index

Notes:
------
- No cleaning or transformations beyond parsing
- Missing values are preserved for downstream handling
"""

import pandas as pd

def load_data(file_path):
    """
    Load the raw dataset from a text file.

    Parameters:
    -----------
    file_path : str
        The path to the raw dataset file.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the loaded dataset with a datetime index (no date and time columns).
    """

    # Load the dataset
    df = pd.read_csv(file_path, sep=';', low_memory=False)

    # Resolve date/time columns in a case-insensitive way.
    normalized_cols = {col.strip().lower(): col for col in df.columns}
    if 'date' not in normalized_cols or 'time' not in normalized_cols:
        raise ValueError(
            "Expected date/time columns in input data. "
            f"Found columns: {list(df.columns)}"
        )

    date_col = normalized_cols['date']
    time_col = normalized_cols['time']

    # Parse DD/MM/YYYY timestamps from the UCI household power dataset.
    df['datetime'] = pd.to_datetime(
        df[date_col].astype(str) + ' ' + df[time_col].astype(str),
        dayfirst=True,
        errors='coerce'
    )
    if df['datetime'].isna().all():
        raise ValueError("Failed to parse datetime column from date/time values.")

    df.set_index('datetime', inplace=True)

    # Drop the original date and time columns
    df.drop(columns=[date_col, time_col], inplace=True)

    # Convert all features to numeric types (if necessary)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

