"""Utilities for preprocessing time-series energy data.

This module centralizes missing-value handling and temporal resampling so the
notebooks and training code follow the same policy.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def _validate_datetime_index(df: pd.DataFrame) -> None:
	"""Ensure a DateTimeIndex is present and sorted for time-aware operations."""
	if not isinstance(df.index, pd.DatetimeIndex):
		raise TypeError("Expected a pandas DateTimeIndex for time-series preprocessing.")
	if not df.index.is_monotonic_increasing:
		raise ValueError("DateTimeIndex must be sorted in ascending order.")


def summarize_missing_gaps(df: pd.DataFrame) -> pd.DataFrame:
	"""Return start/end/length metadata for consecutive missing-row gaps."""
	_validate_datetime_index(df)

	missing_rows = df.isna().any(axis=1)
	if not missing_rows.any():
		return pd.DataFrame(columns=["start", "end", "length"])  # pragma: no cover

	groups = (missing_rows != missing_rows.shift(fill_value=False)).cumsum()
	records = []

	for _, segment in missing_rows.groupby(groups):
		if not bool(segment.iloc[0]):
			continue
		records.append(
			{
				"start": segment.index[0],
				"end": segment.index[-1],
				"length": int(segment.sum()),
			}
		)

	return pd.DataFrame.from_records(records)


def apply_missing_value_policy(
	df: pd.DataFrame,
	short_gap_limit: int = 3,
	interpolation_method: Literal["time", "linear"] = "time",
) -> pd.DataFrame:
	"""Apply a two-stage missing-value strategy.

	Policy:
	- Interpolate short consecutive missing blocks up to ``short_gap_limit`` rows.
	- Preserve long missing blocks as NaN (structural outages).
	- Add explicit indicators for historical missing rows and long-gap rows.
	"""
	_validate_datetime_index(df)

	if short_gap_limit < 1:
		raise ValueError("short_gap_limit must be >= 1")

	original_missing_rows = df.isna().any(axis=1)
	groups = (original_missing_rows != original_missing_rows.shift(fill_value=False)).cumsum()

	long_gap_mask = pd.Series(False, index=df.index)
	for _, segment in original_missing_rows.groupby(groups):
		if bool(segment.iloc[0]) and int(segment.sum()) > short_gap_limit:
			long_gap_mask.loc[segment.index] = True

	cleaned = df.interpolate(
		method=interpolation_method,
		limit=short_gap_limit,
		limit_direction="both",
	)

	# Keep structural outages explicit after interpolation.
	cleaned.loc[long_gap_mask, :] = np.nan

	cleaned["is_missing_row"] = original_missing_rows.astype(int)
	cleaned["is_long_gap"] = long_gap_mask.astype(int)

	return cleaned


def resample_numeric(
	df: pd.DataFrame,
	rule: str = "h",
	agg: Literal["mean", "median", "sum"] = "mean",
) -> pd.DataFrame:
	"""Resample only numeric columns to a lower frequency."""
	_validate_datetime_index(df)

	numeric_df = df.select_dtypes(include=["number"])
	if numeric_df.empty:
		raise ValueError("No numeric columns available for resampling.")

	if agg == "mean":
		return numeric_df.resample(rule).mean()
	if agg == "median":
		return numeric_df.resample(rule).median()
	if agg == "sum":
		return numeric_df.resample(rule).sum()

	raise ValueError(f"Unsupported agg method: {agg}")


def preprocess_time_series(
	raw_df: pd.DataFrame,
	short_gap_limit: int = 3,
	interpolation_method: Literal["time", "linear"] = "time",
	resample_rule: str = "h",
	resample_agg: Literal["mean", "median", "sum"] = "mean",
) -> pd.DataFrame:
	"""Full preprocessing pipeline: missing policy + resampling."""
	cleaned = apply_missing_value_policy(
		raw_df,
		short_gap_limit=short_gap_limit,
		interpolation_method=interpolation_method,
	)
	return resample_numeric(cleaned, rule=resample_rule, agg=resample_agg)