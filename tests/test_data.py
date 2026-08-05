"""Tests for data_loader and sampling modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import pytest

from data_loader import load_rfq_data, filter_active_window, validate_dataset
from sampling import sample_representative, get_trading_days


# ---------------------------------------------------------------------------
# Shared fixture: load dataset once per session for performance
# ---------------------------------------------------------------------------

def _get_df() -> pd.DataFrame:
    """Return the synthetic RFQ dataset (auto-generated on first call)."""
    return load_rfq_data()


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------

def test_synthetic_data_shape():
    """Dataset has > 0 rows, expected columns, and 216 feature dimensions."""
    df = _get_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0

    feature_cols = [c for c in df.columns if c.startswith('f_')]
    assert len(feature_cols) == 216, (
        f"Expected 216 feature columns (f_000…f_215), got {len(feature_cols)}"
    )
    for required in ('event_id', 'timestamp', 'bond_id', 'fill'):
        assert required in df.columns, f"Missing required column: {required}"


def test_feature_range():
    """All classical features must lie in the closed interval [-1, 1]."""
    df = _get_df()
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feature_cols].to_numpy(dtype=float)

    assert X.min() >= -1.0 - 1e-9, (
        f"Feature values go below -1: min={X.min()}"
    )
    assert X.max() <= 1.0 + 1e-9, (
        f"Feature values exceed +1: max={X.max()}"
    )


def test_binary_labels():
    """The 'fill' label column must contain only {0, 1} with no NaN."""
    df = _get_df()
    assert not df['fill'].isna().any(), "'fill' contains NaN values"
    unique_vals = set(df['fill'].unique())
    assert unique_vals.issubset({0, 1}), (
        f"'fill' contains non-binary values: {unique_vals - {0, 1}}"
    )


def test_filter_active_window_removes_maturing():
    """filter_active_window must remove all bonds whose bond_id ends with '_MATURING'."""
    df = _get_df()

    # Confirm the raw dataset contains some maturing bonds
    n_maturing_before = df['bond_id'].str.endswith('_MATURING').sum()
    assert n_maturing_before > 0, (
        "Test requires at least one _MATURING bond in the raw dataset"
    )

    filtered = filter_active_window(df, start='2024-07-24', end='2024-10-29')

    n_maturing_after = filtered['bond_id'].str.endswith('_MATURING').sum()
    assert n_maturing_after == 0, (
        f"{n_maturing_after} maturing bonds remain after filter_active_window"
    )


def test_validate_dataset_raises_on_bad_fill():
    """validate_dataset must raise ValueError when 'fill' has non-binary entries."""
    df = _get_df().head(100).copy()
    df.loc[0, 'fill'] = 2  # inject bad value

    with pytest.raises(ValueError, match="fill"):
        validate_dataset(df)


# ---------------------------------------------------------------------------
# sampling tests
# ---------------------------------------------------------------------------

def test_sample_covers_all_days():
    """sample_representative must include events from every trading day present
    in the source when sample size is sufficiently large relative to day count.
    """
    df = _get_df()

    # Narrow to one business week to keep the test fast and deterministic
    ts = df['timestamp']
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('UTC')
    else:
        ts = ts.dt.tz_convert('UTC')

    week_start = pd.Timestamp('2024-07-22', tz='UTC')
    week_end   = pd.Timestamp('2024-07-29', tz='UTC')
    week_mask  = (ts >= week_start) & (ts < week_end)
    df_week = df.loc[week_mask].reset_index(drop=True)

    if len(df_week) == 0:
        pytest.skip("No events found in test week 2024-07-22 – 2024-07-28.")

    original_days = set(get_trading_days(df_week))
    n_days = len(original_days)
    if n_days == 0:
        pytest.skip("Zero trading days in test week.")

    # Allocate at least 20 events per trading day so Hamilton's method
    # guarantees every day gets at least 1 sample.
    n_sample = min(n_days * 20, len(df_week))

    sample = sample_representative(df_week, n=n_sample, seed=42)
    sample_days = set(get_trading_days(sample))

    assert original_days == sample_days, (
        f"Sample is missing trading days: {original_days - sample_days}"
    )


def test_sample_size_exact():
    """sample_representative must return exactly n rows."""
    df = _get_df()
    n = 500
    sample = sample_representative(df, n=n, seed=42)
    assert len(sample) == n, f"Expected {n} rows, got {len(sample)}"


def test_deterministic_sampling():
    """Two calls with the same seed must return identical DataFrames."""
    df = _get_df()
    n = 500
    s1 = sample_representative(df, n=n, seed=99)
    s2 = sample_representative(df, n=n, seed=99)
    pd.testing.assert_frame_equal(
        s1.reset_index(drop=True),
        s2.reset_index(drop=True),
    )
