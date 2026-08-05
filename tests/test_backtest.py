"""Tests for backtest engine and model factory."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import pytest

from backtest import TemporalBacktest
from models import get_model_and_grid, train_best_model, MODEL_FAMILIES


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

_SECS_PER_DAY: float = 86_400.0


def _make_sequential_df(
    n_events: int = 300,
    n_features: int = 5,
    hours_per_event: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a minimal DataFrame with sequential UTC timestamps.

    Events are spaced *hours_per_event* apart starting from 2024-07-01 09:00 UTC.
    Binary fill labels are assigned randomly.

    Returns
    -------
    (df, feature_cols)
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp('2024-07-01 09:00:00', tz='UTC')
    timestamps = [base + pd.Timedelta(hours=hours_per_event * i) for i in range(n_events)]

    X = rng.standard_normal((n_events, n_features))
    y = (rng.uniform(size=n_events) > 0.5).astype(int)

    feature_cols = [f'f_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['fill'] = y
    df['timestamp'] = timestamps
    return df, feature_cols


def _make_bt(
    df: pd.DataFrame,
    feature_cols: list[str],
    training_sizes: list[int] = None,
    blinding_days: list[int] = None,
) -> TemporalBacktest:
    """Construct a TemporalBacktest with low min_test_events for CI speed."""
    return TemporalBacktest(
        df,
        feature_cols=feature_cols,
        label_col='fill',
        training_sizes=training_sizes or [50],
        blinding_days=blinding_days or [0],
        min_test_events=1,
    )


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------

def test_model_families_all_present():
    """MODEL_FAMILIES must include all four expected families."""
    for family in ('LR', 'XGB', 'RF', 'NN'):
        assert family in MODEL_FAMILIES, f"'{family}' missing from MODEL_FAMILIES"


def test_lr_c_values_count():
    """LR hyperparameter grid must have exactly 11 C values."""
    _, grid = get_model_and_grid('LR')
    assert len(grid['C']) == 11, (
        f"Expected 11 LR C values, got {len(grid['C'])}: {grid['C']}"
    )


def test_get_model_and_grid_invalid_family():
    """get_model_and_grid must raise ValueError for unknown family strings."""
    with pytest.raises(ValueError, match="Unknown"):
        get_model_and_grid('UNKNOWN_FAMILY')


# ---------------------------------------------------------------------------
# Blinding gap semantics
# ---------------------------------------------------------------------------

def test_blinding_gap_0d_lt_24h():
    """With blinding_days=0, the enforced gap must be < 86 400 s (one day)."""
    df, feature_cols = _make_sequential_df()
    bt = _make_bt(df, feature_cols, blinding_days=[0])
    gap_secs = bt._get_blind_gap_seconds(0)
    assert gap_secs < _SECS_PER_DAY, (
        f"blinding_days=0 gap should be < 86 400 s, got {gap_secs}"
    )
    assert gap_secs > 0, "Gap must be strictly positive (at least 1 µs)"


def test_blinding_gap_1d_gte_1day():
    """With blinding_days=1, the enforced gap must be >= 86 400 s (one day)."""
    df, feature_cols = _make_sequential_df()
    bt = _make_bt(df, feature_cols, blinding_days=[1])
    gap_secs = bt._get_blind_gap_seconds(1)
    assert gap_secs >= _SECS_PER_DAY, (
        f"blinding_days=1 gap should be >= 86 400 s, got {gap_secs}"
    )


def test_blinding_gap_negative_raises():
    """_get_blind_gap_seconds must raise ValueError for negative blinding_days."""
    df, feature_cols = _make_sequential_df()
    bt = _make_bt(df, feature_cols)
    with pytest.raises(ValueError):
        bt._get_blind_gap_seconds(-1)


# ---------------------------------------------------------------------------
# No-leakage verification
# ---------------------------------------------------------------------------

def test_temporal_backtest_no_leakage():
    """For any blinding_day=D, the training cutoff is D days before the first
    test event, ensuring training data never includes same-day or future events.

    Verified indirectly: the gap in seconds is >= D × 86 400 s, and
    run_single returns a valid list (no exceptions from the leakage guard).
    """
    df, feature_cols = _make_sequential_df(n_events=300, hours_per_event=2.0)
    blinding_day = 2
    bt = TemporalBacktest(
        df,
        feature_cols=feature_cols,
        label_col='fill',
        training_sizes=[50],
        blinding_days=[blinding_day],
        min_test_events=1,
    )

    gap_secs = bt._get_blind_gap_seconds(blinding_day)
    assert gap_secs >= blinding_day * _SECS_PER_DAY, (
        f"Gap {gap_secs}s < {blinding_day} × 86 400 s – leakage risk"
    )

    results = bt.run_single('LR', training_size=50, blinding_day=blinding_day)
    assert isinstance(results, list), "run_single must return a list"


# ---------------------------------------------------------------------------
# Return-type and structure tests
# ---------------------------------------------------------------------------

def test_backtest_returns_dataframe():
    """run_all must return a pd.DataFrame (possibly empty) with expected columns."""
    df, feature_cols = _make_sequential_df(n_events=400, hours_per_event=2.0)
    bt = _make_bt(df, feature_cols, training_sizes=[50], blinding_days=[0])
    result = bt.run_all(families=['LR'])

    assert isinstance(result, pd.DataFrame), (
        f"run_all must return a DataFrame, got {type(result)}"
    )

    if not result.empty:
        for col in ('family', 'training_size', 'blinding_day', 'auc', 'n_train', 'n_test'):
            assert col in result.columns, f"Expected column '{col}' in results"


def test_run_single_valid_auc_range():
    """AUC values returned by run_single must be in [0, 1]."""
    df, feature_cols = _make_sequential_df(n_events=400, hours_per_event=2.0)
    bt = _make_bt(df, feature_cols, training_sizes=[50], blinding_days=[0])
    results = bt.run_single('LR', training_size=50, blinding_day=0)

    for r in results:
        assert 0.0 <= r['auc'] <= 1.0, (
            f"AUC out of [0, 1]: {r['auc']}"
        )
