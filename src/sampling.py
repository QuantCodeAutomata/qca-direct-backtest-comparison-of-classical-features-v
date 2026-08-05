"""Stratified sampling utilities for the PQFM RFQ fill-prediction experiment.

Provides proportional-by-trading-day sampling that ensures every business day
present in the source DataFrame contributes at least one event to the drawn
sample, preserving the temporal distribution of the full dataset.
"""

import logging

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


def get_trading_days(df: pd.DataFrame) -> list:
    """Extract the sorted list of unique trading days from a DataFrame's timestamps.

    Days are identified by flooring each UTC timestamp to midnight, so all events
    occurring on the same calendar date (UTC) belong to the same trading day.

    Parameters
    ----------
    df:
        DataFrame containing a ``timestamp`` column of datetime dtype.

    Returns
    -------
    list[pd.Timestamp]
        Sorted list of timezone-aware ``pd.Timestamp`` objects at midnight UTC,
        one entry per unique trading day present in *df*.

    Raises
    ------
    KeyError
        If *df* does not contain a ``timestamp`` column.
    """
    if "timestamp" not in df.columns:
        raise KeyError("DataFrame must contain a 'timestamp' column.")

    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    day_bins: pd.DatetimeIndex = ts.dt.floor("D").unique()
    return sorted(day_bins.tolist())


def sample_representative(
    df: pd.DataFrame,
    n: int = 16_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw *n* events from *df* ensuring every trading day is represented.

    **Methodology**

    1. Group events by trading day (UTC calendar date).
    2. Compute each day's exact proportional quota:
       ``quota_d = (events_on_day_d / total_events) * n``.
    3. Apply Hamilton's largest-remainder method to convert floating-point
       quotas to integers that sum exactly to *n*, guaranteeing at least one
       event per non-empty day whenever the day has any events.
    4. Cap each day's allocation at its actual event count (so sampling is
       always without replacement).
    5. Within each day draw the allocated count uniformly at random using an
       independent per-day seed derived from the master *seed*.
    6. Concatenate and sort the result by timestamp.

    The per-day sampled counts and the master seed are emitted at INFO level.

    Parameters
    ----------
    df:
        Source DataFrame containing at least a ``timestamp`` column of
        datetime dtype.  Must have ``len(df) >= n``.
    n:
        Target sample size.  Defaults to ``16_000`` (``N_SAMPLE`` from config).
    seed:
        Master integer seed for full reproducibility.  Individual per-day
        seeds are derived from this value via ``numpy.random.default_rng``.

    Returns
    -------
    pd.DataFrame
        Sampled subset with a reset integer index, sorted ascending by
        ``timestamp``.  The ``_day`` helper column is not included in the
        output.

    Raises
    ------
    KeyError
        If *df* has no ``timestamp`` column.
    ValueError
        If *n* exceeds ``len(df)`` or if *df* contains no events.
    """
    if "timestamp" not in df.columns:
        raise KeyError("DataFrame must contain a 'timestamp' column.")
    if len(df) == 0:
        raise ValueError("Cannot sample from an empty DataFrame.")
    if n > len(df):
        raise ValueError(
            f"Requested n={n:,} exceeds dataset size {len(df):,}."
        )
    if n == len(df):
        _LOG.info("sample_representative: n equals dataset size, returning full dataset.")
        return df.reset_index(drop=True)

    _LOG.info("sample_representative: n=%d, seed=%d", n, seed)

    master_rng = np.random.default_rng(seed)

    # ---- Build day grouping ----
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    day_col: pd.Series = ts.dt.floor("D")
    trading_days: list = sorted(day_col.unique().tolist())
    n_days: int = len(trading_days)

    day_sizes: dict = {day: int((day_col == day).sum()) for day in trading_days}
    total: int = len(df)

    # ---- Hamilton's largest-remainder proportional allocation ----
    exact_quotas: dict = {day: day_sizes[day] / total * n for day in trading_days}
    floor_allocs: dict = {day: int(exact_quotas[day]) for day in trading_days}
    remainder: int = n - sum(floor_allocs.values())

    # Sort by fractional part descending to distribute remainder
    frac_order = sorted(
        trading_days,
        key=lambda d: exact_quotas[d] - floor_allocs[d],
        reverse=True,
    )
    final_allocs: dict = dict(floor_allocs)
    for day in frac_order[:remainder]:
        final_allocs[day] += 1

    # Cap at actual day size (avoids sampling more rows than exist)
    for day in trading_days:
        if final_allocs[day] > day_sizes[day]:
            final_allocs[day] = day_sizes[day]

    # ---- Per-day sampling ----
    sampled_frames: list[pd.DataFrame] = []
    for day in trading_days:
        alloc: int = final_allocs[day]
        if alloc == 0:
            continue
        day_mask: pd.Series = day_col == day
        day_df: pd.DataFrame = df.loc[day_mask]
        # Derive a deterministic per-day integer seed from the master RNG
        day_seed: int = int(master_rng.integers(0, 2**31))
        sampled: pd.DataFrame = day_df.sample(
            n=alloc, replace=False, random_state=day_seed
        )
        sampled_frames.append(sampled)
        _LOG.info(
            "  Day %s: allocated %d / %d events (seed=%d)",
            day.date(),
            alloc,
            day_sizes[day],
            day_seed,
        )

    result: pd.DataFrame = (
        pd.concat(sampled_frames, ignore_index=True)
        .sort_values("timestamp", ignore_index=True)
    )
    _LOG.info(
        "sample_representative complete: %d events drawn from %d trading days.",
        len(result),
        n_days,
    )
    return result
