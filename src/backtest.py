"""Temporal backtesting engine for PQFM RFQ fill-prediction experiment."""

import logging
import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from models import MODEL_FAMILIES, train_best_model, predict_proba
except ImportError:
    from src.models import MODEL_FAMILIES, train_best_model, predict_proba

logger = logging.getLogger(__name__)

_DEFAULT_TRAINING_SIZES: list[int] = [500, 1000, 2000, 5000]
_DEFAULT_BLINDING_DAYS: list[int] = [0, 1, 2, 5, 10]

# Seconds in one calendar day (trading days are approximated as calendar days)
_SECS_PER_DAY: float = 86_400.0
# Minimum gap for zero-day blinding: 1 microsecond
_MIN_GAP_SECS: float = 1e-6


class TemporalBacktest:
    """Strict temporal backtesting engine with a configurable blinding window.

    The engine walks forward through time day-by-day using only events that
    strictly predate each test window.  An optional *blinding* gap prevents
    models from inadvertently exploiting information that was not yet
    actionable at prediction time (e.g., post-trade latency, reporting delays).

    Blinding semantics
    ------------------
    * ``blinding_days=0``   – gap ≥ 1 µs and < 24 h (microsecond exclusion only).
    * ``blinding_days=N≥1`` – gap ≥ N × 86 400 s (N calendar days).

    No future leakage: training events are always strictly before
    ``first_test_event_timestamp − blind_gap``.

    Parameters
    ----------
    df:
        Dataset sorted (or sortable) by *timestamp_col*.
    feature_cols:
        Ordered list of feature column names present in *df*.
    label_col:
        Binary fill-indicator column (0 = no fill, 1 = filled).
    timestamp_col:
        Datetime column used for temporal ordering.
    training_sizes:
        List of rolling-window sizes (number of events).  Defaults to
        ``[500, 1000, 2000, 5000]``.
    blinding_days:
        Candidate blinding windows in days.  Defaults to
        ``[0, 1, 2, 5, 10]``.
    min_test_events:
        Minimum number of test events required per day; days with fewer
        events are silently skipped.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = 'fill',
        timestamp_col: str = 'timestamp',
        training_sizes: Optional[list[int]] = None,
        blinding_days: Optional[list[int]] = None,
        min_test_events: int = 50,
    ) -> None:
        missing = [c for c in feature_cols + [label_col, timestamp_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in df: {missing}")

        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.timestamp_col = timestamp_col
        self.training_sizes = training_sizes if training_sizes is not None else list(_DEFAULT_TRAINING_SIZES)
        self.blinding_days = blinding_days if blinding_days is not None else list(_DEFAULT_BLINDING_DAYS)
        self.min_test_events = min_test_events

        # Ensure datetime dtype and strict temporal ordering
        self.df = df.copy()
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self.df = self.df.sort_values(timestamp_col).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_blind_gap_seconds(self, blinding_days: int) -> float:
        """Convert a blinding-day integer to a gap expressed in seconds.

        Parameters
        ----------
        blinding_days:
            Non-negative integer number of blinding days.

        Returns
        -------
        float
            Gap in seconds to subtract from the first test-event timestamp in
            order to obtain the training-data cutoff:

            * ``blinding_days == 0``: returns ``1e-6`` (1 µs).  This is the
              minimum valid gap (≥ 1 µs, < 86 400 s) that ensures the test
              events themselves are excluded without discarding the preceding
              calendar day.
            * ``blinding_days >= 1``: returns ``blinding_days × 86 400``.
              Trading days are approximated as calendar days.

        Raises
        ------
        ValueError
            If *blinding_days* is negative.
        """
        if blinding_days < 0:
            raise ValueError(f"blinding_days must be non-negative, got {blinding_days}.")
        if blinding_days == 0:
            return _MIN_GAP_SECS
        return blinding_days * _SECS_PER_DAY

    # ------------------------------------------------------------------
    # Core backtest loop
    # ------------------------------------------------------------------

    def run_single(
        self,
        family: str,
        training_size: int,
        blinding_day: int,
        cv: int = 3,
        random_state: int = 42,
        fast: bool = False,
        ultrafast: bool = False,
    ) -> list[dict]:
        """Run the backtest for one (family, training_size, blinding_day) combination.

        The algorithm walks forward day-by-day:

        1. Extract unique trading dates from *timestamp_col*.
        2. For each date, collect that day's events as the **test window**.
        3. Compute ``cutoff = first_test_timestamp − blind_gap``.
        4. All events with ``timestamp < cutoff`` become **training candidates**.
        5. Skip the date if fewer than *training_size* candidates are available.
        6. Use the **most recent** *training_size* candidates as the training set.
        7. Train the model with grid-search CV; skip on failure.
        8. Compute ROC-AUC on the test window; skip if single class.

        Parameters
        ----------
        family:
            Model family string – one of ``MODEL_FAMILIES``.
        training_size:
            Number of most-recent events to use for training.
        blinding_day:
            Blinding window in days (0 = microsecond gap only).
        cv:
            Cross-validation folds passed to ``train_best_model``.
        random_state:
            RNG seed passed to ``train_best_model``.

        Returns
        -------
        list[dict]
            One dict per successfully evaluated test day, each with keys:
            ``{family, training_size, blinding_day, test_timestamp, auc,
            n_train, n_test}``.
        """
        blind_gap_secs = self._get_blind_gap_seconds(blinding_day)

        timestamps = self.df[self.timestamp_col]
        dates = timestamps.dt.date
        unique_dates = sorted(dates.unique())

        results: list[dict] = []

        for date in unique_dates:
            test_mask = dates == date
            test_events = self.df[test_mask]

            if len(test_events) < self.min_test_events:
                continue

            first_test_ts = timestamps[test_mask].min()
            cutoff = first_test_ts - pd.Timedelta(seconds=blind_gap_secs)

            train_candidates = self.df[timestamps < cutoff]

            if len(train_candidates) < training_size:
                continue

            train_events = train_candidates.iloc[-training_size:]

            X_train = train_events[self.feature_cols].to_numpy(dtype=float)
            y_train = train_events[self.label_col].to_numpy()
            X_test = test_events[self.feature_cols].to_numpy(dtype=float)
            y_test = test_events[self.label_col].to_numpy()

            # Skip days where the test set has only one class
            if len(np.unique(y_test)) < 2:
                continue

            model = train_best_model(
                X_train, y_train, family, cv=cv, random_state=random_state,
                fast=fast, ultrafast=ultrafast,
            )
            if model is None:
                continue

            try:
                probas = predict_proba(model, X_test)
                auc = float(roc_auc_score(y_test, probas))
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"run_single ({family}, ts={training_size}, bd={blinding_day}, "
                    f"date={date}): AUC computation failed – {exc}",
                    stacklevel=2,
                )
                continue

            results.append({
                'family': family,
                'training_size': training_size,
                'blinding_day': blinding_day,
                'test_timestamp': first_test_ts,
                'auc': auc,
                'n_train': len(X_train),
                'n_test': len(X_test),
            })

        return results

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def run_all(
        self,
        families: Optional[list[str]] = None,
        cv: int = 3,
        random_state: int = 42,
        fast: bool = False,
        ultrafast: bool = False,
    ) -> pd.DataFrame:
        """Run the backtest for every (family × training_size × blinding_day) combination.

        Parameters
        ----------
        families:
            Subset of model families to evaluate.  Defaults to all
            ``MODEL_FAMILIES`` when ``None``.
        cv:
            Cross-validation folds forwarded to ``run_single``.
        random_state:
            RNG seed forwarded to ``run_single``.

        Returns
        -------
        pd.DataFrame
            Concatenated results from all ``run_single`` calls with columns:
            ``family``, ``training_size``, ``blinding_day``,
            ``test_timestamp``, ``auc``, ``n_train``, ``n_test``.
            Returns an empty DataFrame (with those columns) if no valid
            windows are found.

        Notes
        -----
        Progress is logged at INFO level every 10 combination iterations.
        """
        if families is None:
            families = list(MODEL_FAMILIES)

        combos = list(product(families, self.training_sizes, self.blinding_days))
        n_combos = len(combos)
        logger.info("run_all: starting %d combinations.", n_combos)

        all_results: list[dict] = []

        for idx, (family, training_size, blinding_day) in enumerate(combos):
            if idx % 10 == 0:
                logger.info(
                    "run_all progress: %d/%d – family=%s, training_size=%d, blinding_day=%d",
                    idx, n_combos, family, training_size, blinding_day,
                )

            results = self.run_single(
                family=family,
                training_size=training_size,
                blinding_day=blinding_day,
                cv=cv,
                random_state=random_state,
                fast=fast,
                ultrafast=ultrafast,
            )
            all_results.extend(results)

        logger.info("run_all: completed %d combinations, %d result rows.", n_combos, len(all_results))

        _empty_cols = [
            'family', 'training_size', 'blinding_day',
            'test_timestamp', 'auc', 'n_train', 'n_test',
        ]
        if not all_results:
            return pd.DataFrame(columns=_empty_cols)

        return pd.DataFrame(all_results)
