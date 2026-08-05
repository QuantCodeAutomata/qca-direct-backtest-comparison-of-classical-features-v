"""
Classical-Quantum Event Matching (CQEM) module.

Assigns hardware PQFM features to unseen events via a deterministic matching
key kappa built from binned classical features.  Implements the paper's
event-matching scheme over N_bins ∈ {4, 10, 30, 60} and feature range [-1, +1].

Event-matching resolution = 1 - 1/N_bins.
"""

import numpy as np
import pandas as pd
from typing import Optional


class CQEMMapper:
    """
    Classical-Quantum Event Matching (CQEM) module.
    Assigns hardware PQFM features to unseen events via deterministic
    matching key kappa built from binned classical features.
    Paper: N_bins in {4, 10, 30, 60}, matching over [-1, +1].
    Event-matching resolution = 1 - 1/N_bins.

    For compact/small-sample runs, ``top_k_features`` restricts the kappa to
    the K highest-variance features in the library, dramatically increasing
    collision probability without changing the paper's algorithm semantics.
    At full scale (16K/128K events) set ``top_k_features=None`` (all 216).
    """

    def __init__(
        self,
        n_bins: int = 30,
        feature_range: tuple = (-1.0, 1.0),
        top_k_features: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_bins:
            Number of equal-width bins per feature dimension.  The paper
            evaluates N_bins ∈ {4, 10, 30, 60}.
        feature_range:
            (lo, hi) bounds assumed for all classical features before binning.
            Values outside this range are clipped before the bin index is
            computed.
        top_k_features:
            If set, kappa is computed using only the ``top_k_features``
            highest-variance features selected during ``build_library``.
            ``None`` (default) uses all features, matching the paper exactly.
            Use a small value (e.g. 6–10) only for compact/test runs where
            the event pool is too small for full-dimensional collisions.

        Notes
        -----
        Bins are left-closed, right-open [lo + k·w, lo + (k+1)·w) for
        k = 0 … n_bins−2, with the last bin including the right edge.
        """
        if n_bins < 1:
            raise ValueError(f"n_bins must be ≥ 1, got {n_bins}.")
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range must satisfy lo < hi, got {feature_range}.")

        self.n_bins: int = n_bins
        self.feature_range: tuple = feature_range
        self.top_k_features: Optional[int] = top_k_features
        self._selected_feature_idx: Optional[np.ndarray] = None  # set by build_library
        self._library: dict = {}          # kappa (tuple of int) -> mean PQFM vector
        self._library_counts: dict = {}   # kappa -> count of contributing events

    # ------------------------------------------------------------------
    # Binning / kappa computation
    # ------------------------------------------------------------------

    def _compute_kappa(self, x: np.ndarray) -> tuple:
        """
        Compute matching key kappa for a single event feature vector x.

        Each feature value is discretised into a bin index in
        [0, n_bins − 1] over ``feature_range`` using uniform-width bins.
        If ``_selected_feature_idx`` is set (via ``build_library``), only those
        feature positions are used, reducing dimensionality for small samples.

        Parameters
        ----------
        x:
            1-D array of shape (n_features,) containing classical feature
            values.  Values outside ``feature_range`` are clipped.

        Returns
        -------
        tuple of int
            Deterministic, hashable bin-index tuple of length n_used_features.
        """
        if self._selected_feature_idx is not None:
            x = x[self._selected_feature_idx]
        lo, hi = self.feature_range
        x_clipped = np.clip(x, lo, hi)
        bin_width = (hi - lo) / self.n_bins
        bins = np.floor((x_clipped - lo) / bin_width).astype(int)
        # Edge case: exactly at the right edge maps to n_bins → clamp to n_bins-1
        bins = np.clip(bins, 0, self.n_bins - 1)
        return tuple(int(b) for b in bins)

    def compute_kappa_batch(self, X: np.ndarray) -> list[tuple]:
        """Vectorised kappa computation for a batch of events.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_events, n_features).

        Returns
        -------
        list of tuple
            One kappa per event row.
        """
        return [self._compute_kappa(X[i]) for i in range(X.shape[0])]

    # ------------------------------------------------------------------
    # Library construction
    # ------------------------------------------------------------------

    def build_library(self, X_classical: np.ndarray, X_pqfm: np.ndarray) -> None:
        """
        Build kappa → PQFM mapping from the library dataset.

        If ``top_k_features`` was set at construction, the K highest-variance
        feature indices are computed from ``X_classical`` here and stored as
        ``_selected_feature_idx``.  This selection is then applied deterministically
        for all subsequent kappa computations (library and unseen events alike).

        If multiple events share the same kappa key, their PQFM vectors are
        averaged (Γ^κ = mean).  Calling ``build_library`` multiple times
        accumulates events; call on a fresh ``CQEMMapper`` instance to
        replace the library.

        Parameters
        ----------
        X_classical:
            Classical feature matrix of shape (n_library, n_classical_features).
        X_pqfm:
            Hardware-long PQFM feature matrix of shape
            (n_library, n_pqfm_features), typically 327 for N=109 qubits.
        """
        # Determine feature subset for kappa (top-K by variance, or all).
        if self.top_k_features is not None:
            k = min(int(self.top_k_features), X_classical.shape[1])
            variances = np.var(X_classical, axis=0)
            self._selected_feature_idx = np.argsort(variances)[-k:]
        else:
            self._selected_feature_idx = None  # use all features

        kappas = self.compute_kappa_batch(X_classical)
        for kappa, pqfm_vec in zip(kappas, X_pqfm):
            if kappa in self._library:
                n = self._library_counts[kappa]
                # Online mean update: avoids accumulating a full list
                self._library[kappa] = (n * self._library[kappa] + pqfm_vec) / (n + 1)
                self._library_counts[kappa] = n + 1
            else:
                self._library[kappa] = pqfm_vec.copy()
                self._library_counts[kappa] = 1

    # ------------------------------------------------------------------
    # Feature assignment
    # ------------------------------------------------------------------

    def assign_features(
        self, X_classical: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign reusable PQFM features to unseen events via kappa matching.

        Parameters
        ----------
        X_classical:
            Classical feature matrix of shape (n_events, n_classical_features)
            for the unseen events.

        Returns
        -------
        X_pqfm_assigned:
            Array of shape (n_matched, n_pqfm_features) containing the mean
            PQFM vectors for matched events.
        matched_mask:
            Boolean array of shape (n_events,) that is ``True`` for events
            whose kappa was found in the library.  Unmatched events are
            excluded from ``X_pqfm_assigned`` and must be filtered by the
            caller.
        """
        n_events = X_classical.shape[0]
        matched_mask = np.zeros(n_events, dtype=bool)

        if not self._library:
            pqfm_dim = 0
            return np.empty((0, pqfm_dim)), matched_mask

        pqfm_dim = next(iter(self._library.values())).shape[0]
        X_pqfm_out = np.empty((n_events, pqfm_dim), dtype=float)

        kappas = self.compute_kappa_batch(X_classical)
        for i, kappa in enumerate(kappas):
            if kappa in self._library:
                X_pqfm_out[i] = self._library[kappa]
                matched_mask[i] = True

        return X_pqfm_out[matched_mask], matched_mask

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_stats(self, X_classical: np.ndarray) -> dict:
        """
        Compute match coverage statistics for a set of unseen events.

        Parameters
        ----------
        X_classical:
            Classical feature matrix of shape (n_events, n_classical_features).

        Returns
        -------
        dict with keys:

        * ``n_total``                   – total number of unseen events.
        * ``n_matched``                 – events whose kappa exists in the library.
        * ``coverage_rate``             – n_matched / n_total  (0.0 if n_total == 0).
        * ``n_unique_kappas_in_library``– number of distinct kappa keys stored.
        """
        kappas = self.compute_kappa_batch(X_classical)
        n_total = len(kappas)
        n_matched = sum(1 for k in kappas if k in self._library)
        return {
            "n_total": n_total,
            "n_matched": n_matched,
            "coverage_rate": n_matched / n_total if n_total > 0 else 0.0,
            "n_unique_kappas_in_library": len(self._library),
        }

    def resolution(self) -> float:
        """Event-matching resolution = 1 − 1/N_bins.

        Higher resolution means finer discretisation: two events that fall
        into different bins are distinguished.  Within a single bin the
        features are treated as identical.

        Returns
        -------
        float
            Resolution value in [0, 1).
        """
        return 1.0 - 1.0 / self.n_bins
