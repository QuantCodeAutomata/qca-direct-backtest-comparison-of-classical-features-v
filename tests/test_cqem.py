"""Tests for the CQEMMapper (Classical-Quantum Event Matching) module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from cqem import CQEMMapper


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_kappa_deterministic():
    """_compute_kappa must return the same tuple for identical inputs."""
    mapper = CQEMMapper(n_bins=10)
    x = np.array([0.5, -0.3, 0.1])
    k1 = mapper._compute_kappa(x)
    k2 = mapper._compute_kappa(x)
    assert k1 == k2, f"Kappa not deterministic: {k1} != {k2}"


# ---------------------------------------------------------------------------
# Edge-value binning
# ---------------------------------------------------------------------------

def test_kappa_edge_values():
    """x = -1 (lower bound) must map to bin 0; x = +1 (upper bound) to bin n_bins-1."""
    mapper = CQEMMapper(n_bins=10)

    k_min = mapper._compute_kappa(np.array([-1.0]))
    assert k_min == (0,), (
        f"x=-1.0 should map to bin 0, got {k_min}"
    )

    k_max = mapper._compute_kappa(np.array([1.0]))
    assert k_max == (mapper.n_bins - 1,), (
        f"x=+1.0 should map to bin {mapper.n_bins - 1}, got {k_max}"
    )


def test_kappa_out_of_range_clipped():
    """Values outside feature_range must be clipped to the nearest boundary bin."""
    mapper = CQEMMapper(n_bins=10)

    k_low  = mapper._compute_kappa(np.array([-2.0]))   # below -1 → bin 0
    k_high = mapper._compute_kappa(np.array([ 5.0]))   # above +1 → bin 9

    assert k_low  == (0,), f"Under-range not clipped to bin 0: {k_low}"
    assert k_high == (9,), f"Over-range not clipped to bin 9: {k_high}"


# ---------------------------------------------------------------------------
# Coverage vs. resolution trade-off
# ---------------------------------------------------------------------------

def test_coverage_decreases_with_more_bins():
    """Higher N_bins → finer kappa grid → fewer unseen events find a library match.

    With 50 library events and 200 unseen events drawn from Uniform(-1,+1)
    across a 3-D feature space:
    - n_bins=4  → 4^3=64   possible kappas: high overlap probability.
    - n_bins=60 → 60^3=216k possible kappas: near-zero overlap probability.
    Coverage for n_bins=4 must be >= coverage for n_bins=60.
    """
    rng = np.random.default_rng(0)
    n_lib, n_unseen, n_feat = 50, 200, 3
    n_pqfm = 10

    X_lib    = rng.uniform(-1.0, 1.0, size=(n_lib,    n_feat))
    X_pqfm   = rng.uniform(-1.0, 1.0, size=(n_lib,    n_pqfm))
    X_unseen = rng.uniform(-1.0, 1.0, size=(n_unseen, n_feat))

    coverages: list[float] = []
    for n_bins in [4, 10, 30, 60]:
        mapper = CQEMMapper(n_bins=n_bins)
        mapper.build_library(X_lib, X_pqfm)
        stats = mapper.coverage_stats(X_unseen)
        coverages.append(stats['coverage_rate'])

    # Coarsest binning (n_bins=4) must give at least as high coverage as
    # finest binning (n_bins=60)
    assert coverages[0] >= coverages[-1], (
        f"Coverage should be non-increasing with n_bins; got {coverages}"
    )


# ---------------------------------------------------------------------------
# Mean aggregation in build_library
# ---------------------------------------------------------------------------

def test_build_library_mean_aggregation():
    """Two events that hash to the same kappa must produce the mean PQFM vector."""
    # With n_bins=10, feature_range=(-1,1), bin_width=0.2
    # Both x=-0.95 and x=-0.99 fall in bin 0 for 1-D feature
    mapper = CQEMMapper(n_bins=10)

    X_classical = np.array([[-0.95], [-0.99]])
    pqfm_a = np.array([1.0, 2.0, 3.0])
    pqfm_b = np.array([3.0, 4.0, 5.0])
    X_pqfm = np.vstack([pqfm_a, pqfm_b])

    mapper.build_library(X_classical, X_pqfm)

    k_a = mapper._compute_kappa(X_classical[0])
    k_b = mapper._compute_kappa(X_classical[1])
    assert k_a == k_b, (
        f"Test setup error: x=-0.95 and x=-0.99 must share the same kappa "
        f"(bin 0), got {k_a} and {k_b}"
    )

    expected_mean = (pqfm_a + pqfm_b) / 2.0
    stored = mapper._library[k_a]
    assert np.allclose(stored, expected_mean, atol=1e-12), (
        f"Library mean mismatch: expected {expected_mean}, got {stored}"
    )


# ---------------------------------------------------------------------------
# Unmatched events excluded
# ---------------------------------------------------------------------------

def test_unmatched_events_excluded():
    """assign_features must exclude events whose kappa is absent from the library.

    Library: one event at (0.0, 0.0) → kappa (5, 5) with n_bins=10.
    Unseen:
      - (0.0, 0.0)   → same kappa (5, 5) → should MATCH.
      - (-0.99, -0.99) → kappa (0, 0)    → should NOT match.
    """
    mapper = CQEMMapper(n_bins=10)

    X_lib  = np.array([[0.0, 0.0]])
    X_pqfm = np.array([[1.0, 2.0, 3.0]])
    mapper.build_library(X_lib, X_pqfm)

    # Verify the two unseen events land in different bins
    k_match = mapper._compute_kappa(np.array([0.0,   0.0]))
    k_miss  = mapper._compute_kappa(np.array([-0.99, -0.99]))
    assert k_match in mapper._library, "Library should contain kappa for (0,0)"
    assert k_miss  not in mapper._library, "Library should not contain kappa for (-0.99,-0.99)"

    X_unseen = np.array([[0.0, 0.0], [-0.99, -0.99]])
    X_assigned, matched_mask = mapper.assign_features(X_unseen)

    assert matched_mask.shape == (2,), f"Mask shape mismatch: {matched_mask.shape}"
    assert matched_mask[0] is np.bool_(True) or bool(matched_mask[0]) is True
    assert matched_mask[1] is np.bool_(False) or bool(matched_mask[1]) is False
    assert matched_mask.sum() == 1, f"Expected 1 match, got {matched_mask.sum()}"
    assert X_assigned.shape == (1, 3), (
        f"Expected X_pqfm_assigned shape (1, 3), got {X_assigned.shape}"
    )


# ---------------------------------------------------------------------------
# Coverage stats
# ---------------------------------------------------------------------------

def test_coverage_stats_structure():
    """coverage_stats must return a dict with the required keys."""
    rng = np.random.default_rng(7)
    mapper = CQEMMapper(n_bins=10)

    X_lib  = rng.uniform(-1.0, 1.0, size=(20, 4))
    X_pqfm = rng.uniform(-1.0, 1.0, size=(20, 8))
    mapper.build_library(X_lib, X_pqfm)

    X_unseen = rng.uniform(-1.0, 1.0, size=(50, 4))
    stats = mapper.coverage_stats(X_unseen)

    for key in ('n_total', 'n_matched', 'coverage_rate', 'n_unique_kappas_in_library'):
        assert key in stats, f"Missing key '{key}' in coverage_stats output"

    assert stats['n_total'] == 50
    assert 0 <= stats['n_matched'] <= 50
    assert 0.0 <= stats['coverage_rate'] <= 1.0
    assert stats['n_unique_kappas_in_library'] <= 20  # at most one kappa per library event


# ---------------------------------------------------------------------------
# Resolution formula
# ---------------------------------------------------------------------------

def test_resolution_formula():
    """mapper.resolution() must equal 1 − 1/N_bins for all paper values."""
    for n_bins in [4, 10, 30, 60]:
        mapper = CQEMMapper(n_bins=n_bins)
        expected = 1.0 - 1.0 / n_bins
        assert abs(mapper.resolution() - expected) < 1e-10, (
            f"n_bins={n_bins}: expected resolution {expected:.6f}, "
            f"got {mapper.resolution():.6f}"
        )


# ---------------------------------------------------------------------------
# top_k_features: kappa dimension reduction
# ---------------------------------------------------------------------------

def test_top_k_features_kappa_length():
    """With top_k_features=k, the kappa tuple must have exactly k elements."""
    rng = np.random.default_rng(42)
    n_feat, k = 10, 3
    X_lib  = rng.uniform(-1.0, 1.0, size=(20, n_feat))
    X_pqfm = rng.uniform(-1.0, 1.0, size=(20, 6))

    mapper = CQEMMapper(n_bins=4, top_k_features=k)
    mapper.build_library(X_lib, X_pqfm)

    # After build_library, _selected_feature_idx must be set and have k elements
    assert mapper._selected_feature_idx is not None
    assert len(mapper._selected_feature_idx) == k

    # Every kappa must have exactly k components
    x = rng.uniform(-1.0, 1.0, size=(n_feat,))
    kappa = mapper._compute_kappa(x)
    assert len(kappa) == k, f"Expected kappa length {k}, got {len(kappa)}"


def test_top_k_features_improves_coverage_on_tiny_sample():
    """top_k_features reduces kappa space → higher coverage on tiny library.

    With 20 library events, n_bins=4, and n_feat=8:
    - Full kappa: 4^8 = 65,536 cells → nearly zero coverage.
    - top_k=2:    4^2 = 16  cells → high coverage expected.
    """
    rng = np.random.default_rng(7)
    n_lib, n_unseen, n_feat = 20, 100, 8
    n_pqfm = 6

    X_lib    = rng.uniform(-1.0, 1.0, size=(n_lib,    n_feat))
    X_pqfm   = rng.uniform(-1.0, 1.0, size=(n_lib,    n_pqfm))
    X_unseen = rng.uniform(-1.0, 1.0, size=(n_unseen, n_feat))

    mapper_full = CQEMMapper(n_bins=4, top_k_features=None)
    mapper_full.build_library(X_lib, X_pqfm)
    stats_full = mapper_full.coverage_stats(X_unseen)

    mapper_topk = CQEMMapper(n_bins=4, top_k_features=2)
    mapper_topk.build_library(X_lib, X_pqfm)
    stats_topk = mapper_topk.coverage_stats(X_unseen)

    assert stats_topk['coverage_rate'] >= stats_full['coverage_rate'], (
        f"top_k=2 coverage ({stats_topk['coverage_rate']:.3f}) should be >= "
        f"full coverage ({stats_full['coverage_rate']:.3f})"
    )
