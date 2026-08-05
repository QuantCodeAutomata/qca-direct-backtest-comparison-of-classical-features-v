"""Tests for pqfm module: normalization, tanh transform, circuit building, observables."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from pqfm import (
    normalize_features,
    apply_tanh_transform,
    build_fiducial_state,
    build_pqfm_circuit,
    build_observables,
    transform_event_noiseless,
)

# ---------------------------------------------------------------------------
# Circuit-size constants
# ---------------------------------------------------------------------------
N_TEST_QUBITS: int = 5
# For blocks=1: M=2 Trotter steps, n_pairs=4 bonds → 2*4=8 coupling values
N_TEST_FEATURES: int = 2 * (N_TEST_QUBITS - 1)   # 8


# ---------------------------------------------------------------------------
# Fixtures (inline – no class needed, just module-level RNG)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _make_x() -> tuple[np.ndarray, float, float]:
    """Return (x, mu, w) for a single normalised event vector."""
    x = _RNG.uniform(-1.0, 1.0, size=N_TEST_FEATURES)
    _, mu, w = normalize_features(x.reshape(1, -1))
    return x, float(mu), float(w)


# ---------------------------------------------------------------------------
# normalize_features
# ---------------------------------------------------------------------------

def test_normalize_features_shape():
    """X_normalized must have the same shape as X; mu and w must be scalars."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, size=(10, N_TEST_FEATURES))
    X_norm, mu, w = normalize_features(X)

    assert X_norm.shape == X.shape
    assert np.isscalar(mu) or (hasattr(mu, 'ndim') and mu.ndim == 0)
    assert np.isscalar(w)  or (hasattr(w,  'ndim') and w.ndim  == 0)
    assert float(w) > 0, "std must be strictly positive"


def test_normalize_features_global_vs_per_feature():
    """use_global=True must produce scalar mu/w; use_global=False produces 1-D arrays."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, size=(20, N_TEST_FEATURES))

    _, mu_g, w_g = normalize_features(X, use_global=True)
    _, mu_f, w_f = normalize_features(X, use_global=False)

    # Global: scalars
    assert np.ndim(mu_g) == 0
    assert np.ndim(w_g)  == 0

    # Per-feature: length-N_TEST_FEATURES arrays
    assert np.asarray(mu_f).shape == (N_TEST_FEATURES,)
    assert np.asarray(w_f).shape  == (N_TEST_FEATURES,)


# ---------------------------------------------------------------------------
# apply_tanh_transform
# ---------------------------------------------------------------------------

def test_tanh_output_bounded():
    """Output must lie strictly in (−2π/3, +2π/3) for any finite input."""
    rng = np.random.default_rng(3)
    # Use extreme values to stress the bound
    X_tilde = rng.normal(0.0, 10.0, size=(200, N_TEST_FEATURES))
    result = apply_tanh_transform(X_tilde)

    bound = 2.0 * np.pi / 3.0
    assert np.all(result >= -bound - 1e-12), (
        f"Output below lower bound: min={result.min():.6f}"
    )
    assert np.all(result <=  bound + 1e-12), (
        f"Output above upper bound: max={result.max():.6f}"
    )


def test_tanh_output_zero_for_zero_input():
    """tanh(0) = 0 → transform of zero vector must be the zero vector."""
    zeros = np.zeros((5, N_TEST_FEATURES))
    result = apply_tanh_transform(zeros)
    assert np.allclose(result, 0.0)


# ---------------------------------------------------------------------------
# build_fiducial_state
# ---------------------------------------------------------------------------

def test_build_fiducial_state_qubit_count():
    """Circuit width must equal n_qubits."""
    qc = build_fiducial_state(N_TEST_QUBITS, seed=0)
    assert qc.num_qubits == N_TEST_QUBITS


def test_build_fiducial_state_different_seeds():
    """Different seeds must produce different circuits (different gate angles)."""
    qc0 = build_fiducial_state(N_TEST_QUBITS, seed=0)
    qc1 = build_fiducial_state(N_TEST_QUBITS, seed=1)
    # Extract Ry/Rz parameters and verify they differ between seeds
    def _gate_params(qc):
        return [float(p) for inst in qc.data
                for p in inst.operation.params]
    params0 = _gate_params(qc0)
    params1 = _gate_params(qc1)
    assert params0 != params1, (
        "Circuits with different seeds should have different gate angles"
    )


# ---------------------------------------------------------------------------
# build_observables
# ---------------------------------------------------------------------------

def test_build_observables_count():
    """Must return exactly 3 × n_qubits observables (X, Y, Z per qubit)."""
    obs = build_observables(N_TEST_QUBITS)
    assert len(obs) == 3 * N_TEST_QUBITS, (
        f"Expected {3 * N_TEST_QUBITS} observables, got {len(obs)}"
    )


def test_build_observables_single_qubit_support():
    """Each observable must act non-trivially on exactly one qubit."""
    obs = build_observables(N_TEST_QUBITS)
    for i, op in enumerate(obs):
        # SparsePauliOp.num_qubits == N_TEST_QUBITS (register width)
        assert op.num_qubits == N_TEST_QUBITS, (
            f"Observable {i} has wrong num_qubits: {op.num_qubits}"
        )
        # Exactly one non-Identity Pauli in the Pauli string
        pauli_str = op.paulis.to_labels()[0]
        n_nontrivial = sum(1 for c in pauli_str if c != 'I')
        assert n_nontrivial == 1, (
            f"Observable {i} should be single-qubit, got Pauli string '{pauli_str}'"
        )


# ---------------------------------------------------------------------------
# transform_event_noiseless (full pipeline)
# ---------------------------------------------------------------------------

def test_transform_event_output_shape():
    """For n_qubits=5 with blocks=1, output shape must be (3*5,) = (15,)."""
    x, mu, w = _make_x()
    result = transform_event_noiseless(
        x, mu, w,
        n_qubits=N_TEST_QUBITS,
        blocks=1,
        alpha=0.1,
        fiducial_seed=0,
    )
    assert result.shape == (3 * N_TEST_QUBITS,), (
        f"Expected shape ({3 * N_TEST_QUBITS},), got {result.shape}"
    )


def test_transform_event_values_bounded():
    """Expectation values of Pauli X/Y/Z operators must lie in [-1, 1]."""
    x, mu, w = _make_x()
    result = transform_event_noiseless(
        x, mu, w,
        n_qubits=N_TEST_QUBITS,
        blocks=1,
        alpha=0.1,
        fiducial_seed=0,
    )
    assert np.all(result >= -1.0 - 1e-10), (
        f"Expectation values below -1: min={result.min()}"
    )
    assert np.all(result <=  1.0 + 1e-10), (
        f"Expectation values above +1: max={result.max()}"
    )


def test_deterministic_with_seed():
    """Two calls with identical inputs must return identical outputs (noiseless)."""
    x, mu, w = _make_x()
    kwargs = dict(n_qubits=N_TEST_QUBITS, blocks=1, alpha=0.1, fiducial_seed=7)

    r1 = transform_event_noiseless(x, mu, w, **kwargs)
    r2 = transform_event_noiseless(x, mu, w, **kwargs)

    assert np.allclose(r1, r2, atol=1e-12), (
        "Noiseless transform returned different values for identical inputs"
    )
