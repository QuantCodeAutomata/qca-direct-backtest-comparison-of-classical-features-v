"""
Noiseless PQFM simulator runner for the corporate bond RFQ fill-prediction
experiment.

For circuits with n_qubits ≤ 20 the exact Qiskit
:class:`~qiskit.primitives.StatevectorEstimator` is used.  For larger systems
(including the full N=109 Heisenberg chain) the Qiskit Aer
:class:`~qiskit_aer.primitives.EstimatorV2` is used with the
``matrix_product_state`` simulation method, which avoids storing the full 2^N
statevector by exploiting the limited entanglement of the 1-D ansatz.
"""

from __future__ import annotations

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

from pqfm import (
    normalize_features,
    apply_tanh_transform,
    build_pqfm_circuit,
    build_observables,
)

# Threshold above which the MPS backend is used instead of the exact
# statevector simulator.
_MPS_QUBIT_THRESHOLD = 20


class SimulatorRunner:
    """Noiseless PQFM feature extractor backed by Qiskit (Aer) simulation.

    For systems with ``n_qubits <= 20`` an exact statevector simulation is
    performed.  For larger systems (including the paper's N=109 circuit) Qiskit
    Aer's matrix-product-state (MPS) method is used, which can handle hundreds
    of qubits for circuits with limited entanglement.

    Typical usage::

        runner = SimulatorRunner(n_qubits=109, blocks=1, alpha=1.0)
        runner.fit_normalization(X_train)
        X_feat = runner.transform(X_train)          # shape (n_train, 327)

    Args:
        n_qubits: Number of qubits N in the Heisenberg chain (default 15 for
            testing; set to 109 for the full experiment).
        blocks: Circuit blocks; M = 2·blocks Trotter steps.
        alpha: Global Heisenberg coupling strength α.
        fiducial_seed: RNG seed for the fiducial product-state preparation.
        use_global_norm: When ``True`` (default) a single scalar μ, w is
            computed over all elements of the training matrix; otherwise
            per-feature statistics are used.
    """

    def __init__(
        self,
        n_qubits: int = 15,
        blocks: int = 1,
        alpha: float = 1.0,
        fiducial_seed: int = 1,
        use_global_norm: bool = True,
    ) -> None:
        self.n_qubits = n_qubits
        self.blocks = blocks
        self.alpha = alpha
        self.fiducial_seed = fiducial_seed
        self.use_global_norm = use_global_norm

        self._mu: float | None = None
        self._w: float | None = None

        # Pre-build the estimator and the shared observable list once.
        self._observables = build_observables(n_qubits)
        self._estimator = self._make_estimator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_estimator(self) -> StatevectorEstimator | AerEstimatorV2:
        """Instantiate the appropriate estimator based on qubit count."""
        if self.n_qubits <= _MPS_QUBIT_THRESHOLD:
            return StatevectorEstimator()

        est = AerEstimatorV2()
        est.options.backend_options = {"method": "matrix_product_state"}
        return est

    def _run_single(self, x_pqfm: np.ndarray) -> np.ndarray:
        """Run one preprocessed event vector through the estimator.

        Args:
            x_pqfm: 1-D array of tanh-transformed coupling values.

        Returns:
            1-D array of shape ``(3*n_qubits,)`` of expectation values.
        """
        circuit = build_pqfm_circuit(
            x_pqfm,
            self.n_qubits,
            self.blocks,
            self.alpha,
            self.fiducial_seed,
        )
        job = self._estimator.run([(circuit, self._observables)])
        result = job.result()
        return np.asarray(result[0].data.evs, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_normalization(self, X: np.ndarray) -> None:
        """Compute and store global normalization statistics from *X*.

        Must be called with training data before calling :meth:`transform` or
        :meth:`transform_single` on any split.

        Args:
            X: Feature matrix of shape ``(n_events, n_features)``.  Statistics
                are computed over all elements when ``use_global_norm=True``.
        """
        _, mu, w = normalize_features(X, use_global=self.use_global_norm)
        self._mu = float(mu) if np.ndim(mu) == 0 else mu  # type: ignore[assignment]
        self._w = float(w) if np.ndim(w) == 0 else w      # type: ignore[assignment]

    def transform(self, X: np.ndarray, batch_size: int = 10) -> np.ndarray:
        """Transform a batch of raw event feature vectors into PQFM features.

        Args:
            X: Raw feature matrix of shape ``(n_events, n_features)``.
            batch_size: Number of events to process before logging progress.
                Has no effect on correctness; included for visibility when
                running large batches.

        Returns:
            Array of shape ``(n_events, 3*n_qubits)`` containing expectation
            values for each event.

        Raises:
            RuntimeError: If :meth:`fit_normalization` has not been called.
        """
        if self._mu is None or self._w is None:
            raise RuntimeError(
                "Call fit_normalization() before transform()."
            )

        n_events = X.shape[0]
        out = np.empty((n_events, 3 * self.n_qubits), dtype=float)

        for i in range(n_events):
            out[i] = self.transform_single(X[i])
            if batch_size > 0 and (i + 1) % batch_size == 0:
                print(f"  SimulatorRunner: processed {i + 1}/{n_events} events")

        return out

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """Transform a single raw event feature vector into PQFM features.

        Args:
            x: 1-D raw feature array of length ``n_features``.

        Returns:
            1-D array of shape ``(3*n_qubits,)`` containing expectation
            values ⟨X₀⟩, ⟨Y₀⟩, ⟨Z₀⟩, …, ⟨Z_{N-1}⟩.

        Raises:
            RuntimeError: If :meth:`fit_normalization` has not been called.
        """
        if self._mu is None or self._w is None:
            raise RuntimeError(
                "Call fit_normalization() before transform_single()."
            )

        x_tilde = (x - self._mu) / self._w
        x_pqfm = apply_tanh_transform(x_tilde)
        return self._run_single(x_pqfm)
