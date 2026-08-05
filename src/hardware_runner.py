"""
Hardware-approximate PQFM runner for the corporate bond RFQ fill-prediction
experiment.

In production this module would dispatch circuits to an IBM Quantum Runtime
Estimator on *ibm_torino* (IBM Heron r1 processor).  For reproducible
development and CI without hardware access it wraps Qiskit Aer's
:class:`~qiskit_aer.primitives.EstimatorV2` with a depolarizing noise model
whose parameters approximate the Heron r1 device characteristics:

* 1-qubit gate error  ≈ 0.1 %  (noise_level_1q = 0.001)
* 2-qubit gate error  ≈ 1.0 %  (noise_level_2q = 0.01)
* Readout shots       = 4 096

The noise model is intentionally kept simple (gate-level depolarizing) so that
the qualitative effect of hardware noise on the PQFM features can be studied
without access to an actual device.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

from pqfm import (
    normalize_features,
    apply_tanh_transform,
    build_pqfm_circuit,
    build_observables,
)


def _build_noise_model(
    noise_level_1q: float,
    noise_level_2q: float,
) -> NoiseModel:
    """Construct a simple all-qubit depolarizing noise model.

    Depolarizing errors are attached to the standard Aer basis-gate sets:
    1-qubit gates ``{'id', 'rz', 'sx', 'x'}`` and 2-qubit gate ``'cx'``.
    Qiskit Aer transpiles the RXX/RYY/RZZ gates in the PQFM circuit to this
    basis before applying the noise.

    Args:
        noise_level_1q: Depolarizing probability for each 1-qubit gate.
        noise_level_2q: Depolarizing probability for each 2-qubit gate.

    Returns:
        A :class:`~qiskit_aer.noise.NoiseModel` ready to be passed to the Aer
        estimator via ``options.backend_options['noise_model']``.
    """
    nm = NoiseModel()

    err_1q = depolarizing_error(noise_level_1q, 1)
    err_2q = depolarizing_error(noise_level_2q, 2)

    # 1-qubit basis gates used by Aer's default transpiler
    nm.add_all_qubit_quantum_error(err_1q, ["id", "rz", "sx", "x"])
    # 2-qubit basis gate
    nm.add_all_qubit_quantum_error(err_2q, ["cx"])

    return nm


class HardwareRunner:
    """Noisy PQFM feature extractor mimicking IBM Heron r1 hardware.

    Wraps Qiskit Aer's ``EstimatorV2`` with a depolarizing noise model so that
    shot-noise and gate errors are both captured in the estimated expectation
    values.  The interface is intentionally identical to
    :class:`~simulator_runner.SimulatorRunner` to allow drop-in replacement.

    In production, replace ``_make_estimator`` with IBM Runtime's
    ``EstimatorV2`` pointed at ``ibm_torino``.

    Args:
        n_qubits: Number of qubits N (default 15 for testing; 109 for paper).
        blocks: Circuit blocks; M = 2·blocks Trotter steps.
        alpha: Global Heisenberg coupling strength α.
        fiducial_seed: RNG seed for the fiducial product-state preparation.
        noise_level_1q: Depolarizing error probability per 1-qubit gate,
            approximating Heron r1 1Q fidelity (≈99.9 %).
        noise_level_2q: Depolarizing error probability per 2-qubit gate,
            approximating Heron r1 2Q fidelity (≈99.0 %).
        shots: Number of measurement shots per expectation-value estimation.
        backend_name: Label stored in metadata (for bookkeeping only; no
            actual IBM connection is made).
        seed: Global RNG seed for Aer shot sampling.
    """

    def __init__(
        self,
        n_qubits: int = 15,
        blocks: int = 2,
        alpha: float = 0.1,
        fiducial_seed: int = 0,
        noise_level_1q: float = 0.001,
        noise_level_2q: float = 0.01,
        shots: int = 4096,
        backend_name: str = "ibm_torino_mock",
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.blocks = blocks
        self.alpha = alpha
        self.fiducial_seed = fiducial_seed
        self.noise_level_1q = noise_level_1q
        self.noise_level_2q = noise_level_2q
        self.shots = shots
        self.backend_name = backend_name
        self.seed = seed

        self._mu: float | None = None
        self._w: float | None = None

        self._observables = build_observables(n_qubits)
        self._estimator = self._make_estimator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_estimator(self) -> AerEstimatorV2:
        """Build an Aer EstimatorV2 with the depolarizing noise model."""
        noise_model = _build_noise_model(self.noise_level_1q, self.noise_level_2q)

        est = AerEstimatorV2()
        est.options.backend_options = {
            "method": "density_matrix",
            "noise_model": noise_model,
        }
        est.options.run_options = {"seed_simulator": self.seed}
        est.options.default_precision = 1.0 / np.sqrt(self.shots)
        return est

    def _run_single(self, x_pqfm: np.ndarray) -> np.ndarray:
        """Run one preprocessed event vector through the noisy estimator.

        Args:
            x_pqfm: 1-D array of tanh-transformed coupling values.

        Returns:
            1-D array of shape ``(3*n_qubits,)`` of noisy expectation values.
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

        Must be called with training data before :meth:`transform`.

        Args:
            X: Feature matrix of shape ``(n_events, n_features)``.  A single
                global scalar μ and w are computed over all elements.
        """
        _, mu, w = normalize_features(X, use_global=True)
        self._mu = float(mu)
        self._w = float(w)

    def transform(self, X: np.ndarray, batch_size: int = 10) -> np.ndarray:
        """Transform a batch of raw event feature vectors using noisy simulation.

        Applies the same preprocessing (normalization → tanh) and circuit as
        :class:`~simulator_runner.SimulatorRunner`, but executes through the
        depolarizing noise model to approximate shot-limited hardware output.

        Args:
            X: Raw feature matrix of shape ``(n_events, n_features)``.
            batch_size: Progress-logging interval (events); does not affect
                correctness.

        Returns:
            Array of shape ``(n_events, 3*n_qubits)`` of noisy expectation
            values.

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
            x_tilde = (X[i] - self._mu) / self._w
            x_pqfm = apply_tanh_transform(x_tilde)
            out[i] = self._run_single(x_pqfm)
            if batch_size > 0 and (i + 1) % batch_size == 0:
                print(f"  HardwareRunner: processed {i + 1}/{n_events} events")

        return out

    def get_metadata(self) -> dict[str, Any]:
        """Return a metadata dictionary describing this hardware configuration.

        The dictionary is intended to be stored alongside experiment results
        for reproducibility.

        Returns:
            Dictionary with keys:

            * ``backend_name`` – label of the (mock) backend
            * ``noise_level_1q`` – per-gate 1-qubit depolarizing probability
            * ``noise_level_2q`` – per-gate 2-qubit depolarizing probability
            * ``shots`` – measurement shots per expectation value
            * ``n_qubits`` – number of qubits N
            * ``blocks`` – circuit block count
            * ``alpha`` – Heisenberg α parameter
            * ``fiducial_seed`` – fiducial state RNG seed
            * ``aer_seed`` – Aer shot-sampling seed
            * ``date`` – ISO-8601 date string at call time
        """
        return {
            "backend_name": self.backend_name,
            "noise_level_1q": self.noise_level_1q,
            "noise_level_2q": self.noise_level_2q,
            "shots": self.shots,
            "n_qubits": self.n_qubits,
            "blocks": self.blocks,
            "alpha": self.alpha,
            "fiducial_seed": self.fiducial_seed,
            "aer_seed": self.seed,
            "date": datetime.date.today().isoformat(),
        }
