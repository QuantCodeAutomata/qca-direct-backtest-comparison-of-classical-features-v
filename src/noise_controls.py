"""
Noise and robustness control runners for the PQFM experiment.

Provides two controlled variants of the PQFM simulation:

* :class:`NoisySimulatorRunner` – identical to :class:`HardwareRunner` but
  exposed here for explicit robustness testing.  Runs the full PQFM circuit
  through Qiskit Aer with a configurable depolarizing + readout noise model.

* :class:`IdleControlRunner` – replaces all 2-qubit entangling gates (RXX,
  RYY, RZZ) with :class:`~qiskit.circuit.Delay` instructions of equivalent
  angular duration, preserving the approximate decoherence-time exposure while
  removing all two-qubit entanglement.  Both variants are run under the same
  depolarizing noise model to isolate whether performance derives from
  entangling structure or simply noise exposure.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Delay
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

from pqfm import (
    normalize_features,
    apply_tanh_transform,
    build_pqfm_circuit,
    build_observables,
    build_fiducial_state,
)


# ---------------------------------------------------------------------------
# Shared noise-model factory
# ---------------------------------------------------------------------------

def _build_noise_model(
    noise_1q: float,
    noise_2q: float,
    readout_error: float = 0.0,
) -> NoiseModel:
    """Build a depolarizing noise model with optional symmetric readout error.

    Parameters
    ----------
    noise_1q:
        Depolarizing probability per 1-qubit gate.
    noise_2q:
        Depolarizing probability per 2-qubit gate.
    readout_error:
        Per-qubit readout bit-flip probability.  Set to 0 to omit.

    Returns
    -------
    NoiseModel
        Aer noise model ready for injection into :class:`AerEstimatorV2`.
    """
    nm = NoiseModel()

    err_1q = depolarizing_error(noise_1q, 1)
    err_2q = depolarizing_error(noise_2q, 2)

    nm.add_all_qubit_quantum_error(err_1q, ["id", "rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(err_2q, ["cx"])

    if readout_error > 0.0:
        p = readout_error
        ro_matrix = [[1.0 - p, p], [p, 1.0 - p]]
        ro_err = ReadoutError(ro_matrix)
        nm.add_all_qubit_readout_error(ro_err)

    return nm


# ---------------------------------------------------------------------------
# Idle-circuit helper
# ---------------------------------------------------------------------------

def _build_idle_pqfm_circuit(
    x_pqfm: np.ndarray,
    n_qubits: int,
    blocks: int,
    alpha: float,
    fiducial_seed: int,
    delay_dt: int = 1000,
) -> QuantumCircuit:
    """Build a PQFM circuit where every 2-qubit entangling gate is replaced
    by a pair of single-qubit :class:`~qiskit.circuit.Delay` instructions.

    The fiducial state preparation and single-qubit structure are preserved;
    only the two-qubit RXX/RYY/RZZ operations are absent.  The delay
    duration (in ``'dt'`` units) approximates the gate time, preserving
    the decoherence exposure of qubits that would have participated in a
    two-qubit interaction.

    Parameters
    ----------
    x_pqfm:
        1-D array of tanh-transformed coupling values (not used for delay
        duration; included for interface symmetry with build_pqfm_circuit).
    n_qubits:
        Number of qubits N.
    blocks:
        Circuit blocks; M = 2·blocks Trotter steps.
    alpha:
        Global α parameter (not used in delay circuit; kept for symmetry).
    fiducial_seed:
        Seed for the fiducial state preparation.
    delay_dt:
        Duration of each inserted Delay instruction in ``'dt'`` units.
        Default 1000 dt ≈ typical 2-qubit gate time.

    Returns
    -------
    QuantumCircuit
        Circuit of width *n_qubits* with fiducial state followed by delay
        layers in place of each Trotter step.
    """
    n_pairs = n_qubits - 1
    M = 2 * blocks

    qc = QuantumCircuit(n_qubits)
    qc.compose(build_fiducial_state(n_qubits, fiducial_seed), inplace=True)

    for _ in range(M):
        # Odd bonds first, then even bonds – mirrors Trotter layer structure
        for j in range(1, n_pairs, 2):          # odd bonds
            qc.append(Delay(delay_dt, unit="dt"), [j])
            qc.append(Delay(delay_dt, unit="dt"), [j + 1])
        for j in range(0, n_pairs, 2):          # even bonds
            qc.append(Delay(delay_dt, unit="dt"), [j])
            qc.append(Delay(delay_dt, unit="dt"), [j + 1])

    return qc


# ---------------------------------------------------------------------------
# NoisySimulatorRunner
# ---------------------------------------------------------------------------

class NoisySimulatorRunner:
    """
    Runs PQFM with a noisy simulation to test whether noise alone
    reproduces the hardware uplift effect.
    Uses depolarizing noise model with configurable error rates.

    Interface mirrors :class:`~simulator_runner.SimulatorRunner` and
    :class:`~hardware_runner.HardwareRunner` for drop-in substitution.

    Parameters
    ----------
    n_qubits:
        Number of qubits N (default 15 for testing; 109 for paper).
    blocks:
        Circuit blocks; M = 2·blocks Trotter steps.
    alpha:
        Global Heisenberg coupling strength α.
    fiducial_seed:
        RNG seed for the fiducial product-state preparation.
    noise_1q:
        Depolarizing error probability per 1-qubit gate.
    noise_2q:
        Depolarizing error probability per 2-qubit gate.
    readout_error:
        Symmetric readout bit-flip probability per qubit.
    shots:
        Measurement shots for precision estimation.
    seed:
        Global Aer shot-sampling seed.
    """

    def __init__(
        self,
        n_qubits: int = 15,
        blocks: int = 2,
        alpha: float = 0.1,
        fiducial_seed: int = 0,
        noise_1q: float = 0.001,
        noise_2q: float = 0.01,
        readout_error: float = 0.02,
        shots: int = 4096,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.blocks = blocks
        self.alpha = alpha
        self.fiducial_seed = fiducial_seed
        self.noise_1q = noise_1q
        self.noise_2q = noise_2q
        self.readout_error = readout_error
        self.shots = shots
        self.seed = seed

        self._mu: float | None = None
        self._w: float | None = None

        self._observables = build_observables(n_qubits)
        self._estimator = self._make_estimator()

    def _make_estimator(self) -> AerEstimatorV2:
        """Build a noisy AerEstimatorV2 with depolarizing + readout errors."""
        noise_model = _build_noise_model(self.noise_1q, self.noise_2q, self.readout_error)
        est = AerEstimatorV2()
        est.options.backend_options = {
            "method": "density_matrix",
            "noise_model": noise_model,
        }
        est.options.run_options = {"seed_simulator": self.seed}
        est.options.default_precision = 1.0 / np.sqrt(self.shots)
        return est

    def fit_normalization(self, X: np.ndarray) -> None:
        """Compute and store global normalisation statistics from *X*.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_events, n_features).
        """
        _, mu, w = normalize_features(X, use_global=True)
        self._mu = float(mu)
        self._w = float(w)

    def transform(self, X: np.ndarray, batch_size: int = 10) -> np.ndarray:
        """Transform a batch of raw feature vectors using the noisy estimator.

        Parameters
        ----------
        X:
            Raw feature matrix of shape (n_events, n_features).
        batch_size:
            Progress-logging interval (events); no effect on correctness.

        Returns
        -------
        np.ndarray
            Shape (n_events, 3·n_qubits) of noisy expectation values.

        Raises
        ------
        RuntimeError
            If :meth:`fit_normalization` has not been called first.
        """
        if self._mu is None or self._w is None:
            raise RuntimeError("Call fit_normalization() before transform().")

        n_events = X.shape[0]
        out = np.empty((n_events, 3 * self.n_qubits), dtype=float)

        for i in range(n_events):
            x_tilde = (X[i] - self._mu) / self._w
            x_pqfm = apply_tanh_transform(x_tilde)
            circuit = build_pqfm_circuit(
                x_pqfm, self.n_qubits, self.blocks, self.alpha, self.fiducial_seed
            )
            job = self._estimator.run([(circuit, self._observables)])
            result = job.result()
            out[i] = np.asarray(result[0].data.evs, dtype=float)
            if batch_size > 0 and (i + 1) % batch_size == 0:
                print(f"  NoisySimulatorRunner: processed {i + 1}/{n_events} events")

        return out


# ---------------------------------------------------------------------------
# IdleControlRunner
# ---------------------------------------------------------------------------

class IdleControlRunner:
    """
    Circuit control variant: replace 2-qubit entangling gates with
    idle delays preserving approximate decoherence time exposure.

    Tests whether PQFM performance depends on the entangling structure of
    the Heisenberg Trotter steps or merely on noise exposure.  By replacing
    all RXX/RYY/RZZ gates with :class:`~qiskit.circuit.Delay` instructions
    of equivalent duration and applying the same depolarizing noise model,
    any performance difference isolates the contribution of entanglement.

    Parameters
    ----------
    n_qubits:
        Number of qubits N.
    blocks:
        Circuit blocks; M = 2·blocks Trotter steps.
    alpha:
        Global α parameter (kept for interface symmetry; not used in delays).
    fiducial_seed:
        Seed for the fiducial state preparation.
    noise_1q:
        Depolarizing error probability per 1-qubit gate.
    noise_2q:
        Depolarizing error probability per 2-qubit gate.
    shots:
        Measurement shots for precision estimation.
    seed:
        Global Aer shot-sampling seed.
    """

    def __init__(
        self,
        n_qubits: int = 15,
        blocks: int = 2,
        alpha: float = 0.1,
        fiducial_seed: int = 0,
        noise_1q: float = 0.001,
        noise_2q: float = 0.01,
        shots: int = 4096,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.blocks = blocks
        self.alpha = alpha
        self.fiducial_seed = fiducial_seed
        self.noise_1q = noise_1q
        self.noise_2q = noise_2q
        self.shots = shots
        self.seed = seed

        self._mu: float | None = None
        self._w: float | None = None

        self._observables = build_observables(n_qubits)
        self._estimator = self._make_estimator()

    def _make_estimator(self) -> AerEstimatorV2:
        """Build an AerEstimatorV2 with the same noise model as NoisySimulatorRunner."""
        noise_model = _build_noise_model(self.noise_1q, self.noise_2q, readout_error=0.0)
        est = AerEstimatorV2()
        est.options.backend_options = {
            "method": "density_matrix",
            "noise_model": noise_model,
        }
        est.options.run_options = {"seed_simulator": self.seed}
        est.options.default_precision = 1.0 / np.sqrt(self.shots)
        return est

    def fit_normalization(self, X: np.ndarray) -> None:
        """Compute and store global normalisation statistics from *X*.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_events, n_features).
        """
        _, mu, w = normalize_features(X, use_global=True)
        self._mu = float(mu)
        self._w = float(w)

    def transform(self, X: np.ndarray, batch_size: int = 10) -> np.ndarray:
        """Build idle-delay circuit for each event and return expectation values.

        All two-qubit entangling gates in the Heisenberg Trotter steps are
        replaced by :class:`~qiskit.circuit.Delay` instructions.  The same
        depolarizing noise model is applied to the single-qubit gates and
        fiducial-state preparation that remain.

        Parameters
        ----------
        X:
            Raw feature matrix of shape (n_events, n_features).
        batch_size:
            Progress-logging interval; no effect on correctness.

        Returns
        -------
        np.ndarray
            Shape (n_events, 3·n_qubits) of expectation values from the
            idle-delay circuit.

        Raises
        ------
        RuntimeError
            If :meth:`fit_normalization` has not been called first.
        """
        if self._mu is None or self._w is None:
            raise RuntimeError("Call fit_normalization() before transform().")

        n_events = X.shape[0]
        out = np.empty((n_events, 3 * self.n_qubits), dtype=float)

        for i in range(n_events):
            x_tilde = (X[i] - self._mu) / self._w
            x_pqfm = apply_tanh_transform(x_tilde)
            circuit = _build_idle_pqfm_circuit(
                x_pqfm, self.n_qubits, self.blocks, self.alpha, self.fiducial_seed
            )
            job = self._estimator.run([(circuit, self._observables)])
            result = job.result()
            out[i] = np.asarray(result[0].data.evs, dtype=float)
            if batch_size > 0 and (i + 1) % batch_size == 0:
                print(f"  IdleControlRunner: processed {i + 1}/{n_events} events")

        return out
