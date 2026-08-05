# PQFM Bond RFQ Fill-Prediction: Classical vs Quantum Feature Maps

Reproduction of the paper's experiments testing whether **Projected Quantum Feature Map (PQFM)** transformed features improve out-of-sample RFQ fill-probability prediction in European corporate bond markets, and whether hardware-generated PQFM features outperform noiseless simulated PQFM features.

---

## Repository Structure

```
qca-direct-backtest-comparison/
│
├── data/                       # Synthetic RFQ dataset generation & storage
│   ├── generate_synthetic_rfq.py   # Synthetic data generator (mimics proprietary data)
│   └── rfq_active_window.parquet   # Pre-generated synthetic dataset (143,912 events)
│
├── src/                        # Core implementation
│   ├── config.py               # Experiment parameters, seeds, hyperparameter grids
│   ├── data_loader.py          # Dataset loading, filtering, active-window construction
│   ├── sampling.py             # Constrained random 16,000-event representative sampling
│   ├── pqfm.py                 # PQFM circuit construction (Heisenberg ansatz, N qubits)
│   ├── simulator_runner.py     # Noiseless Aer simulation of PQFM expectation values
│   ├── hardware_runner.py      # IBM Quantum hardware runner (ibm_torino stub + metadata)
│   ├── noise_controls.py       # Noisy-simulation controls (Aer noise models)
│   ├── cqem.py                 # Classical-Quantum Event Matching (CQEM) mapper
│   ├── backtest.py             # Temporal blinded backtesting engine
│   ├── models.py               # LR / XGB / RF / NN model families + hyperparameter grids
│   └── evaluation.py           # AUC aggregation, uplift calculations, plot generation
│
├── exp/                        # Experiment entry points
│   ├── exp1_direct_backtest.py     # Exp 1: Classical vs Sim-PQFM vs HW-PQFM
│   ├── exp2_cqem.py               # Exp 2: CQEM reuse of hardware features
│   └── exp3_robustness.py         # Exp 3: Cross-device, drift, and circuit controls
│
├── tests/                      # Pytest test suite (38 tests)
│   ├── test_data.py
│   ├── test_pqfm.py
│   ├── test_cqem.py
│   └── test_backtest.py
│
├── results/                    # Experiment artifacts
│   ├── RESULTS.md              # Master summary of all findings
│   ├── RESULTS_exp1.md / RESULTS_exp2.md / RESULTS_exp3.md
│   ├── metrics_exp*.csv        # Per-run AUC metrics
│   ├── auc_vs_blinding_exp*.png
│   └── feature_histograms_*.png
│
├── run_experiments.py          # Master runner (all three experiments)
├── AGENTS.md                   # Persistent agent memory
├── .gitignore
└── README.md
```

---

## Experiment Overview

### Experiment 1 — Direct Backtest Comparison

Evaluates four input representations over 16,000 sampled RFQ events:

| Variant | Features | Notes |
|---------|----------|-------|
| `classical` | 216 engineered features | Baseline |
| `simulation` | 327 PQFM features | Noiseless Qiskit Aer simulation |
| `hardware_short` | 327 PQFM features | Shorter circuit (blocks=1, α=1.0, seed=1) |
| `hardware_long` | 327 PQFM features | Longer circuit (blocks=2, α=0.1, seed=0) |

Models: LR, XGB, RF, NN  
Backtesting: Blinded temporal splits (0d–4d blinding windows)  
Expected ordering: `hardware_long > hardware_short >> classical ≈ simulation`

### Experiment 2 — CQEM Reuse

Tests whether hardware-long PQFM features from the 16,000-event library can be matched to **unseen events** via discretized classical feature bins (`N_bins ∈ {4, 10, 30, 60}`). Best performance near N_bins=30 (~97% matching resolution).

### Experiment 3 — Hardware Robustness

- Cross-device reproducibility (ibm_torino / ibm_fez / ibm_marrakesh stubs)
- Repeated identical-input drift analysis
- Circuit control variants (idle operations preserving decoherence time)
- Noisy classical simulation controls

---

## PQFM Methodology

The Projected Quantum Feature Map uses a Heisenberg-ansatz Trotterized embedding:

1. **Preprocessing**: `x̃_i = (x_i − μ) / w`, then `x_i^(PQFM) = (2π/3) · tanh(x̃_i)`
2. **Hamiltonian**: `H_J = (1/4) Σ_j J_j (X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1})`
3. **Unitary**: `U_{J,α} = (∏_odd U_j · ∏_even U_j)^M`, where `U_j = exp(−iα/(2M) · H_j)`
4. **Observables**: All 1-local Pauli {X_j, Y_j, Z_j} on N qubits → q = 3N features

**Short circuit**: blocks=1, α=1.0, seed=1 | **Long circuit**: blocks=2, α=0.1, seed=0

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt   # or install the listed packages

# Generate synthetic dataset (excluded from git due to ~300 MB size)
python data/generate_synthetic_rfq.py

# Run all experiments (compact/fast mode)
python run_experiments.py

# Run individual experiments
python exp/exp1_direct_backtest.py
python exp/exp2_cqem.py
python exp/exp3_robustness.py

# Run tests
pytest tests/ -v
```

---

## Data Note

The paper uses **proprietary** European corporate bond RFQ data (2024-07-24 to 2024-10-29, 143,912 events, 216 engineered features). This repository uses a **fully synthetic dataset** generated by `data/generate_synthetic_rfq.py` that:
- Preserves the statistical structure (216 features ∈ [−1, +1], ~40% fill rate)
- Covers the same 69 trading days
- Enables end-to-end methodology verification without proprietary data access

Replace `rfq_active_window.parquet` with your own dataset conforming to the schema to reproduce results on real data.

---

## Configuration

Key parameters are controlled via `src/config.py`:

```python
N_QUBITS = 8           # Paper: 109 (reduced for tractable simulation)
N_EVENTS_SAMPLE = 200  # Paper: 16,000 (compact run)
TRAINING_SIZES = [50, 100]          # Paper: [500, 1000, 1500, 2000]
BLINDING_DAYS = [0, 1]              # Paper: [0, 1, 2, 3, 4]
```

---

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the full experiment summary, AUC tables, and analysis.

---

## Dependencies

- `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`
- `qiskit >= 2.0`, `qiskit-aer >= 0.17`
- `matplotlib`, `seaborn`
- `pytest`, `mlflow` (optional experiment tracking)
