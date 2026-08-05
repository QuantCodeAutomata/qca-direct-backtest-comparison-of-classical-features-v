# Repository Knowledge — PQFM Corporate Bond RFQ

## Project layout
```
src/
  pqfm.py             # Core circuit-building primitives (paper formulas)
  simulator_runner.py # Noiseless PQFM feature extractor (Qiskit / Aer MPS)
  hardware_runner.py  # Noisy PQFM stub (depolarizing noise, ibm_torino_mock)
```

## Installed packages (tested versions)
- qiskit 2.5.1
- qiskit-aer 0.17.2
- numpy 2.2.6
- scipy 1.14.1

## Key design decisions

### Paper formulas (pqfm.py)
- **Normalize**: `x̃ = (x − μ) / w`  — global scalar μ, w over all values in the batch.
- **Bounded transform**: `x_pqfm = (2π/3) · tanh(x̃)` — keeps coupling angles finite.
- **Heisenberg bond j**: `H_j = J_j (XX + YY + ZZ)` between qubits j and j+1.
- **Trotter angle**: `θ = α · J_j / (2M)` where M = 2·blocks.
- **Bond unitary**: `RXX(2θ) · RYY(2θ) · RZZ(2θ)` — first-order product-formula Trotter.
- **Layer order**: odd bonds (j=1,3,…) first, then even bonds (j=0,2,…).
- **Fiducial state**: per-qubit `Ry(θ) Rz(φ)` with angles from `np.random.default_rng(seed)`.
- **Observables**: `[X₀,Y₀,Z₀, X₁,Y₁,Z₁, …]` — 3·N SparsePauliOps.
  Qiskit convention: rightmost Pauli string char = qubit 0.
  For qubit j of N: `'I'*(N-1-j) + 'P' + 'I'*j`.

### Feature indexing
- Input `X_pqfm` always has `2·n_pairs` elements (= 216 for N=109).
- Step k (1-indexed) uses coupling `X_pqfm[((k-1)*n_pairs + j) % len(X_pqfm)]`.
- For blocks=2 (M=4), steps 3 and 4 cyclically reuse steps 1 and 2's features.

### Qiskit Primitives API (V2)
- Both `StatevectorEstimator` and `AerEstimatorV2` share the same PUB interface:
  `job = estimator.run([(circuit, list_of_SparsePauliOp)])`
  `evs = job.result()[0].data.evs  # shape (n_obs,)`
- MPS backend: `est.options.backend_options = {'method': 'matrix_product_state'}`.
- Noisy backend: `est.options.backend_options = {'method': 'density_matrix', 'noise_model': nm}`.
- Aer seed: `est.options.run_options = {'seed_simulator': seed}` (not a constructor arg).
- Shot precision: `est.options.default_precision = 1/sqrt(shots)`.

### Runner classes
- `SimulatorRunner`: uses `StatevectorEstimator` for n_qubits ≤ 20, else Aer MPS.
- `HardwareRunner`: always uses `AerEstimatorV2` + `density_matrix` + depolarizing noise.
  `get_metadata()` returns dict with backend_name, noise params, shots, date, seeds.
- Both require `fit_normalization(X_train)` before `transform()`.

### Tested with n_qubits=5 (n_pairs=4, feature_len=8, 15 output features)
- For production N=109: 108 pairs, 216 features in, 327 features out.

---

## Data pipeline (added in session 2)

### Directory layout additions
```
src/
  config.py          # All experiment constants and hyperparameter grids
  data_loader.py     # load_rfq_data / filter_active_window / validate_dataset
  sampling.py        # sample_representative / get_trading_days
data/
  __init__.py
  generate_synthetic_rfq.py   # Synthetic RFQ generator
  rfq_active_window.parquet   # Generated on first run (143,912 rows)
exp/
  __init__.py
tests/
  __init__.py
results/
  .gitkeep
```

### Synthetic dataset (generate_synthetic_rfq.py)
- 143,912 events × 222 columns: event_id, timestamp, bond_id, ticker, sector, f_000…f_215, fill
- 70 business days: 2024-07-24 → 2024-10-29; events distributed via multinomial(n, equal_weights)
- Timestamps: UTC, microsecond precision, within 08:00–17:00 each day
  - Computed as: `day.value // 1000` (ns→µs) + `8*3600*1e6` + random µs offset
- Features: float64 Uniform(−1, +1); 500 bonds, 10 sectors
- Fill labels: σ(0.3·Σ_{k<5} f_k + log(0.4/0.6)) → ~40.4 % fill rate
- 5 % of bonds (25/500) have `_MATURING` suffix for test exclusion

### data_loader.py design
- `load_rfq_data(path=None)`: reads parquet; if missing AND path is None,
  generates via `importlib.util.spec_from_file_location` (no sys.path mutation)
- `filter_active_window(df, start, end)`: temporal [start, end] + removes _MATURING bonds
- `validate_dataset(df)`: checks _REQUIRED_COLS, feature pattern `f_\d+`, datetime dtype,
  NaN-free features, binary fill; raises ValueError on failure

### sampling.py design
- `get_trading_days(df)`: floors UTC timestamps to day, returns sorted list of pd.Timestamps
- `sample_representative(df, n=16000, seed=42)`: Hamilton's largest-remainder proportional
  allocation across trading days; per-day seeds derived from master_rng.integers(0, 2^31);
  result sorted by timestamp; every non-empty day is represented

### config.py notes
- `import numpy as np` is at top of file; LR_PARAM_GRID['C'] built with `10**np.array([...])`
- `N_PQFM_FEATURES = 327 = 3 * N_QUBITS`; `N_PAIRS = 108 = N_QUBITS - 1`
- Simulation small mode: N=15 qubits, 28 features in, 45 features out

---

## ML models and backtesting engine (added in session 3)

### New files
```
src/
  models.py      # Model factory + hyperparameter grids (LR, XGB, RF, NN)
  backtest.py    # TemporalBacktest class (strict temporal CV)
  evaluation.py  # Aggregation, uplift, and visualization helpers
```

### models.py
- `MODEL_FAMILIES = ['LR', 'XGB', 'RF', 'NN']`
- `get_model_and_grid(family)` → `(estimator, param_grid)`:
  - LR: `penalty='l2'`, `max_iter=10000`, `solver` ∈ 6 options, C ∈ 10^[-4…4] (step 0.8, 11 values)
  - XGB: `eval_metric='logloss'`, `use_label_encoder=False`, `verbosity=0`; max_depth 5 vals, n_estimators 6 vals, lr 4 vals
  - RF: criterion 3 vals, n_estimators 6 vals
  - NN: `max_iter=10000`; 8 architectures, 3 activations, 3 lr schedules
- `train_best_model(X_train, y_train, family, cv=3, n_jobs=1, random_state=42)`:
  - Returns `None` for single-class y_train or when `counts.min() < cv`
  - Uses `GridSearchCV(scoring='roc_auc', error_score=np.nan, refit=True)`
  - Suppresses `ConvergenceWarning` inside `warnings.catch_warnings()` context
  - Injects `random_state` via `estimator.set_params(random_state=...)` when supported
- `predict_proba(model, X_test)` → `model.predict_proba(X_test)[:, 1]`

### backtest.py
- `TemporalBacktest.__init__`: copies df, converts timestamp to datetime, sorts by timestamp
- Default `training_sizes=[500, 1000, 2000, 5000]`, `blinding_days=[0, 1, 2, 5, 10]`, `min_test_events=50`
- `_get_blind_gap_seconds(n)`: returns `1e-6` for n=0, else `n * 86400.0`
- `run_single`: iterates `df[timestamp_col].dt.date` unique dates; cutoff = `first_test_ts − Timedelta(seconds=gap)`; training = `train_candidates.iloc[-training_size:]`; skips if test has single class or train is None
- `run_all`: iterates `product(families, training_sizes, blinding_days)`; logs INFO every 10 combos; returns empty DataFrame with correct columns when no results

### evaluation.py
- `aggregate_auc_results`: groups by `[dataset_type?, family, training_size, blinding_day]`; produces `median_auc`, `std_auc`, `count`
- `compute_relative_uplift`: aggregates internally, merges baseline rows on `[family, training_size, blinding_day]`, computes `(auc − baseline_auc) / baseline_auc * 100`; raises ValueError if `dataset_type` absent or baseline not found
- `plot_auc_vs_blinding`: rows=dataset_types × cols=training_sizes; lines per family with `fill_between` ±std bands; uses seaborn whitegrid + tab10 palette
- `plot_feature_histograms`: overlaid density histograms per feature index; filters NaN/inf with `np.isfinite`
- `plot_predicted_probabilities`: one subplot per dataset_type; two histograms per subplot (label 0=steelblue, label 1=tomato)
- `save_results_table`: agg_df → `path`, uplift_df → `{stem}_uplift.csv`; auto-creates dirs
- `generate_results_markdown`: uses `df.to_markdown()` (requires tabulate); falls back to `to_string()`; emits key-findings bullets (best uplift, best AUC)

### Import path pattern for backtest.py
Uses `try: from models import ... except ImportError: from src.models import ...` to support both
running from `src/` and from project root with `src/` on sys.path.

---

## CQEM, noise controls, experiments, and tests (added in session 4)

### New files
```
src/
  cqem.py            # CQEMMapper: kappa hashing, library build, feature assignment
  noise_controls.py  # NoisySimulatorRunner, IdleControlRunner, _build_noise_model
exp/
  exp1_direct_backtest.py  # 4-variant comparison (classical + sim_short + sim_long + hw)
  exp2_cqem.py             # CQEM n_bins sweep {4, 10, 30, 60}
  exp3_robustness.py       # Noiseless / noisy / idle control + drift analysis
tests/
  test_data.py     # 8 tests: shape, feature range, binary labels, filter, sampling
  test_pqfm.py     # 11 tests: normalize, tanh, fiducial, observables, transform
  test_backtest.py # 9 tests: model families, LR grid, blinding gaps, no-leakage, AUC
  test_cqem.py     # 8 tests: determinism, edge bins, coverage, mean agg, resolution
```

### cqem.py — CQEMMapper
- `__init__(n_bins=30, feature_range=(-1,1))`: validates n_bins ≥ 1, lo < hi
- `_compute_kappa(x)`: clips to feature_range; bin_index = floor((x_clip − lo) / bin_width); clips to [0, n_bins−1]; returns tuple of ints
- `compute_kappa_batch(X)`: list comprehension over `_compute_kappa`
- `build_library(X_classical, X_pqfm)`: online mean update (avoids storing full list); each kappa → mean PQFM vector; accumulates across multiple calls
- `assign_features(X_classical)` → `(X_pqfm_assigned, matched_mask)`: matched_mask is bool (n_events,); X_pqfm_assigned is (n_matched, n_pqfm); unmatched rows excluded (not imputed)
- `coverage_stats(X_classical)` → dict with `n_total, n_matched, coverage_rate, n_unique_kappas_in_library`
- `resolution()` → `1 - 1/n_bins`

### noise_controls.py
- `_build_noise_model(noise_1q, noise_2q, readout_error)`: applies `depolarizing_error` to gates `[id, rz, sx, x]` and `[cx]`; adds symmetric `ReadoutError` if readout_error > 0
- `NoisySimulatorRunner`: mirrors HardwareRunner interface; uses `density_matrix` Aer method; `fit_normalization` + `transform` with batch_size progress logging
- `IdleControlRunner`: replaces 2Q entangling gates with `Delay(delay_dt, unit='dt')` on both qubits of each bond; same noise model (no readout error); `_build_idle_pqfm_circuit` builds idle circuit with fiducial state preserved
- Idle circuit structure: fiducial state → M layers of (odd-bond delays, then even-bond delays); each bond's two qubits each get a Delay instruction
- Both classes: import `normalize_features, apply_tanh_transform, build_pqfm_circuit / build_fiducial_state, build_observables` from `pqfm.py`

### Experiment scripts (exp/)
All three scripts share the same pattern:
1. `sys.path.insert(0, '../src')` for src imports
2. `load_rfq_data() → filter_active_window() → sample_representative()`
3. Runner(s) instantiated; `fit_normalization()` then `transform()`
4. `TemporalBacktest → run_all()` for each variant
5. `aggregate_auc_results`, `save_results_table`, `generate_results_markdown`, plots saved to `save_dir/`
6. Errors caught with `logger.warning`; returns dict with result DataFrames

- **exp1**: 4 variants = classical + sim_short (blocks=1) + sim_long (blocks=2) + hardware; save to `metrics_exp1.csv`, `RESULTS_exp1.md`, `auc_vs_blinding_exp1.png`, `feature_histograms_exp1.png`, `config_exp1.json`
- **exp2**: CQEM loop over `n_bins_list`; pool split 50/50 library/unseen; hardware-long PQFM for library; coverage stats saved to `coverage_exp2.csv`; backtest only when n_matched ≥ 10
- **exp3**: three runners (noiseless/noisy/idle); drift analysis via `_compute_drift_stats` which runs `NoisySimulatorRunner` × 3 different seeds, stacks results, computes per-feature std; drift saved to `drift_exp3.csv`

### Test conventions
- All tests use direct `assert` statements (no mocks, no pytest.mock)
- `test_data.py`: calls `load_rfq_data()` (auto-generates parquet on first call); uses a single business week (2024-07-22–28) to keep sampling tests fast
- `test_pqfm.py`: N_TEST_QUBITS=5, N_TEST_FEATURES=8=2*(5-1); uses `_make_x()` to generate a normalised event; `QuantumCircuit.qasm()` NOT available in Qiskit 1.x — use gate parameter comparison instead
- `test_backtest.py`: creates synthetic sequential-timestamp DataFrames in-memory; `_get_blind_gap_seconds` and `run_single` are the primary interfaces under test
- `test_cqem.py`: uses controlled kappas (x=-0.95 and x=-0.99 both map to bin 0 with n_bins=10); coverage monotonicity test uses 50 lib / 200 unseen / 3 features
