"""
Experiment 3: Noise and Circuit Robustness

Compares three PQFM variants on the same event sample:
  1. **noiseless** – SimulatorRunner (statevector, zero noise)
  2. **noisy**     – NoisySimulatorRunner (depolarizing + readout noise)
  3. **idle**      – IdleControlRunner (2Q gates replaced by Delay instructions)

Additionally, runs the noisy simulation three times on a fixed 10-event subset
and reports per-feature standard deviation across runs, quantifying output drift
due to shot noise.

Results are saved to results/metrics_exp3.csv and comparison plots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_rfq_data, filter_active_window, validate_dataset
from sampling import sample_representative
from simulator_runner import SimulatorRunner
from noise_controls import NoisySimulatorRunner, IdleControlRunner
from backtest import TemporalBacktest
from evaluation import (
    aggregate_auc_results,
    plot_auc_vs_blinding,
    plot_feature_histograms,
    save_results_table,
    generate_results_markdown,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Number of events used for the drift / shot-noise analysis
_N_DRIFT_EVENTS: int = 10
_N_DRIFT_RUNS: int = 3


def _feature_cols_from_df(df: pd.DataFrame) -> list[str]:
    """Return sorted list of 'f_XXX' column names from *df*."""
    return sorted(c for c in df.columns if c.startswith('f_'))


def _pqfm_df(df_source: pd.DataFrame, X_pqfm: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Wrap PQFM matrix in a DataFrame alongside timestamp and fill."""
    pqfm_cols = [f'pqfm_{i:03d}' for i in range(X_pqfm.shape[1])]
    out = df_source[['timestamp', 'fill']].copy().reset_index(drop=True)
    for i, col in enumerate(pqfm_cols):
        out[col] = X_pqfm[:, i]
    return out, pqfm_cols


def _compute_drift_stats(
    X_subset: np.ndarray,
    n_qubits: int,
    noise_seeds: list[int],
    **runner_kwargs,
) -> pd.DataFrame:
    """Run NoisySimulatorRunner *len(noise_seeds)* times and return per-feature std.

    Parameters
    ----------
    X_subset:
        Feature matrix of shape (n_events, n_features) for the drift subset.
    n_qubits:
        Qubit count for PQFM circuits.
    noise_seeds:
        List of Aer seeds; one run per seed.
    **runner_kwargs:
        Extra keyword arguments forwarded to NoisySimulatorRunner.

    Returns
    -------
    pd.DataFrame
        One row per PQFM feature dimension with columns:
        feature_idx, mean_std, max_std, min_std.
    """
    run_results: list[np.ndarray] = []
    for seed in noise_seeds:
        runner = NoisySimulatorRunner(n_qubits=n_qubits, seed=seed, **runner_kwargs)
        runner.fit_normalization(X_subset)
        X_out = runner.transform(X_subset)
        run_results.append(X_out)

    stacked = np.stack(run_results, axis=0)  # shape: (n_runs, n_events, n_pqfm)
    # Per-feature std: average across events, then per feature
    per_event_std = np.std(stacked, axis=0)  # (n_events, n_pqfm)
    mean_std = per_event_std.mean(axis=0)    # (n_pqfm,)
    max_std = per_event_std.max(axis=0)
    min_std = per_event_std.min(axis=0)

    return pd.DataFrame({
        'feature_idx': np.arange(len(mean_std)),
        'mean_std': mean_std,
        'max_std': max_std,
        'min_std': min_std,
    })


def run_exp3(
    n_events_sample: int = 100,
    n_qubits: int = 15,
    training_sizes: list = None,
    blinding_days: list = None,
    model_families: list = None,
    save_dir: str = 'results',
    random_seed: int = 42,
    fast: bool = True,
    ultrafast: bool = False,
) -> dict:
    """
    Run the robustness experiment.

    Workflow
    --------
    1. Load and sample *n_events_sample* events.
    2. Compute PQFM features for three circuit variants (noiseless, noisy, idle).
    3. Run the same temporal backtest on all three variants.
    4. Compute per-feature output drift on a fixed 10-event subset by repeating
       the noisy simulation with three different Aer seeds.
    5. Save results and comparison plots.

    Parameters
    ----------
    n_events_sample:
        Number of events to process.
    n_qubits:
        Qubit count for PQFM circuits.
    training_sizes:
        Rolling-window sizes.  Defaults to [50, 100].
    blinding_days:
        Blinding windows.  Defaults to [0, 1].
    model_families:
        Model families.  Defaults to ['LR'].
    save_dir:
        Directory for output files.
    random_seed:
        Master RNG seed.

    Returns
    -------
    dict
        Keys: 'agg', 'raw', 'drift'.
    """
    if training_sizes is None:
        training_sizes = [50, 100]
    if blinding_days is None:
        blinding_days = [0, 1]
    if model_families is None:
        model_families = ['LR']

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data loading & sampling
    # ------------------------------------------------------------------
    logger.info("Loading RFQ data …")
    try:
        df_full = load_rfq_data()
        validate_dataset(df_full)
        df_active = filter_active_window(df_full, start='2024-07-24', end='2024-10-29')
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        raise

    logger.info("Sampling %d events …", n_events_sample)
    df = sample_representative(df_active, n=n_events_sample, seed=random_seed)
    df = df.reset_index(drop=True)

    feature_cols = _feature_cols_from_df(df)
    X_classical = df[feature_cols].to_numpy(dtype=float)
    logger.info("Sample ready: %d events, %d features", len(df), X_classical.shape[1])

    # ------------------------------------------------------------------
    # 2. PQFM feature extraction for three variants
    # ------------------------------------------------------------------
    runner_configs: dict[str, object] = {
        'noiseless': SimulatorRunner(
            n_qubits=n_qubits, blocks=2, alpha=0.1, fiducial_seed=0
        ),
        'noisy': NoisySimulatorRunner(
            n_qubits=n_qubits, blocks=2, alpha=0.1, fiducial_seed=0,
            noise_1q=0.001, noise_2q=0.01, readout_error=0.02,
            shots=4096, seed=random_seed,
        ),
        'idle': IdleControlRunner(
            n_qubits=n_qubits, blocks=2, alpha=0.1, fiducial_seed=0,
            noise_1q=0.001, noise_2q=0.01, shots=4096, seed=random_seed,
        ),
    }

    variant_datasets: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    feature_arrays: dict[str, np.ndarray] = {'classical': X_classical}

    for variant, runner in runner_configs.items():
        logger.info("Computing PQFM features: %s …", variant)
        try:
            runner.fit_normalization(X_classical)
            X_pqfm = runner.transform(X_classical)
            df_pqfm, pqfm_cols = _pqfm_df(df, X_pqfm)
            variant_datasets[variant] = (df_pqfm, pqfm_cols)
            feature_arrays[variant] = X_pqfm
            logger.info("  %s: shape %s", variant, X_pqfm.shape)
        except Exception as exc:
            logger.warning("PQFM extraction failed for %s: %s", variant, exc)

    # ------------------------------------------------------------------
    # 3. Temporal backtesting
    # ------------------------------------------------------------------
    all_results: list[pd.DataFrame] = []

    for variant, (df_pqfm, pqfm_cols) in variant_datasets.items():
        logger.info("Running backtest: %s …", variant)
        try:
            bt = TemporalBacktest(
                df_pqfm,
                feature_cols=pqfm_cols,
                label_col='fill',
                training_sizes=training_sizes,
                blinding_days=blinding_days,
                min_test_events=1,
            )
            raw = bt.run_all(families=model_families, fast=fast, ultrafast=ultrafast)
            raw['dataset_type'] = variant
            all_results.append(raw)
        except Exception as exc:
            logger.warning("Backtest failed for %s: %s", variant, exc)

    # ------------------------------------------------------------------
    # 4. Drift analysis (noisy simulation × 3 runs on 10-event subset)
    # ------------------------------------------------------------------
    logger.info("Computing drift statistics (%d events × %d runs) …",
                _N_DRIFT_EVENTS, _N_DRIFT_RUNS)
    drift_df = pd.DataFrame()
    try:
        n_drift = min(_N_DRIFT_EVENTS, len(X_classical))
        X_drift = X_classical[:n_drift]
        drift_seeds = [random_seed + i for i in range(_N_DRIFT_RUNS)]
        drift_df = _compute_drift_stats(
            X_drift,
            n_qubits=n_qubits,
            noise_seeds=drift_seeds,
            blocks=2,
            alpha=0.1,
            fiducial_seed=0,
            noise_1q=0.001,
            noise_2q=0.01,
            readout_error=0.02,
            shots=4096,
        )
        drift_path = save_path / 'drift_exp3.csv'
        drift_df.to_csv(drift_path, index=False)
        logger.info(
            "Drift stats saved → %s (mean std across features: %.4f)",
            drift_path,
            drift_df['mean_std'].mean(),
        )
    except Exception as exc:
        logger.warning("Drift analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    if all_results:
        raw_all = pd.concat(all_results, ignore_index=True)
        agg = aggregate_auc_results(raw_all)
        save_results_table(agg, None, path=str(save_path / 'metrics_exp3.csv'))

        try:
            plot_auc_vs_blinding(
                agg, save_path=str(save_path / 'auc_vs_blinding_exp3.png')
            )
            plot_feature_histograms(
                feature_arrays,
                save_path=str(save_path / 'feature_histograms_exp3.png'),
            )
        except Exception as exc:
            logger.warning("Plotting failed: %s", exc)

        md_text = generate_results_markdown(agg, None, exp_name='exp3_robustness')
        (save_path / 'RESULTS_exp3.md').write_text(md_text)
    else:
        logger.warning("No backtest results produced.")
        raw_all = pd.DataFrame()
        agg = pd.DataFrame()

    summary = {
        'n_events_sample': n_events_sample,
        'n_qubits': n_qubits,
        'training_sizes': training_sizes,
        'blinding_days': blinding_days,
        'model_families': model_families,
        'n_drift_events': _N_DRIFT_EVENTS,
        'n_drift_runs': _N_DRIFT_RUNS,
    }
    (save_path / 'config_exp3.json').write_text(json.dumps(summary, indent=2))
    logger.info("Experiment 3 complete.")

    return {'agg': agg, 'raw': raw_all, 'drift': drift_df}


if __name__ == '__main__':
    results = run_exp3()
    print('Exp 3 completed')
    if not results['drift'].empty:
        print(f"Mean per-feature std: {results['drift']['mean_std'].mean():.4f}")
