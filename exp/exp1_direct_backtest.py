"""
Experiment 1: Direct Backtest Comparison
Classical vs Simulated PQFM vs Hardware-Short PQFM vs Hardware-Long PQFM

For practical reproducibility, uses:
- Synthetic RFQ data (proprietary data not available)
- Quantum simulation via Qiskit Aer (hardware not available)
- Small qubit count (N_QUBITS_SMALL=15) for fast execution
  Full 109-qubit execution requires IBM hardware access
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
from hardware_runner import HardwareRunner
from backtest import TemporalBacktest
from evaluation import (
    aggregate_auc_results,
    compute_relative_uplift,
    plot_auc_vs_blinding,
    plot_feature_histograms,
    save_results_table,
    generate_results_markdown,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _extract_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract the numerical feature matrix and ordered column names from *df*.

    Returns
    -------
    X:
        Float64 matrix of shape (n_events, n_features).
    feature_cols:
        Ordered list of column names that constitute *X*.
    """
    feature_cols = sorted([c for c in df.columns if c.startswith('f_')])
    X = df[feature_cols].to_numpy(dtype=float)
    return X, feature_cols


def _pqfm_to_dataframe(
    df_source: pd.DataFrame,
    X_pqfm: np.ndarray,
    label_col: str = 'fill',
    timestamp_col: str = 'timestamp',
) -> tuple[pd.DataFrame, list[str]]:
    """Bundle PQFM feature matrix with timestamps and labels into a DataFrame.

    Parameters
    ----------
    df_source:
        Source DataFrame supplying *timestamp_col* and *label_col*.
    X_pqfm:
        PQFM feature matrix of shape (n_events, n_pqfm_features).

    Returns
    -------
    df_pqfm:
        DataFrame with columns timestamp, fill, pqfm_000, pqfm_001, …
    pqfm_cols:
        Ordered list of PQFM column names.
    """
    n_pqfm = X_pqfm.shape[1]
    pqfm_cols = [f'pqfm_{i:03d}' for i in range(n_pqfm)]
    df_pqfm = df_source[[timestamp_col, label_col]].copy().reset_index(drop=True)
    for i, col in enumerate(pqfm_cols):
        df_pqfm[col] = X_pqfm[:, i]
    return df_pqfm, pqfm_cols


def run_exp1(
    n_events_sample: int = 500,      # small for CI speed; paper uses 16 000
    n_qubits: int = 15,               # small for speed; paper uses 109
    training_sizes: list = None,
    blinding_days: list = None,
    model_families: list = None,
    save_dir: str = 'results',
    random_seed: int = 42,
    min_test_events: int = 3,        # scaled down from 50 (paper) for small sample
    fast: bool = True,                 # use fast reduced grids for CI runs
    ultrafast: bool = False,           # skip GridSearchCV entirely for speed
) -> dict:
    """
    Run the direct comparison experiment.

    Workflow
    --------
    1. Load / auto-generate synthetic RFQ data.
    2. Filter to the active analysis window and remove maturing bonds.
    3. Sample *n_events_sample* events proportionally across trading days.
    4. Apply four feature representations:

       * **classical**  – raw normalised features (no PQFM).
       * **sim_short**  – noiseless PQFM, short circuit (blocks=1).
       * **sim_long**   – noiseless PQFM, long circuit  (blocks=2).
       * **hardware**   – noisy PQFM via HardwareRunner (density-matrix Aer).

    5. Run a temporal backtest for each representation and model family.
    6. Aggregate, compute relative uplift, and save results.

    Parameters
    ----------
    n_events_sample:
        Number of events to draw after filtering.
    n_qubits:
        Qubit count for PQFM circuits.
    training_sizes:
        List of rolling-window sizes.  Defaults to [100, 200].
    blinding_days:
        List of blinding windows.  Defaults to [0, 1].
    model_families:
        List of model families.  Defaults to ['LR', 'XGB'].
    save_dir:
        Directory for result CSV, plots, and RESULTS.md.
    random_seed:
        Master RNG seed.

    Returns
    -------
    dict
        Keys: 'agg', 'uplift', 'raw' (concatenated raw results DataFrame).
    """
    if training_sizes is None:
        training_sizes = [100, 200]
    if blinding_days is None:
        blinding_days = [0, 1]
    if model_families is None:
        model_families = ['LR', 'XGB']

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data loading & sampling
    # ------------------------------------------------------------------
    logger.info("Loading RFQ data …")
    try:
        df_full = load_rfq_data()
        validate_dataset(df_full)
        df_active = filter_active_window(
            df_full, start='2024-07-24', end='2024-10-29'
        )
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        raise

    logger.info("Sampling %d events …", n_events_sample)
    df = sample_representative(df_active, n=n_events_sample, seed=random_seed)
    df = df.reset_index(drop=True)

    X_classical, feature_cols = _extract_features(df)
    y = df['fill'].to_numpy()
    logger.info(
        "Sample ready: %d events, %d classical features, fill rate=%.3f",
        len(df), X_classical.shape[1], y.mean(),
    )

    # ------------------------------------------------------------------
    # 2. PQFM feature extraction for three quantum variants
    # ------------------------------------------------------------------
    pqfm_configs: dict[str, dict] = {
        'sim_short': {'runner_cls': SimulatorRunner,
                      'kwargs': {'n_qubits': n_qubits, 'blocks': 1,
                                 'alpha': 1.0, 'fiducial_seed': 1}},
        'sim_long':  {'runner_cls': SimulatorRunner,
                      'kwargs': {'n_qubits': n_qubits, 'blocks': 2,
                                 'alpha': 0.1, 'fiducial_seed': 0}},
        'hardware':  {'runner_cls': HardwareRunner,
                      'kwargs': {'n_qubits': n_qubits, 'blocks': 2,
                                 'alpha': 0.1, 'fiducial_seed': 0}},
    }

    pqfm_datasets: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for variant, cfg in pqfm_configs.items():
        logger.info("Computing PQFM features: %s …", variant)
        try:
            runner = cfg['runner_cls'](**cfg['kwargs'])
            runner.fit_normalization(X_classical)
            X_pqfm = runner.transform(X_classical)
            df_pqfm, pqfm_cols = _pqfm_to_dataframe(df, X_pqfm)
            pqfm_datasets[variant] = (df_pqfm, pqfm_cols)
            logger.info("  %s: shape %s", variant, X_pqfm.shape)
        except Exception as exc:
            logger.warning("PQFM extraction failed for %s: %s", variant, exc)

    # ------------------------------------------------------------------
    # 3. Backtesting all four datasets
    # ------------------------------------------------------------------
    all_results: list[pd.DataFrame] = []

    # Classical baseline
    logger.info("Running backtest: classical …")
    try:
        bt_classical = TemporalBacktest(
            df, feature_cols=feature_cols, label_col='fill',
            training_sizes=training_sizes, blinding_days=blinding_days,
            min_test_events=min_test_events,
        )
        raw_classical = bt_classical.run_all(families=model_families, fast=fast, ultrafast=ultrafast)
        raw_classical['dataset_type'] = 'classical'
        all_results.append(raw_classical)
    except Exception as exc:
        logger.warning("Classical backtest failed: %s", exc)

    # PQFM variants
    for variant, (df_pqfm, pqfm_cols) in pqfm_datasets.items():
        logger.info("Running backtest: %s …", variant)
        try:
            bt = TemporalBacktest(
                df_pqfm, feature_cols=pqfm_cols, label_col='fill',
                training_sizes=training_sizes, blinding_days=blinding_days,
                min_test_events=min_test_events,
            )
            raw = bt.run_all(families=model_families, fast=fast, ultrafast=ultrafast)
            raw['dataset_type'] = variant
            all_results.append(raw)
        except Exception as exc:
            logger.warning("Backtest failed for %s: %s", variant, exc)

    if not all_results:
        logger.error("No backtest results produced.")
        return {'agg': pd.DataFrame(), 'uplift': pd.DataFrame(), 'raw': pd.DataFrame()}

    raw_all = pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # 4. Aggregation, uplift, and output
    # ------------------------------------------------------------------
    logger.info("Aggregating results …")
    agg = aggregate_auc_results(raw_all)

    try:
        uplift = compute_relative_uplift(raw_all, baseline='classical')
    except Exception as exc:
        logger.warning("Uplift computation failed: %s", exc)
        uplift = pd.DataFrame()

    # Save CSV and markdown
    metrics_path = save_path / 'metrics_exp1.csv'
    save_results_table(agg, uplift, path=str(metrics_path))
    logger.info("Saved metrics → %s", metrics_path)

    md_text = generate_results_markdown(agg, uplift, exp_name='exp1_direct_backtest')
    results_md = save_path / 'RESULTS_exp1.md'
    results_md.write_text(md_text)
    logger.info("Saved markdown → %s", results_md)

    # Plots
    try:
        plot_auc_vs_blinding(agg, save_path=str(save_path / 'auc_vs_blinding_exp1.png'))
        feature_arrays = {
            'classical': X_classical,
            **{v: pqfm_datasets[v][0][[c for c in pqfm_datasets[v][1]]].to_numpy()
               for v in pqfm_datasets},
        }
        plot_feature_histograms(
            feature_arrays,
            save_path=str(save_path / 'feature_histograms_exp1.png'),
        )
    except Exception as exc:
        logger.warning("Plotting failed: %s", exc)

    # Save raw results JSON-compatible summary
    summary = {
        'n_events_sample': n_events_sample,
        'n_qubits': n_qubits,
        'training_sizes': training_sizes,
        'blinding_days': blinding_days,
        'model_families': model_families,
        'n_result_rows': len(raw_all),
    }
    (save_path / 'config_exp1.json').write_text(json.dumps(summary, indent=2))

    logger.info("Experiment 1 complete.")
    return {'agg': agg, 'uplift': uplift, 'raw': raw_all}


if __name__ == '__main__':
    results = run_exp1()
    print('Exp 1 completed')
    print(results['agg'].to_string())
