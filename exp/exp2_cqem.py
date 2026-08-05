"""
Experiment 2: Classical-Quantum Event Matching (CQEM)

Tests the reuse-via-matching scheme from the paper: a library of
hardware-long PQFM feature vectors is built from a set of reference events,
and unseen events are assigned library vectors via a deterministic kappa key
constructed from binned classical features.

For each N_bins ∈ {4, 10, 30, 60} we report:
  - Coverage rate (fraction of unseen events that find a kappa match)
  - AUC on the matched subset for each model family

Results are saved to results/metrics_exp2.csv and plots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import load_rfq_data, filter_active_window, validate_dataset
from sampling import sample_representative
from hardware_runner import HardwareRunner
from cqem import CQEMMapper
from backtest import TemporalBacktest
from evaluation import (
    aggregate_auc_results,
    plot_auc_vs_blinding,
    save_results_table,
    generate_results_markdown,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Paper values; use a small subset for CI speed
_DEFAULT_N_BINS_LIST: list[int] = [4, 10, 30, 60]


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


def run_exp2(
    n_events_sample: int = 500,
    n_qubits: int = 15,
    n_bins_list: list = None,
    training_sizes: list = None,
    blinding_days: list = None,
    model_families: list = None,
    library_fraction: float = 0.5,
    save_dir: str = 'results',
    random_seed: int = 42,
    fast: bool = True,
    ultrafast: bool = False,
    top_k_features: Optional[int] = None,
) -> dict:
    """
    Run the CQEM experiment.

    Workflow
    --------
    1. Load and sample 2 × *n_events_sample* events.
    2. Split evenly into **library** (first half) and **unseen** (second half).
    3. Compute hardware-long PQFM features for the library split.
    4. For each N_bins in *n_bins_list*:

       a. Build a CQEM mapper from the library split.
       b. Assign PQFM features to unseen events via kappa matching.
       c. Collect coverage statistics.
       d. Run a temporal backtest on the matched unseen subset.

    5. Aggregate AUC results and save.

    Parameters
    ----------
    n_events_sample:
        Number of events in each split (library and unseen).
    n_qubits:
        Qubit count for PQFM circuits.
    n_bins_list:
        List of N_bins values.  Defaults to [4, 10, 30, 60].
    training_sizes:
        Rolling-window sizes for the backtest.  Defaults to [100, 200].
    blinding_days:
        Blinding windows.  Defaults to [0, 1].
    model_families:
        Model families to evaluate.  Defaults to ['LR', 'XGB'].
    library_fraction:
        Fraction of the combined pool used as the library.
    save_dir:
        Directory for output files.
    random_seed:
        Master RNG seed.

    Returns
    -------
    dict
        Keys: 'coverage', 'agg', 'raw'.
    """
    if n_bins_list is None:
        n_bins_list = _DEFAULT_N_BINS_LIST
    if training_sizes is None:
        training_sizes = [100, 200]
    if blinding_days is None:
        blinding_days = [0, 1]
    if model_families is None:
        model_families = ['LR', 'XGB']

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and split data
    # ------------------------------------------------------------------
    total_events = int(n_events_sample / library_fraction)
    logger.info("Loading RFQ data (need ~%d events) …", total_events)
    try:
        df_full = load_rfq_data()
        validate_dataset(df_full)
        df_active = filter_active_window(df_full, start='2024-07-24', end='2024-10-29')
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        raise

    logger.info("Sampling %d events …", total_events)
    df_pool = sample_representative(df_active, n=total_events, seed=random_seed)
    df_pool = df_pool.reset_index(drop=True)

    n_library = int(len(df_pool) * library_fraction)
    df_library = df_pool.iloc[:n_library].reset_index(drop=True)
    df_unseen = df_pool.iloc[n_library:].reset_index(drop=True)
    logger.info("Library: %d events, Unseen: %d events", len(df_library), len(df_unseen))

    feature_cols = _feature_cols_from_df(df_pool)
    X_library = df_library[feature_cols].to_numpy(dtype=float)
    X_unseen = df_unseen[feature_cols].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 2. Hardware-long PQFM for the library split
    # ------------------------------------------------------------------
    logger.info("Computing hardware-long PQFM for library events …")
    try:
        hw_runner = HardwareRunner(n_qubits=n_qubits, blocks=2, alpha=0.1, fiducial_seed=0)
        hw_runner.fit_normalization(X_library)
        X_pqfm_library = hw_runner.transform(X_library)
        logger.info("Library PQFM shape: %s", X_pqfm_library.shape)
    except Exception as exc:
        logger.error("Library PQFM computation failed: %s", exc)
        raise

    # ------------------------------------------------------------------
    # 3. CQEM loop over N_bins values
    # ------------------------------------------------------------------
    coverage_records: list[dict] = []
    all_results: list[pd.DataFrame] = []

    for n_bins in n_bins_list:
        logger.info("CQEM with N_bins=%d …", n_bins)

        mapper = CQEMMapper(n_bins=n_bins, top_k_features=top_k_features)
        mapper.build_library(X_library, X_pqfm_library)

        stats = mapper.coverage_stats(X_unseen)
        stats['n_bins'] = n_bins
        stats['resolution'] = mapper.resolution()
        coverage_records.append(stats)
        logger.info(
            "  N_bins=%d: coverage=%.3f (%d/%d), kappas=%d",
            n_bins,
            stats['coverage_rate'],
            stats['n_matched'],
            stats['n_total'],
            stats['n_unique_kappas_in_library'],
        )

        X_pqfm_assigned, matched_mask = mapper.assign_features(X_unseen)
        if matched_mask.sum() < 10:
            logger.warning("  Too few matched events (%d) for backtest, skipping.", matched_mask.sum())
            continue

        df_matched = df_unseen.loc[matched_mask].reset_index(drop=True)
        df_bt, pqfm_cols = _pqfm_df(df_matched, X_pqfm_assigned)

        try:
            bt = TemporalBacktest(
                df_bt,
                feature_cols=pqfm_cols,
                label_col='fill',
                training_sizes=training_sizes,
                blinding_days=blinding_days,
                min_test_events=1,
            )
            raw = bt.run_all(families=model_families, fast=fast, ultrafast=ultrafast)
            raw['dataset_type'] = f'cqem_nbins{n_bins}'
            raw['n_bins'] = n_bins
            raw['coverage_rate'] = stats['coverage_rate']
            all_results.append(raw)
        except Exception as exc:
            logger.warning("  Backtest failed for N_bins=%d: %s", n_bins, exc)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    coverage_df = pd.DataFrame(coverage_records)
    coverage_path = save_path / 'coverage_exp2.csv'
    coverage_df.to_csv(coverage_path, index=False)
    logger.info("Saved coverage stats → %s", coverage_path)

    if all_results:
        raw_all = pd.concat(all_results, ignore_index=True)
        agg = aggregate_auc_results(raw_all)
        save_results_table(agg, None, path=str(save_path / 'metrics_exp2.csv'))
        try:
            plot_auc_vs_blinding(agg, save_path=str(save_path / 'auc_vs_blinding_exp2.png'))
        except Exception as exc:
            logger.warning("Plotting failed: %s", exc)

        md_text = generate_results_markdown(agg, None, exp_name='exp2_cqem')
        (save_path / 'RESULTS_exp2.md').write_text(md_text)
    else:
        logger.warning("No backtest results produced.")
        raw_all = pd.DataFrame()
        agg = pd.DataFrame()

    summary = {
        'n_events_sample': n_events_sample,
        'n_qubits': n_qubits,
        'n_bins_list': n_bins_list,
        'training_sizes': training_sizes,
        'blinding_days': blinding_days,
        'model_families': model_families,
    }
    (save_path / 'config_exp2.json').write_text(json.dumps(summary, indent=2))
    logger.info("Experiment 2 complete.")

    return {'coverage': coverage_df, 'agg': agg, 'raw': raw_all}


if __name__ == '__main__':
    results = run_exp2(
        n_events_sample=200,
        n_qubits=8,
        n_bins_list=[4, 10, 30],
        training_sizes=[50, 100],
        blinding_days=[0, 1],
        model_families=['LR', 'XGB'],
        fast=True,
        ultrafast=True,
        top_k_features=4,
    )
    print('Exp 2 completed')
    print(results['coverage'].to_string())
