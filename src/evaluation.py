"""Metrics aggregation and visualization for PQFM RFQ fill-prediction experiment."""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend – must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_GROUP_COLS_BASE = ['family', 'training_size', 'blinding_day']


def aggregate_auc_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-day AUC scores into summary statistics per configuration.

    Parameters
    ----------
    results_df:
        Raw results returned by :meth:`TemporalBacktest.run_all`, optionally
        enriched with a ``dataset_type`` column by the caller before
        concatenation across multiple datasets.

    Returns
    -------
    pd.DataFrame
        One row per unique ``(dataset_type?, family, training_size,
        blinding_day)`` group with columns:

        * ``median_auc`` – median AUC across test days.
        * ``std_auc``    – standard deviation (``NaN`` for groups with one
          observation).
        * ``count``      – number of test days that contributed to the group.
    """
    if results_df.empty:
        base = ['dataset_type'] + _GROUP_COLS_BASE if 'dataset_type' in results_df.columns else _GROUP_COLS_BASE
        return pd.DataFrame(columns=base + ['median_auc', 'std_auc', 'count'])

    group_cols = list(_GROUP_COLS_BASE)
    if 'dataset_type' in results_df.columns:
        group_cols = ['dataset_type'] + group_cols

    agg = (
        results_df
        .groupby(group_cols, sort=True)['auc']
        .agg(median_auc='median', std_auc='std', count='count')
        .reset_index()
    )
    return agg


def compute_relative_uplift(
    results_df: pd.DataFrame,
    baseline: str = 'classical',
) -> pd.DataFrame:
    """Compute the relative AUC uplift of each dataset variant over the baseline.

    The uplift is defined per ``(family, training_size, blinding_day)`` group as::

        uplift (%) = (median_AUC_variant − median_AUC_baseline) /
                      median_AUC_baseline × 100

    Parameters
    ----------
    results_df:
        Raw results with a ``dataset_type`` column that contains at least
        one entry equal to *baseline*.
    baseline:
        Value of ``dataset_type`` to use as the reference (default
        ``'classical'``).

    Returns
    -------
    pd.DataFrame
        Merged table with an additional ``relative_uplift`` column (in %).
        Rows for which a matching baseline group is absent receive ``NaN``.

    Raises
    ------
    ValueError
        If ``dataset_type`` is not present in *results_df* or the baseline
        label is absent.
    """
    if 'dataset_type' not in results_df.columns:
        raise ValueError(
            "compute_relative_uplift requires a 'dataset_type' column in results_df."
        )

    agg = aggregate_auc_results(results_df)

    if baseline not in agg['dataset_type'].values:
        raise ValueError(
            f"Baseline dataset_type '{baseline}' not found in results_df. "
            f"Available: {sorted(agg['dataset_type'].unique().tolist())}"
        )

    baseline_df = (
        agg[agg['dataset_type'] == baseline]
        [_GROUP_COLS_BASE + ['median_auc']]
        .rename(columns={'median_auc': 'baseline_auc'})
    )

    merged = agg.merge(baseline_df, on=_GROUP_COLS_BASE, how='left')
    merged['relative_uplift'] = (
        (merged['median_auc'] - merged['baseline_auc'])
        / merged['baseline_auc']
        * 100
    )
    return merged


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _resolve_dataset_types(agg_df: pd.DataFrame) -> list[str]:
    """Return ordered list of dataset types; ``['all']`` when column absent."""
    if 'dataset_type' in agg_df.columns:
        return sorted(agg_df['dataset_type'].unique().tolist())
    return ['all']


def plot_auc_vs_blinding(
    agg_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """Plot median AUC ± std vs blinding days for each dataset type.

    Layout: **rows** = dataset types, **columns** = training sizes.
    Each subplot draws one line per model family with a shaded ±1 std band.

    Parameters
    ----------
    agg_df:
        Aggregated results as returned by :func:`aggregate_auc_results`.
        Must contain columns ``family``, ``training_size``, ``blinding_day``,
        ``median_auc``, ``std_auc``.  An optional ``dataset_type`` column adds
        one row of subplots per type.
    save_path:
        If provided, the figure is saved to this path (PNG/PDF inferred from
        extension).  Directories are created automatically.

    Notes
    -----
    Uses seaborn whitegrid style for the background and matplotlib tab10
    palette for model families.
    """
    sns.set_theme(style='whitegrid', font_scale=0.9)

    dataset_types = _resolve_dataset_types(agg_df)
    training_sizes = sorted(agg_df['training_size'].unique().tolist())
    families = sorted(agg_df['family'].unique().tolist())

    n_rows = len(dataset_types)
    n_cols = len(training_sizes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(4, 4 * n_cols), max(3, 3.5 * n_rows)),
        squeeze=False,
    )

    palette = sns.color_palette('tab10', n_colors=max(len(families), 1))
    color_map = {f: palette[i] for i, f in enumerate(families)}

    for row, ds_type in enumerate(dataset_types):
        for col, ts in enumerate(training_sizes):
            ax = axes[row, col]

            if 'dataset_type' in agg_df.columns:
                subset = agg_df[
                    (agg_df['dataset_type'] == ds_type) &
                    (agg_df['training_size'] == ts)
                ]
            else:
                subset = agg_df[agg_df['training_size'] == ts]

            for family in families:
                fam_data = (
                    subset[subset['family'] == family]
                    .sort_values('blinding_day')
                )
                if fam_data.empty:
                    continue
                x = fam_data['blinding_day'].to_numpy()
                y = fam_data['median_auc'].to_numpy()
                yerr = fam_data['std_auc'].fillna(0).to_numpy()

                ax.plot(x, y, label=family, color=color_map[family],
                        marker='o', linewidth=1.8)
                ax.fill_between(x, y - yerr, y + yerr,
                                color=color_map[family], alpha=0.15)

            title = f"training_size={ts}"
            if 'dataset_type' in agg_df.columns:
                title = f"{ds_type}  |  training_size={ts}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Blinding Days')
            ax.set_ylabel('Median AUC')
            ax.legend(fontsize=7, loc='best')

    fig.suptitle('Median AUC vs Blinding Window', fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.close(fig)


def plot_feature_histograms(
    feature_arrays: dict[str, np.ndarray],
    n_features_to_plot: int = 5,
    save_path: str = None,
) -> None:
    """Plot overlaid feature histograms for each dataset type.

    Parameters
    ----------
    feature_arrays:
        Mapping from dataset-type name to a 2-D NumPy array of shape
        ``(n_samples, n_features)``.  The first *n_features_to_plot* columns
        are shown.
    n_features_to_plot:
        How many feature columns (counting from index 0) to include.
    save_path:
        File path for saving; directories are created automatically.

    Notes
    -----
    Each subplot corresponds to one feature index.  Dataset types are
    overlaid with semi-transparent histograms so distributions can be
    compared directly.
    """
    if not feature_arrays:
        warnings.warn("plot_feature_histograms: feature_arrays is empty – nothing to plot.")
        return

    # Determine the actual number of features available across all arrays
    n_feats_available = min(arr.shape[1] for arr in feature_arrays.values() if arr.ndim == 2)
    n_plot = min(n_features_to_plot, n_feats_available)
    if n_plot == 0:
        warnings.warn("plot_feature_histograms: no features to plot.")
        return

    sns.set_theme(style='whitegrid', font_scale=0.9)
    palette = sns.color_palette('tab10', n_colors=max(len(feature_arrays), 1))

    fig, axes = plt.subplots(1, n_plot, figsize=(max(4, 4 * n_plot), 4), squeeze=False)

    for feat_idx in range(n_plot):
        ax = axes[0, feat_idx]
        for color, (ds_type, arr) in zip(palette, feature_arrays.items()):
            col_data = arr[:, feat_idx]
            col_data = col_data[np.isfinite(col_data)]
            if len(col_data) == 0:
                continue
            ax.hist(col_data, bins=40, alpha=0.55, label=ds_type,
                    color=color, density=True, edgecolor='none')
        ax.set_title(f'Feature {feat_idx}', fontsize=9)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        if feat_idx == 0:
            ax.legend(fontsize=7, loc='best')

    fig.suptitle('Feature Distributions by Dataset Type', fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.close(fig)


def plot_predicted_probabilities(
    proba_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str = None,
) -> None:
    """Plot predicted-probability distributions split by realized fill outcome.

    Parameters
    ----------
    proba_dict:
        Mapping from dataset-type name to ``(probas, labels)`` where *probas*
        is a 1-D array of predicted P(fill=1) and *labels* is the matching
        array of realized outcomes (0 or 1).
    save_path:
        File path for saving; directories are created automatically.

    Notes
    -----
    Each subplot covers one dataset type.  Two overlaid histograms show the
    predicted-probability distribution for quotes that did (label=1) and did
    not (label=0) fill, enabling visual calibration assessment.
    """
    if not proba_dict:
        warnings.warn("plot_predicted_probabilities: proba_dict is empty – nothing to plot.")
        return

    sns.set_theme(style='whitegrid', font_scale=0.9)
    n_types = len(proba_dict)
    fig, axes = plt.subplots(
        1, n_types,
        figsize=(max(4, 4 * n_types), 4),
        squeeze=False,
    )

    for col, (ds_type, (probas, labels)) in enumerate(proba_dict.items()):
        ax = axes[0, col]
        probas = np.asarray(probas, dtype=float)
        labels = np.asarray(labels)

        mask_pos = labels == 1
        mask_neg = labels == 0

        p_pos = probas[mask_pos]
        p_neg = probas[mask_neg]

        if len(p_neg) > 0:
            ax.hist(p_neg, bins=40, alpha=0.60, label='No fill (0)',
                    color='steelblue', density=True, edgecolor='none')
        if len(p_pos) > 0:
            ax.hist(p_pos, bins=40, alpha=0.60, label='Filled (1)',
                    color='tomato', density=True, edgecolor='none')

        ax.set_title(ds_type, fontsize=9)
        ax.set_xlabel('Predicted P(fill=1)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='best')

    fig.suptitle('Predicted Probability Distributions by Realized Outcome', fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results_table(
    agg_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    path: str = 'results/metrics_exp1.csv',
) -> None:
    """Save aggregated results and uplift table to CSV files.

    Parameters
    ----------
    agg_df:
        Aggregated AUC table (output of :func:`aggregate_auc_results`).
    uplift_df:
        Relative-uplift table (output of :func:`compute_relative_uplift`).
        May be ``None`` or empty; in that case only *agg_df* is saved.
    path:
        Destination path for *agg_df*.  The uplift table is written to a
        sibling file with ``_uplift`` inserted before the ``.csv`` suffix.

    Notes
    -----
    Parent directories are created automatically.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out, index=False)

    if uplift_df is not None and not (isinstance(uplift_df, pd.DataFrame) and uplift_df.empty):
        uplift_path = out.with_stem(out.stem + '_uplift') if hasattr(out, 'with_stem') \
            else out.parent / (out.stem + '_uplift' + out.suffix)
        pd.DataFrame(uplift_df).to_csv(uplift_path, index=False)


def generate_results_markdown(
    agg_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    exp_name: str = 'exp1',
) -> str:
    """Generate a Markdown-formatted results summary suitable for ``RESULTS.md``.

    Parameters
    ----------
    agg_df:
        Aggregated AUC table from :func:`aggregate_auc_results`.
    uplift_df:
        Relative-uplift table from :func:`compute_relative_uplift`, or
        ``None`` / empty DataFrame to omit the uplift section.
    exp_name:
        Experiment identifier used in the section heading.

    Returns
    -------
    str
        Markdown string containing:

        1. A ``## Median AUC by Configuration`` table.
        2. An optional ``## Relative Uplift vs Classical Timestamps`` table.
        3. A ``## Key Findings`` bullet list derived from the uplift data.
    """
    lines: list[str] = [
        f'# Experiment: {exp_name}',
        '',
        '## Median AUC by Configuration',
        '',
    ]

    # Round numeric columns for readability
    display_agg = agg_df.copy()
    for col in ('median_auc', 'std_auc'):
        if col in display_agg.columns:
            display_agg[col] = display_agg[col].round(4)

    try:
        lines.append(display_agg.to_markdown(index=False))
    except Exception:
        lines.append(display_agg.to_string(index=False))
    lines.append('')

    has_uplift = (
        uplift_df is not None
        and isinstance(uplift_df, pd.DataFrame)
        and not uplift_df.empty
        and 'relative_uplift' in uplift_df.columns
    )

    if has_uplift:
        lines += [
            '## Relative Uplift vs Classical Timestamps',
            '',
        ]

        display_up = uplift_df.copy()
        for col in ('median_auc', 'baseline_auc', 'std_auc', 'relative_uplift'):
            if col in display_up.columns:
                display_up[col] = display_up[col].round(4)

        try:
            lines.append(display_up.to_markdown(index=False))
        except Exception:
            lines.append(display_up.to_string(index=False))
        lines.append('')

        lines.append('## Key Findings')
        lines.append('')

        # Best uplift
        valid_up = uplift_df.dropna(subset=['relative_uplift'])
        if not valid_up.empty:
            best = valid_up.loc[valid_up['relative_uplift'].idxmax()]
            family_str = best.get('family', 'N/A')
            ds_str = best.get('dataset_type', 'N/A')
            ts_str = best.get('training_size', 'N/A')
            bd_str = best.get('blinding_day', 'N/A')
            uplift_val = best['relative_uplift']
            lines.append(
                f'- **Best relative uplift**: {uplift_val:.2f}% for model family '
                f'`{family_str}` on dataset `{ds_str}` '
                f'(training_size={ts_str}, blinding_day={bd_str}).'
            )

        # Best overall median AUC
        if not agg_df.empty and 'median_auc' in agg_df.columns:
            best_auc_row = agg_df.loc[agg_df['median_auc'].idxmax()]
            lines.append(
                f'- **Highest median AUC**: {best_auc_row["median_auc"]:.4f} '
                f'(family=`{best_auc_row.get("family", "N/A")}`, '
                f'training_size={best_auc_row.get("training_size", "N/A")}, '
                f'blinding_day={best_auc_row.get("blinding_day", "N/A")}).'
            )
        lines.append('')

    return '\n'.join(lines)
