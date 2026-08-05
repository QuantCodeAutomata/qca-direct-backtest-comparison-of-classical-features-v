"""Data loading and validation utilities for the PQFM RFQ fill-prediction experiment.

All public functions operate on DataFrames produced by
``data.generate_synthetic_rfq.generate_rfq_dataset``.  The module is
side-effect free at import time: no I/O, no sys.path mutation.
"""

import importlib.util
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

_LOG = logging.getLogger(__name__)

# Absolute path anchors derived from this file's location at import time.
# These are pure Path computations – no I/O.
_SRC_DIR: Path = Path(__file__).resolve().parent
_PROJECT_ROOT: Path = _SRC_DIR.parent
_DEFAULT_DATA_PATH: Path = _PROJECT_ROOT / "data" / "rfq_active_window.parquet"
_GENERATOR_MODULE_PATH: Path = _PROJECT_ROOT / "data" / "generate_synthetic_rfq.py"

# Required non-feature columns that every valid RFQ DataFrame must contain.
_REQUIRED_COLS: tuple[str, ...] = ("event_id", "timestamp", "bond_id", "ticker", "sector", "fill")

# Pattern that identifies feature columns produced by the generator (f_000 … f_NNN).
_FEATURE_COL_RE: re.Pattern = re.compile(r"^f_\d+$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature column names matching pattern ``f_\\d+``."""
    return [c for c in df.columns if _FEATURE_COL_RE.match(c)]


def _load_generator_module():
    """Dynamically load ``data.generate_synthetic_rfq`` without mutating sys.path.

    Returns the loaded module object.  Raises ``ImportError`` if the generator
    source file cannot be located.
    """
    if not _GENERATOR_MODULE_PATH.exists():
        raise ImportError(
            f"Synthetic data generator not found at {_GENERATOR_MODULE_PATH}. "
            "Cannot auto-generate dataset."
        )
    spec = importlib.util.spec_from_file_location(
        "data.generate_synthetic_rfq", _GENERATOR_MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rfq_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the RFQ dataset from a Parquet file, generating it if necessary.

    When *path* is ``None`` the loader looks for the canonical file at
    ``<project_root>/data/rfq_active_window.parquet``.  If that file does not
    yet exist the synthetic generator is invoked, the resulting DataFrame is
    persisted to disk, and then returned.

    Parameters
    ----------
    path:
        Explicit filesystem path to a Parquet file.  Pass ``None`` to use the
        project-default location.

    Returns
    -------
    pd.DataFrame
        Raw dataset as produced by ``generate_rfq_dataset``.  No filtering is
        applied; call :func:`filter_active_window` afterwards.

    Raises
    ------
    FileNotFoundError
        If an explicit *path* is supplied but the file does not exist.
    ImportError
        If the synthetic generator source cannot be found when auto-generation
        is attempted.
    """
    parquet_path: Path = _DEFAULT_DATA_PATH if path is None else Path(path)

    if path is not None and not parquet_path.exists():
        raise FileNotFoundError(
            f"Explicitly requested Parquet file not found: {parquet_path}"
        )

    if not parquet_path.exists():
        _LOG.info("Parquet not found at %s – generating synthetic dataset …", parquet_path)
        mod = _load_generator_module()
        df: pd.DataFrame = mod.generate_rfq_dataset()
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        _LOG.info("Synthetic dataset saved (%d rows) → %s", len(df), parquet_path)
        return df

    _LOG.info("Loading RFQ data from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    _LOG.info("Loaded %d rows, %d columns.", len(df), df.shape[1])
    return df


def filter_active_window(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Filter the RFQ DataFrame to the active analysis window.

    Two filters are applied in sequence:

    1. **Temporal filter** – retains only rows whose ``timestamp`` falls in the
       closed interval [*start*, *end*] (both inclusive, day-level precision).
    2. **Maturity filter** – removes bonds whose ``bond_id`` ends with the
       ``_MATURING`` suffix, which flags bonds excluded from the experiment's
       test set.

    The count of removed events for each filter step is logged at INFO level.

    Parameters
    ----------
    df:
        Raw RFQ DataFrame as returned by :func:`load_rfq_data`.
    start:
        ISO-8601 date string for the window start (e.g. ``'2024-07-24'``).
    end:
        ISO-8601 date string for the window end (e.g. ``'2024-10-29'``).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df* with the original index reset.

    Raises
    ------
    KeyError
        If the DataFrame is missing the ``timestamp`` or ``bond_id`` column.
    """
    for col in ("timestamp", "bond_id"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    n_initial: int = len(df)

    # ---- Temporal filter ----
    ts_start = pd.Timestamp(start, tz="UTC")
    # End is inclusive at day level: extend to last microsecond of end date
    ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    timestamps = df["timestamp"]
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("UTC")

    time_mask = (timestamps >= ts_start) & (timestamps <= ts_end)
    df_filtered: pd.DataFrame = df.loc[time_mask].copy()
    n_after_time: int = len(df_filtered)
    _LOG.info(
        "Temporal filter [%s, %s]: removed %d events (%d → %d).",
        start,
        end,
        n_initial - n_after_time,
        n_initial,
        n_after_time,
    )

    # ---- Maturity filter ----
    maturity_mask = ~df_filtered["bond_id"].str.endswith("_MATURING", na=False)
    df_filtered = df_filtered.loc[maturity_mask].copy()
    n_after_maturity: int = len(df_filtered)
    _LOG.info(
        "Maturity filter: removed %d events with maturing bonds (%d → %d).",
        n_after_time - n_after_maturity,
        n_after_time,
        n_after_maturity,
    )

    return df_filtered.reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> None:
    """Assert that a DataFrame conforms to the expected RFQ schema.

    Checks performed:

    * All required non-feature columns are present
      (``event_id``, ``timestamp``, ``bond_id``, ``ticker``, ``sector``, ``fill``).
    * At least one feature column matching the pattern ``f_\\d+`` exists.
    * ``timestamp`` column has a datetime dtype.
    * No ``NaN`` values appear in any feature column.
    * ``fill`` contains only the values ``{0, 1}`` with no missing entries.

    Parameters
    ----------
    df:
        DataFrame to validate.

    Raises
    ------
    ValueError
        On any schema violation.  The message describes the specific problem.
    """
    # --- Required column presence ---
    missing_cols = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # --- Feature columns ---
    feature_cols = _get_feature_cols(df)
    if not feature_cols:
        raise ValueError(
            "Dataset contains no feature columns matching pattern 'f_\\d+'. "
            "Expected at least one column named f_000, f_001, …"
        )

    # --- Timestamp dtype ---
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError(
            f"Column 'timestamp' must be a datetime dtype, "
            f"got {df['timestamp'].dtype!r} instead."
        )

    # --- No NaN in feature columns ---
    nan_counts: pd.Series = df[feature_cols].isna().sum()
    bad_cols = nan_counts[nan_counts > 0]
    if not bad_cols.empty:
        detail = ", ".join(f"{col}={cnt}" for col, cnt in bad_cols.items())
        raise ValueError(f"NaN values found in feature columns: {detail}")

    # --- Binary fill labels ---
    if df["fill"].isna().any():
        raise ValueError("Column 'fill' contains NaN values.")
    unique_fill = set(df["fill"].unique())
    if not unique_fill.issubset({0, 1}):
        bad_vals = sorted(unique_fill - {0, 1})
        raise ValueError(
            f"Column 'fill' must contain only {{0, 1}}, "
            f"but found additional values: {bad_vals}"
        )

    _LOG.info(
        "Dataset validated: %d rows, %d feature columns, fill rate=%.4f.",
        len(df),
        len(feature_cols),
        float(df["fill"].mean()),
    )
