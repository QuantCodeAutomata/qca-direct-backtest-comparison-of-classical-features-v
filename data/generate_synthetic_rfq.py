"""Synthetic RFQ dataset generator for the PQFM bond fill-prediction experiment.

Generates a realistic but fully synthetic Request-For-Quote (RFQ) dataset covering
a corporate bond trading window.  Events are distributed across business days,
features are drawn from Uniform(−1, +1), and fill labels are generated with a
mild logistic signal on the first five features so that the overall fill rate is
approximately 40 %.  Five percent of the bond universe is flagged as
'maturing_during_window' by appending '_MATURING' to those bond IDs.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH: Path = Path(__file__).resolve().parent / "rfq_active_window.parquet"

_SECTORS: list[str] = [
    "Financials",
    "Energy",
    "Technology",
    "Healthcare",
    "Consumer",
    "Industrials",
    "Materials",
    "Utilities",
    "Real_Estate",
    "Communication",
]

# Trading window constants (µs from midnight UTC)
_TRADING_START_US: int = 8 * 3600 * 1_000_000   # 08:00 UTC
_TRADING_RANGE_US: int = 9 * 3600 * 1_000_000   # 9-hour window → 17:00 UTC


def generate_rfq_dataset(
    n_events: int = 143_912,
    n_features: int = 216,
    start_date: str = "2024-07-24",
    end_date: str = "2024-10-29",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic RFQ dataset for the PQFM bond fill-prediction experiment.

    Parameters
    ----------
    n_events:
        Total number of RFQ events to generate.
    n_features:
        Number of classical feature dimensions, producing columns
        ``f_000`` … ``f_{n_features-1}``.
    start_date:
        ISO-8601 date string for the inclusive window start.
    end_date:
        ISO-8601 date string for the inclusive window end.
    seed:
        Integer seed passed to ``numpy.random.default_rng`` for full
        reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns in order:

        * ``event_id``  – unique string identifier ``RFQ_XXXXXXXX``
        * ``timestamp`` – UTC datetime64 with microsecond precision
        * ``bond_id``   – bond identifier; 5 % end with ``_MATURING``
        * ``ticker``    – three-character ticker derived from bond index
        * ``sector``    – one of ten industry sectors
        * ``f_000`` … ``f_{n-1}`` – float64 features scaled to [−1, +1]
        * ``fill``      – int8 binary label (1 = filled, 0 = not filled)

    Raises
    ------
    ValueError
        If no business days exist between *start_date* and *end_date*.
    """
    if n_events < 1:
        raise ValueError(f"n_events must be ≥ 1, got {n_events}.")
    if n_features < 5:
        raise ValueError(f"n_features must be ≥ 5 (signal uses first 5), got {n_features}.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Trading day schedule
    # ------------------------------------------------------------------
    trading_days: pd.DatetimeIndex = pd.bdate_range(
        start=start_date, end=end_date, tz="UTC"
    )
    n_days = len(trading_days)
    if n_days == 0:
        raise ValueError(
            f"No business days found between {start_date!r} and {end_date!r}."
        )
    _LOG.info("Trading window: %s → %s  (%d business days)", start_date, end_date, n_days)

    # ------------------------------------------------------------------
    # 2. Bond universe  (500 bonds, 5 % flagged as maturing)
    # ------------------------------------------------------------------
    n_bonds: int = 500
    n_maturing: int = max(1, round(n_bonds * 0.05))
    n_normal: int = n_bonds - n_maturing

    normal_bond_ids: list[str] = [f"BOND_{i:04d}" for i in range(n_normal)]
    maturing_bond_ids: list[str] = [
        f"BOND_{i:04d}_MATURING" for i in range(n_normal, n_bonds)
    ]
    all_bond_ids: np.ndarray = np.array(normal_bond_ids + maturing_bond_ids)

    bond_tickers: np.ndarray = np.array(
        [f"T{i:03d}" for i in range(n_bonds)]
    )
    bond_sectors: np.ndarray = np.array(
        [_SECTORS[i % len(_SECTORS)] for i in range(n_bonds)]
    )

    # ------------------------------------------------------------------
    # 3. Distribute events proportionally across trading days (equal weight)
    # ------------------------------------------------------------------
    day_weights = np.ones(n_days, dtype=np.float64) / n_days
    day_counts: np.ndarray = rng.multinomial(n_events, day_weights)

    # ------------------------------------------------------------------
    # 4. Timestamps with microsecond precision (UTC, 08:00–17:00)
    # ------------------------------------------------------------------
    ts_chunks: list[np.ndarray] = []
    for day, count in zip(trading_days, day_counts):
        if count == 0:
            continue
        # pandas Timestamp.value is nanoseconds since Unix epoch
        day_base_us: int = int(day.value) // 1_000  # ns → µs
        offsets_us: np.ndarray = rng.integers(
            0, _TRADING_RANGE_US, size=int(count), dtype=np.int64
        )
        ts_chunks.append(day_base_us + _TRADING_START_US + offsets_us)

    all_times_us: np.ndarray = np.concatenate(ts_chunks)
    timestamps = pd.to_datetime(all_times_us, unit="us", utc=True)

    # ------------------------------------------------------------------
    # 5. Features: float64 uniform in [−1, +1]
    # ------------------------------------------------------------------
    feature_cols: list[str] = [f"f_{i:03d}" for i in range(n_features)]
    features: np.ndarray = rng.uniform(-1.0, 1.0, size=(n_events, n_features)).astype(
        np.float64
    )

    # ------------------------------------------------------------------
    # 6. Fill labels: ~40 % fill rate with logistic signal on first 5 features
    #    P(fill=1) = σ(0.3 · Σ_{k<5} f_k + bias),  bias = log(0.4/0.6) ≈ −0.405
    # ------------------------------------------------------------------
    signal: np.ndarray = features[:, :5].sum(axis=1)
    bias: float = float(np.log(0.40 / 0.60))
    probs: np.ndarray = 1.0 / (1.0 + np.exp(-(signal * 0.3 + bias)))
    fill: np.ndarray = (rng.uniform(size=n_events) < probs).astype(np.int8)

    # ------------------------------------------------------------------
    # 7. Bond metadata per event (vectorised index lookup)
    # ------------------------------------------------------------------
    bond_idx: np.ndarray = rng.integers(0, n_bonds, size=n_events)
    bond_id_arr: np.ndarray = all_bond_ids[bond_idx]
    ticker_arr: np.ndarray = bond_tickers[bond_idx]
    sector_arr: np.ndarray = bond_sectors[bond_idx]

    # ------------------------------------------------------------------
    # 8. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(features, columns=feature_cols)
    df.insert(0, "event_id", [f"RFQ_{i:08d}" for i in range(n_events)])
    df.insert(1, "timestamp", timestamps)
    df.insert(2, "bond_id", bond_id_arr)
    df.insert(3, "ticker", ticker_arr)
    df.insert(4, "sector", sector_arr)
    df["fill"] = fill

    actual_fill_rate: float = float(df["fill"].mean())
    _LOG.info(
        "Dataset ready: %d events | fill rate=%.3f | maturing bonds=%d/%d",
        n_events,
        actual_fill_rate,
        n_maturing,
        n_bonds,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = generate_rfq_dataset()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEFAULT_OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows → {DEFAULT_OUTPUT_PATH}")
    print(df.dtypes.to_string())
    print(f"Fill rate: {df['fill'].mean():.4f}")
