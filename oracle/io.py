"""I/O helpers for the oracle harness.

``save_df`` writes a DataFrame to BOTH parquet (canonical, fast reload)
AND CSV (human-readable for inspection / Kaggle output preview). All
notebook outputs use this so no data is lost in a notebook output that
only previews CSVs.

``load_df`` prefers the parquet (faster) and falls back to CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def save_df(
    df: pd.DataFrame,
    parquet_path: str | Path,
    *,
    index: bool = False,
) -> Tuple[Path, Path]:
    """Save ``df`` as both ``.parquet`` and ``.csv`` (alongside).

    Args:
        df: DataFrame to persist.
        parquet_path: target path (with .parquet extension). The CSV is
            written next to it with the same stem and ``.csv`` extension.
        index: forwarded to ``to_csv`` (parquet always drops index).

    Returns:
        ``(parquet_path, csv_path)``.
    """
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")

    df.to_parquet(parquet_path)
    df.to_csv(csv_path, index=index)
    return parquet_path, csv_path


def load_df(parquet_path: str | Path) -> pd.DataFrame:
    """Load a DataFrame, preferring parquet but falling back to CSV.

    Useful when running across environments (some Kaggle setups have
    pyarrow issues; the CSV fallback keeps notebooks resilient).
    """
    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = parquet_path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Neither {parquet_path} nor {csv_path} exists"
    )


def is_complete(parquet_path: str | Path, expected_rows: int) -> bool:
    """Return True iff ``parquet_path`` has at least ``expected_rows`` rows.

    Use this — never raw ``path.exists()`` — to gate redo-vs-resume on
    long Kaggle runs: a half-written parquet from a preempted session
    exists on disk but should still trigger a resume.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return False
    try:
        return len(pd.read_parquet(parquet_path)) >= int(expected_rows)
    except Exception:
        return False


def load_or_empty(parquet_path: str | Path) -> pd.DataFrame:
    """Load a parquet if present, else return an empty DataFrame.

    Mid-run resumption seeds its in-memory accumulator from whatever the
    previous session checkpointed; returns empty on miss or read error.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        return pd.DataFrame()
