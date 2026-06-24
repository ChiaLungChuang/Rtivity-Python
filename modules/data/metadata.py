"""
Metadata loader — reads the conditions xlsx that maps monitor files +
channels to experimental conditions.

Expected xlsx columns:
  file           : monitor filename  e.g. "Monitor 1.txt"
  start_datetime : analysis window start  "2021-05-03 12:11:00"
  stop_datetime  : analysis window end    "2021-05-07 07:10:00"
  region_id      : channel number (1–32)
  labels         : condition label  e.g. "Condition A"
  order          : display order for the condition (integer)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"file", "start_datetime", "stop_datetime", "region_id", "labels"}


def read_metadata(path: str | Path) -> pd.DataFrame:
    """
    Load a metadata xlsx file and return a validated, typed DataFrame.

    Parameters
    ----------
    path : path to the .xlsx metadata file

    Returns
    -------
    DataFrame with columns:
        file (str), start_datetime (Timestamp), stop_datetime (Timestamp),
        region_id (int), labels (str), order (int)

    Raises
    ------
    FileNotFoundError  : file does not exist
    ValueError         : missing required columns or unparseable datetimes
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Could not read metadata file {path.name}: {exc}") from exc

    # Normalise column names (strip whitespace, lower-case)
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing required column(s): {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["stop_datetime"]  = pd.to_datetime(df["stop_datetime"],  errors="coerce")

    bad_start = df["start_datetime"].isna().sum()
    bad_stop  = df["stop_datetime"].isna().sum()
    if bad_start or bad_stop:
        raise ValueError(
            f"Metadata has {bad_start} unparseable start_datetime and "
            f"{bad_stop} unparseable stop_datetime values."
        )

    df["region_id"] = df["region_id"].astype(float).astype(int)
    df["labels"]    = df["labels"].astype(str).str.strip()

    if "order" in df.columns:
        df["order"] = df["order"].astype(float).astype(int)
    else:
        # Assign order by first appearance of each label
        label_order = {lab: i + 1 for i, lab in enumerate(df["labels"].unique())}
        df["order"] = df["labels"].map(label_order)

    invalid = df[df["start_datetime"] >= df["stop_datetime"]]
    if not invalid.empty:
        raise ValueError(
            f"{len(invalid)} row(s) have start_datetime ≥ stop_datetime:\n"
            + invalid[["file", "start_datetime", "stop_datetime"]].to_string(index=False)
        )

    return df.reset_index(drop=True)


def list_files(meta: pd.DataFrame) -> list[str]:
    """Return unique monitor filenames in the order they appear."""
    return list(dict.fromkeys(meta["file"].tolist()))


def list_labels(meta: pd.DataFrame) -> list[str]:
    """Return unique condition labels sorted by their 'order' column."""
    return (
        meta[["labels", "order"]]
        .drop_duplicates()
        .sort_values("order")["labels"]
        .tolist()
    )


def channels_for_file(meta: pd.DataFrame, filename: str) -> list[int]:
    """Return the channel numbers assigned to a given monitor file."""
    return sorted(meta.loc[meta["file"] == filename, "region_id"].tolist())


def label_for_channel(meta: pd.DataFrame, filename: str, channel: int) -> str | None:
    """Return the condition label for a specific file + channel, or None."""
    mask = (meta["file"] == filename) & (meta["region_id"] == channel)
    result = meta.loc[mask, "labels"]
    return result.iloc[0] if not result.empty else None


def time_window_for_file(
    meta: pd.DataFrame, filename: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return (start_datetime, stop_datetime) for a given monitor file.
    All channels in a file must share the same window.
    """
    sub = meta[meta["file"] == filename]
    if sub.empty:
        raise KeyError(f"File '{filename}' not found in metadata.")
    return sub["start_datetime"].iloc[0], sub["stop_datetime"].iloc[0]
