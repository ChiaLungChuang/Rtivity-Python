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


# ── manual metadata builder (UI "Manual Setup" mode, no xlsx upload) ──────────
#
# Equivalent to the R app's addcondition observer (Data_selection_server.R):
# the user repeatedly names a condition and assigns it a set of channels for
# the currently selected file. Here, a file can have any number of groups
# (1..N), not a fixed count.

def parse_channel_spec(spec: str) -> list[int]:
    """
    Parse a channel spec like '1-16' or '1,3,5-10' into a sorted list of
    unique 1-based channel numbers.

    Raises
    ------
    ValueError : spec is empty, malformed, or contains channels outside 1-32.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Channel spec is empty.")

    channels: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                lo, hi = int(lo_str), int(hi_str)
            except ValueError:
                raise ValueError(f"Invalid channel range: '{part}'") from None
            if lo > hi:
                lo, hi = hi, lo
            channels.update(range(lo, hi + 1))
        else:
            try:
                channels.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid channel number: '{part}'") from None

    if not channels:
        raise ValueError(f"No channels parsed from spec: '{spec}'")
    if any(not (1 <= c <= 32) for c in channels):
        raise ValueError(f"Channel numbers must be between 1 and 32: '{spec}'")

    return sorted(channels)


def _normalize_channels(channels) -> list[int]:
    """
    Accept either a spec string ('1-16', '1,3,5-10') or an iterable of channel
    numbers (the checkbox UI hands us a list), and return a sorted, validated,
    de-duplicated list of 1-based channel numbers in 1..32.
    """
    if isinstance(channels, str):
        return parse_channel_spec(channels)

    try:
        nums = {int(c) for c in channels}
    except (TypeError, ValueError):
        raise ValueError(f"Invalid channel selection: {channels!r}") from None
    if not nums:
        raise ValueError("No channels selected.")
    if any(not (1 <= c <= 32) for c in nums):
        raise ValueError(f"Channel numbers must be between 1 and 32: {sorted(nums)}")
    return sorted(nums)


def build_manual_conditions(file_entries: list[dict]) -> pd.DataFrame:
    """
    Build a conditions DataFrame (same shape as read_metadata()'s return
    value) from manually entered file/group/channel assignments — used when
    the user picks "Manual Setup" instead of uploading a metadata .xlsx.

    Parameters
    ----------
    file_entries : one dict per DAM file, each shaped:
        {
            "file": "Monitor 1.txt",
            "start_datetime": "2026-06-17 15:28:00",
            "stop_datetime":  "2026-06-22 10:51:00",
            "groups": [
                {"label": "Control", "channels": "1-16"},
                {"label": "Experimental", "channels": "17-32"},
                ...                                       # any count, >= 1
            ],
        }

    Returns
    -------
    DataFrame with columns file, start_datetime, stop_datetime, region_id,
    labels, order — one row per channel. A label's `order` is assigned by
    first appearance across all files, so the same condition name on
    different files shares one order value (mirrors R's behaviour).

    Raises
    ------
    ValueError : empty label, unparseable channel spec, channels reused by
                 more than one group within the same file, or
                 start_datetime >= stop_datetime.
    """
    rows: list[dict] = []
    label_order: dict[str, int] = {}

    for entry in file_entries:
        filename = entry["file"]

        start = pd.to_datetime(entry.get("start_datetime"), errors="coerce")
        stop = pd.to_datetime(entry.get("stop_datetime"), errors="coerce")
        if pd.isna(start) or pd.isna(stop):
            raise ValueError(
                f"{filename}: start/finish datetime could not be parsed."
            )
        if start >= stop:
            raise ValueError(
                f"{filename}: start datetime must be before finish datetime."
            )

        groups = entry.get("groups") or []
        if not groups:
            raise ValueError(f"{filename}: no conditions defined.")

        seen_channels: set[int] = set()
        for group in groups:
            label = (group.get("label") or "").strip()
            if not label:
                raise ValueError(f"{filename}: condition name cannot be empty.")

            channels = _normalize_channels(group.get("channels", ""))
            overlap = seen_channels & set(channels)
            if overlap:
                raise ValueError(
                    f"{filename}: channel(s) {sorted(overlap)} assigned to "
                    "more than one condition."
                )
            seen_channels.update(channels)

            if label not in label_order:
                label_order[label] = len(label_order) + 1

            for ch in channels:
                rows.append({
                    "file": filename,
                    "start_datetime": start,
                    "stop_datetime": stop,
                    "region_id": ch,
                    "labels": label,
                    "order": label_order[label],
                })

    if not rows:
        raise ValueError("No conditions defined for any file.")

    return pd.DataFrame(rows)
