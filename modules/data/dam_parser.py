"""
DAM file parser — replaces the R `damr` package.

DAM (Drosophila Activity Monitor) files from Trikinetics have this format
(whitespace-delimited, no header):

  col 0  : row index
  col 1  : day   (e.g. 3)
  col 2  : month (e.g. May)
  col 3  : year  (2-digit, e.g. 21)
  col 4  : time  (HH:MM:SS)
  col 5  : status (1 = normal, other values may indicate issues)
  col 6-8: unused flags
  col 9  : label ("Ct")
  col 10-41: activity counts for channels 1–32
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Mapping of 3-letter month abbreviations (Trikinetics uses English abbrevs)
_MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

N_CHANNELS = 32
_CHANNEL_COLS = [f"ch{i:02d}" for i in range(1, N_CHANNELS + 1)]


def read_dam_file(
    path: str | Path,
    start: pd.Timestamp | None = None,
    stop: pd.Timestamp | None = None,
    zt0: str = "08:00:00",
    ld_period: float = 24.0,
    light_hours: float = 12.0,
) -> pd.DataFrame:
    """
    Read a single DAM monitor file and return a wide-format DataFrame.

    Parameters
    ----------
    path : path to the DAM .txt file
    start : optional start datetime filter (inclusive)
    stop  : optional stop datetime filter (inclusive)
    zt0   : light-onset time as "HH:MM:SS" (Zeitgeber Time 0), default 08:00
    ld_period : length of one LD cycle in hours (default 24)
    light_hours : duration of the light phase in hours (default 12)

    Returns
    -------
    DataFrame with columns:
        datetime, status, ch01…ch32, zt, phase (Light/Dark)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DAM file not found: {path}")

    raw = _read_raw(path)
    raw["datetime"] = _parse_datetime(raw)
    raw = raw.drop(columns=["date_str", "time"])

    # Apply time window
    if start is not None:
        raw = raw[raw["datetime"] >= pd.Timestamp(start)]
    if stop is not None:
        raw = raw[raw["datetime"] <= pd.Timestamp(stop)]

    if raw.empty:
        raise ValueError(
            f"No data in [{start}, {stop}] for file {path.name}"
        )

    raw = raw.reset_index(drop=True)
    raw["zt"] = _compute_zt(raw["datetime"], zt0, ld_period)
    raw["phase"] = _compute_phase(raw["zt"], light_hours)

    return raw


def to_long_format(
    df: pd.DataFrame,
    channels: list[int] | None = None,
    monitor_name: str = "",
) -> pd.DataFrame:
    """
    Melt a wide DAM DataFrame into long format.

    Parameters
    ----------
    df       : output of read_dam_file()
    channels : list of 1-based channel numbers to keep (None = all 32)
    monitor_name : label for the 'monitor' column

    Returns
    -------
    Long DataFrame with columns:
        datetime, zt, phase, monitor, channel, activity
    """
    if channels is not None:
        cols = [f"ch{c:02d}" for c in channels]
    else:
        cols = _CHANNEL_COLS

    keep = ["datetime", "zt", "phase"] + cols
    sub = df[keep].copy()

    long = sub.melt(
        id_vars=["datetime", "zt", "phase"],
        value_vars=cols,
        var_name="channel",
        value_name="activity",
    )
    # channel "ch01" → integer 1
    long["channel"] = long["channel"].str.lstrip("ch").astype(int)
    long["monitor"] = monitor_name
    long["activity"] = pd.to_numeric(long["activity"], errors="coerce")

    return long.sort_values(["channel", "datetime"]).reset_index(drop=True)


def get_time_range(path: str | Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (first_datetime, last_datetime) without loading all data."""
    path = Path(path)
    raw = _read_raw(path)
    dt = _parse_datetime(raw)
    return dt.iloc[0], dt.iloc[-1]


# ── internal helpers ──────────────────────────────────────────────────────────

def _read_raw(path: Path) -> pd.DataFrame:
    """Read DAM file into a normalised DataFrame.

    DAM files are TAB-delimited with 42 fields per row (no header):
      [0]  : row index
      [1]  : date string  e.g. "3 May 21"
      [2]  : time         e.g. "11:09:00"
      [3]  : status       (1 = normal, 24 = first row of recording)
      [4-8]: unused flags / label
      [9-40]: activity counts for channels 1–32
      [41] : trailing field (ignored)
    """
    try:
        raw = pd.read_csv(
            path,
            sep="\t",
            header=None,
            dtype=str,
            engine="python",
        )
    except Exception as exc:
        raise ValueError(f"Could not read DAM file {path.name}: {exc}") from exc

    if raw.shape[1] < 42:
        raise ValueError(
            f"{path.name}: expected 42 tab-separated columns, got {raw.shape[1]}. "
            "Is this a valid DAM file?"
        )

    # Build a clean DataFrame from positional columns
    df = pd.DataFrame()
    df["date_str"] = raw.iloc[:, 1].astype(str).str.strip()
    df["time"]     = raw.iloc[:, 2].astype(str).str.strip()
    df["status"]   = pd.to_numeric(raw.iloc[:, 3], errors="coerce").fillna(0).astype(int)

    for i, col in enumerate(_CHANNEL_COLS):
        df[col] = pd.to_numeric(raw.iloc[:, 9 + i], errors="coerce").fillna(0).astype(int)

    return df


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    """Parse 'date_str' ("3 May 21") + 'time' ("11:09:00") into Timestamps."""
    # date_str is e.g. "3 May 21" — day month(abbrev) 2-digit-year
    dt_str = df["date_str"].str.strip() + " " + df["time"].str.strip()
    parsed = pd.to_datetime(dt_str, format="%d %b %y %H:%M:%S", errors="coerce")

    if parsed.isna().any():
        n_bad = parsed.isna().sum()
        raise ValueError(
            f"{n_bad} rows could not be parsed as datetime. "
            f"Sample bad values: {dt_str[parsed.isna()].head(3).tolist()}"
        )
    return parsed


def _compute_zt(
    dt: pd.Series,
    zt0: str,
    ld_period: float,
) -> pd.Series:
    """
    Compute Zeitgeber Time (hours since last ZT0).

    ZT0 is lights-on; ZT12 (in a 12:12 LD) is lights-off.
    """
    h0, m0, s0 = map(int, zt0.split(":"))
    onset_seconds = h0 * 3600 + m0 * 60 + s0
    period_seconds = ld_period * 3600

    seconds_of_day = (
        dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    )
    zt_seconds = (seconds_of_day - onset_seconds) % period_seconds
    return (zt_seconds / 3600).round(6)


def _compute_phase(zt: pd.Series, light_hours: float) -> pd.Series:
    """Return 'Light' or 'Dark' based on ZT and light duration."""
    return pd.Categorical(
        np.where(zt < light_hours, "Light", "Dark"),
        categories=["Light", "Dark"],
    )
