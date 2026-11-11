"""
Activity analysis — replaces Activity_server.R analysis functions.

All public functions accept a long-format DataFrame from DataStore.data
and return tidy summary DataFrames ready for plotting or statistics.

Summary output format (mirrors R's summaryDT_melted):
    id, labels, order, channel, monitor, x, y
where x is the grouping label ("Light", "Dark", "Day_0", …)
and y is the measured value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bins import runs


# ── bout detection ────────────────────────────────────────────────────────────

def detect_activity_bouts(
    df: pd.DataFrame,
    threshold_min: float = 5.0,
) -> pd.DataFrame:
    """
    Detect activity bouts for each animal and add three columns:

      bout_start      (bool)  : True at the first row of each activity bout
      bout_duration_s (float) : duration of the bout in seconds (0 elsewhere)
      bout_activity   (int)   : total activity counts within the bout (0 elsewhere)

    A bout is a maximal period where the animal is active (activity > 0).
    Inactive gaps ≤ `threshold_min` minutes within an otherwise active stretch
    are bridged (treated as part of the bout) — mirrors R's ActivityBoutWindow.

    Parameters
    ----------
    df            : DataStore.data long-format DataFrame
    threshold_min : gap-bridging threshold in minutes

    Returns
    -------
    df with the three new columns added (original df is not modified).
    """
    df = df.copy()
    df["bout_start"]      = False
    df["bout_duration_s"] = 0.0
    df["bout_activity"]   = 0

    threshold_s = threshold_min * 60.0

    for aid, grp in df.groupby("id", sort=False):
        idx = grp.index
        active = grp["activity"].values > 0
        t      = grp["t"].values
        act    = grp["activity"].values

        bouts = _merge_bouts(_raw_bouts(active, t), threshold_s)

        for (t_start, t_end) in bouts:
            # row that contains t_start
            start_pos = np.searchsorted(t, t_start)
            end_pos   = np.searchsorted(t, t_end, side="left")   # t_end exclusive
            if start_pos >= len(t):
                continue

            row_idx = idx[start_pos]
            duration = float(t_end - t_start)
            total_act = int(act[start_pos:end_pos].sum())

            df.at[row_idx, "bout_start"]      = True
            df.at[row_idx, "bout_duration_s"] = duration
            df.at[row_idx, "bout_activity"]   = total_act

    return df


def _raw_bouts(
    active: np.ndarray, t: np.ndarray
) -> list[tuple[float, float]]:
    """(start_t, end_t) for each contiguous active run; end_t is exclusive, so
    end_t - start_t is the run's bin count × bin width (see bins.py), and the
    gap to the next run's start is exactly the quiet time between them."""
    return runs(np.asarray(active, dtype=bool), t)


def _merge_bouts(
    bouts: list[tuple[float, float]], threshold_s: float
) -> list[tuple[float, float]]:
    """Merge bouts whose gap is ≤ threshold_s (bridges inactive gaps)."""
    if not bouts:
        return []
    merged = [bouts[0]]
    for start, end in bouts[1:]:
        if start - merged[-1][1] <= threshold_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


# ── activity summaries ────────────────────────────────────────────────────────

def activity_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total activity per animal per light phase (Light / Dark) across all days.
    Mirrors R's activitySummary("lightDark").
    """
    result = (
        df.groupby(["id", "labels", "order", "channel", "monitor", "phase"],
                   observed=True)
        ["activity"]
        .sum()
        .reset_index()
        .rename(columns={"phase": "x", "activity": "y"})
    )
    return _ensure_cols(result)


def activity_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total activity per animal per experiment day.
    Mirrors R's activitySummary("day").
    """
    result = (
        df.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["activity"]
        .sum()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    result = result.drop(columns="experiment_day").rename(columns={"activity": "y"})
    return _ensure_cols(result)


def activity_by_day_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total activity per animal per experiment day × light phase.
    Mirrors R's activitySummary("dayLightDark").
    """
    result = (
        df.groupby(
            ["id", "labels", "order", "channel", "monitor",
             "experiment_day", "phase"],
            observed=True,
        )
        ["activity"]
        .sum()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str) + "_" + result["phase"].astype(str)
    result = result.drop(columns=["experiment_day", "phase"]).rename(columns={"activity": "y"})
    return _ensure_cols(result)


def activity_by_custom_bins(
    df: pd.DataFrame, bin_size_cycles: float = 1.0, ld_period_h: float = 24.0
) -> pd.DataFrame:
    """
    Total activity per animal per custom time bin.
    Mirrors R's activitySummary("custom") / SleepBoxPlot_time grouping.
    """
    bin_s = bin_size_cycles * ld_period_h * 3600
    df = df.copy()
    df["_bin"] = (df["t"] // bin_s).astype(int)
    result = (
        df.groupby(["id", "labels", "order", "channel", "monitor", "_bin"])
        ["activity"]
        .sum()
        .reset_index()
    )
    result["x"] = "Group_" + result["_bin"].astype(str)
    result = result.drop(columns="_bin").rename(columns={"activity": "y"})
    return _ensure_cols(result)


# ── bout summaries ────────────────────────────────────────────────────────────

def bout_activity_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean activity per bout per animal per phase.
    Mirrors R's boutActivitySummary("lightDark").
    Requires detect_activity_bouts() to have been run.
    """
    _require_bouts(df)
    bouts = df[df["bout_start"] & (df["bout_activity"] > 0)]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "phase"],
                      observed=True)
        ["bout_activity"]
        .mean()
        .reset_index()
        .rename(columns={"phase": "x", "bout_activity": "y"})
    )
    return _ensure_cols(result)


def bout_activity_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean activity per bout per animal per experiment day.
    Mirrors R's boutActivitySummary("day").
    """
    _require_bouts(df)
    bouts = df[df["bout_start"] & (df["bout_activity"] > 0)]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["bout_activity"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    result = result.drop(columns="experiment_day").rename(columns={"bout_activity": "y"})
    return _ensure_cols(result)


def bout_activity_by_day_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean activity per bout per animal per day × phase.
    Mirrors R's boutActivitySummary("dayLightDark").
    """
    _require_bouts(df)
    bouts = df[df["bout_start"] & (df["bout_activity"] > 0)]
    result = (
        bouts.groupby(
            ["id", "labels", "order", "channel", "monitor",
             "experiment_day", "phase"],
            observed=True,
        )
        ["bout_activity"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str) + "_" + result["phase"].astype(str)
    result = result.drop(columns=["experiment_day", "phase"]).rename(columns={"bout_activity": "y"})
    return _ensure_cols(result)


def bout_duration_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean bout duration (seconds) per animal per phase.
    Mirrors R's boutTimeSummary("lightDark").
    """
    _require_bouts(df)
    bouts = df[df["bout_start"] & (df["bout_duration_s"] > 0)]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "phase"],
                      observed=True)
        ["bout_duration_s"]
        .mean()
        .reset_index()
        .rename(columns={"phase": "x", "bout_duration_s": "y"})
    )
    return _ensure_cols(result)


def bout_duration_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean bout duration (seconds) per animal per experiment day.
    Mirrors R's boutTimeSummary("day").
    """
    _require_bouts(df)
    bouts = df[df["bout_start"] & (df["bout_duration_s"] > 0)]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["bout_duration_s"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    result = result.drop(columns="experiment_day").rename(columns={"bout_duration_s": "y"})
    return _ensure_cols(result)


def bout_duration_by_day_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean bout duration (seconds) per animal per day × phase.
    Mirrors R's boutTimeSummary("dayLightDark").
    """
    _require_bouts(df)
    bouts = df[df["bout_start"] & (df["bout_duration_s"] > 0)]
    result = (
        bouts.groupby(
            ["id", "labels", "order", "channel", "monitor",
             "experiment_day", "phase"],
            observed=True,
        )
        ["bout_duration_s"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str) + "_" + result["phase"].astype(str)
    result = result.drop(columns=["experiment_day", "phase"]).rename(columns={"bout_duration_s": "y"})
    return _ensure_cols(result)


# ── statistics report ─────────────────────────────────────────────────────────

def summary_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics from any activity/bout summary DataFrame.
    Mirrors R's statisticsReport() function.

    Input columns: id, labels, order, x, y
    Output columns: condition, x, n, mean, sem, sd, median, q1, q3
    """
    def _sem(s: pd.Series) -> float:
        return s.std(ddof=1) / np.sqrt(s.count()) if s.count() > 1 else np.nan

    grp = long_df.dropna(subset=["y"]).groupby(["labels", "x"], sort=False, observed=True)["y"]

    stats = grp.agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
    ).reset_index()
    stats["sem"] = grp.apply(_sem).values
    stats["q1"]  = grp.quantile(0.25).values
    stats["q3"]  = grp.quantile(0.75).values

    stats = stats.round({"mean": 3, "sem": 3, "sd": 3, "median": 3, "q1": 3, "q3": 3})
    stats = stats.sort_values(["labels", "x"]).reset_index(drop=True)
    return stats


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee standard column order and replace NaN y-values with 0."""
    df["y"] = df["y"].fillna(0)
    cols = ["id", "labels", "order", "channel", "monitor", "x", "y"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra].reset_index(drop=True)


def _require_bouts(df: pd.DataFrame) -> None:
    if "bout_start" not in df.columns:
        raise ValueError(
            "DataFrame is missing bout columns. "
            "Run detect_activity_bouts() first."
        )
