"""
Sleep analysis — replaces Sleep_server.R analysis functions.

Pipeline:
  1. annotate_sleep()      → adds `asleep` column (replaces sleep_dam_annotation)
  2. detect_sleep_bouts()  → adds sleep/wake bout columns (replaces sleepBoutAnalysis)
  3. Summary functions     → tidy DataFrames for plotting / statistics

All summaries share the same output schema (mirrors R's summaryDT_melted):
    id, labels, order, channel, monitor, x, y
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── sleep annotation ──────────────────────────────────────────────────────────

def annotate_sleep(
    df: pd.DataFrame,
    threshold_min: float = 5.0,
) -> pd.DataFrame:
    """
    Add an `asleep` column — True where the animal is in an inactivity bout
    that lasts ≥ threshold_min minutes.

    Mirrors R's sleepr::sleep_dam_annotation(data, min_time_immobile=threshold_s).

    Parameters
    ----------
    df            : DataStore.data long-format DataFrame
    threshold_min : minimum immobility duration to call sleep (minutes)

    Returns
    -------
    Copy of df with `asleep` (bool) column added.
    """
    df = df.copy()
    df["asleep"] = False
    threshold_s = threshold_min * 60.0

    for aid, grp in df.groupby("id", sort=False):
        active = grp["activity"].values > 0
        t      = grp["t"].values

        for start_t, end_t in _inactive_bouts(active, t):
            if (end_t - start_t) >= threshold_s:
                mask = (grp["t"] >= start_t) & (grp["t"] <= end_t)
                df.loc[grp.index[mask], "asleep"] = True

    return df


# ── bout detection ────────────────────────────────────────────────────────────

def detect_sleep_bouts(
    df: pd.DataFrame,
    ld_period_h: float = 24.0,
) -> pd.DataFrame:
    """
    Detect sleep bouts and wake bouts; compute sleep latency.

    Adds five columns (mirrors R's sleepBoutAnalysis):
      sleep_bout_start    (bool)  : True at first row of each sleep bout
      sleep_bout_duration (float) : sleep bout duration in seconds (0 elsewhere)
      wake_bout_start     (bool)  : True at first row of each wake bout
      wake_bout_duration  (float) : wake bout duration in seconds (0 elsewhere)
      sleep_latency_s     (float) : seconds from lights-off to first sleep bout
                                    (non-zero only at the latency row per day)

    Requires `asleep` column — call annotate_sleep() first.

    Parameters
    ----------
    df           : DataFrame with `asleep` column
    ld_period_h  : LD cycle length in hours (used to clip bouts at cycle boundary)
    """
    _require_asleep(df)
    df = df.copy()
    df["sleep_bout_start"]    = False
    df["sleep_bout_duration"] = 0.0
    df["wake_bout_start"]     = False
    df["wake_bout_duration"]  = 0.0
    df["sleep_latency_s"]     = 0.0

    period_s = ld_period_h * 3600.0

    for aid, grp in df.groupby("id", sort=False):
        idx    = grp.index
        asleep = grp["asleep"].values
        t      = grp["t"].values

        # ── sleep bouts ──────────────────────────────────────────────────────
        for start_t, end_t in _state_bouts(asleep, t, state=True):
            # Clip bout at LD-cycle boundary (mirrors R's ceiling adjustment)
            clipped_end = _clip_to_cycle(start_t, end_t, period_s)
            duration    = clipped_end - start_t
            if duration <= 0:
                continue
            row_pos = np.searchsorted(t, start_t)
            if row_pos < len(t):
                df.at[idx[row_pos], "sleep_bout_start"]    = True
                df.at[idx[row_pos], "sleep_bout_duration"] = duration

        # ── wake bouts (WASO denominator) ────────────────────────────────────
        for start_t, end_t in _state_bouts(asleep, t, state=False):
            clipped_end = _clip_to_cycle(start_t, end_t, period_s)
            duration    = clipped_end - start_t
            if duration <= 0:
                continue
            row_pos = np.searchsorted(t, start_t)
            if row_pos < len(t):
                df.at[idx[row_pos], "wake_bout_start"]    = True
                df.at[idx[row_pos], "wake_bout_duration"] = duration

        # ── sleep latency (time from lights-off to first dark-phase sleep) ───
        days = grp["experiment_day"].unique()
        for day in sorted(days):
            day_mask  = grp["experiment_day"] == day
            dark_mask = day_mask & ~grp["day_night"]
            dark_grp  = grp[dark_mask]
            if dark_grp.empty:
                continue

            dark_t      = dark_grp["t"].values
            dark_asleep = dark_grp["asleep"].values
            dark_idx    = dark_grp.index

            # First row in the dark phase that is asleep
            first_sleep = np.argmax(dark_asleep)
            if not dark_asleep[first_sleep]:
                continue  # no sleep this dark phase

            # Latency = time elapsed since dark phase started
            latency = dark_t[first_sleep] - dark_t[0]
            if latency > 0:
                # Store latency at the row just before first sleep
                store_pos = max(first_sleep - 1, 0)
                df.at[dark_idx[store_pos], "sleep_latency_s"] = latency

    return df


# ── sleep-ratio summaries ─────────────────────────────────────────────────────

def sleep_ratio_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sleep ratio (fraction of time asleep) per animal per light phase.
    Mirrors R's sleepSummary("lightDark").
    """
    _require_asleep(df)
    result = (
        df.groupby(["id", "labels", "order", "channel", "monitor", "phase"],
                   observed=True)
        ["asleep"]
        .mean()
        .reset_index()
        .rename(columns={"phase": "x", "asleep": "y"})
    )
    return _ensure_cols(result)


def sleep_ratio_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sleep ratio per animal per experiment day.
    Mirrors R's sleepSummary("day").
    """
    _require_asleep(df)
    result = (
        df.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["asleep"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    return _ensure_cols(result.drop(columns="experiment_day").rename(columns={"asleep": "y"}))


def sleep_ratio_by_day_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sleep ratio per animal per day × phase.
    Mirrors R's sleepSummary("dayLightDark").
    """
    _require_asleep(df)
    result = (
        df.groupby(
            ["id", "labels", "order", "channel", "monitor",
             "experiment_day", "phase"],
            observed=True,
        )
        ["asleep"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str) + "_" + result["phase"].astype(str)
    return _ensure_cols(
        result.drop(columns=["experiment_day", "phase"]).rename(columns={"asleep": "y"})
    )


# ── TST and WASO ──────────────────────────────────────────────────────────────

def tst_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean total sleep time (hours) per animal per dark phase, averaged over days.
    Mirrors R's TST_WASOSummary("TST", "all").
    """
    _require_bouts(df)
    n_days = df["experiment_day"].nunique()
    dark   = df[~df["day_night"]]
    result = (
        dark.groupby(["id", "labels", "order", "channel", "monitor"])
        ["sleep_bout_duration"]
        .sum()
        .reset_index()
    )
    result["y"] = result["sleep_bout_duration"] / 3600 / max(n_days, 1)
    result["x"] = "TotalSleepTime"
    return _ensure_cols(result.drop(columns="sleep_bout_duration"))


def tst_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total sleep time (hours) per animal per dark phase day.
    Mirrors R's TST_WASOSummary("TST", "day").
    """
    _require_bouts(df)
    dark   = df[~df["day_night"]]
    result = (
        dark.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["sleep_bout_duration"]
        .sum()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    result["y"] = result["sleep_bout_duration"] / 3600
    return _ensure_cols(result.drop(columns=["sleep_bout_duration", "experiment_day"]))


def waso_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean wake time in dark phase (hours) per animal, averaged over days.
    Mirrors R's TST_WASOSummary("WASO", "all").
    """
    _require_bouts(df)
    n_days = df["experiment_day"].nunique()
    dark   = df[~df["day_night"]]
    result = (
        dark.groupby(["id", "labels", "order", "channel", "monitor"])
        ["wake_bout_duration"]
        .sum()
        .reset_index()
    )
    result["y"] = result["wake_bout_duration"] / 3600 / max(n_days, 1)
    result["x"] = "WakeAfterSleepOnset"
    return _ensure_cols(result.drop(columns="wake_bout_duration"))


def waso_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wake time in dark phase (hours) per animal per day.
    Mirrors R's TST_WASOSummary("WASO", "day").
    """
    _require_bouts(df)
    dark   = df[~df["day_night"]]
    result = (
        dark.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["wake_bout_duration"]
        .sum()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    result["y"] = result["wake_bout_duration"] / 3600
    return _ensure_cols(result.drop(columns=["wake_bout_duration", "experiment_day"]))


# ── sleep latency ─────────────────────────────────────────────────────────────

def sleep_latency_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sleep latency (minutes) per animal across all days.
    Mirrors R's boutSleepLatencySummary("all").
    """
    _require_bouts(df)
    lat = df[df["sleep_latency_s"] > 0]
    result = (
        lat.groupby(["id", "labels", "order", "channel", "monitor"])
        ["sleep_latency_s"]
        .mean()
        .reset_index()
    )
    result["y"] = result["sleep_latency_s"] / 60
    result["x"] = "Sleep_latency"
    return _ensure_cols(result.drop(columns="sleep_latency_s"))


def sleep_latency_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sleep latency (minutes) per animal per day.
    Mirrors R's boutSleepLatencySummary("day").
    """
    _require_bouts(df)
    lat = df[df["sleep_latency_s"] > 0]
    result = (
        lat.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["sleep_latency_s"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    result["y"] = result["sleep_latency_s"] / 60
    return _ensure_cols(result.drop(columns=["sleep_latency_s", "experiment_day"]))


# ── sleep bout duration ───────────────────────────────────────────────────────

def sleep_bout_duration_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sleep bout duration (minutes) per animal per phase.
    Mirrors R's boutSleepTimeSummary("lightDark").
    """
    _require_bouts(df)
    bouts = df[df["sleep_bout_start"] & (df["sleep_bout_duration"] > 0)]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "phase"],
                      observed=True)
        ["sleep_bout_duration"]
        .mean()
        .reset_index()
        .rename(columns={"phase": "x", "sleep_bout_duration": "y"})
    )
    result["y"] = result["y"] / 60
    return _ensure_cols(result)


def sleep_bout_duration_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sleep bout duration (minutes) per animal per day.
    Mirrors R's boutSleepTimeSummary("day").
    """
    _require_bouts(df)
    bouts = df[df["sleep_bout_start"] & (df["sleep_bout_duration"] > 0)]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        ["sleep_bout_duration"]
        .mean()
        .reset_index()
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    result["y"] = result["sleep_bout_duration"] / 60
    return _ensure_cols(result.drop(columns=["sleep_bout_duration", "experiment_day"]))


# ── sleep bout count ──────────────────────────────────────────────────────────

def sleep_bout_count_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Number of sleep bouts per animal per phase.
    Mirrors R's boutSleepCountSummary("lightDark").
    """
    _require_bouts(df)
    bouts = df[df["sleep_bout_start"]]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "phase"],
                      observed=True)
        .size()
        .reset_index(name="y")
        .rename(columns={"phase": "x"})
    )
    return _ensure_cols(result)


def sleep_bout_count_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Number of sleep bouts per animal per day.
    Mirrors R's boutSleepCountSummary("day").
    """
    _require_bouts(df)
    bouts = df[df["sleep_bout_start"]]
    result = (
        bouts.groupby(["id", "labels", "order", "channel", "monitor", "experiment_day"])
        .size()
        .reset_index(name="y")
    )
    result["x"] = "Day_" + result["experiment_day"].astype(str)
    return _ensure_cols(result.drop(columns="experiment_day"))


# ── internal helpers ──────────────────────────────────────────────────────────

def _inactive_bouts(
    active: np.ndarray, t: np.ndarray
) -> list[tuple[float, float]]:
    """Return (start_t, end_t) for each contiguous inactive run."""
    bouts: list[tuple[float, float]] = []
    in_bout = False
    for i, is_active in enumerate(active):
        if not is_active and not in_bout:
            start = t[i]
            in_bout = True
        elif is_active and in_bout:
            bouts.append((start, t[i - 1]))
            in_bout = False
    if in_bout:
        bouts.append((start, t[-1]))
    return bouts


def _state_bouts(
    state_arr: np.ndarray,
    t: np.ndarray,
    *,
    state: bool = True,
) -> list[tuple[float, float]]:
    bouts: list[tuple[float, float]] = []
    in_bout = False
    for i, val in enumerate(state_arr):
        if val == state and not in_bout:
            start = t[i]
            in_bout = True
        elif val != state and in_bout:
            bouts.append((start, t[i - 1]))
            in_bout = False
    if in_bout:
        bouts.append((start, t[-1]))
    return bouts


def _clip_to_cycle(start_t: float, end_t: float, period_s: float) -> float:
    """
    Clip a bout's end time to the LD-cycle boundary if the bout crosses it.
    Mirrors R: duration <- (ceiling(t/period) - t/period) * period
    """
    start_cycle = start_t // period_s
    end_cycle   = end_t   // period_s
    if end_cycle > start_cycle:
        return (start_cycle + 1) * period_s
    return end_t


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["y"] = df["y"].fillna(0)
    cols  = ["id", "labels", "order", "channel", "monitor", "x", "y"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra].reset_index(drop=True)


def _require_asleep(df: pd.DataFrame) -> None:
    if "asleep" not in df.columns:
        raise ValueError(
            "DataFrame is missing 'asleep' column. "
            "Run annotate_sleep() first."
        )


def _require_bouts(df: pd.DataFrame) -> None:
    if "sleep_bout_start" not in df.columns:
        raise ValueError(
            "DataFrame is missing sleep bout columns. "
            "Run detect_sleep_bouts() first."
        )
