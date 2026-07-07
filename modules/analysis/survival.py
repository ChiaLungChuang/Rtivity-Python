"""
Survival / lifespan analysis — replaces the survival R package usage in Rtivity.

Pipeline:
  1. build_survival_data()    → per-animal duration + event table
  2. kaplan_meier_curves()    → KM survival curves per group
  3. logrank_test()           → pairwise log-rank p-values
  4. survival_summary()       → median survival + CI per group

Death detection (mirrors R's detect_dead_animals_rle() in
Survival_analysis_server.R, with the death time taken as the *first* qualifying
silence rather than the longest): scan the recording forward, and call the
animal dead at the start of the **first** continuous zero-activity run that
lasts at least `dead_window_h` hours.  Whether a *later* silent run happens to
be longer is irrelevant — the first one to cross the threshold marks death.
If no silent run reaches the threshold the animal is right-censored, regardless
of whether the recording happens to end in silence.  The scan covers the whole
DataStore series, which is already trimmed to the user's selected start/finish
window upstream, so "within the selected period" is enforced by construction.

All tabular outputs use the standard schema where applicable:
    id, labels, order, channel, monitor, x, y
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test as _pairwise_logrank
from lifelines.statistics import multivariate_logrank_test


# ── build per-animal survival table ──────────────────────────────────────────

def _first_lethal_silence(
    t: np.ndarray, activity: np.ndarray, window_s: float
) -> float | None:
    """
    Return the start time of the **first** contiguous zero-activity run whose
    span is at least `window_s` seconds, scanning forward. None if no run
    qualifies.

    A later run being longer does not matter — the first stretch of silence to
    reach the threshold is the one that marks death.  Span is measured from the
    first to the last zero sample of the run via `t`, so it does not assume a
    fixed 1-minute sampling interval the way R's row-count / 60 does.
    """
    inactive = activity == 0
    n = len(inactive)
    i = 0
    while i < n:
        if not inactive[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inactive[j + 1]:
            j += 1
        if t[j] - t[i] >= window_s:
            return float(t[i])
        i = j + 1
    return None


def build_survival_data(
    df: pd.DataFrame,
    dead_window_h: float = 24.0,
) -> pd.DataFrame:
    """
    Infer lifespan from DAM activity data.

    An animal is classified as **dead** if any continuous silent run reaches
    `dead_window_h` hours; death is dated to the experiment_day on which the
    **first** such run began.  A later, longer silent run does not move the
    death time.

    Animals whose silent runs never reach the threshold are **right-censored**
    (event = 0) with duration = last recorded experiment_day — even if the
    recording happens to end in silence.

    Parameters
    ----------
    df            : DataStore.data long-format DataFrame
    dead_window_h : minimum hours of unbroken silence to call death

    Returns
    -------
    DataFrame with columns:
        id, labels, order, channel, monitor, duration (days), event (0/1)
    """
    window_s = dead_window_h * 3600.0
    records  = []

    for aid, grp in df.groupby("id", sort=False):
        meta = grp[["id", "labels", "order", "channel", "monitor"]].iloc[0]
        grp  = grp.sort_values("t")
        t    = grp["t"].to_numpy()
        act  = grp["activity"].to_numpy()

        death_start = _first_lethal_silence(t, act, window_s)

        if death_start is not None:
            death_day = grp.loc[grp["t"] == death_start, "experiment_day"].iloc[0]
            duration  = int(death_day)
            event     = 1
        else:
            duration = int(grp["experiment_day"].max())
            event    = 0

        records.append({**meta.to_dict(), "duration": duration, "event": event})

    return pd.DataFrame(records)


def flag_ambiguous_deaths(
    df: pd.DataFrame,
    dead_window_h: float = 24.0,
) -> pd.DataFrame:
    """
    Find dead animals that recorded activity *after* their inferred time of
    death (the first silence reaching `dead_window_h`).

    build_survival_data() deliberately treats that later movement as artifact
    and keeps the early death time, so these animals don't show up as unusual
    in the survival curves. They are flagged here so the user can sanity-check
    them — a monitor glitch, a dislodged fly, or a threshold set too short can
    make a living animal look briefly dead.

    Returns one row per flagged animal (empty DataFrame with the right columns
    if none):
        id, labels, death_day, post_death_counts,
        post_death_active_bins, last_active_day
    """
    window_s = dead_window_h * 3600.0
    rows = []

    for aid, grp in df.groupby("id", sort=False):
        grp = grp.sort_values("t")
        t   = grp["t"].to_numpy()
        act = grp["activity"].to_numpy()

        death_start = _first_lethal_silence(t, act, window_s)
        if death_start is None:
            continue  # censored / alive

        after = t > death_start
        post_counts = int(act[after].sum())
        if post_counts <= 0:
            continue  # clean death — silent through to the end of the record

        meta = grp[["id", "labels"]].iloc[0]
        death_day = int(grp.loc[grp["t"] == death_start, "experiment_day"].iloc[0])
        post_bins = int((after & (act > 0)).sum())
        last_active_day = int(grp.loc[act > 0, "experiment_day"].iloc[-1])

        rows.append({
            "id": meta["id"],
            "labels": meta["labels"],
            "death_day": death_day,
            "post_death_counts": post_counts,
            "post_death_active_bins": post_bins,
            "last_active_day": last_active_day,
        })

    return pd.DataFrame(rows, columns=[
        "id", "labels", "death_day", "post_death_counts",
        "post_death_active_bins", "last_active_day",
    ])


# ── Kaplan-Meier curves ───────────────────────────────────────────────────────

def kaplan_meier_curves(survival_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a Kaplan-Meier estimator for each condition group.

    Mirrors R: survfit(Surv(duration, event) ~ labels, data=…)

    Returns
    -------
    Long-format DataFrame:
        labels, time, survival, ci_lower, ci_upper
    """
    _require_survival(survival_df)
    records = []

    for label, grp in survival_df.groupby("labels", sort=False):
        kmf = KaplanMeierFitter(label=str(label))
        kmf.fit(grp["duration"], event_observed=grp["event"])

        sf  = kmf.survival_function_
        ci  = kmf.confidence_interval_

        for t in kmf.timeline:
            records.append({
                "labels":   label,
                "time":     float(t),
                "survival": float(sf.at[t, str(label)]),
                "ci_lower": float(ci.at[t, f"{label}_lower_0.95"]),
                "ci_upper": float(ci.at[t, f"{label}_upper_0.95"]),
            })

    return pd.DataFrame(records)


# ── log-rank tests ────────────────────────────────────────────────────────────

def logrank_test(survival_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise log-rank tests between all condition groups.

    Mirrors R: survdiff(Surv(duration, event) ~ labels, data=…)

    Returns
    -------
    DataFrame with columns:
        group1, group2, test_statistic, p_value
    """
    _require_survival(survival_df)
    labels = survival_df["labels"].unique().tolist()

    if len(labels) < 2:
        return pd.DataFrame(columns=["group1", "group2", "test_statistic", "p_value"])

    records = []
    for g1, g2 in itertools.combinations(labels, 2):
        a = survival_df[survival_df["labels"] == g1]
        b = survival_df[survival_df["labels"] == g2]

        result = _pairwise_logrank(
            a["duration"], b["duration"],
            a["event"],   b["event"],
        )
        records.append({
            "group1":          g1,
            "group2":          g2,
            "test_statistic":  round(float(result.test_statistic), 4),
            "p_value":         round(float(result.p_value), 6),
        })

    return pd.DataFrame(records)


def multigroup_logrank_test(survival_df: pd.DataFrame) -> dict:
    """
    Overall (multivariate) log-rank test across all groups.

    Returns dict with keys: test_statistic, p_value, degrees_of_freedom
    """
    _require_survival(survival_df)
    result = multivariate_logrank_test(
        survival_df["duration"],
        survival_df["labels"],
        survival_df["event"],
    )
    return {
        "test_statistic":  round(float(result.test_statistic), 4),
        "p_value":         round(float(result.p_value), 6),
        "degrees_of_freedom": len(survival_df["labels"].unique()) - 1,
    }


# ── survival summary ──────────────────────────────────────────────────────────

def survival_summary(survival_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-group descriptive statistics.

    Mirrors R's summary(survfit(…)).

    Returns standard long-format DataFrame:
        id=NaN, labels, order=NaN, channel=NaN, monitor=NaN,
        x ∈ {"n", "events", "median", "mean"}, y
    """
    _require_survival(survival_df)
    records = []

    for label, grp in survival_df.groupby("labels", sort=False):
        kmf = KaplanMeierFitter()
        kmf.fit(grp["duration"], event_observed=grp["event"])

        n      = len(grp)
        events = int(grp["event"].sum())
        median = kmf.median_survival_time_
        mean   = float(kmf.survival_function_["KM_estimate"].sum()
                       * (kmf.timeline[1] - kmf.timeline[0])
                       if len(kmf.timeline) > 1 else np.nan)

        for metric, val in [("n", n), ("events", events),
                             ("median", median), ("mean", mean)]:
            records.append({"labels": label, "x": metric, "y": float(val)})

    df = pd.DataFrame(records)
    for col in ["id", "order", "channel", "monitor"]:
        df[col] = np.nan
    return _ensure_cols(df)


def median_survival(survival_df: pd.DataFrame) -> pd.DataFrame:
    """
    Median survival (days) per group in standard x/y format.
    x = group label, y = median days survived.

    CI bounds (ci_lower, ci_upper) are the times where the lower / upper
    95% confidence bands cross the 0.5 survival probability level.
    """
    _require_survival(survival_df)
    records = []

    for label, grp in survival_df.groupby("labels", sort=False):
        kmf = KaplanMeierFitter()
        kmf.fit(grp["duration"], event_observed=grp["event"])

        ci = kmf.confidence_interval_
        lower_col = ci.columns[0]   # e.g. "KM_estimate_lower_0.95"
        upper_col = ci.columns[1]   # e.g. "KM_estimate_upper_0.95"

        ci_lower = _crossing_time(ci.index.values, ci[lower_col].values, 0.5)
        ci_upper = _crossing_time(ci.index.values, ci[upper_col].values, 0.5)

        records.append({
            "labels":   label,
            "x":        str(label),
            "y":        float(kmf.median_survival_time_),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    df = pd.DataFrame(records)
    for col in ["id", "order", "channel", "monitor"]:
        df[col] = np.nan
    return _ensure_cols(df)


def _crossing_time(t: np.ndarray, s: np.ndarray, level: float = 0.5) -> float:
    """Return the first time where s crosses below `level`; NaN if never."""
    idx = np.searchsorted(-s, -level)   # first index where s <= level
    if idx >= len(t):
        return np.nan
    return float(t[idx])


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_survival(df: pd.DataFrame) -> None:
    for col in ("duration", "event", "labels"):
        if col not in df.columns:
            raise ValueError(
                f"Survival DataFrame is missing '{col}' column. "
                "Run build_survival_data() first."
            )


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    cols  = ["id", "labels", "order", "channel", "monitor", "x", "y"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra].reset_index(drop=True)
