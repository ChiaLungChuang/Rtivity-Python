"""
Circadian rhythm analysis — replaces zeitgebr, ActCR, nparACT R packages.

Pipeline (per animal):
  1. chi_sq_periodogram() / lomb_scargle_periodogram()  → find dominant period
  2. fit_cosinor()                                       → amplitude, acrophase, mesor
  3. is_iv_ra()                                          → IS, IV, RA non-parametric metrics

All multi-animal wrappers return tidy DataFrames with the standard schema:
    id, labels, order, channel, monitor, x, y
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import chi2


# ── periodogram helpers ───────────────────────────────────────────────────────

_DEFAULT_PERIODS_H = np.arange(16.0, 32.5, 0.5)   # 16–32 h in 0.5 h steps


def chi_sq_periodogram(
    activity: np.ndarray,
    t: np.ndarray,
    periods_h: np.ndarray | None = None,
    bin_size_min: float = 30.0,
) -> pd.DataFrame:
    """
    Sokolove & Bushell (1978) chi-square periodogram for a single animal.

    For each test period τ, the time series is folded and the variance of
    column (phase) means is compared to the total variance via a chi-square
    statistic Q.  The period with maximum Q is the dominant rhythm.

    Parameters
    ----------
    activity     : 1-D array of activity counts (pre-binned or raw 1-min data)
    t            : 1-D array of elapsed time in seconds (same length)
    periods_h    : test periods in hours  (default: 16–32 h, step 0.5 h)
    bin_size_min : bin size for resampling before folding (minutes)

    Returns
    -------
    DataFrame with columns: period_h, Q, p_value
    """
    if periods_h is None:
        periods_h = _DEFAULT_PERIODS_H

    # ── resample to uniform bins ─────────────────────────────────────────────
    x = _bin_series(activity, t, bin_size_min)
    N = len(x)
    if N < 3:
        return pd.DataFrame({"period_h": periods_h, "Q": np.nan, "p_value": np.nan})

    grand_var = x.var(ddof=1)
    grand_mean = x.mean()

    records = []
    for tau_h in periods_h:
        p = max(round(tau_h * 60.0 / bin_size_min), 2)   # bins per period
        k = N // p                                         # complete cycles
        if k < 1:
            records.append({"period_h": tau_h, "Q": np.nan, "p_value": np.nan})
            continue

        mat = x[: k * p].reshape(k, p)
        col_means = mat.mean(axis=0)

        if grand_var == 0:
            Q = 0.0
        else:
            Q = float(k * p * np.sum((col_means - grand_mean) ** 2) / grand_var)

        p_val = float(1.0 - chi2.cdf(Q, df=p - 1))
        records.append({"period_h": tau_h, "Q": Q, "p_value": p_val})

    return pd.DataFrame(records)


def lomb_scargle_periodogram(
    activity: np.ndarray,
    t: np.ndarray,
    periods_h: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Lomb-Scargle periodogram for a single animal (unevenly spaced friendly).

    Uses scipy.signal.lombscargle.  The normalised power is in [0, 1].

    Returns
    -------
    DataFrame with columns: period_h, power
    """
    if periods_h is None:
        periods_h = _DEFAULT_PERIODS_H

    t_sec = np.asarray(t, dtype=float)
    x     = np.asarray(activity, dtype=float)
    x     = x - x.mean()          # Lomb-Scargle expects zero-mean signal

    if len(x) < 3 or x.std() == 0:
        return pd.DataFrame({"period_h": periods_h, "power": np.nan})

    angular_freqs = 2.0 * np.pi / (np.asarray(periods_h) * 3600.0)
    power = signal.lombscargle(t_sec, x, angular_freqs, normalize=True)

    return pd.DataFrame({"period_h": periods_h, "power": power})


def dominant_period(pgram: pd.DataFrame, value_col: str = "Q") -> float:
    """Return the period_h with the highest Q (or power) in a periodogram DataFrame."""
    valid = pgram.dropna(subset=[value_col])
    if valid.empty:
        return np.nan
    return float(valid.loc[valid[value_col].idxmax(), "period_h"])


# ── per-animal periodogram wrappers ───────────────────────────────────────────

def periodogram_by_animal(
    df: pd.DataFrame,
    method: str = "chi_sq",
    periods_h: np.ndarray | None = None,
    bin_size_min: float = 30.0,
) -> pd.DataFrame:
    """
    Run chi-square or Lomb-Scargle periodogram for every animal.

    Returns a long-format DataFrame with columns:
        id, labels, order, channel, monitor, period_h, Q (or power), p_value
    Mirrors R's zeitgebr::chi_sq_periodogram() applied per animal.

    Parameters
    ----------
    method : "chi_sq" (default) | "lomb_scargle"
    """
    records = []
    for aid, grp in df.groupby("id", sort=False):
        meta = grp[["id", "labels", "order", "channel", "monitor"]].iloc[0]
        act  = grp["activity"].values
        t    = grp["t"].values

        if method == "lomb_scargle":
            pg = lomb_scargle_periodogram(act, t, periods_h)
        else:
            pg = chi_sq_periodogram(act, t, periods_h, bin_size_min)

        for col in ("id", "labels", "order", "channel", "monitor"):
            pg[col] = meta[col]

        records.append(pg)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def dominant_period_by_animal(
    df: pd.DataFrame,
    method: str = "chi_sq",
    periods_h: np.ndarray | None = None,
    bin_size_min: float = 30.0,
) -> pd.DataFrame:
    """
    Return the dominant circadian period for each animal.

    Output schema: id, labels, order, channel, monitor, x="DominantPeriod", y=period_h
    """
    pgram = periodogram_by_animal(df, method=method, periods_h=periods_h,
                                   bin_size_min=bin_size_min)
    value_col = "power" if method == "lomb_scargle" else "Q"
    result = (
        pgram.dropna(subset=[value_col])
        .groupby(["id", "labels", "order", "channel", "monitor"], sort=False)
        .apply(lambda g: g.loc[g[value_col].idxmax(), "period_h"],
               include_groups=False)
        .reset_index(name="y")
    )
    result["x"] = "DominantPeriod"
    return _ensure_cols(result)


# ── cosinor analysis ──────────────────────────────────────────────────────────

def fit_cosinor(
    activity: np.ndarray,
    t: np.ndarray,
    period_h: float = 24.0,
    bin_size_min: float = 30.0,
) -> dict:
    """
    Fit a single-component cosinor model to one animal's activity.

    Model: y(t) = M + β·cos(2π·t/T) + γ·sin(2π·t/T)
           ↔  y(t) = M + A·cos(2π·t/T + φ)

    where:
      M (mesor)     = rhythm-adjusted mean
      A (amplitude) = half the peak-to-trough range
      φ (acrophase) = time of peak in hours within the period [0, T)

    Mirrors R's ActCR::ActCosinor().

    Returns
    -------
    dict with keys: mesor, amplitude, acrophase_h, r2, period_h
      acrophase_h is in [0, period_h)
    """
    x = _bin_series(activity, t, bin_size_min)
    dt_s = bin_size_min * 60.0
    t_bin = np.arange(len(x)) * dt_s

    omega = 2.0 * np.pi / (period_h * 3600.0)

    # Design matrix [1, cos, sin]
    A_mat = np.column_stack([
        np.ones(len(t_bin)),
        np.cos(omega * t_bin),
        np.sin(omega * t_bin),
    ])

    try:
        coeffs, *_ = np.linalg.lstsq(A_mat, x, rcond=None)
    except np.linalg.LinAlgError:
        return {"mesor": np.nan, "amplitude": np.nan, "acrophase_h": np.nan,
                "r2": np.nan, "period_h": period_h}

    mesor, beta, gamma = coeffs
    amplitude  = float(np.sqrt(beta ** 2 + gamma ** 2))
    acrophase_rad = float(np.arctan2(-gamma, beta))
    acrophase_h   = float((acrophase_rad % (2 * np.pi)) / (2 * np.pi) * period_h)

    # R²
    y_hat  = A_mat @ coeffs
    ss_res = float(np.sum((x - y_hat) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "mesor":       float(mesor),
        "amplitude":   amplitude,
        "acrophase_h": acrophase_h,
        "r2":          r2,
        "period_h":    period_h,
    }


def cosinor_by_animal(
    df: pd.DataFrame,
    period_h: float = 24.0,
    bin_size_min: float = 30.0,
) -> pd.DataFrame:
    """
    Fit cosinor model for every animal.

    Returns a wide DataFrame:
        id, labels, order, channel, monitor,
        mesor, amplitude, acrophase_h, r2, period_h
    Mirrors R's ActCR::ActCosinor() applied per animal.
    """
    records = []
    for aid, grp in df.groupby("id", sort=False):
        meta   = grp[["id", "labels", "order", "channel", "monitor"]].iloc[0].to_dict()
        result = fit_cosinor(grp["activity"].values, grp["t"].values,
                             period_h=period_h, bin_size_min=bin_size_min)
        records.append({**meta, **result})

    return pd.DataFrame(records) if records else pd.DataFrame()


def cosinor_summary(df: pd.DataFrame, period_h: float = 24.0) -> pd.DataFrame:
    """
    Cosinor fit per animal returned in the standard x/y long format
    (one row per metric per animal).

    x ∈ {"mesor", "amplitude", "acrophase_h", "r2"}
    """
    wide = cosinor_by_animal(df, period_h=period_h)
    metrics = ["mesor", "amplitude", "acrophase_h", "r2"]
    frames = []
    for m in metrics:
        tmp = wide[["id", "labels", "order", "channel", "monitor", m]].copy()
        tmp = tmp.rename(columns={m: "y"})
        tmp["x"] = m
        frames.append(tmp)
    return _ensure_cols(pd.concat(frames, ignore_index=True))


# ── non-parametric metrics (IS, IV, RA) ───────────────────────────────────────

def is_iv_ra(
    df: pd.DataFrame,
    bin_size_min: float = 60.0,
) -> pd.DataFrame:
    """
    Compute three non-parametric circadian rhythm metrics per animal.

    IS  (Interdaily Stability)   : degree of coupling to the 24-h cycle [0-1]
    IV  (Intradaily Variability) : fragmentation of the rest-activity rhythm [0-2]
    RA  (Relative Amplitude)     : (M10 - L5) / (M10 + L5)               [0-1]

    Mirrors R's nparACT::nparACT_base().

    Returns standard long-format DataFrame:
        id, labels, order, channel, monitor, x ∈ {"IS","IV","RA"}, y
    """
    records = []
    for aid, grp in df.groupby("id", sort=False):
        meta = grp[["id", "labels", "order", "channel", "monitor"]].iloc[0].to_dict()
        x    = _bin_series(grp["activity"].values, grp["t"].values, bin_size_min)
        bins_per_day = round(24.0 * 60.0 / bin_size_min)

        IS_val = _interdaily_stability(x, bins_per_day)
        IV_val = _intradaily_variability(x)
        RA_val = _relative_amplitude(x, bins_per_day)

        for metric, val in [("IS", IS_val), ("IV", IV_val), ("RA", RA_val)]:
            records.append({**meta, "x": metric, "y": val})

    return _ensure_cols(pd.DataFrame(records)) if records else pd.DataFrame()


def _interdaily_stability(x: np.ndarray, bins_per_day: int) -> float:
    """
    IS = between-bin SS / total SS  (always in [0, 1]).

    Equivalent to the R² of folding the time-series at the 24-h period.
    Formula: IS = k * Σ_h(x̄_h - x̄)² / Σ_i(x_i - x̄)²
    where k = number of complete days, h_mean = per-bin means across days.
    """
    N = len(x)
    total_ss = float(np.sum((x - x.mean()) ** 2))
    if N < bins_per_day or total_ss == 0:
        return np.nan

    n_complete = (N // bins_per_day) * bins_per_day
    k   = n_complete // bins_per_day
    mat = x[:n_complete].reshape(k, bins_per_day)
    h_mean = mat.mean(axis=0)   # mean profile across days

    between_ss = float(k * np.sum((h_mean - x.mean()) ** 2))
    return between_ss / total_ss


def _intradaily_variability(x: np.ndarray) -> float:
    """IV = mean squared successive difference / overall variance."""
    N = len(x)
    if N < 2 or x.var(ddof=1) == 0:
        return np.nan

    numerator   = N * np.sum(np.diff(x) ** 2)
    denominator = (N - 1) * np.sum((x - x.mean()) ** 2)

    return float(numerator / denominator)


def _relative_amplitude(x: np.ndarray, bins_per_day: int) -> float:
    """RA = (M10 - L5) / (M10 + L5) based on average-day profile."""
    n_complete = (len(x) // bins_per_day) * bins_per_day
    if n_complete == 0:
        return np.nan

    mat    = x[:n_complete].reshape(-1, bins_per_day)
    h_mean = mat.mean(axis=0)

    # M10: mean of the 10 most active consecutive hours
    m10_bins = round(10.0 * 60.0 / (1440.0 / bins_per_day))  # approximate
    l5_bins  = round( 5.0 * 60.0 / (1440.0 / bins_per_day))

    m10_bins = max(m10_bins, 1)
    l5_bins  = max(l5_bins, 1)

    m10 = _max_window_mean(h_mean, m10_bins)
    l5  = _min_window_mean(h_mean, l5_bins)

    denom = m10 + l5
    if denom == 0:
        return np.nan
    return float((m10 - l5) / denom)


def _max_window_mean(arr: np.ndarray, w: int) -> float:
    """Mean of the w-length sliding window with the highest sum (circular)."""
    n = len(arr)
    w = min(w, n)
    extended = np.concatenate([arr, arr[:w-1]])
    sums = np.convolve(extended, np.ones(w), mode="valid")[:n]
    return float(sums.max() / w)


def _min_window_mean(arr: np.ndarray, w: int) -> float:
    """Mean of the w-length sliding window with the lowest sum (circular)."""
    n = len(arr)
    w = min(w, n)
    extended = np.concatenate([arr, arr[:w-1]])
    sums = np.convolve(extended, np.ones(w), mode="valid")[:n]
    return float(sums.min() / w)


# ── internal helpers ──────────────────────────────────────────────────────────

def _bin_series(
    activity: np.ndarray,
    t: np.ndarray,
    bin_size_min: float,
) -> np.ndarray:
    """
    Sum activity into fixed-width time bins.
    Returns a 1-D numpy array of binned counts.
    """
    if len(activity) == 0:
        return np.array([], dtype=float)

    bin_s  = bin_size_min * 60.0
    t0     = t[0]
    bin_id = ((t - t0) / bin_s).astype(int)
    n_bins = int(bin_id.max()) + 1

    binned = np.zeros(n_bins, dtype=float)
    np.add.at(binned, bin_id, activity)
    return binned


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)
    cols  = ["id", "labels", "order", "channel", "monitor", "x", "y"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra].reset_index(drop=True)
