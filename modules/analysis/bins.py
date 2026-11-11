"""
Bin width and bout spans — shared by sleep and activity bout detection.

A DAM read summarises the bin that ends at its timestamp, so a run of n
consecutive bins covers n × bin seconds. Measuring a run as
t_last − t_first (as Rtivity-Python did before 0.13.0) counts the gaps
between reads, not the reads, and makes every bout one bin short — an error
whose size is the bin width, so the same behaviour measured differently on a
30 s rig and a 60 s rig.

The bin width is derived from the timestamps, never assumed.
"""

from __future__ import annotations

import numpy as np


def bin_seconds(t: np.ndarray) -> float:
    """Bin width in seconds: the median gap between consecutive timestamps.

    The median, not the mode or the first gap, so one dropped read does not
    change it. 0.0 when there are fewer than two reads, in which case a run's
    span cannot be known and is reported as zero rather than guessed.
    """
    if len(t) < 2:
        return 0.0
    gaps = np.diff(np.asarray(t, dtype=float))
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if len(gaps) else 0.0


def runs(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) for each maximal run where `mask` is True.

    `end` is EXCLUSIVE: the timestamp of the run's last bin plus one bin
    width, so `end - start` is n_bins × bin seconds and a row at time `x`
    belongs to the run when `start <= x < end`.
    """
    width = bin_seconds(t)
    out: list[tuple[float, float]] = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = t[i]
        elif not val and start is not None:
            out.append((float(start), float(t[i - 1] + width)))
            start = None
    if start is not None:
        out.append((float(start), float(t[-1] + width)))
    return out
