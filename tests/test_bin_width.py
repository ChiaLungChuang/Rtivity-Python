"""Bout durations are bin count × bin width, at any bin width (0.13.0).

Before 0.13.0, `_inactive_bouts`, `_state_bouts` and activity's `_raw_bouts`
ended a run at the timestamp of its LAST row and measured it as
t_last - t_first, so every bout was one bin shorter than the time it covers.
A DAM read summarises the bin up to its timestamp, so n bins span
n × bin seconds. The error was one bin, so its size was the bin width:

  * the 5-minute sleep rule was really 6 min at 60 s bins and 5.5 min at 30 s;
  * the same 10-minute bout measured 540 s at 60 s bins and 570 s at 30 s;
  * activity bouts were one bin short, and the gap bridged between them one
    bin narrower than `threshold_min` said.

The first three tests were committed failing (Rtivity-Python PR #2) and pass
with the fix; the rest pin the new definition at 60 s, for activity bouts, and
for the bin-width helper. Frames are built directly — no parser and no
external data — so every test here runs in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.analysis import activity as A
from modules.analysis import sleep as S
from modules.analysis.bins import bin_seconds, runs


def _frame(bin_s: int, pattern: list[tuple[bool, int]]) -> pd.DataFrame:
    """One animal; `pattern` is [(active, n_bins), ...]."""
    act: list[int] = []
    for active, n in pattern:
        act += [1 if active else 0] * n
    return pd.DataFrame({
        "id": "a",
        "activity": act,
        "t": [float(i * bin_s) for i in range(len(act))],
        "experiment_day": 0,
        "day_night": True,          # all light: the latency loop is not exercised
    })


def _bout_seconds(bin_s: int, quiet_bins: int) -> float:
    df = S.annotate_sleep(_frame(bin_s, [(True, 4), (False, quiet_bins), (True, 4)]),
                          threshold_min=5)
    return float(S.detect_sleep_bouts(df)["sleep_bout_duration"].sum())


def _asleep_rows(bin_s: int, quiet_bins: int) -> int:
    df = S.annotate_sleep(_frame(bin_s, [(True, 4), (False, quiet_bins), (True, 4)]),
                          threshold_min=5)
    return int(df["asleep"].sum())


# ── committed failing in PR #2 ────────────────────────────────────────────────

def test_five_minutes_of_immobility_at_30s_bins_is_sleep():
    """Ten quiet 30 s bins are five minutes of immobility — the threshold."""
    assert _asleep_rows(30, 10) == 10


def test_a_ten_minute_bout_at_30s_bins_lasts_600_seconds():
    assert _bout_seconds(30, 20) == 600.0


def test_the_same_bout_measures_the_same_at_30s_and_60s():
    """Identical behaviour recorded at two bin widths must give one duration."""
    assert _bout_seconds(30, 20) == _bout_seconds(60, 10)


# ── the same definition at 60 s: these changed too ────────────────────────────

def test_five_minutes_of_immobility_at_60s_bins_is_sleep():
    """Before 0.13.0 five quiet 60 s bins measured 240 s and were not sleep."""
    assert _asleep_rows(60, 5) == 5


def test_four_minutes_of_immobility_is_not_sleep_at_either_bin_width():
    assert _asleep_rows(60, 4) == 0
    assert _asleep_rows(30, 9) == 0


def test_a_ten_minute_bout_at_60s_bins_lasts_600_seconds():
    """540 s before 0.13.0."""
    assert _bout_seconds(60, 10) == 600.0


def test_the_row_after_a_sleep_bout_is_not_marked_asleep():
    """The run's end is exclusive: the first moving row stays awake."""
    df = S.annotate_sleep(_frame(60, [(True, 2), (False, 6), (True, 2)]), threshold_min=5)
    assert df["asleep"].tolist() == [False] * 2 + [True] * 6 + [False] * 2


# ── activity bouts ────────────────────────────────────────────────────────────

def _activity_bouts(bin_s: int, pattern, threshold_min: float = 5.0) -> pd.DataFrame:
    out = A.detect_activity_bouts(_frame(bin_s, pattern), threshold_min=threshold_min)
    return out[out["bout_start"]]


def test_an_activity_bout_lasts_its_bin_count_times_the_bin_width():
    """Five minutes of movement is 300 s at 30 s and at 60 s bins."""
    a30 = _activity_bouts(30, [(False, 20), (True, 10), (False, 20)])
    a60 = _activity_bouts(60, [(False, 10), (True, 5), (False, 10)])
    assert a30["bout_duration_s"].tolist() == [300.0]
    assert a60["bout_duration_s"].tolist() == [300.0]


def test_activity_bout_counts_cover_every_active_row():
    bouts = _activity_bouts(60, [(False, 3), (True, 4), (False, 8)])
    assert bouts["bout_activity"].tolist() == [4]


def test_a_quiet_gap_of_exactly_the_threshold_is_bridged_and_one_bin_more_is_not():
    """threshold_min=5 bridges five quiet minutes, not six. Before 0.13.0 the
    gap was measured one bin wide, so only four quiet 60 s bins were bridged."""
    bridged = _activity_bouts(60, [(True, 3), (False, 5), (True, 3)])
    split = _activity_bouts(60, [(True, 3), (False, 6), (True, 3)])
    assert bridged["bout_duration_s"].tolist() == [660.0]
    assert split["bout_duration_s"].tolist() == [180.0, 180.0]


# ── the bin-width helper ──────────────────────────────────────────────────────

def test_bin_width_is_read_from_the_timestamps():
    assert bin_seconds(np.arange(10) * 30.0) == 30.0
    assert bin_seconds(np.arange(10) * 60.0) == 60.0


def test_one_dropped_read_does_not_change_the_bin_width():
    t = np.delete(np.arange(20) * 60.0, 7)
    assert bin_seconds(t) == 60.0


def test_a_single_read_has_no_bin_width_rather_than_an_assumed_one():
    assert bin_seconds(np.array([0.0])) == 0.0
    assert runs(np.array([True]), np.array([0.0])) == [(0.0, 0.0)]


def test_runs_end_one_bin_after_their_last_read():
    t = np.arange(6) * 30.0
    assert runs(np.array([False, True, True, False, True, True]), t) == [
        (30.0, 90.0), (120.0, 180.0)]
