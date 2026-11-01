"""Bin-width dependence of sleep durations — FINDING, recorded as failing tests.

These tests are expected to FAIL on the current code. They are not to be
skipped, xfailed or relaxed; they pass when the defect is fixed.

No metric here derives a duration from a row count: annotate_sleep,
detect_sleep_bouts, tst_*, waso_* and the survival death window all work in
seconds from `t`. But `_inactive_bouts` and `_state_bouts` end a run at the
timestamp of its LAST row and measure it as t_last - t_first, so every bout
is one bin shorter than the time it covers. A bin of DAM data represents the
interval up to its timestamp, so n inactive bins span n x bin seconds — which
is how R's sleepr (`run length x dt`), the function this module mirrors,
counts it.

The error is one bin, so its size depends on the bin width — which is what
makes it a cross-rig defect, not a rounding detail:

  * the 5-minute sleep rule is really 6 min at 60 s bins and 5.5 min at 30 s;
  * the same 10-minute bout measures 540 s at 60 s bins and 570 s at 30 s;
  * so total sleep time differs between a 30 s rig and a 60 s rig recording
    identical behaviour.

Frames are built directly (no parser), so these tests are independent of the
DAM column mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.analysis import sleep as S


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


def test_five_minutes_of_immobility_at_30s_bins_is_sleep():
    """Ten quiet 30 s bins are five minutes of immobility — the threshold."""
    df = S.annotate_sleep(_frame(30, [(True, 4), (False, 10), (True, 4)]),
                          threshold_min=5)
    assert int(df["asleep"].sum()) == 10


def test_a_ten_minute_bout_at_30s_bins_lasts_600_seconds():
    assert _bout_seconds(30, 20) == 600.0


def test_the_same_bout_measures_the_same_at_30s_and_60s():
    """Identical behaviour recorded at two bin widths must give one duration."""
    assert _bout_seconds(30, 20) == _bout_seconds(60, 10)
