"""
Pin down build_survival_data()'s death-detection semantics with synthetic
data (no real fixture files needed): an animal is dead iff some silent run
reaches the threshold, and death is dated to the FIRST run that does — not the
longest, and not merely whether the recording ends in silence.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.analysis.survival import (
    _first_lethal_silence,
    build_survival_data,
    flag_ambiguous_deaths,
)

def _animal(activity_by_hour: list[float]) -> pd.DataFrame:
    """One synthetic animal: one row per hour, activity as given."""
    rows = []
    for h, act in enumerate(activity_by_hour):
        rows.append({
            "id": "a1", "labels": "X", "order": 1, "channel": 1, "monitor": "m1.txt",
            "t": h * 3600.0,
            "activity": act,
            "experiment_day": h // 24,
        })
    return pd.DataFrame(rows)


class TestFirstLethalSilence:
    def test_returns_first_qualifying_run_not_longest(self):
        # hourly bins: active, 3 silent (3h), active, 5 silent (5h), active.
        # A run of n bins lasts n x bin width (0.13.0); before, this comment read
        # "span 2h / span 4h" and the second window was 3h.
        t = np.arange(11) * 3600.0
        act = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        # window 3h: first run (3h) qualifies -> its start at t=1h
        assert _first_lethal_silence(t, act, 3 * 3600.0) == 1 * 3600.0
        # window 4h: first run (3h) too short, second run (5h) qualifies -> t=5h
        assert _first_lethal_silence(t, act, 4 * 3600.0) == 5 * 3600.0

    def test_none_when_always_active(self):
        t = np.arange(3) * 3600.0
        act = np.array([1, 2, 3])
        assert _first_lethal_silence(t, act, 3600.0) is None

    def test_none_when_no_run_reaches_threshold(self):
        t = np.arange(5) * 3600.0
        act = np.array([1, 0, 0, 1, 0])  # longest silent run is 2h
        assert _first_lethal_silence(t, act, 12 * 3600.0) is None

    def test_trailing_silence_qualifies(self):
        t = np.arange(5) * 3600.0
        act = np.array([1, 1, 0, 0, 0])  # silent from t=2h, 3h long
        assert _first_lethal_silence(t, act, 1 * 3600.0) == 2 * 3600.0


class TestBuildSurvivalDataSemantics:
    def test_death_dated_to_first_qualifying_silence_not_longest(self):
        """The defining case for the user's rule. Threshold 12h. The animal has
        a 12h silence early (hours 1-13), resumes, then has a longer 40h silence
        later (starting hour 54, day 2). Death must be dated to the FIRST run
        that crosses the threshold (day 0), not the longest later run (day 2)."""
        hours = [1] + [0] * 13 + [1] * 40 + [0] * 40
        df = _animal(hours)
        surv = build_survival_data(df, dead_window_h=12.0)

        assert surv.loc[0, "event"] == 1
        assert surv.loc[0, "duration"] == 0  # first 12h silence began on day 0

    def test_dead_called_at_mid_recording_silence_even_if_active_at_end(self):
        """A fly goes silent past the threshold mid-recording, then resumes and
        is active at the end. It is still dead, dated to that mid-recording
        silence — not censored just because it moved later."""
        hours = [1] * 24 + [0] * 30 + [1] * 24  # active 24h, silent 30h, active 24h
        df = _animal(hours)
        surv = build_survival_data(df, dead_window_h=24.0)

        assert surv.loc[0, "event"] == 1
        # the 30h silent run starts at hour 24 -> day 1
        assert surv.loc[0, "duration"] == 1

    def test_censored_when_only_brief_mid_silence_and_active_at_end(self):
        """A silence shorter than the threshold, anywhere, with activity
        resuming and continuing through the end of the recording -> alive
        (censored), not dead."""
        hours = [1] * 24 + [0] * 5 + [1] * 24  # only 5h silent gap
        df = _animal(hours)
        surv = build_survival_data(df, dead_window_h=24.0)

        assert surv.loc[0, "event"] == 0
        assert surv.loc[0, "duration"] == df["experiment_day"].max()

    def test_never_active_but_recording_shorter_than_threshold_is_censored(self):
        """An animal that never moves is only called dead once the silence
        itself has lasted the full threshold -- a too-short recording can't
        meet that bar yet, so it's censored rather than auto-declared dead."""
        hours = [0] * 5  # 5h total, all silent, threshold is 24h
        df = _animal(hours)
        surv = build_survival_data(df, dead_window_h=24.0)

        assert surv.loc[0, "event"] == 0
        assert surv.loc[0, "duration"] == df["experiment_day"].max()

    def test_never_active_and_recording_longer_than_threshold_is_dead_at_day_zero(self):
        hours = [0] * 30  # 30h, all silent, threshold is 24h
        df = _animal(hours)
        surv = build_survival_data(df, dead_window_h=24.0)

        assert surv.loc[0, "event"] == 1
        assert surv.loc[0, "duration"] == 0

    def test_always_active_is_censored_at_max_day(self):
        hours = [1] * 30
        df = _animal(hours)
        surv = build_survival_data(df, dead_window_h=24.0)

        assert surv.loc[0, "event"] == 0
        assert surv.loc[0, "duration"] == df["experiment_day"].max()


class TestFlagAmbiguousDeaths:
    def test_flags_animal_active_after_death(self):
        # dead at first 24h+ silence (starts hour 24 -> day 1), then moves again
        hours = [1] * 24 + [0] * 30 + [1] * 24
        df = _animal(hours)
        flagged = flag_ambiguous_deaths(df, dead_window_h=24.0)

        assert len(flagged) == 1
        assert flagged.loc[0, "death_day"] == 1
        assert flagged.loc[0, "post_death_counts"] == 24       # 24 active hrs after
        assert flagged.loc[0, "post_death_active_bins"] == 24
        assert flagged.loc[0, "last_active_day"] == 3          # hour 77 -> day 3

    def test_no_flag_for_clean_death(self):
        # silent straight through to the end of the record -> not ambiguous
        hours = [1] * 24 + [0] * 30
        df = _animal(hours)
        flagged = flag_ambiguous_deaths(df, dead_window_h=24.0)
        assert flagged.empty

    def test_no_flag_for_alive_animal(self):
        hours = [1] * 48
        df = _animal(hours)
        flagged = flag_ambiguous_deaths(df, dead_window_h=24.0)
        assert flagged.empty

    def test_empty_result_still_has_columns(self):
        df = _animal([1] * 48)
        flagged = flag_ambiguous_deaths(df, dead_window_h=24.0)
        for col in ["id", "labels", "death_day", "post_death_counts",
                    "post_death_active_bins", "last_active_day"]:
            assert col in flagged.columns
