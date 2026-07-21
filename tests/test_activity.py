"""Tests for activity.py using real data from DataStore."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.analysis.activity import (
    detect_activity_bouts,
    activity_by_phase,
    activity_by_day,
    activity_by_day_phase,
    activity_by_custom_bins,
    bout_activity_by_phase,
    bout_activity_by_day,
    bout_activity_by_day_phase,
    bout_duration_by_phase,
    bout_duration_by_day,
    bout_duration_by_day_phase,
    summary_stats,
)

DAM_DIR   = Path.home() / "Desktop/Rtivity_main"
META_PATH = DAM_DIR / "metadata.xlsx"

pytestmark = pytest.mark.skipif(
    not META_PATH.exists(), reason="metadata.xlsx not found"
)


# `data` is the session-scoped fixture in conftest.py (loaded once for the run).


@pytest.fixture(scope="module")
def data_with_bouts(data):
    return detect_activity_bouts(data, threshold_min=5)


# ── bout detection ────────────────────────────────────────────────────────────

class TestDetectBouts:
    def test_adds_expected_columns(self, data_with_bouts):
        for col in ["bout_start", "bout_duration_s", "bout_activity"]:
            assert col in data_with_bouts.columns

    def test_bout_start_is_bool(self, data_with_bouts):
        assert data_with_bouts["bout_start"].dtype == bool

    def test_bout_duration_non_negative(self, data_with_bouts):
        assert (data_with_bouts["bout_duration_s"] >= 0).all()

    def test_bout_activity_non_negative(self, data_with_bouts):
        assert (data_with_bouts["bout_activity"] >= 0).all()

    def test_bouts_exist(self, data_with_bouts):
        assert data_with_bouts["bout_start"].sum() > 0

    def test_bout_duration_only_at_start_rows(self, data_with_bouts):
        non_start = data_with_bouts[~data_with_bouts["bout_start"]]
        assert (non_start["bout_duration_s"] == 0).all()

    def test_bout_activity_only_at_start_rows(self, data_with_bouts):
        non_start = data_with_bouts[~data_with_bouts["bout_start"]]
        assert (non_start["bout_activity"] == 0).all()

    def test_does_not_modify_original(self, data, data_with_bouts):
        assert "bout_start" not in data.columns


# ── activity summaries ────────────────────────────────────────────────────────

class TestActivityByPhase:
    def test_returns_dataframe(self, data):
        assert isinstance(activity_by_phase(data), pd.DataFrame)

    def test_has_required_columns(self, data):
        df = activity_by_phase(data)
        for col in ["id", "labels", "x", "y"]:
            assert col in df.columns

    def test_x_values_are_light_dark(self, data):
        df = activity_by_phase(data)
        assert set(df["x"].unique()).issubset({"Light", "Dark"})

    def test_y_non_negative(self, data):
        assert (activity_by_phase(data)["y"] >= 0).all()

    def test_one_row_per_animal_per_phase(self, data):
        df = activity_by_phase(data)
        assert df.groupby(["id", "x"], observed=False).size().max() == 1


class TestActivityByDay:
    def test_returns_dataframe(self, data):
        assert isinstance(activity_by_day(data), pd.DataFrame)

    def test_x_values_start_with_day(self, data):
        df = activity_by_day(data)
        assert df["x"].str.startswith("Day_").all()

    def test_y_non_negative(self, data):
        assert (activity_by_day(data)["y"] >= 0).all()

    def test_multiple_days(self, data):
        df = activity_by_day(data)
        assert df["x"].nunique() > 1


class TestActivityByDayPhase:
    def test_returns_dataframe(self, data):
        assert isinstance(activity_by_day_phase(data), pd.DataFrame)

    def test_x_contains_day_and_phase(self, data):
        df = activity_by_day_phase(data)
        assert df["x"].str.contains("Day_").all()
        assert df["x"].str.contains("Light|Dark").any()

    def test_y_non_negative(self, data):
        assert (activity_by_day_phase(data)["y"] >= 0).all()


class TestActivityByCustomBins:
    def test_returns_dataframe(self, data):
        df = activity_by_custom_bins(data, bin_size_cycles=1.0)
        assert isinstance(df, pd.DataFrame)

    def test_x_starts_with_group(self, data):
        df = activity_by_custom_bins(data)
        assert df["x"].str.startswith("Group_").all()


# ── bout summaries ────────────────────────────────────────────────────────────

class TestBoutActivityByPhase:
    def test_returns_dataframe(self, data_with_bouts):
        assert isinstance(bout_activity_by_phase(data_with_bouts), pd.DataFrame)

    def test_x_values(self, data_with_bouts):
        df = bout_activity_by_phase(data_with_bouts)
        assert set(df["x"].unique()).issubset({"Light", "Dark"})

    def test_y_positive(self, data_with_bouts):
        df = bout_activity_by_phase(data_with_bouts)
        assert (df["y"] > 0).all()

    def test_requires_bouts_column(self, data):
        with pytest.raises(ValueError, match="bout columns"):
            bout_activity_by_phase(data)


class TestBoutDurationByPhase:
    def test_returns_dataframe(self, data_with_bouts):
        assert isinstance(bout_duration_by_phase(data_with_bouts), pd.DataFrame)

    def test_y_positive(self, data_with_bouts):
        df = bout_duration_by_phase(data_with_bouts)
        assert (df["y"] > 0).all()


class TestBoutActivityByDay:
    def test_returns_dataframe(self, data_with_bouts):
        assert isinstance(bout_activity_by_day(data_with_bouts), pd.DataFrame)

    def test_x_starts_with_day(self, data_with_bouts):
        df = bout_activity_by_day(data_with_bouts)
        assert df["x"].str.startswith("Day_").all()


class TestBoutDurationByDay:
    def test_returns_dataframe(self, data_with_bouts):
        assert isinstance(bout_duration_by_day(data_with_bouts), pd.DataFrame)


class TestBoutActivityByDayPhase:
    def test_returns_dataframe(self, data_with_bouts):
        assert isinstance(bout_activity_by_day_phase(data_with_bouts), pd.DataFrame)


class TestBoutDurationByDayPhase:
    def test_returns_dataframe(self, data_with_bouts):
        assert isinstance(bout_duration_by_day_phase(data_with_bouts), pd.DataFrame)


# ── statistics report ─────────────────────────────────────────────────────────

class TestSummaryStats:
    def test_returns_dataframe(self, data):
        summary = activity_by_phase(data)
        stats = summary_stats(summary)
        assert isinstance(stats, pd.DataFrame)

    def test_has_required_columns(self, data):
        stats = summary_stats(activity_by_phase(data))
        for col in ["labels", "x", "n", "mean", "sem", "sd", "median", "q1", "q3"]:
            assert col in stats.columns

    def test_n_matches_animal_count_per_group(self, data):
        summary = activity_by_phase(data)
        stats = summary_stats(summary)
        # Each (label, phase) group should have N = number of animals in that condition
        for _, row in stats.iterrows():
            expected_n = summary[
                (summary["labels"] == row["labels"]) & (summary["x"] == row["x"])
            ]["id"].nunique()
            assert row["n"] == expected_n

    def test_mean_is_finite(self, data):
        stats = summary_stats(activity_by_phase(data))
        assert stats["mean"].apply(np.isfinite).all()

    def test_sem_non_negative(self, data):
        stats = summary_stats(activity_by_phase(data))
        assert (stats["sem"] >= 0).all()
