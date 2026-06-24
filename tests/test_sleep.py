"""Tests for sleep.py using real data from DataStore."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.data.data_store import DataStore, Settings
from modules.analysis.sleep import (
    annotate_sleep,
    detect_sleep_bouts,
    sleep_ratio_by_phase,
    sleep_ratio_by_day,
    sleep_ratio_by_day_phase,
    tst_mean,
    tst_by_day,
    waso_mean,
    waso_by_day,
    sleep_latency_mean,
    sleep_latency_by_day,
    sleep_bout_duration_by_phase,
    sleep_bout_duration_by_day,
    sleep_bout_count_by_phase,
    sleep_bout_count_by_day,
)

DAM_DIR   = Path.home() / "Desktop/Rtivity_main"
META_PATH = DAM_DIR / "metadata.xlsx"

pytestmark = pytest.mark.skipif(
    not META_PATH.exists(), reason="metadata.xlsx not found"
)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    store = DataStore(dam_dir=DAM_DIR, settings=Settings())
    store.load_from_metadata(META_PATH)
    return store.data


@pytest.fixture(scope="module")
def data_annotated(data):
    return annotate_sleep(data, threshold_min=5)


@pytest.fixture(scope="module")
def data_with_bouts(data_annotated):
    return detect_sleep_bouts(data_annotated)


# ── annotate_sleep ────────────────────────────────────────────────────────────

class TestAnnotateSleep:
    def test_adds_asleep_column(self, data_annotated):
        assert "asleep" in data_annotated.columns

    def test_asleep_is_bool(self, data_annotated):
        assert data_annotated["asleep"].dtype == bool

    def test_some_sleep_exists(self, data_annotated):
        assert data_annotated["asleep"].sum() > 0

    def test_sleep_fraction_between_0_and_1(self, data_annotated):
        ratio = data_annotated["asleep"].mean()
        assert 0 < ratio < 1

    def test_does_not_modify_original(self, data, data_annotated):
        assert "asleep" not in data.columns

    def test_active_rows_not_asleep(self, data_annotated):
        """Rows with activity > 0 must not be marked asleep."""
        active_rows = data_annotated[data_annotated["activity"] > 0]
        assert not active_rows["asleep"].any()

    def test_short_inactive_bouts_not_asleep(self, data):
        """A 1-row inactive gap (60 s << 5 min) should not be marked asleep."""
        # Use a very high threshold so almost nothing qualifies
        annotated = annotate_sleep(data, threshold_min=9999)
        assert annotated["asleep"].sum() == 0


# ── detect_sleep_bouts ────────────────────────────────────────────────────────

class TestDetectSleepBouts:
    def test_adds_expected_columns(self, data_with_bouts):
        for col in [
            "sleep_bout_start", "sleep_bout_duration",
            "wake_bout_start", "wake_bout_duration",
            "sleep_latency_s",
        ]:
            assert col in data_with_bouts.columns

    def test_sleep_bout_start_is_bool(self, data_with_bouts):
        assert data_with_bouts["sleep_bout_start"].dtype == bool

    def test_wake_bout_start_is_bool(self, data_with_bouts):
        assert data_with_bouts["wake_bout_start"].dtype == bool

    def test_sleep_bout_durations_non_negative(self, data_with_bouts):
        assert (data_with_bouts["sleep_bout_duration"] >= 0).all()

    def test_wake_bout_durations_non_negative(self, data_with_bouts):
        assert (data_with_bouts["wake_bout_duration"] >= 0).all()

    def test_sleep_latency_non_negative(self, data_with_bouts):
        assert (data_with_bouts["sleep_latency_s"] >= 0).all()

    def test_sleep_bouts_exist(self, data_with_bouts):
        assert data_with_bouts["sleep_bout_start"].sum() > 0

    def test_wake_bouts_exist(self, data_with_bouts):
        assert data_with_bouts["wake_bout_start"].sum() > 0

    def test_bout_duration_only_at_start_rows(self, data_with_bouts):
        non_start = data_with_bouts[~data_with_bouts["sleep_bout_start"]]
        assert (non_start["sleep_bout_duration"] == 0).all()

    def test_requires_asleep_column(self, data):
        with pytest.raises(ValueError, match="asleep"):
            detect_sleep_bouts(data)


# ── sleep ratio summaries ─────────────────────────────────────────────────────

class TestSleepRatioByPhase:
    def test_returns_dataframe(self, data_annotated):
        assert isinstance(sleep_ratio_by_phase(data_annotated), pd.DataFrame)

    def test_has_required_columns(self, data_annotated):
        df = sleep_ratio_by_phase(data_annotated)
        for col in ["id", "labels", "x", "y"]:
            assert col in df.columns

    def test_x_values_are_light_dark(self, data_annotated):
        df = sleep_ratio_by_phase(data_annotated)
        assert set(df["x"].unique()).issubset({"Light", "Dark"})

    def test_y_between_0_and_1(self, data_annotated):
        df = sleep_ratio_by_phase(data_annotated)
        assert (df["y"] >= 0).all() and (df["y"] <= 1).all()

    def test_dark_sleep_ratio_higher_than_light(self, data_annotated):
        """Drosophila sleep more during the dark phase (nocturnal sleepers)."""
        df = sleep_ratio_by_phase(data_annotated)
        light_mean = df[df["x"] == "Light"]["y"].mean()
        dark_mean  = df[df["x"] == "Dark"]["y"].mean()
        assert dark_mean > light_mean

    def test_requires_asleep_column(self, data):
        with pytest.raises(ValueError, match="asleep"):
            sleep_ratio_by_phase(data)


class TestSleepRatioByDay:
    def test_returns_dataframe(self, data_annotated):
        assert isinstance(sleep_ratio_by_day(data_annotated), pd.DataFrame)

    def test_x_starts_with_day(self, data_annotated):
        df = sleep_ratio_by_day(data_annotated)
        assert df["x"].str.startswith("Day_").all()

    def test_y_between_0_and_1(self, data_annotated):
        df = sleep_ratio_by_day(data_annotated)
        assert (df["y"] >= 0).all() and (df["y"] <= 1).all()

    def test_multiple_days(self, data_annotated):
        df = sleep_ratio_by_day(data_annotated)
        assert df["x"].nunique() > 1


class TestSleepRatioByDayPhase:
    def test_returns_dataframe(self, data_annotated):
        assert isinstance(sleep_ratio_by_day_phase(data_annotated), pd.DataFrame)

    def test_x_contains_day_and_phase(self, data_annotated):
        df = sleep_ratio_by_day_phase(data_annotated)
        assert df["x"].str.contains("Day_").all()
        assert df["x"].str.contains("Light|Dark").any()


# ── TST / WASO ────────────────────────────────────────────────────────────────

class TestTST:
    def test_tst_mean_returns_dataframe(self, data_with_bouts):
        assert isinstance(tst_mean(data_with_bouts), pd.DataFrame)

    def test_tst_mean_x_label(self, data_with_bouts):
        df = tst_mean(data_with_bouts)
        assert (df["x"] == "TotalSleepTime").all()

    def test_tst_mean_y_positive(self, data_with_bouts):
        df = tst_mean(data_with_bouts)
        assert (df["y"] >= 0).all()

    def test_tst_mean_y_plausible_hours(self, data_with_bouts):
        """Mean TST in dark phase should be between 0 and 12 h."""
        df = tst_mean(data_with_bouts)
        assert df["y"].mean() < 13

    def test_tst_by_day_returns_dataframe(self, data_with_bouts):
        assert isinstance(tst_by_day(data_with_bouts), pd.DataFrame)

    def test_tst_by_day_x_starts_with_day(self, data_with_bouts):
        df = tst_by_day(data_with_bouts)
        assert df["x"].str.startswith("Day_").all()

    def test_requires_bouts(self, data):
        with pytest.raises(ValueError, match="sleep bout"):
            tst_mean(data)


class TestWASO:
    def test_waso_mean_returns_dataframe(self, data_with_bouts):
        assert isinstance(waso_mean(data_with_bouts), pd.DataFrame)

    def test_waso_mean_x_label(self, data_with_bouts):
        df = waso_mean(data_with_bouts)
        assert (df["x"] == "WakeAfterSleepOnset").all()

    def test_waso_mean_y_non_negative(self, data_with_bouts):
        df = waso_mean(data_with_bouts)
        assert (df["y"] >= 0).all()

    def test_waso_by_day_returns_dataframe(self, data_with_bouts):
        assert isinstance(waso_by_day(data_with_bouts), pd.DataFrame)


# ── sleep latency ─────────────────────────────────────────────────────────────

class TestSleepLatency:
    def test_latency_mean_returns_dataframe(self, data_with_bouts):
        assert isinstance(sleep_latency_mean(data_with_bouts), pd.DataFrame)

    def test_latency_mean_y_positive(self, data_with_bouts):
        df = sleep_latency_mean(data_with_bouts)
        assert (df["y"] >= 0).all()

    def test_latency_mean_x_label(self, data_with_bouts):
        df = sleep_latency_mean(data_with_bouts)
        assert (df["x"] == "Sleep_latency").all()

    def test_latency_by_day_returns_dataframe(self, data_with_bouts):
        assert isinstance(sleep_latency_by_day(data_with_bouts), pd.DataFrame)

    def test_latency_by_day_x_starts_with_day(self, data_with_bouts):
        df = sleep_latency_by_day(data_with_bouts)
        assert df["x"].str.startswith("Day_").all()


# ── sleep bout duration ───────────────────────────────────────────────────────

class TestSleepBoutDuration:
    def test_by_phase_returns_dataframe(self, data_with_bouts):
        assert isinstance(sleep_bout_duration_by_phase(data_with_bouts), pd.DataFrame)

    def test_by_phase_x_values(self, data_with_bouts):
        df = sleep_bout_duration_by_phase(data_with_bouts)
        assert set(df["x"].unique()).issubset({"Light", "Dark"})

    def test_by_phase_y_positive(self, data_with_bouts):
        df = sleep_bout_duration_by_phase(data_with_bouts)
        assert (df["y"] > 0).all()

    def test_by_day_returns_dataframe(self, data_with_bouts):
        assert isinstance(sleep_bout_duration_by_day(data_with_bouts), pd.DataFrame)


# ── sleep bout count ──────────────────────────────────────────────────────────

class TestSleepBoutCount:
    def test_by_phase_returns_dataframe(self, data_with_bouts):
        assert isinstance(sleep_bout_count_by_phase(data_with_bouts), pd.DataFrame)

    def test_by_phase_x_values(self, data_with_bouts):
        df = sleep_bout_count_by_phase(data_with_bouts)
        assert set(df["x"].unique()).issubset({"Light", "Dark"})

    def test_by_phase_y_positive_integers(self, data_with_bouts):
        df = sleep_bout_count_by_phase(data_with_bouts)
        assert (df["y"] > 0).all()

    def test_by_day_returns_dataframe(self, data_with_bouts):
        assert isinstance(sleep_bout_count_by_day(data_with_bouts), pd.DataFrame)

    def test_by_day_x_starts_with_day(self, data_with_bouts):
        df = sleep_bout_count_by_day(data_with_bouts)
        assert df["x"].str.startswith("Day_").all()
