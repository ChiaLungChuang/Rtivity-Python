"""Tests for dam_parser using the real Monitor files from Rtivity_main."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.data.dam_parser import (
    N_CHANNELS,
    _CHANNEL_COLS,
    get_time_range,
    read_dam_file,
    to_long_format,
)

MONITOR_1 = Path.home() / "Desktop/Rtivity_main/Monitor 1.txt"
MONITOR_2 = Path.home() / "Desktop/Rtivity_main/Monitor 2.txt"

# Skip all tests if sample files are missing
pytestmark = pytest.mark.skipif(
    not MONITOR_1.exists(), reason="Monitor 1.txt not found"
)


class TestReadDamFile:
    def test_returns_dataframe(self):
        df = read_dam_file(MONITOR_1)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = read_dam_file(MONITOR_1)
        for col in ["datetime", "status", "zt", "phase"] + _CHANNEL_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_datetime_is_timestamp(self):
        df = read_dam_file(MONITOR_1)
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])

    def test_datetime_is_monotonic(self):
        df = read_dam_file(MONITOR_1)
        assert df["datetime"].is_monotonic_increasing

    def test_channel_values_are_non_negative_integers(self):
        df = read_dam_file(MONITOR_1)
        for col in _CHANNEL_COLS:
            assert (df[col] >= 0).all(), f"{col} has negative values"

    def test_zt_range(self):
        df = read_dam_file(MONITOR_1)
        assert df["zt"].between(0, 24, inclusive="left").all()

    def test_phase_values(self):
        df = read_dam_file(MONITOR_1)
        assert set(df["phase"].unique()).issubset({"Light", "Dark"})

    def test_time_window_filter(self):
        df_full = read_dam_file(MONITOR_1)
        mid = df_full["datetime"].iloc[len(df_full) // 2]
        df_filtered = read_dam_file(MONITOR_1, start=mid)
        assert df_filtered["datetime"].min() >= mid

    def test_stop_filter(self):
        df_full = read_dam_file(MONITOR_1)
        mid = df_full["datetime"].iloc[len(df_full) // 2]
        df_filtered = read_dam_file(MONITOR_1, stop=mid)
        assert df_filtered["datetime"].max() <= mid

    def test_custom_zt0(self):
        df = read_dam_file(MONITOR_1, zt0="12:00:00")
        assert df["zt"].between(0, 24, inclusive="left").all()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_dam_file("/nonexistent/path/monitor.txt")

    def test_monitor_2(self):
        if not MONITOR_2.exists():
            pytest.skip("Monitor 2.txt not found")
        df = read_dam_file(MONITOR_2)
        assert not df.empty


class TestToLongFormat:
    def test_basic_shape(self):
        df = read_dam_file(MONITOR_1)
        long = to_long_format(df)
        assert len(long) == len(df) * N_CHANNELS

    def test_channel_subset(self):
        df = read_dam_file(MONITOR_1)
        long = to_long_format(df, channels=[1, 2, 3])
        assert set(long["channel"].unique()) == {1, 2, 3}
        assert len(long) == len(df) * 3

    def test_monitor_name(self):
        df = read_dam_file(MONITOR_1)
        long = to_long_format(df, monitor_name="M1")
        assert (long["monitor"] == "M1").all()

    def test_activity_non_negative(self):
        df = read_dam_file(MONITOR_1)
        long = to_long_format(df)
        assert (long["activity"] >= 0).all()

    def test_has_required_columns(self):
        df = read_dam_file(MONITOR_1)
        long = to_long_format(df)
        for col in ["datetime", "zt", "phase", "channel", "activity", "monitor"]:
            assert col in long.columns


class TestGetTimeRange:
    def test_returns_two_timestamps(self):
        start, stop = get_time_range(MONITOR_1)
        assert isinstance(start, pd.Timestamp)
        assert isinstance(stop, pd.Timestamp)

    def test_start_before_stop(self):
        start, stop = get_time_range(MONITOR_1)
        assert start < stop
