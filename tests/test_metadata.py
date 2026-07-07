"""Tests for metadata.py using the real metadata.xlsx from Rtivity_main."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.data.metadata import (
    build_manual_conditions,
    channels_for_file,
    label_for_channel,
    list_files,
    list_labels,
    parse_channel_spec,
    read_metadata,
    time_window_for_file,
)

META_PATH = Path.home() / "Desktop/Rtivity_main/metadata.xlsx"

pytestmark = pytest.mark.skipif(
    not META_PATH.exists(), reason="metadata.xlsx not found"
)


class TestReadMetadata:
    def test_returns_dataframe(self):
        df = read_metadata(META_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        df = read_metadata(META_PATH)
        for col in ["file", "start_datetime", "stop_datetime", "region_id", "labels", "order"]:
            assert col in df.columns

    def test_datetimes_are_timestamps(self):
        df = read_metadata(META_PATH)
        assert pd.api.types.is_datetime64_any_dtype(df["start_datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df["stop_datetime"])

    def test_start_before_stop(self):
        df = read_metadata(META_PATH)
        assert (df["start_datetime"] < df["stop_datetime"]).all()

    def test_region_id_is_int(self):
        df = read_metadata(META_PATH)
        assert df["region_id"].dtype in [int, "int64", "int32"]

    def test_region_id_in_valid_range(self):
        df = read_metadata(META_PATH)
        assert df["region_id"].between(1, 32).all()

    def test_no_empty_labels(self):
        df = read_metadata(META_PATH)
        assert (df["labels"].str.len() > 0).all()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_metadata("/nonexistent/metadata.xlsx")

    def test_row_count(self):
        df = read_metadata(META_PATH)
        assert len(df) == 48  # 49 rows minus header


class TestHelpers:
    def setup_method(self):
        self.meta = read_metadata(META_PATH)

    def test_list_files(self):
        files = list_files(self.meta)
        assert "Monitor 1.txt" in files
        assert "Monitor 2.txt" in files

    def test_list_files_order(self):
        files = list_files(self.meta)
        # Monitor 1 should come before Monitor 2
        assert files.index("Monitor 1.txt") < files.index("Monitor 2.txt")

    def test_list_labels(self):
        labels = list_labels(self.meta)
        assert len(labels) >= 2
        assert "Condition A" in labels

    def test_list_labels_sorted_by_order(self):
        labels = list_labels(self.meta)
        orders = [
            self.meta.loc[self.meta["labels"] == lab, "order"].iloc[0]
            for lab in labels
        ]
        assert orders == sorted(orders)

    def test_channels_for_file(self):
        channels = channels_for_file(self.meta, "Monitor 1.txt")
        assert len(channels) > 0
        assert all(1 <= c <= 32 for c in channels)

    def test_label_for_channel(self):
        channels = channels_for_file(self.meta, "Monitor 1.txt")
        label = label_for_channel(self.meta, "Monitor 1.txt", channels[0])
        assert isinstance(label, str)
        assert len(label) > 0

    def test_label_for_unknown_channel_returns_none(self):
        label = label_for_channel(self.meta, "Monitor 1.txt", 99)
        assert label is None

    def test_time_window_for_file(self):
        start, stop = time_window_for_file(self.meta, "Monitor 1.txt")
        assert isinstance(start, pd.Timestamp)
        assert isinstance(stop, pd.Timestamp)
        assert start < stop

    def test_time_window_unknown_file_raises(self):
        with pytest.raises(KeyError):
            time_window_for_file(self.meta, "nonexistent.txt")


# ── manual metadata builder (no metadata.xlsx required) ───────────────────────

class TestParseChannelSpec:
    def test_simple_range(self):
        assert parse_channel_spec("1-16") == list(range(1, 17))

    def test_comma_list(self):
        assert parse_channel_spec("1,3,5") == [1, 3, 5]

    def test_mixed_list_and_range(self):
        assert parse_channel_spec("1,3,5-10") == [1, 3, 5, 6, 7, 8, 9, 10]

    def test_dedupes_overlapping_entries(self):
        assert parse_channel_spec("1-5,3-7") == [1, 2, 3, 4, 5, 6, 7]

    def test_reversed_range_is_normalised(self):
        assert parse_channel_spec("16-1") == list(range(1, 17))

    def test_whitespace_is_tolerated(self):
        assert parse_channel_spec(" 1 , 3 , 5 - 8 ") == [1, 3, 5, 6, 7, 8]

    def test_empty_spec_raises(self):
        with pytest.raises(ValueError):
            parse_channel_spec("")

    def test_blank_spec_raises(self):
        with pytest.raises(ValueError):
            parse_channel_spec("   ")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_channel_spec("a-b")

    def test_out_of_range_channel_raises(self):
        with pytest.raises(ValueError):
            parse_channel_spec("1-33")

    def test_zero_channel_raises(self):
        with pytest.raises(ValueError):
            parse_channel_spec("0-5")


class TestBuildManualConditions:
    def _entry(self, **overrides):
        entry = {
            "file": "Monitor 1.txt",
            "start_datetime": "2026-06-17 15:28:00",
            "stop_datetime": "2026-06-22 10:51:00",
            "groups": [
                {"label": "Control", "channels": "1-16"},
                {"label": "Experimental", "channels": "17-32"},
            ],
        }
        entry.update(overrides)
        return entry

    def test_returns_dataframe_with_expected_columns(self):
        df = build_manual_conditions([self._entry()])
        for col in ["file", "start_datetime", "stop_datetime", "region_id", "labels", "order"]:
            assert col in df.columns

    def test_one_row_per_channel(self):
        df = build_manual_conditions([self._entry()])
        assert len(df) == 32
        assert sorted(df["region_id"]) == list(range(1, 33))

    def test_single_group_supported(self):
        entry = self._entry(groups=[{"label": "All flies", "channels": "1-32"}])
        df = build_manual_conditions([entry])
        assert df["labels"].unique().tolist() == ["All flies"]
        assert len(df) == 32

    def test_channels_as_int_list_supported(self):
        # the checkbox UI hands channels in as a list of ints, not a spec string
        entry = self._entry(groups=[
            {"label": "Control", "channels": [1, 2, 3, 4]},
            {"label": "Experimental", "channels": [5, 6, 7, 8]},
        ])
        df = build_manual_conditions([entry])
        assert sorted(df.loc[df["labels"] == "Control", "region_id"]) == [1, 2, 3, 4]
        assert sorted(df.loc[df["labels"] == "Experimental", "region_id"]) == [5, 6, 7, 8]

    def test_int_list_overlap_detected(self):
        entry = self._entry(groups=[
            {"label": "A", "channels": [1, 2, 3]},
            {"label": "B", "channels": [3, 4, 5]},
        ])
        with pytest.raises(ValueError):
            build_manual_conditions([entry])

    def test_empty_int_list_raises(self):
        entry = self._entry(groups=[{"label": "A", "channels": []}])
        with pytest.raises(ValueError):
            build_manual_conditions([entry])

    def test_more_than_two_groups_supported(self):
        entry = self._entry(groups=[
            {"label": "A", "channels": "1-8"},
            {"label": "B", "channels": "9-16"},
            {"label": "C", "channels": "17-24"},
            {"label": "D", "channels": "25-32"},
        ])
        df = build_manual_conditions([entry])
        assert sorted(df["labels"].unique()) == ["A", "B", "C", "D"]
        assert df["order"].nunique() == 4

    def test_order_assigned_by_first_appearance(self):
        df = build_manual_conditions([self._entry()])
        control_order = df.loc[df["labels"] == "Control", "order"].iloc[0]
        exp_order = df.loc[df["labels"] == "Experimental", "order"].iloc[0]
        assert control_order == 1
        assert exp_order == 2

    def test_same_label_across_files_shares_order(self):
        entries = [
            self._entry(),
            self._entry(file="Monitor 2.txt"),
        ]
        df = build_manual_conditions(entries)
        orders = df.groupby("labels")["order"].nunique()
        assert (orders == 1).all()

    def test_datetimes_are_parsed(self):
        df = build_manual_conditions([self._entry()])
        assert pd.api.types.is_datetime64_any_dtype(df["start_datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df["stop_datetime"])

    def test_empty_label_raises(self):
        entry = self._entry(groups=[{"label": "  ", "channels": "1-16"}])
        with pytest.raises(ValueError):
            build_manual_conditions([entry])

    def test_overlapping_channels_raise(self):
        entry = self._entry(groups=[
            {"label": "Control", "channels": "1-16"},
            {"label": "Experimental", "channels": "10-32"},
        ])
        with pytest.raises(ValueError):
            build_manual_conditions([entry])

    def test_start_after_stop_raises(self):
        entry = self._entry(start_datetime="2026-06-22 10:51:00",
                             stop_datetime="2026-06-17 15:28:00")
        with pytest.raises(ValueError):
            build_manual_conditions([entry])

    def test_unparseable_datetime_raises(self):
        entry = self._entry(start_datetime="not a date")
        with pytest.raises(ValueError):
            build_manual_conditions([entry])

    def test_no_groups_raises(self):
        entry = self._entry(groups=[])
        with pytest.raises(ValueError):
            build_manual_conditions([entry])

    def test_no_entries_raises(self):
        with pytest.raises(ValueError):
            build_manual_conditions([])
