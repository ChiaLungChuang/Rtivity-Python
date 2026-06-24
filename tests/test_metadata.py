"""Tests for metadata.py using the real metadata.xlsx from Rtivity_main."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.data.metadata import (
    channels_for_file,
    label_for_channel,
    list_files,
    list_labels,
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
