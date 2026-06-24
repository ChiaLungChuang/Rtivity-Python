"""Tests for DataStore using the real Monitor files + metadata.xlsx."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.data.data_store import DataStore, Settings

DAM_DIR  = Path.home() / "Desktop/Rtivity_main"
META_PATH = DAM_DIR / "metadata.xlsx"

pytestmark = pytest.mark.skipif(
    not META_PATH.exists(), reason="metadata.xlsx not found"
)


@pytest.fixture(scope="module")
def store():
    s = DataStore(dam_dir=DAM_DIR, settings=Settings())
    s.load_from_metadata(META_PATH)
    return s


class TestDataStore:
    def test_loads_without_error(self, store):
        assert store is not None

    def test_data_is_dataframe(self, store):
        assert isinstance(store.data, pd.DataFrame)

    def test_data_not_empty(self, store):
        assert len(store.data) > 0

    def test_required_columns_present(self, store):
        for col in [
            "id", "datetime", "t", "zt", "phase",
            "activity", "moving", "auc", "time_diff",
            "labels", "order", "channel", "monitor",
            "experiment_day", "day_night", "period_t",
        ]:
            assert col in store.data.columns, f"Missing column: {col}"

    def test_metadata_has_one_row_per_animal(self, store):
        assert len(store.metadata) == store.data["id"].nunique()

    def test_t_starts_at_zt_offset(self, store):
        # t starts at (recording_start - ZT0) in seconds, not at 0.
        # ZT0=08:00, Monitor 1 starts at 12:11 → t₀ ≈ 15060s (4h11m)
        # All animals in the same file share the same starting t.
        first_t = store.data.groupby("id")["t"].first()
        assert (first_t >= 0).all(), "t should be non-negative"
        assert (first_t < 86400).all(), "t offset should be less than one full day"

    def test_t_monotonic_per_animal(self, store):
        for aid, grp in store.data.groupby("id"):
            assert grp["t"].is_monotonic_increasing, f"t not monotonic for {aid}"

    def test_activity_non_negative(self, store):
        assert (store.data["activity"] >= 0).all()

    def test_moving_is_bool(self, store):
        assert store.data["moving"].dtype == bool

    def test_moving_consistent_with_activity(self, store):
        assert (store.data["moving"] == (store.data["activity"] > 0)).all()

    def test_auc_non_decreasing_per_animal(self, store):
        for _, grp in store.data.groupby("id"):
            assert grp["auc"].is_monotonic_increasing

    def test_experiment_day_non_negative(self, store):
        assert (store.data["experiment_day"] >= 0).all()

    def test_day_night_is_bool(self, store):
        assert store.data["day_night"].dtype == bool

    def test_labels_not_empty(self, store):
        assert store.data["labels"].notna().all()
        assert (store.data["labels"].str.len() > 0).all()

    def test_labels_match_metadata_labels(self, store):
        data_labels = set(store.data["labels"].unique())
        meta_labels = set(store.labels)
        assert data_labels == meta_labels

    def test_channels_in_valid_range(self, store):
        assert store.data["channel"].between(1, 32).all()


class TestGetters:
    def test_get_animal_returns_single_id(self, store):
        aid = store.animal_ids[0]
        sub = store.get_animal(aid)
        assert (sub["id"] == aid).all()

    def test_get_label_returns_correct_group(self, store):
        label = store.labels[0]
        sub = store.get_label(label)
        assert (sub["labels"] == label).all()
        assert len(sub) > 0

    def test_animal_ids_count(self, store):
        # metadata.xlsx has 48 channel rows = 48 animals
        assert len(store.animal_ids) == 48


class TestSettings:
    def test_default_ld_period(self):
        s = Settings()
        assert s.ld_period == 24.0

    def test_per_file_zt0_override(self):
        s = Settings(zt0="08:00:00", zt0_per_file={"Monitor 1.txt": "10:00:00"})
        assert s.zt0_for("Monitor 1.txt") == "10:00:00"
        assert s.zt0_for("Monitor 2.txt") == "08:00:00"
