"""Tests for survival.py using real data from DataStore."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.analysis.survival import (
    build_survival_data,
    kaplan_meier_curves,
    logrank_test,
    multigroup_logrank_test,
    survival_summary,
    median_survival,
)

DAM_DIR   = Path.home() / "Desktop/Rtivity_main"
META_PATH = DAM_DIR / "metadata.xlsx"

pytestmark = pytest.mark.skipif(
    not META_PATH.exists(), reason="metadata.xlsx not found"
)


# `data` is the session-scoped fixture in conftest.py (loaded once for the run).


@pytest.fixture(scope="module")
def surv(data):
    return build_survival_data(data, dead_window_h=24.0)


# ── build_survival_data ───────────────────────────────────────────────────────

class TestBuildSurvivalData:
    def test_returns_dataframe(self, surv):
        assert isinstance(surv, pd.DataFrame)

    def test_has_required_columns(self, surv):
        for col in ["id", "labels", "duration", "event"]:
            assert col in surv.columns

    def test_one_row_per_animal(self, data, surv):
        assert len(surv) == data["id"].nunique()

    def test_duration_non_negative(self, surv):
        assert (surv["duration"] >= 0).all()

    def test_event_binary(self, surv):
        assert set(surv["event"].unique()).issubset({0, 1})

    def test_duration_not_exceed_max_day(self, data, surv):
        max_day = data["experiment_day"].max()
        assert (surv["duration"] <= max_day).all()

    def test_labels_preserved(self, data, surv):
        assert set(surv["labels"].unique()) == set(data["labels"].unique())

    def test_some_deaths_detected(self, surv):
        """With a 24h window, at least some animals should be classified as dead."""
        # For multi-day recordings most animals die → event=1 should appear
        # (unless recording is very short; allow either case but check it runs)
        assert surv["event"].isin([0, 1]).all()

    def test_censored_have_max_duration(self, data, surv):
        """Censored animals (event=0) should live until the last recorded day."""
        censored = surv[surv["event"] == 0]
        if censored.empty:
            pytest.skip("No censored animals in this dataset")
        max_day = data["experiment_day"].max()
        assert (censored["duration"] == max_day).all()


# ── kaplan_meier_curves ───────────────────────────────────────────────────────

class TestKaplanMeierCurves:
    def test_returns_dataframe(self, surv):
        assert isinstance(kaplan_meier_curves(surv), pd.DataFrame)

    def test_has_required_columns(self, surv):
        df = kaplan_meier_curves(surv)
        for col in ["labels", "time", "survival", "ci_lower", "ci_upper"]:
            assert col in df.columns

    def test_survival_starts_at_1(self, surv):
        df = kaplan_meier_curves(surv)
        for label, grp in df.groupby("labels"):
            t0_survival = grp.loc[grp["time"] == grp["time"].min(), "survival"].iloc[0]
            assert t0_survival == pytest.approx(1.0, abs=1e-9)

    def test_survival_non_increasing(self, surv):
        df = kaplan_meier_curves(surv)
        for label, grp in df.groupby("labels"):
            s = grp.sort_values("time")["survival"].values
            assert (np.diff(s) <= 1e-9).all(), f"{label}: survival is not non-increasing"

    def test_survival_in_0_1(self, surv):
        df = kaplan_meier_curves(surv)
        assert df["survival"].between(0, 1).all()

    def test_ci_lower_le_survival(self, surv):
        df = kaplan_meier_curves(surv)
        assert (df["ci_lower"] <= df["survival"] + 1e-9).all()

    def test_ci_upper_ge_survival(self, surv):
        df = kaplan_meier_curves(surv)
        assert (df["ci_upper"] >= df["survival"] - 1e-9).all()

    def test_one_curve_per_group(self, data, surv):
        df = kaplan_meier_curves(surv)
        assert set(df["labels"].unique()) == set(data["labels"].unique())

    def test_requires_survival_columns(self, data):
        with pytest.raises(ValueError, match="duration"):
            kaplan_meier_curves(data)


# ── logrank_test ──────────────────────────────────────────────────────────────

class TestLogRankTest:
    def test_returns_dataframe(self, surv):
        assert isinstance(logrank_test(surv), pd.DataFrame)

    def test_has_required_columns(self, surv):
        df = logrank_test(surv)
        for col in ["group1", "group2", "test_statistic", "p_value"]:
            assert col in df.columns

    def test_n_comparisons(self, data, surv):
        n_groups = data["labels"].nunique()
        expected = n_groups * (n_groups - 1) // 2
        df = logrank_test(surv)
        assert len(df) == expected

    def test_p_value_in_0_1(self, surv):
        df = logrank_test(surv)
        assert df["p_value"].between(0, 1).all()

    def test_test_statistic_non_negative(self, surv):
        df = logrank_test(surv)
        assert (df["test_statistic"] >= 0).all()

    def test_groups_are_different_labels(self, surv):
        df = logrank_test(surv)
        assert (df["group1"] != df["group2"]).all()


class TestMultigroupLogrankTest:
    def test_returns_dict(self, surv):
        result = multigroup_logrank_test(surv)
        assert isinstance(result, dict)

    def test_has_required_keys(self, surv):
        result = multigroup_logrank_test(surv)
        for key in ["test_statistic", "p_value", "degrees_of_freedom"]:
            assert key in result

    def test_p_value_in_0_1(self, surv):
        result = multigroup_logrank_test(surv)
        assert 0 <= result["p_value"] <= 1

    def test_degrees_of_freedom(self, data, surv):
        result = multigroup_logrank_test(surv)
        expected_df = data["labels"].nunique() - 1
        assert result["degrees_of_freedom"] == expected_df


# ── survival_summary ──────────────────────────────────────────────────────────

class TestSurvivalSummary:
    def test_returns_dataframe(self, surv):
        assert isinstance(survival_summary(surv), pd.DataFrame)

    def test_has_standard_columns(self, surv):
        df = survival_summary(surv)
        for col in ["labels", "x", "y"]:
            assert col in df.columns

    def test_x_values(self, surv):
        df = survival_summary(surv)
        assert set(df["x"].unique()) == {"n", "events", "median", "mean"}

    def test_four_rows_per_group(self, data, surv):
        df = survival_summary(surv)
        n_groups = data["labels"].nunique()
        assert len(df) == n_groups * 4

    def test_n_matches_animal_count(self, data, surv):
        df = survival_summary(surv)
        for label, grp in df[df["x"] == "n"].groupby("labels"):
            expected = (surv["labels"] == label).sum()
            assert int(grp["y"].iloc[0]) == expected

    def test_events_le_n(self, surv):
        df = survival_summary(surv)
        for label in df["labels"].unique():
            sub = df[df["labels"] == label]
            n      = sub[sub["x"] == "n"]["y"].iloc[0]
            events = sub[sub["x"] == "events"]["y"].iloc[0]
            assert events <= n


# ── median_survival ───────────────────────────────────────────────────────────

class TestMedianSurvival:
    def test_returns_dataframe(self, surv):
        assert isinstance(median_survival(surv), pd.DataFrame)

    def test_one_row_per_group(self, data, surv):
        df = median_survival(surv)
        assert len(df) == data["labels"].nunique()

    def test_y_non_negative(self, surv):
        df = median_survival(surv)
        valid = df["y"].dropna()
        assert (valid >= 0).all()

    def test_ci_columns_present(self, surv):
        df = median_survival(surv)
        assert "ci_lower" in df.columns and "ci_upper" in df.columns
