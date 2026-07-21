"""Tests for rhythm.py using real data from DataStore."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.analysis.rhythm import (
    chi_sq_periodogram,
    lomb_scargle_periodogram,
    dominant_period,
    periodogram_by_animal,
    dominant_period_by_animal,
    fit_cosinor,
    cosinor_by_animal,
    cosinor_summary,
    is_iv_ra,
)

DAM_DIR   = Path.home() / "Desktop/Rtivity_main"
META_PATH = DAM_DIR / "metadata.xlsx"

pytestmark = pytest.mark.skipif(
    not META_PATH.exists(), reason="metadata.xlsx not found"
)


# `data` is the session-scoped fixture in conftest.py (loaded once for the run).


@pytest.fixture(scope="module")
def one_animal(data):
    aid = data["id"].unique()[0]
    return data[data["id"] == aid]


# ── chi-square periodogram ────────────────────────────────────────────────────

class TestChiSqPeriodogram:
    def test_returns_dataframe(self, one_animal):
        pg = chi_sq_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        assert isinstance(pg, pd.DataFrame)

    def test_has_expected_columns(self, one_animal):
        pg = chi_sq_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        for col in ["period_h", "Q", "p_value"]:
            assert col in pg.columns

    def test_Q_non_negative(self, one_animal):
        pg = chi_sq_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        assert (pg["Q"].dropna() >= 0).all()

    def test_p_value_in_0_1(self, one_animal):
        pg = chi_sq_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        pv = pg["p_value"].dropna()
        assert (pv >= 0).all() and (pv <= 1).all()

    def test_periods_cover_requested_range(self, one_animal):
        periods = np.arange(20.0, 30.0, 1.0)
        pg = chi_sq_periodogram(
            one_animal["activity"].values, one_animal["t"].values,
            periods_h=periods,
        )
        assert len(pg) == len(periods)
        np.testing.assert_array_almost_equal(pg["period_h"].values, periods)

    def test_dominant_period_near_24h(self, one_animal):
        """Wild-type flies should show a ~24 h dominant period."""
        pg = chi_sq_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        dp = dominant_period(pg, value_col="Q")
        assert 20.0 <= dp <= 28.0, f"Dominant period {dp:.1f} h outside expected range"


# ── Lomb-Scargle periodogram ──────────────────────────────────────────────────

class TestLombScarglePeriodogram:
    def test_returns_dataframe(self, one_animal):
        pg = lomb_scargle_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        assert isinstance(pg, pd.DataFrame)

    def test_has_expected_columns(self, one_animal):
        pg = lomb_scargle_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        for col in ["period_h", "power"]:
            assert col in pg.columns

    def test_power_in_0_1(self, one_animal):
        pg = lomb_scargle_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        pwr = pg["power"].dropna()
        assert (pwr >= 0).all() and (pwr <= 1.01).all()

    def test_dominant_period_near_24h(self, one_animal):
        pg = lomb_scargle_periodogram(
            one_animal["activity"].values, one_animal["t"].values
        )
        dp = dominant_period(pg, value_col="power")
        assert 20.0 <= dp <= 28.0, f"LS dominant period {dp:.1f} h outside range"


# ── per-animal wrappers ───────────────────────────────────────────────────────

class TestPeriodogramByAnimal:
    def test_returns_dataframe(self, data):
        pg = periodogram_by_animal(data, method="chi_sq")
        assert isinstance(pg, pd.DataFrame)

    def test_has_animal_columns(self, data):
        pg = periodogram_by_animal(data, method="chi_sq")
        for col in ["id", "labels", "period_h", "Q"]:
            assert col in pg.columns

    def test_one_entry_per_animal_per_period(self, data):
        periods = np.arange(22.0, 26.0, 1.0)
        pg = periodogram_by_animal(data, method="chi_sq", periods_h=periods)
        n_animals = data["id"].nunique()
        assert len(pg) == n_animals * len(periods)

    def test_lomb_scargle_method(self, data):
        pg = periodogram_by_animal(data, method="lomb_scargle")
        assert "power" in pg.columns


class TestDominantPeriodByAnimal:
    def test_returns_dataframe(self, data):
        result = dominant_period_by_animal(data)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_animal(self, data):
        result = dominant_period_by_animal(data)
        assert len(result) == data["id"].nunique()

    def test_x_label(self, data):
        result = dominant_period_by_animal(data)
        assert (result["x"] == "DominantPeriod").all()

    def test_y_in_plausible_range(self, data):
        result = dominant_period_by_animal(data)
        assert result["y"].between(16, 32).all()

    def test_has_standard_columns(self, data):
        result = dominant_period_by_animal(data)
        for col in ["id", "labels", "order", "channel", "monitor", "x", "y"]:
            assert col in result.columns


# ── cosinor analysis ──────────────────────────────────────────────────────────

class TestFitCosinor:
    def test_returns_dict(self, one_animal):
        result = fit_cosinor(
            one_animal["activity"].values, one_animal["t"].values
        )
        assert isinstance(result, dict)

    def test_has_expected_keys(self, one_animal):
        result = fit_cosinor(
            one_animal["activity"].values, one_animal["t"].values
        )
        for key in ["mesor", "amplitude", "acrophase_h", "r2", "period_h"]:
            assert key in result

    def test_amplitude_non_negative(self, one_animal):
        result = fit_cosinor(
            one_animal["activity"].values, one_animal["t"].values
        )
        assert result["amplitude"] >= 0

    def test_acrophase_in_valid_range(self, one_animal):
        result = fit_cosinor(
            one_animal["activity"].values, one_animal["t"].values
        )
        assert 0.0 <= result["acrophase_h"] < 24.0

    def test_r2_in_0_1(self, one_animal):
        result = fit_cosinor(
            one_animal["activity"].values, one_animal["t"].values
        )
        assert -0.01 <= result["r2"] <= 1.01

    def test_period_h_preserved(self, one_animal):
        result = fit_cosinor(
            one_animal["activity"].values, one_animal["t"].values,
            period_h=24.0,
        )
        assert result["period_h"] == 24.0


class TestCosinorByAnimal:
    def test_returns_dataframe(self, data):
        assert isinstance(cosinor_by_animal(data), pd.DataFrame)

    def test_one_row_per_animal(self, data):
        df = cosinor_by_animal(data)
        assert len(df) == data["id"].nunique()

    def test_has_expected_columns(self, data):
        df = cosinor_by_animal(data)
        for col in ["id", "labels", "mesor", "amplitude", "acrophase_h", "r2"]:
            assert col in df.columns

    def test_amplitude_non_negative(self, data):
        df = cosinor_by_animal(data)
        assert (df["amplitude"] >= 0).all()

    def test_acrophase_in_range(self, data):
        df = cosinor_by_animal(data)
        assert df["acrophase_h"].between(0, 24).all()


class TestCosinorSummary:
    def test_returns_dataframe(self, data):
        assert isinstance(cosinor_summary(data), pd.DataFrame)

    def test_x_values(self, data):
        df = cosinor_summary(data)
        assert set(df["x"].unique()) == {"mesor", "amplitude", "acrophase_h", "r2"}

    def test_has_standard_columns(self, data):
        df = cosinor_summary(data)
        for col in ["id", "labels", "order", "channel", "monitor", "x", "y"]:
            assert col in df.columns

    def test_four_rows_per_animal(self, data):
        df = cosinor_summary(data)
        assert len(df) == data["id"].nunique() * 4


# ── IS / IV / RA ──────────────────────────────────────────────────────────────

class TestIsIvRa:
    def test_returns_dataframe(self, data):
        assert isinstance(is_iv_ra(data), pd.DataFrame)

    def test_x_values(self, data):
        df = is_iv_ra(data)
        assert set(df["x"].unique()) == {"IS", "IV", "RA"}

    def test_three_rows_per_animal(self, data):
        df = is_iv_ra(data)
        assert len(df) == data["id"].nunique() * 3

    def test_IS_in_0_1(self, data):
        df = is_iv_ra(data)
        IS = df[df["x"] == "IS"]["y"]
        assert IS.between(0, 1).all(), f"IS out of [0,1]: {IS.describe()}"

    def test_IV_non_negative(self, data):
        df = is_iv_ra(data)
        IV = df[df["x"] == "IV"]["y"]
        assert (IV >= 0).all()

    def test_RA_in_0_1(self, data):
        df = is_iv_ra(data)
        RA = df[df["x"] == "RA"]["y"]
        assert RA.between(0, 1).all(), f"RA out of [0,1]: {RA.describe()}"

    def test_has_standard_columns(self, data):
        df = is_iv_ra(data)
        for col in ["id", "labels", "order", "channel", "monitor", "x", "y"]:
            assert col in df.columns
