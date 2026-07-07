"""
DataStore — replaces the R `behavr` table + damr linking step.

Loads multiple DAM files, filters them by the metadata time windows,
joins condition metadata, and produces a single analysis-ready long-format
DataFrame (one row per animal per time-point).

Equivalent to R's:
    data <- link_dam_metadata(Conditions$df, Directory())
    damData$dt <- load_dam(data, FUN = sleep_dam_annotation, ...)
    # ... plus the post-processing in observeEvent(input$startanalysis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .dam_parser import read_dam_file, to_long_format
from .metadata import (
    channels_for_file,
    label_for_channel,
    list_files,
    list_labels,
    read_metadata,
    time_window_for_file,
)


@dataclass
class Settings:
    """Analysis settings — mirrors the Settings sheet saved in the xlsx."""
    ld_period: float = 24.0       # LD cycle period in hours
    light_hours: float = 12.0     # light phase duration in hours
    sleep_threshold_min: int = 5  # inactivity minutes to call sleep
    bin_size_min: int = 60        # moving-average bin size in minutes
    bout_threshold_min: int = 5   # activity bout gap threshold in minutes
    zt0: str = "08:00:00"         # default lights-on time (overridden per file)

    # Per-file ZT0 overrides: {"Monitor 1.txt": "08:00:00", ...}
    zt0_per_file: dict[str, str] = field(default_factory=dict)

    def zt0_for(self, filename: str) -> str:
        return self.zt0_per_file.get(filename, self.zt0)


class DataStore:
    """
    Central data container, equivalent to R's damData$dt (behavr table).

    Usage
    -----
    store = DataStore(dam_dir="/path/to/monitors", settings=Settings())
    store.load_from_metadata("/path/to/metadata.xlsx")
    df = store.data          # long-format DataFrame, all animals
    meta = store.metadata    # per-animal metadata
    """

    def __init__(
        self,
        dam_dir: str | Path,
        settings: Settings | None = None,
    ):
        self.dam_dir = Path(dam_dir)
        self.settings = settings or Settings()

        self._data: pd.DataFrame | None = None      # long-format animal data
        self._meta: pd.DataFrame | None = None      # per-animal metadata row
        self._conditions: pd.DataFrame | None = None  # raw metadata xlsx rows

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise RuntimeError("No data loaded. Call load_from_metadata() first.")
        return self._data

    @property
    def metadata(self) -> pd.DataFrame:
        if self._meta is None:
            raise RuntimeError("No data loaded. Call load_from_metadata() first.")
        return self._meta

    @property
    def conditions(self) -> pd.DataFrame:
        if self._conditions is None:
            raise RuntimeError("No conditions loaded.")
        return self._conditions

    @property
    def labels(self) -> list[str]:
        return list_labels(self.conditions)

    @property
    def animal_ids(self) -> list[str]:
        return self.data["id"].unique().tolist()

    def load_from_metadata(self, meta_path: str | Path) -> None:
        """
        Main entry point. Reads a metadata xlsx and loads all referenced
        DAM files, producing a unified long-format DataFrame.
        """
        meta_path = Path(meta_path)
        self._conditions = read_metadata(meta_path)
        segments = []

        for filename in list_files(self._conditions):
            dam_path = self.dam_dir / filename
            if not dam_path.exists():
                raise FileNotFoundError(
                    f"DAM file '{filename}' listed in metadata not found in {self.dam_dir}"
                )

            start, stop = time_window_for_file(self._conditions, filename)
            zt0 = self.settings.zt0_for(filename)

            wide = read_dam_file(
                dam_path,
                start=start,
                stop=stop,
                zt0=zt0,
                ld_period=self.settings.ld_period,
                light_hours=self.settings.light_hours,
            )

            channels = channels_for_file(self._conditions, filename)
            long = to_long_format(wide, channels=channels, monitor_name=filename)

            # Join condition metadata
            long["labels"] = long.apply(
                lambda row: label_for_channel(
                    self._conditions, filename, row["channel"]
                ),
                axis=1,
            )
            long["order"] = long["labels"].map(
                self._conditions.drop_duplicates("labels")
                .set_index("labels")["order"]
            )

            segments.append(long)

        combined = pd.concat(segments, ignore_index=True)
        self._data, self._meta = self._build_analysis_columns(combined)

    def load_from_dataframe(
        self, conditions: pd.DataFrame, dam_dir: str | Path | None = None
    ) -> None:
        """Load from an already-parsed conditions DataFrame (e.g. built in UI)."""
        if dam_dir is not None:
            self.dam_dir = Path(dam_dir)
        self._conditions = conditions
        # Re-use the metadata loading path
        tmp_path = self.dam_dir / "_tmp_meta.xlsx"
        conditions.to_excel(tmp_path, index=False)
        self.load_from_metadata(tmp_path)
        tmp_path.unlink(missing_ok=True)

    def get_animal(self, animal_id: str) -> pd.DataFrame:
        """Return all rows for a single animal."""
        return self.data[self.data["id"] == animal_id].copy()

    def get_label(self, label: str) -> pd.DataFrame:
        """Return all rows for animals in a condition group."""
        return self.data[self.data["labels"] == label].copy()

    # ── internal ─────────────────────────────────────────────────────────────

    def _build_analysis_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Add derived columns that mirror R's post-load computations:
          id, t, period_t, experiment_day, day_night,
          moving, auc, time_diff
        """
        # Unique animal identifier: "filename|channel"
        df["id"] = df["monitor"] + "|" + df["channel"].astype(str).str.zfill(2)

        # Sort so each animal's rows are chronological
        df = df.sort_values(["id", "datetime"]).reset_index(drop=True)

        # t: seconds elapsed since each animal's first recorded timepoint
        # (mirrors R's behavr 't' column — ZT-adjusted below)
        df["t"] = (
            df.groupby("id")["datetime"]
            .transform(lambda s: (s - s.iloc[0]).dt.total_seconds())
        )

        # ZT offset: shift t so t=0 aligns with ZT0 of that animal's file
        # (mirrors R: damData$dt[, t := t + (minDiff[n] * 60)])
        zt0_offsets = {
            fname: self._zt0_offset_seconds(fname, start)
            for fname in self._conditions["file"].unique()
            for start in [time_window_for_file(self._conditions, fname)[0]]
        }
        df["t"] = df.apply(
            lambda row: row["t"] + zt0_offsets.get(row["monitor"], 0),
            axis=1,
        )

        df["period_t"] = df["t"]

        lp_sec = self.settings.ld_period * 3600
        lh_sec = self.settings.light_hours * 3600

        df["experiment_day"] = (df["t"] // lp_sec).astype(int)
        df["day_night"] = ((df["t"] - df["experiment_day"] * lp_sec) < lh_sec)

        df["moving"] = df["activity"] > 0
        df["auc"] = df.groupby("id")["activity"].cumsum()

        # time_diff: seconds between consecutive rows per animal
        df["time_diff"] = df.groupby("id")["t"].diff()

        # Per-animal metadata summary
        meta = (
            df[["id", "monitor", "channel", "labels", "order"]]
            .drop_duplicates("id")
            .reset_index(drop=True)
        )

        return df, meta

    def _zt0_offset_seconds(self, filename: str, start: pd.Timestamp) -> float:
        """
        Seconds to add to t so that the recording start aligns with ZT0.
        Mirrors: minDiff = difftime(start_time, zt0_time, units='mins')
        """
        zt0_str = self.settings.zt0_for(filename)
        h, m, s = map(int, zt0_str.split(":"))
        zt0_seconds_of_day = h * 3600 + m * 60 + s
        start_seconds_of_day = (
            start.hour * 3600 + start.minute * 60
            # truncate seconds for clean minute alignment (mirrors R code)
        )
        return (start_seconds_of_day - zt0_seconds_of_day)
