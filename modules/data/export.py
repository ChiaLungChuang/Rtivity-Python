"""
Metric export — reproduces the R Rtivity per-metric download workbook.

Each analysis tab in the R app can download the metric behind its plot as a
multi-sheet .xlsx. This module builds the same workbook from the standard
summary schema used throughout the Python rewrite:

    id, labels, order, channel, monitor, x, y

Sheets produced (mirroring the R export, e.g. "Sleep ratio per light phase"):
  * Settings   — the analysis settings, as name/value rows
  * Statistics — per (condition, x) descriptive stats: N, mean, SEM, SD,
                 median, 1st & 3rd quartile
  * All data   — one row per animal: File, Labels, Channels, Start/End date,
                 then one column per x-level (the per-animal value)
  * <x-level>  — one sheet per x-level (e.g. Dark, Light, or Day 0 …) holding a
                 condition-by-animal matrix, ready to paste into GraphPad Prism
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..analysis.activity import summary_stats

_STATS_RENAME = {
    "labels": "Condition",
    "x": "Label",
    "n": "N",
    "mean": "Data_Mean",
    "sem": "Data_SEM",
    "sd": "Data_SD",
    "median": "Data_Median",
    "q1": "Data_1st_Quartile",
    "q3": "Data_3rd_Quartile",
}
_STATS_ORDER = list(_STATS_RENAME.values())

# Characters Excel forbids in a sheet name, plus the 31-char limit.
_BAD_SHEET_CHARS = str.maketrans({c: "_" for c in r"[]:*?/\\"})


def _ordered_labels(summary: pd.DataFrame) -> list[str]:
    """Condition labels in display order (by the `order` column if present)."""
    if "order" in summary.columns:
        pairs = summary[["labels", "order"]].drop_duplicates().sort_values("order")
        return pairs["labels"].tolist()
    return list(dict.fromkeys(summary["labels"]))


def _ordered_x(summary: pd.DataFrame) -> list:
    """x-levels in first-appearance order (handles Categorical and plain)."""
    return list(dict.fromkeys(summary["x"].tolist()))


def _safe_sheet_name(name: str, used: set[str]) -> str:
    """Excel-safe, <=31 char, unique sheet name."""
    base = str(name).translate(_BAD_SHEET_CHARS)[:31] or "Sheet"
    candidate, i = base, 1
    while candidate in used:
        suffix = f"_{i}"
        candidate = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(candidate)
    return candidate


def _statistics_sheet(summary: pd.DataFrame) -> pd.DataFrame:
    stats = summary_stats(summary).rename(columns=_STATS_RENAME)
    return stats[[c for c in _STATS_ORDER if c in stats.columns]]


def _monitor_of(animal_id, meta: pd.DataFrame):
    if "monitor" in meta.columns and pd.notna(meta.at[animal_id, "monitor"]):
        return meta.at[animal_id, "monitor"]
    return str(animal_id).split("|")[0]


def _channel_of(animal_id, meta: pd.DataFrame):
    if "channel" in meta.columns and pd.notna(meta.at[animal_id, "channel"]):
        return meta.at[animal_id, "channel"]
    parts = str(animal_id).split("|")
    return parts[1] if len(parts) > 1 else ""


def _all_data_sheet(
    summary: pd.DataFrame,
    dates_by_id: dict[str, tuple] | None,
) -> pd.DataFrame:
    """One row per animal, one column per x-level, plus id/metadata columns.

    Tolerant of summaries that omit monitor/channel/order (e.g. rhythm tables):
    File/Channels are then derived from the ``monitor|channel`` id.
    """
    x_levels = _ordered_x(summary)
    wide = summary.pivot_table(index="id", columns="x", values="y",
                               observed=True, sort=False).reindex(columns=x_levels)
    meta = summary.drop_duplicates("id").set_index("id")
    dates_by_id = dates_by_id or {}

    out = pd.DataFrame(index=wide.index)
    out["File"]     = [_monitor_of(i, meta) for i in wide.index]
    out["Labels"]   = [meta.at[i, "labels"] for i in wide.index]
    out["Channels"] = [_channel_of(i, meta) for i in wide.index]
    out["Start_date"] = [dates_by_id.get(i, ("", ""))[0] for i in wide.index]
    out["End_date"]   = [dates_by_id.get(i, ("", ""))[1] for i in wide.index]
    for x in x_levels:
        out[x] = wide[x].values

    if "order" in meta.columns:
        out["_order"] = [meta.at[i, "order"] for i in wide.index]
        out = out.sort_values(["_order", "Channels"]).drop(columns="_order")
    return out.reset_index(drop=True)


def _matrix_sheet(summary: pd.DataFrame, x_level, labels: list[str]) -> pd.DataFrame:
    """Condition-by-animal matrix for one x-level (Prism-wide layout)."""
    sub = summary[summary["x"] == x_level]
    columns: dict[str, list] = {}
    for label in labels:
        rows = sub[sub["labels"] == label]
        if "channel" in rows.columns:
            rows = rows.sort_values("channel")
        columns[str(label)] = rows["y"].tolist()

    max_len = max((len(v) for v in columns.values()), default=0)
    for label in columns:
        columns[label] += [None] * (max_len - len(columns[label]))
    return pd.DataFrame(columns)


def write_metric_workbook(
    path: str | Path,
    summary: pd.DataFrame,
    settings_rows: list[tuple[str, object]],
    dates_by_id: dict[str, tuple] | None = None,
) -> Path:
    """
    Write the multi-sheet metric workbook to `path`.

    Parameters
    ----------
    path          : output .xlsx path
    summary       : standard summary DataFrame (id/labels/order/channel/
                    monitor/x/y) from any analysis function
    settings_rows : [(name, value), …] for the Settings sheet
    dates_by_id   : optional {id: (start, end)} to fill Start/End_date columns

    Returns the written path.
    """
    path = Path(path)
    if summary is None or summary.empty:
        raise ValueError("No data to export — run the analysis first.")

    settings_df = pd.DataFrame(settings_rows, columns=["Settings", "Values"])
    stats_df = _statistics_sheet(summary)
    all_df = _all_data_sheet(summary, dates_by_id)

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        settings_df.to_excel(xw, sheet_name="Settings", index=False)
        stats_df.to_excel(xw, sheet_name="Statistics", index=False)
        all_df.to_excel(xw, sheet_name="All data", index=False)

        used = {"Settings", "Statistics", "All data"}
        labels = _ordered_labels(summary)
        for x_level in _ordered_x(summary):
            sheet = _safe_sheet_name(x_level, used)
            _matrix_sheet(summary, x_level, labels).to_excel(
                xw, sheet_name=sheet, index=False)

    return path
