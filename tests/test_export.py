"""Tests for the R-style multi-sheet metric exporter."""

import sys
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.data.export import write_metric_workbook


def _summary():
    """Minimal standard-schema summary: 2 conditions x 2 phases x 3 animals."""
    rows = []
    for cond, order, base_ch in [("Control", 1, 1), ("Mutant", 2, 4)]:
        for ch in range(base_ch, base_ch + 3):
            for x, y in [("Light", 0.5 + ch * 0.01), ("Dark", 0.8 + ch * 0.01)]:
                rows.append({
                    "id": f"Monitor 1.txt|{ch:02d}", "labels": cond, "order": order,
                    "channel": ch, "monitor": "Monitor 1.txt", "x": x, "y": y,
                })
    return pd.DataFrame(rows)


SETTINGS = [("LD cycle (h)", 24), ("Light hours per cycle (h)", 12),
            ("Sleep threshold (min)", 5)]


class TestWriteMetricWorkbook:
    def test_sheets_match_r_layout(self, tmp_path):
        out = tmp_path / "metric.xlsx"
        write_metric_workbook(out, _summary(), SETTINGS)
        wb = openpyxl.load_workbook(out)
        # Settings, Statistics, All data, then one sheet per x-level
        assert wb.sheetnames[:3] == ["Settings", "Statistics", "All data"]
        assert set(wb.sheetnames[3:]) == {"Light", "Dark"}

    def test_statistics_columns(self, tmp_path):
        out = tmp_path / "metric.xlsx"
        write_metric_workbook(out, _summary(), SETTINGS)
        wb = openpyxl.load_workbook(out)
        header = [c.value for c in next(wb["Statistics"].iter_rows())]
        assert header == ["Condition", "Label", "N", "Data_Mean", "Data_SEM",
                          "Data_SD", "Data_Median", "Data_1st_Quartile",
                          "Data_3rd_Quartile"]

    def test_statistics_row_count(self, tmp_path):
        # 2 conditions x 2 phases = 4 stat rows (+ header)
        out = tmp_path / "metric.xlsx"
        write_metric_workbook(out, _summary(), SETTINGS)
        wb = openpyxl.load_workbook(out)
        assert wb["Statistics"].max_row == 5

    def test_all_data_one_row_per_animal(self, tmp_path):
        out = tmp_path / "metric.xlsx"
        write_metric_workbook(out, _summary(), SETTINGS)
        wb = openpyxl.load_workbook(out)
        ws = wb["All data"]
        header = [c.value for c in next(ws.iter_rows())]
        assert header[:5] == ["File", "Labels", "Channels", "Start_date", "End_date"]
        assert "Light" in header and "Dark" in header
        assert ws.max_row == 7  # 6 animals + header

    def test_matrix_sheet_is_condition_by_animal(self, tmp_path):
        out = tmp_path / "metric.xlsx"
        write_metric_workbook(out, _summary(), SETTINGS)
        wb = openpyxl.load_workbook(out)
        ws = wb["Light"]
        header = [c.value for c in next(ws.iter_rows())]
        assert header == ["Control", "Mutant"]
        assert ws.max_row == 4  # 3 animals per condition + header

    def test_dates_are_filled_when_provided(self, tmp_path):
        out = tmp_path / "metric.xlsx"
        dates = {f"Monitor 1.txt|{ch:02d}": ("2021-05-03 11:09:00",
                                             "2021-05-07 10:48:00")
                 for ch in range(1, 7)}
        write_metric_workbook(out, _summary(), SETTINGS, dates_by_id=dates)
        wb = openpyxl.load_workbook(out)
        ws = wb["All data"]
        second_row = [c.value for c in list(ws.iter_rows())[1]]
        assert second_row[3] == "2021-05-03 11:09:00"

    def test_works_without_monitor_channel_columns(self, tmp_path):
        # rhythm-style summary lacking monitor/channel — derived from the id
        summary = _summary().drop(columns=["monitor", "channel"])
        out = tmp_path / "metric.xlsx"
        write_metric_workbook(out, summary, SETTINGS)
        wb = openpyxl.load_workbook(out)
        ws = wb["All data"]
        row2 = [c.value for c in list(ws.iter_rows())[1]]
        assert row2[0] == "Monitor 1.txt"       # File derived from id
        assert str(row2[2]) in {"1", "01"}       # Channels derived from id

    def test_empty_summary_raises(self, tmp_path):
        with pytest.raises(ValueError):
            write_metric_workbook(tmp_path / "x.xlsx", pd.DataFrame(), SETTINGS)

    def test_sheet_name_sanitised(self, tmp_path):
        # x-levels with illegal / long names must not break the writer
        s = _summary().copy()
        s["x"] = s["x"].map({"Light": "Day [1]/2", "Dark": "Dark"})
        out = tmp_path / "metric.xlsx"
        write_metric_workbook(out, s, SETTINGS)
        wb = openpyxl.load_workbook(out)
        assert all(not set(name) & set(r"[]:*?/\\") for name in wb.sheetnames)
