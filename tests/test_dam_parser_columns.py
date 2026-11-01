"""Column mapping of the DAM2 parser, on a synthetic file built in the test.

Every other parser test reads the real files in ~/Desktop/Rtivity_main and skips
where they are absent — which is CI, and every machine but one. Those tests also
assert only structure (columns exist, values are non-negative integers), which a
one-column shift satisfies: the light column is a valid non-negative integer
too. That is how v0.12.0 shipped reading the light sensor as channel 1, channel
k-1 as channel k, and never reading channel 32.

This fixture needs no external file and gives every field a value no other
field has, so reading any channel from the wrong column fails: the light column
is 7, the field before it is 9, channel k is 100 + k.

Layout (TAB-separated, 42 fields, 1-indexed), as exported by the rig:
  1 index | 2 date | 3 time | 4 status | 5 extras | 6 monitor | 7 tube |
  8 data type ("Ct") | 9 unused | 10 light | 11-42 channels 1-32
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.data.dam_parser import N_CHANNELS, _CHANNEL_COLS, read_dam_file

LIGHT = 7
UNUSED = 9


def _write_dam(path: Path, n_rows: int = 10) -> Path:
    lines = []
    for i in range(n_rows):
        fields = [str(i + 1), "28 Jan 26", f"15:{i:02d}:00", "1", "0", "7", "0",
                  "Ct", str(UNUSED), str(LIGHT)]
        fields += [str(100 + ch) for ch in range(1, N_CHANNELS + 1)]
        assert len(fields) == 42
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def dam(tmp_path):
    return read_dam_file(_write_dam(tmp_path / "Monitor1.txt"))


@pytest.mark.parametrize("ch", range(1, N_CHANNELS + 1))
def test_each_channel_reads_its_own_column(dam, ch):
    assert set(dam[f"ch{ch:02d}"]) == {100 + ch}


def test_channel_1_is_not_the_light_column(dam):
    assert LIGHT not in set(dam["ch01"])


def test_channel_32_is_read(dam):
    """v0.12.0 never read column 42: its 'channel 32' was column 41."""
    assert set(dam["ch32"]) == {132}


def test_no_channel_carries_the_light_or_unused_field(dam):
    for col in _CHANNEL_COLS:
        assert not ({LIGHT, UNUSED} & set(dam[col])), col
