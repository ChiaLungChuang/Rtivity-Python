"""Shared fixtures.

The DataStore load of the real Monitor files is the expensive step in this suite.
It used to run once per test module (five copies of the same load); here it is
session scoped, so the whole run pays it once. The loaded frame is treated as
read-only — the analysis functions copy before they mutate — so sharing it across
modules is safe.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data.data_store import DataStore, Settings

DAM_DIR = Path.home() / "Desktop/Rtivity_main"
META_PATH = DAM_DIR / "metadata.xlsx"


@pytest.fixture(scope="session")
def store():
    """The real experiment loaded once for the whole test session."""
    if not META_PATH.exists():
        pytest.skip("metadata.xlsx not found")
    s = DataStore(dam_dir=DAM_DIR, settings=Settings())
    s.load_from_metadata(META_PATH)
    return s


@pytest.fixture(scope="session")
def data(store):
    """Long-format analysis DataFrame. Read-only; derive copies, do not mutate."""
    return store.data
