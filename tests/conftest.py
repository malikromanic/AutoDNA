import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "stm32" / "bin_parser"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA = Path(__file__).parent.parent / "data" / "training_data"
_RAW    = _DATA / "raw_data"
_PARSED = _DATA / "parsed_data"
_LABELS = _DATA / "labeled_data_json"


@pytest.fixture
def log001_bin():
    path = _RAW / "LOG001.BIN"
    if not path.exists():
        pytest.skip(f"Not found: {path}")
    return path


@pytest.fixture
def log001_npz():
    path = _PARSED / "LOG001.npz"
    if not path.exists():
        pytest.skip(f"Not found: {path}")
    return path


@pytest.fixture
def log001_labels():
    path = _LABELS / "LOG001_labels.json"
    if not path.exists():
        pytest.skip(f"Not found: {path}")
    return path


@pytest.fixture
def training_npz():
    path = _PARSED / "LOG001.npz"
    if not path.exists():
        pytest.skip(f"Not found: {path}")
    return path


@pytest.fixture
def training_labels():
    path = _LABELS / "LOG001_labels.json"
    if not path.exists():
        pytest.skip(f"Not found: {path}")
    return path