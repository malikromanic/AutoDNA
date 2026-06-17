import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "stm32" / "bin_parser"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

VOZNJE_DIR        = Path(r"C:\Users\marom\Desktop\Voznje")
TRAINING_DATA_DIR = Path(__file__).parent.parent / "data" / "training_data"


# ── Desktop\Voznje recordings (skipped if absent) ─────────────────────────────

@pytest.fixture
def log001_bin():
    path = VOZNJE_DIR / "1.Dom-malo pred martinom" / "LOG001.BIN"
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return path


@pytest.fixture
def log001_npz():
    path = VOZNJE_DIR / "1.Dom-malo pred martinom" / "LOG001.npz"
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return path


@pytest.fixture
def log001_labels():
    path = VOZNJE_DIR / "1.Dom-malo pred martinom" / "LOG001_labels.json"
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return path


# ── Committed training data (always present in repo, runs in CI) ──────────────

@pytest.fixture
def training_npz():
    path = TRAINING_DATA_DIR / "parsed_data" / "LOG001.npz"
    if not path.exists():
        pytest.skip(f"Training NPZ not found: {path}")
    return path


@pytest.fixture
def training_labels():
    path = TRAINING_DATA_DIR / "labeled_data_json" / "LOG001_labels.json"
    if not path.exists():
        pytest.skip(f"Training labels not found: {path}")
    return path