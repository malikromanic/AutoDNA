import sys
from pathlib import Path

import pytest

# Allow "from stm_utils import ..." / "from packet import ..." in test files
sys.path.insert(0, str(Path(__file__).parent.parent / "stm32" / "bin_parser"))
# Allow "from AutoDNA.X.Y import ..." — repo's parent must be on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

VOZNJE_DIR = Path(r"C:\Users\marom\Desktop\Voznje")


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
