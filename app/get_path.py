"""Resolves where AutoDNA reads and writes its persistent data.

Used by every module that touches ``data/`` (drive stores, caches, the
crash log) so the path logic only lives in one place.
"""

import sys
from pathlib import Path

def get_data_dir() -> Path:
    """Return the data directory, creating it if needed.

    When running from a PyInstaller-frozen executable, this resolves to
    an ``AutoDNA_data`` folder next to the executable instead of the
    ``data/`` folder inside the source tree, since the source tree isn't
    available in a frozen build.

    Returns:
        Path: Absolute path to the data directory.
    """
    if getattr(sys, 'frozen', False):
        data_dir = Path(sys.executable).parent / 'AutoDNA_data'
    else:
        data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir