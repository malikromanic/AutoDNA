import sys
from pathlib import Path

def get_data_dir() -> Path:
    if getattr(sys, 'frozen', False):
        data_dir = Path(sys.executable).parent / 'AutoDNA_data'
    else:
        data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir