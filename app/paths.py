"""UI-free path helpers for AutoDNA.

Kept separate from ``autodna_app`` so they can be imported and tested without
pulling in PyQt6.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent   # …/AutoDNA


def resolve_drive_path(p) -> Path:
    """
    Resolve a stored drive path on THIS machine.

    Drive paths saved in the store may be absolute paths from another teammate's
    computer. If the path doesn't exist locally, re-root it onto this repo's
    data/drive_data folder (the drive folders are committed, so they resolve
    on every machine). Unresolvable paths are returned unchanged so the caller
    surfaces a clear "not found" error.
    """
    path = Path(p)
    if path.exists():
        return path
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i] == "data" and parts[i + 1] == "drive_data":
            candidate = REPO_ROOT.joinpath(*parts[i:])
            if candidate.exists():
                return candidate
            break
    return path
