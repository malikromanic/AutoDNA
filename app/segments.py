"""Shared turn/hill segment helpers, used by both the map widget and segment_records.

A "segment" is a consecutive run of GPS points with the same turn/hill
class. These helpers turn the per-point class arrays on
:class:`~app.data_loader.DriveData` into segments using one canonical
filter + gap-merge pass, so the map's Turns/Hills view and the fuel
records engine always agree on where an event starts and ends.
"""

from __future__ import annotations

import numpy as np

# Minimum run length (GPS points) for an event to count; gaps up to MERGE_GAP
# between same-direction runs are bridged. Mirrors the map widget defaults.
MIN_TURN_RUN = 2
MIN_HILL_RUN = 2
MERGE_GAP    = 5


def route_segments(arr: np.ndarray):
    """Return list of (i0, i1_exclusive, value) for consecutive equal runs."""
    segs, i, n = [], 0, len(arr)
    while i < n:
        v = int(arr[i])
        j = i
        while j < n and int(arr[j]) == v:
            j += 1
        segs.append((i, j, v))
        i = j
    return segs


def filter_preds(preds: np.ndarray, min_run: int) -> np.ndarray:
    """Zero out non-zero runs shorter than min_run."""
    out = preds.copy()
    for i0, i1, p in route_segments(preds):
        if p != 0 and (i1 - i0) < min_run:
            out[i0:i1] = 0
    return out.astype(np.int32)


def merge_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill zero gaps of <= max_gap points between same-value non-zero runs."""
    out = arr.copy()
    while True:
        changed = False
        segs = route_segments(out)
        for k in range(len(segs) - 2):
            _, _, pred_a = segs[k]
            i0_z, i1_z, pred_z = segs[k + 1]
            _, _, pred_b = segs[k + 2]
            if pred_z == 0 and pred_a != 0 and pred_a == pred_b and (i1_z - i0_z) <= max_gap:
                out[i0_z:i1_z] = pred_a
                changed = True
        if not changed:
            break
    return out.astype(np.int32)


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def cleaned_preds(preds: np.ndarray, min_run: int) -> np.ndarray:
    """Apply the canonical filter + gap-merge used everywhere."""
    return merge_gaps(filter_preds(np.asarray(preds).astype(np.int32), min_run), MERGE_GAP)


def event_segments(d) -> list[dict]:
    """
    Lightweight event segments for the records engine.

    Each dict: kind ('turn'|'hill'), direction (1|2), i0, i1 (inclusive),
    duration_s, magnitude (turn = net heading change deg; hill = mean abs(grade) %).
    Uses the same canonical filtering as the default map view, so the records
    coloring lines up with the Turns/Hills views.
    """
    ts   = d.gps_timestamps
    head = d.gps_heading
    grade = d.grade_pct
    out: list[dict] = []

    for kind, preds, min_run in (
        ('turn', cleaned_preds(d.turn_preds, MIN_TURN_RUN), MIN_TURN_RUN),
        ('hill', cleaned_preds(d.hill_preds, MIN_HILL_RUN), MIN_HILL_RUN),
    ):
        for i0, i1x, val in route_segments(preds):
            if val == 0:
                continue
            i1 = i1x - 1                       # inclusive end
            dur = float(ts[i1] - ts[i0]) if i1 > i0 else 0.0
            if kind == 'turn':
                magnitude = abs(_wrap180(float(head[i1]) - float(head[i0])))
            else:
                magnitude = float(np.mean(np.abs(grade[i0:i1x])))
            out.append({
                'kind': kind, 'direction': int(val),
                'i0': int(i0), 'i1': int(i1),
                'duration_s': dur, 'magnitude': magnitude,
            })
    return out
