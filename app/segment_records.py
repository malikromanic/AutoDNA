# ============================================================================
# AutoDNA — per-segment fuel records & potential-savings (UI-free)
#
# Every detected turn/hill segment is bucketed by type + magnitude
# (e.g. "left_turn_0-10", "uphill_2-5"). For each bucket we keep the LOWEST
# fuel-consumption rate ever seen (the "record"), per vehicle.
#
# On each drive, for every segment:
#   • new best  → overwrite the record, no savings.
#   • worse     → savings_L = (rate - record) × duration_s.
# Summed over all segments → total potential savings the driver could have made.
#
# Each segment also gets a performance class for the map:
#   1 = green  (record set / tied)
#   2 = orange (worse, but within ORANGE_MAX_RATIO of the record)
#   3 = red    (significantly worse)
# ============================================================================

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from AutoDNA.app.segments import event_segments

_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "segment_records.json"

# A segment is "orange" if at most this fraction worse than the record, else red.
ORANGE_MAX_RATIO = 0.25

# Hill magnitude bins by |grade| %  (upper edges); last bucket is open-ended.
_HILL_EDGES = (2.0, 5.0, 10.0)
# Turn magnitude bin width (degrees); angles >= _TURN_CAP go in one open bucket.
_TURN_BIN_DEG = 15.0
_TURN_CAP = 90.0


# ── Keys ────────────────────────────────────────────────────────────────────
def vehicle_key(drive_path: str) -> str:
    """Derive a vehicle id from the drive path (folder under data/drive_data)."""
    parts = Path(drive_path).parts
    if "drive_data" in parts:
        i = parts.index("drive_data")
        if i + 1 < len(parts):
            return parts[i + 1]
    # fallback: parent folder of the CSV
    p = Path(drive_path)
    return p.parent.parent.name or p.parent.name or "default"


def _turn_bin(angle_deg: float) -> str:
    a = abs(angle_deg)
    if a >= _TURN_CAP:
        return f"{int(_TURN_CAP)}+"
    lo = int(a // _TURN_BIN_DEG) * int(_TURN_BIN_DEG)
    return f"{lo}-{lo + int(_TURN_BIN_DEG)}"


def _hill_bin(slope_pct: float) -> str:
    s = abs(slope_pct)
    lo = 0.0
    for hi in _HILL_EDGES:
        if s < hi:
            return f"{_fmt(lo)}-{_fmt(hi)}"
        lo = hi
    return f"{_fmt(_HILL_EDGES[-1])}+"


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def bucket_key(seg: dict) -> str:
    if seg["kind"] == "turn":
        side = "left_turn" if seg["direction"] == 1 else "right_turn"
        return f"{side}_{_turn_bin(seg['magnitude'])}"
    side = "uphill" if seg["direction"] == 1 else "downhill"
    return f"{side}_{_hill_bin(seg['magnitude'])}"


# ── Store I/O ───────────────────────────────────────────────────────────────
def _load_store(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_store(store: dict, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, indent=2))
    except OSError:
        pass


def _segment_rate(fuel_rate: np.ndarray | None, i0: int, i1: int) -> float:
    """Mean instant fuel rate (L/s) over the segment's GPS points."""
    if fuel_rate is None or len(fuel_rate) == 0:
        return 0.0
    seg = np.asarray(fuel_rate)[i0:i1 + 1]
    return float(seg.mean()) if len(seg) else 0.0


# ── Main entry point ──────────────────────────────────────────────────────--
def evaluate_drive(d, store_path: Path = _STORE_PATH) -> dict:
    """
    Bucket every segment, compare to the per-vehicle records, accumulate
    potential savings, and attach results to the DriveData instance:
      d.vehicle, d.turn_perf, d.hill_perf, d.total_savings_l, d.savings_breakdown

    The records store is updated in place (new bests are persisted).
    Returns a summary dict (also suitable for the Stats view).
    """
    store = _load_store(store_path)
    # Global records for now — no per-vehicle / per-user differentiation.
    # (vehicle_key() is kept for when we re-enable per-vehicle scoping.)
    vehicle = "all"
    vrec: dict = store.setdefault(vehicle, {})
    # Compare every segment against the records as they stood BEFORE this drive,
    # so the first drive (empty store) yields zero savings and repeats of the
    # same bucket within one drive are judged against history, not each other.
    snapshot = dict(vrec)

    fuel_rate = getattr(d, "fuel_rate_l_s", None)
    n = d.n_points
    turn_perf = np.zeros(n, dtype=np.int32)
    hill_perf = np.zeros(n, dtype=np.int32)
    total_savings = 0.0
    breakdown: list[dict] = []

    for seg in event_segments(d):
        i0, i1 = seg["i0"], seg["i1"]
        dur = seg["duration_s"]
        rate = _segment_rate(fuel_rate, i0, i1)
        bucket = bucket_key(seg)
        best = snapshot.get(bucket)            # the record to beat (pre-drive)

        if best is None or rate <= best + 1e-12:
            perf, savings = 1, 0.0             # new / tied record → green, no savings
        else:
            savings = (rate - best) * dur
            total_savings += savings
            ratio = (rate - best) / best if best > 1e-12 else float("inf")
            perf = 2 if ratio <= ORANGE_MAX_RATIO else 3

        # fold this drive's result into the stored record (keep the historical min)
        if bucket not in vrec or rate < vrec[bucket]:
            vrec[bucket] = rate

        (turn_perf if seg["kind"] == "turn" else hill_perf)[i0:i1 + 1] = perf
        breakdown.append({
            "bucket": bucket, "kind": seg["kind"], "direction": seg["direction"],
            "magnitude": round(seg["magnitude"], 1), "duration_s": round(dur, 1),
            "rate_l_s": rate, "record_l_s": float(best) if best is not None else rate,
            "savings_l": savings, "perf": perf,
        })

    _save_store(store, store_path)

    d.vehicle = vehicle
    d.turn_perf = turn_perf
    d.hill_perf = hill_perf
    d.total_savings_l = total_savings
    d.savings_breakdown = breakdown

    return {
        "vehicle": vehicle,
        "total_savings_l": total_savings,
        "n_segments": len(breakdown),
        "n_worse": sum(1 for b in breakdown if b["perf"] >= 2),
        "breakdown": breakdown,
    }
