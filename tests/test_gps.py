# ============================================================================
# GPS app tests (gps_analysis + elevation + data_loader + paths + segment_records)
# — core behaviors only.
# ============================================================================
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from AutoDNA.app.gps_analysis import (
    cumulative_distance, compute_turns, compute_hills, elevation_gain_loss, _iter_segments,
)
from AutoDNA.app.elevation import _ElevationCache, get_default_provider, OnlineElevationProvider
from AutoDNA.app.data_loader import (
    _compute_heading, _align_speed, _extract_fuel_l, _fuel_rate_per_point,
)
from AutoDNA.app.paths import resolve_drive_path, REPO_ROOT
from AutoDNA.app.segment_records import evaluate_drive, bucket_key


def _even_track(n=60, spacing=10.0):
    return np.arange(n, dtype=float) * spacing


# ── Turns / hills / distance (gps_analysis) ──────────────────────────────--
def test_cumulative_distance_one_degree_lat_about_111km():
    cum = cumulative_distance(np.array([45.0, 46.0]), np.array([15.0, 15.0]))
    assert 110_000 < cum[-1] < 112_000


def test_straight_track_has_no_turns():
    cum = _even_track()
    assert np.all(compute_turns(np.zeros(len(cum)), cum)[0] == 0)


def test_right_and_left_turns_detected():
    cum = _even_track()
    r_head = np.zeros(len(cum)); r_head[30:] = 90.0          # clockwise N→E = right
    l_head = np.full(len(cum), 90.0); l_head[30:] = 0.0      # E→N = left
    assert compute_turns(r_head, cum)[0][30] == 2
    assert compute_turns(l_head, cum)[0][30] == 1


def test_gradual_turn_is_one_continuous_segment():
    cum = _even_track(n=60, spacing=10.0)
    heading = np.zeros(60)
    for k in range(20, 41):
        heading[k] = (k - 20) * 4.5                          # ramp 0° → 90°
    heading[41:] = heading[40]
    segs = [(s, e, v) for s, e, v in _iter_segments(compute_turns(heading, cum)[0]) if v != 0]
    assert len(segs) == 1 and segs[0][2] == 2
    assert cum[segs[0][1]] - cum[segs[0][0]] >= 100          # long & continuous


def test_flat_uphill_downhill():
    cum = _even_track()
    assert np.all(compute_hills(np.full(len(cum), 400.0), cum)[0] == 0)   # flat
    assert compute_hills(400.0 + 0.05 * cum, cum)[0][len(cum) // 2] == 1  # uphill
    assert compute_hills(400.0 - 0.05 * cum, cum)[0][len(cum) // 2] == 2  # downhill


def test_all_nan_elevation_disables_hills():
    cum = _even_track()
    assert np.all(compute_hills(np.full(len(cum), np.nan), cum)[0] == 0)


def test_elevation_gain_loss():
    gain, loss = elevation_gain_loss(np.array([100.0, 110.0, 105.0, 120.0]))
    assert gain == 25.0 and loss == 5.0


# ── Elevation provider ───────────────────────────────────────────────────--
def test_elevation_cache_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    c = _ElevationCache(path)
    c.put(46.123456, 15.123456, 410.5)
    c.flush()
    assert _ElevationCache(path).get(46.123456, 15.123456) == 410.5


def test_default_provider_is_online():
    assert isinstance(get_default_provider(), OnlineElevationProvider)


# ── data_loader GPS/fuel helpers ─────────────────────────────────────────--
def test_heading_due_north_is_zero():
    h = _compute_heading(np.array([46.0, 46.001, 46.002]), np.array([15.0, 15.0, 15.0]))
    assert min(h[1] % 360, 360 - (h[1] % 360)) < 5


def test_align_speed_interpolates():
    df = pd.DataFrame({"seconds": [0.0, 10.0], "value": [0.0, 100.0]})
    assert np.allclose(_align_speed(df, np.array([0.0, 5.0, 10.0])), [0, 50, 100], atol=1e-3)


def test_extract_fuel_from_pid_and_fallback():
    used = pd.DataFrame({"pid": ["Fuel used"] * 3, "value": [1.0, 1.2, 1.5], "seconds": [0, 1, 2]})
    assert abs(_extract_fuel_l(used, 5.0) - 0.5) < 1e-9                 # cumulative → last-first
    none = pd.DataFrame({"pid": ["Vehicle speed"], "value": ["0"], "seconds": [0]})
    assert abs(_extract_fuel_l(none, 10.0) - 0.8) < 1e-9               # 8 L/100km fallback


def test_fuel_rate_from_instant_pid():
    df = pd.DataFrame({"pid": ["Calculated instant fuel rate"] * 3,
                       "value": [3.6, 3.6, 3.6], "seconds": [0.0, 1.0, 2.0]})
    rate = _fuel_rate_per_point(df, np.array([0.0, 1.0, 2.0]), total_fuel_l=0.002)
    assert np.allclose(rate, 0.001, atol=1e-6)                         # 3.6 L/h → 0.001 L/s


# ── Drive-path resolution (the "click a stored drive" fix) ───────────────--
def test_foreign_path_rerooted_to_local_repo():
    resolved = resolve_drive_path(os.path.join("nonexistent_root", "x", "data", "drive_data"))
    assert resolved == REPO_ROOT / "data" / "drive_data" and resolved.exists()


def test_unresolvable_path_returned_unchanged():
    foreign = os.path.join("nonexistent_root", "whatever", "x.csv")
    assert resolve_drive_path(foreign) == Path(foreign)


# ── Fuel records & savings ───────────────────────────────────────────────--
def _fake_drive(rate):
    """6-point track: a left turn (pts 1–2) and an uphill (pts 3–4)."""
    return SimpleNamespace(
        gps_file=r"data\drive_data\malik-toyota\d\d.csv", n_points=6,
        gps_timestamps=np.arange(6, dtype=float),
        gps_heading=np.array([0, 0, 30, 30, 30, 30], dtype=float),
        grade_pct=np.array([0, 0, 0, 4.0, 4.0, 0], dtype=float),
        turn_preds=np.array([0, 1, 1, 0, 0, 0], dtype=np.int32),
        hill_preds=np.array([0, 0, 0, 1, 1, 0], dtype=np.int32),
        fuel_rate_l_s=np.full(6, rate, dtype=float),
    )


def test_bucket_key():
    assert bucket_key({"kind": "turn", "direction": 1, "magnitude": 8}) == "left_turn_0-15"
    assert bucket_key({"kind": "hill", "direction": 2, "magnitude": 12}) == "downhill_10+"


def test_first_drive_sets_records_no_savings(tmp_path):
    d = _fake_drive(0.001)
    res = evaluate_drive(d, store_path=tmp_path / "rec.json")
    assert res["total_savings_l"] == 0.0
    assert set(np.unique(d.turn_perf)) <= {0, 1}          # only none / record (green)


def test_worse_drive_accrues_savings(tmp_path):
    store = tmp_path / "rec.json"
    evaluate_drive(_fake_drive(0.001), store_path=store)   # set records
    d = _fake_drive(0.003)                                 # 3× worse
    res = evaluate_drive(d, store_path=store)
    assert abs(res["total_savings_l"] - 0.004) < 1e-9      # 2 segs × (0.003-0.001)×1s
    assert 3 in np.unique(d.turn_perf)                     # red


def test_records_are_global(tmp_path):
    store = tmp_path / "rec.json"
    evaluate_drive(_fake_drive(0.001), store_path=store)
    other = _fake_drive(0.003)
    other.gps_file = r"data\drive_data\miha-hyundai\d\d.csv"   # different car/path
    assert evaluate_drive(other, store_path=store)["total_savings_l"] > 0.0
