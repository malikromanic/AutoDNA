"""Unit tests for the per-segment fuel records / savings engine (network-free)."""

from types import SimpleNamespace

import numpy as np

from AutoDNA.app.segment_records import (
    evaluate_drive, vehicle_key, bucket_key, _turn_bin, _hill_bin,
)


# ── key derivation ───────────────────────────────────────────────────────────
def test_vehicle_key_from_drive_data_path():
    p = r"C:\x\data\drive_data\malik-toyota\center-salek\center-salek.csv"
    assert vehicle_key(p) == "malik-toyota"


def test_vehicle_key_fallback():
    # no drive_data marker → falls back to a folder name (never crashes)
    assert vehicle_key("/tmp/foo/bar/drive.csv")


def test_turn_bins():
    assert _turn_bin(5) == "0-15"
    assert _turn_bin(20) == "15-30"
    assert _turn_bin(88) == "75-90"
    assert _turn_bin(95) == "90+"


def test_hill_bins():
    assert _hill_bin(1.0) == "0-2"
    assert _hill_bin(3.0) == "2-5"
    assert _hill_bin(12.0) == "10+"


def test_bucket_key():
    assert bucket_key({"kind": "turn", "direction": 1, "magnitude": 8}) == "left_turn_0-15"
    assert bucket_key({"kind": "turn", "direction": 2, "magnitude": 95}) == "right_turn_90+"
    assert bucket_key({"kind": "hill", "direction": 1, "magnitude": 3}) == "uphill_2-5"
    assert bucket_key({"kind": "hill", "direction": 2, "magnitude": 12}) == "downhill_10+"


# ── evaluate_drive ───────────────────────────────────────────────────────────
def _fake_drive(rate):
    """6-point track: a left turn (pts 1-2) and an uphill (pts 3-4)."""
    return SimpleNamespace(
        gps_file=r"data\drive_data\malik-toyota\d\d.csv",
        n_points=6,
        gps_timestamps=np.arange(6, dtype=float),                  # 1 s apart
        gps_heading=np.array([0, 0, 30, 30, 30, 30], dtype=float),  # 30° left turn
        grade_pct=np.array([0, 0, 0, 4.0, 4.0, 0], dtype=float),    # +4% uphill
        turn_preds=np.array([0, 1, 1, 0, 0, 0], dtype=np.int32),
        hill_preds=np.array([0, 0, 0, 1, 1, 0], dtype=np.int32),
        fuel_rate_l_s=np.full(6, rate, dtype=float),
    )


def test_first_drive_sets_records_no_savings(tmp_path):
    store = tmp_path / "rec.json"
    d = _fake_drive(0.001)
    res = evaluate_drive(d, store_path=store)
    assert res["total_savings_l"] == 0.0
    assert d.total_savings_l == 0.0
    assert set(np.unique(d.turn_perf)) <= {0, 1}   # only none / record (green)
    assert d.vehicle == "all"                      # global records for now
    assert store.exists()


def test_worse_drive_accrues_savings_and_red(tmp_path):
    store = tmp_path / "rec.json"
    evaluate_drive(_fake_drive(0.001), store_path=store)          # set records
    d = _fake_drive(0.003)                                        # 3x worse
    res = evaluate_drive(d, store_path=store)
    # turn seg (1 s) + hill seg (1 s), each (0.003-0.001)*1 = 0.002 L
    assert abs(res["total_savings_l"] - 0.004) < 1e-9
    assert 3 in np.unique(d.turn_perf)   # red (>25% over record)
    assert 3 in np.unique(d.hill_perf)


def test_better_drive_updates_record(tmp_path):
    store = tmp_path / "rec.json"
    evaluate_drive(_fake_drive(0.002), store_path=store)
    d = _fake_drive(0.001)                                        # new best
    res = evaluate_drive(d, store_path=store)
    assert res["total_savings_l"] == 0.0
    assert set(np.unique(d.turn_perf)) <= {0, 1}


def test_records_are_global(tmp_path):
    store = tmp_path / "rec.json"
    evaluate_drive(_fake_drive(0.001), store_path=store)          # sets global records
    other = _fake_drive(0.003)
    other.gps_file = r"data\drive_data\miha-hyundai\d\d.csv"      # different car/path
    res = evaluate_drive(other, store_path=store)
    # records are global now → the second drive is compared to the first's records
    assert res["vehicle"] == "all"
    assert res["total_savings_l"] > 0.0
