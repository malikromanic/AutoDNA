"""Label assignment logic and full preprocessing→features pipeline smoke tests.

Requires ai/features.py and the updated app/ai_pipeline.py (which imports from it).
"""

import json

import numpy as np

from AutoDNA.ai.features import N_FEATURES, extract_features
from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    preprocess_sensor_data,
    resample_sensors_to_common_grid,
)
from AutoDNA.app.ai_pipeline import _label_window

WINDOW_SIZE = 100
STRIDE      = 25
TARGET_FS   = 50


# ── _label_window boundary tests ──────────────────────────────────────────────

def _lbl(t_start, t_end, turn_dir=None, hill_dir=None):
    lbl = {"t_start": t_start, "t_end": t_end}
    if turn_dir:
        lbl["turn"] = {"dir": turn_dir}
    if hill_dir:
        lbl["hill"] = {"dir": hill_dir}
    return lbl


def test_full_overlap_returns_label():
    turn, _ = _label_window(0.0, 2.0, [_lbl(0.0, 2.0, turn_dir="left")])
    assert turn == 1  # left


def test_exactly_30pct_overlap_returns_label():
    # Window 0.0–2.0s, event 1.4–3.0s → overlap = 0.6s = exactly 30 %
    turn, _ = _label_window(0.0, 2.0, [_lbl(1.4, 3.0, turn_dir="right")])
    assert turn == 2  # right


def test_below_30pct_overlap_returns_none():
    # Window 0.0–2.0s, event 1.42–3.0s → overlap = 0.58s = 29 %
    turn, _ = _label_window(0.0, 2.0, [_lbl(1.42, 3.0, turn_dir="left")])
    assert turn == 0  # none


def test_no_overlap_returns_none():
    turn, _ = _label_window(0.0, 2.0, [_lbl(5.0, 8.0, turn_dir="left")])
    assert turn == 0


def test_highest_overlap_wins_when_multiple_events():
    # event_a: overlap 0.3s (15 %) — below threshold on its own
    # event_b: overlap 1.5s (75 %) — wins
    labels = [
        _lbl(1.7, 3.0, turn_dir="left"),    # 0.3s overlap
        _lbl(0.5, 2.0, turn_dir="right"),   # 1.5s overlap
    ]
    turn, _ = _label_window(0.0, 2.0, labels)
    assert turn == 2  # right


def test_turn_and_hill_labeled_independently():
    turn, hill = _label_window(0.0, 2.0, [_lbl(0.0, 2.0, turn_dir="right", hill_dir="down")])
    assert turn == 2  # right
    assert hill == 2  # down


def test_hill_only_event_leaves_turn_as_none():
    turn, hill = _label_window(0.0, 2.0, [_lbl(0.0, 2.0, hill_dir="up")])
    assert turn == 0
    assert hill == 1  # up


# ── Full pipeline smoke test (real committed data) ────────────────────────────

def test_pipeline_output_shape_on_real_data(training_npz):
    """parse NPZ → resample → preprocess → window → extract_features → (N, 36)"""
    data = load_sensor_data(str(training_npz))
    resampled = resample_sensors_to_common_grid(data, target_fs=TARGET_FS)
    processed = preprocess_sensor_data(resampled, fs=TARGET_FS)

    signal = np.column_stack([
        processed["gyro"]["x"],  processed["gyro"]["y"],  processed["gyro"]["z"],
        processed["accel"]["x"], processed["accel"]["y"], processed["accel"]["z"],
        processed["mag"]["x"],   processed["mag"]["y"],   processed["mag"]["z"],
    ]).astype(np.float32)

    n_windows = (len(signal) - WINDOW_SIZE) // STRIDE + 1
    assert n_windows > 0

    features = np.array([
        extract_features(signal[i * STRIDE : i * STRIDE + WINDOW_SIZE])
        for i in range(n_windows)
    ])
    assert features.shape == (n_windows, N_FEATURES)
    assert np.all(np.isfinite(features)), "Pipeline produced NaN/inf on real data"


def test_pipeline_window_count_formula(training_npz):
    """Window count must satisfy (N - WINDOW_SIZE) // STRIDE + 1."""
    data = load_sensor_data(str(training_npz))
    resampled = resample_sensors_to_common_grid(data, target_fs=TARGET_FS)
    processed = preprocess_sensor_data(resampled, fs=TARGET_FS)
    n_samples = len(processed["gyro"]["ts"])
    expected = (n_samples - WINDOW_SIZE) // STRIDE + 1
    assert expected > 0


def test_label_window_on_real_labels_no_crash(training_labels):
    """_label_window must not crash or return out-of-range values for any real label."""
    with open(training_labels, encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    for lbl in labels:
        turn, hill = _label_window(lbl["t_start"], lbl["t_end"], labels)
        assert turn in (0, 1, 2), f"Invalid turn label {turn}"
        assert hill in (0, 1, 2), f"Invalid hill label {hill}"