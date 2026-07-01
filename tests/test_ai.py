# ============================================================================
# IMU preprocessing + AI feature/pipeline tests — core behaviors only.
# ============================================================================
import json

import numpy as np
import pytest

from AutoDNA.ai.features import N_FEATURES, extract_features
from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    lowpass_filter,
    preprocess_sensor_data,
    resample_sensors_to_common_grid,
    smooth_signal,
)
from AutoDNA.app.ai_pipeline import _label_window

WINDOW_SIZE = 100
STRIDE = 25


# ── smooth_signal ────────────────────────────────────────────────────────--
def test_smooth_window_1_is_identity():
    sig = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    np.testing.assert_array_equal(smooth_signal(sig, window_size=1), sig)


def test_smooth_reduces_spike():
    sig = np.zeros(20)
    sig[10] = 100.0
    assert smooth_signal(sig, window_size=5)[10] < 100.0


# ── lowpass_filter ───────────────────────────────────────────────────────--
def test_lowpass_raises_when_cutoff_exceeds_nyquist():
    with pytest.raises(ValueError):
        lowpass_filter(np.ones(100), fs=50, cutoff=25.0)


def test_lowpass_attenuates_high_frequency():
    fs = 50.0
    t = np.linspace(0, 4, int(fs * 4), endpoint=False)
    low = np.sin(2 * np.pi * 1.0 * t)
    high = np.sin(2 * np.pi * 20.0 * t)
    filtered = lowpass_filter(low + high, fs=fs, cutoff=5.0)
    # interior only — filtfilt has edge transients at both ends
    np.testing.assert_allclose(filtered[20:-20], low[20:-20], atol=0.1)


def test_lowpass_preserves_dc():
    result = lowpass_filter(np.ones(200) * 5.0, fs=50, cutoff=5.0)
    np.testing.assert_allclose(result[50:], 5.0, atol=0.01)


# ── resample_sensors_to_common_grid ─────────────────────────────────────--
def _make_sensor(ts, amp=1.0):
    return {"ts": np.array(ts, dtype=float),
            "x": np.ones(len(ts)) * amp, "y": np.ones(len(ts)) * amp, "z": np.ones(len(ts)) * amp}


def test_resample_all_sensors_share_timestamps():
    sensors = {"gyro": _make_sensor(np.linspace(0, 2, 200)),
               "accel": _make_sensor(np.linspace(0, 2, 50)),
               "mag": _make_sensor(np.linspace(0, 2, 20))}
    result = resample_sensors_to_common_grid(sensors, target_fs=50)
    np.testing.assert_array_equal(result["gyro"]["ts"], result["accel"]["ts"])
    np.testing.assert_array_equal(result["gyro"]["ts"], result["mag"]["ts"])


def test_resample_converts_millisecond_timestamps():
    # ~10 ms apart → detected as ms, converted to s (≈2 s, not ~2000 s)
    result = resample_sensors_to_common_grid({"gyro": _make_sensor(np.linspace(0, 2000, 200))}, target_fs=50)
    assert result["gyro"]["ts"][-1] < 10.0


# ── preprocess_sensor_data: axis-ratio invariant (regression guard) ──────--
def test_preprocess_does_not_normalize_axes():
    n, fs = 500, 50
    t = np.arange(n) / fs
    sensors = {"gyro": {"ts": t,
                        "x": np.sin(2 * np.pi * t) * 10.0,
                        "y": np.sin(2 * np.pi * t) * 1.0,
                        "z": np.sin(2 * np.pi * t) * 0.1}}
    r = preprocess_sensor_data(sensors, fs=fs)
    assert np.std(r["gyro"]["x"]) > np.std(r["gyro"]["y"]) * 5
    assert np.std(r["gyro"]["y"]) > np.std(r["gyro"]["z"]) * 5


# ── extract_features (IMU feature vector) ────────────────────────────────--
def _window(gyro_z=0.0, accel_y=0.0, n=100):
    w = np.zeros((n, 9), dtype=np.float32)
    w[:, 2] = gyro_z
    w[:, 4] = accel_y
    return w


def test_feature_output_shape_and_dtype():
    feats = extract_features(np.random.default_rng(0).random((100, 9)).astype(np.float32))
    assert feats.shape == (N_FEATURES,)
    assert feats.dtype == np.float32


def test_constant_gyro_z_stats():
    feats = extract_features(_window(gyro_z=2.5))
    assert feats[0] == pytest.approx(2.5, rel=1e-5)   # max
    assert feats[1] == pytest.approx(2.5, rel=1e-5)   # min
    assert feats[3] == pytest.approx(0.0, abs=1e-5)   # std of a constant = 0


def test_features_no_nan_on_constant_and_zero_windows():
    assert not np.any(np.isnan(extract_features(_window(gyro_z=1.0, accel_y=1.0))))
    assert not np.any(np.isnan(extract_features(np.zeros((100, 9), dtype=np.float32))))


# ── _label_window (window → turn/hill label) ─────────────────────────────--
def _lbl(t_start, t_end, turn_dir=None, hill_dir=None):
    lbl = {"t_start": t_start, "t_end": t_end}
    if turn_dir:
        lbl["turn"] = {"dir": turn_dir}
    if hill_dir:
        lbl["hill"] = {"dir": hill_dir}
    return lbl


def test_label_full_overlap():
    assert _label_window(0.0, 2.0, [_lbl(0.0, 2.0, turn_dir="left")])[0] == 1


def test_label_below_30pct_overlap_is_none():
    # window 0–2 s, event 1.42–3.0 → 0.58 s overlap = 29 % < 30 %
    assert _label_window(0.0, 2.0, [_lbl(1.42, 3.0, turn_dir="left")])[0] == 0


def test_label_highest_overlap_wins():
    labels = [_lbl(1.7, 3.0, turn_dir="left"), _lbl(0.5, 2.0, turn_dir="right")]
    assert _label_window(0.0, 2.0, labels)[0] == 2


def test_label_turn_and_hill_independent():
    turn, hill = _label_window(0.0, 2.0, [_lbl(0.0, 2.0, turn_dir="right", hill_dir="down")])
    assert turn == 2 and hill == 2


# ── Real committed data (auto-skips if the .npz isn't present) ───────────--
def _load_signal_and_ts(npz_path):
    processed = preprocess_sensor_data(
        resample_sensors_to_common_grid(load_sensor_data(str(npz_path)), target_fs=50), fs=50)
    signal = np.column_stack([
        processed["gyro"]["x"], processed["gyro"]["y"], processed["gyro"]["z"],
        processed["accel"]["x"], processed["accel"]["y"], processed["accel"]["z"],
        processed["mag"]["x"], processed["mag"]["y"], processed["mag"]["z"],
    ]).astype(np.float32)
    return signal, processed["gyro"]["ts"]


def test_features_finite_on_real_data(training_npz):
    signal, ts = _load_signal_and_ts(training_npz)
    n = (len(signal) - WINDOW_SIZE) // STRIDE + 1
    features = np.array([extract_features(signal[i * STRIDE:i * STRIDE + WINDOW_SIZE]) for i in range(n)])
    assert np.all(np.isfinite(features))


def test_gyro_z_std_higher_in_turn_windows(training_npz, training_labels):
    """Core signal assumption behind the turn classifier — fails if preprocessing
    ever corrupts the gyro_z turn signal."""
    signal, ts = _load_signal_and_ts(training_npz)
    with open(training_labels, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    n = (len(signal) - WINDOW_SIZE) // STRIDE + 1
    turn_stds, none_stds = [], []
    for i in range(n):
        i0, i1 = i * STRIDE, i * STRIDE + WINDOW_SIZE
        t0, t1 = float(ts[i0]), float(ts[i1 - 1])
        dur = t1 - t0
        is_turn = any(
            min(t1, lbl["t_end"]) - max(t0, lbl["t_start"]) >= 0.3 * dur and lbl.get("turn")
            for lbl in labels
        )
        (turn_stds if is_turn else none_stds).append(float(np.std(signal[i0:i1, 2])))
    assert turn_stds and none_stds
    assert np.mean(turn_stds) / np.mean(none_stds) > 1.2
