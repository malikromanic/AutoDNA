import json

import numpy as np
import pytest

from AutoDNA.ai.features import N_FEATURES, extract_features
from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    preprocess_sensor_data,
    resample_sensors_to_common_grid,
)

WINDOW_SIZE = 100
STRIDE      = 25


def _window(gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            accel_x=0.0, accel_y=0.0, accel_z=0.0,
            mag_x=0.0, mag_y=0.0, mag_z=0.0, n=100):
    """Constant (n, 9) window for testing individual features."""
    w = np.zeros((n, 9), dtype=np.float32)
    w[:, 0] = gyro_x
    w[:, 1] = gyro_y
    w[:, 2] = gyro_z
    w[:, 3] = accel_x
    w[:, 4] = accel_y
    w[:, 5] = accel_z
    w[:, 6] = mag_x
    w[:, 7] = mag_y
    w[:, 8] = mag_z
    return w


# ── Shape and dtype ───────────────────────────────────────────────────────────

def test_output_shape_matches_n_features():
    w = np.random.default_rng(0).random((100, 9)).astype(np.float32)
    assert extract_features(w).shape == (N_FEATURES,)


def test_output_dtype_is_float32():
    w = np.random.default_rng(0).random((100, 9)).astype(np.float32)
    assert extract_features(w).dtype == np.float32


# ── Specific feature values ───────────────────────────────────────────────────

def test_constant_gyro_z_max_min_mean_std():
    feats = extract_features(_window(gyro_z=2.5))
    assert feats[0] == pytest.approx(2.5, rel=1e-5)   # max
    assert feats[1] == pytest.approx(2.5, rel=1e-5)   # min
    assert feats[2] == pytest.approx(2.5, rel=1e-5)   # mean
    assert feats[3] == pytest.approx(0.0, abs=1e-5)   # std of constant signal = 0


def test_positive_fraction_all_above_threshold():
    # All gz = 1.0 > 0.1 → fraction should be 1.0
    assert extract_features(_window(gyro_z=1.0))[6] == pytest.approx(1.0, rel=1e-5)


def test_negative_fraction_all_below_threshold():
    # All gz = -1.0 < -0.1 → fraction should be 1.0
    assert extract_features(_window(gyro_z=-1.0))[7] == pytest.approx(1.0, rel=1e-5)


def test_zero_gyro_z_has_zero_fractions():
    feats = extract_features(_window(gyro_z=0.0))
    assert feats[6] == pytest.approx(0.0, abs=1e-5)   # none above 0.1
    assert feats[7] == pytest.approx(0.0, abs=1e-5)   # none below -0.1


# ── NaN safety ────────────────────────────────────────────────────────────────

def test_constant_gyro_z_does_not_produce_nan():
    # std(gz) == 0 triggers the explicit 0.0 branch in the correlation guard.
    # Without that guard np.corrcoef returns NaN.
    feats = extract_features(_window(gyro_z=1.0, accel_y=1.0))
    assert not np.any(np.isnan(feats))


def test_all_zeros_window_no_nan():
    assert not np.any(np.isnan(extract_features(np.zeros((100, 9), dtype=np.float32))))


# ── Real data: the assumption the classifier is built on ─────────────────────

def _load_signal_and_ts(npz_path):
    data = load_sensor_data(str(npz_path))
    resampled = resample_sensors_to_common_grid(data, target_fs=50)
    processed = preprocess_sensor_data(resampled, fs=50)
    signal = np.column_stack([
        processed["gyro"]["x"],  processed["gyro"]["y"],  processed["gyro"]["z"],
        processed["accel"]["x"], processed["accel"]["y"], processed["accel"]["z"],
        processed["mag"]["x"],   processed["mag"]["y"],   processed["mag"]["z"],
    ]).astype(np.float32)
    return signal, processed["gyro"]["ts"]


def test_feature_vectors_all_finite_on_real_data(training_npz):
    signal, ts = _load_signal_and_ts(training_npz)
    n = (len(signal) - WINDOW_SIZE) // STRIDE + 1
    features = np.array([extract_features(signal[i*STRIDE : i*STRIDE+WINDOW_SIZE]) for i in range(n)])
    assert np.all(np.isfinite(features)), "Feature vectors contain NaN or inf on real recording"


def test_gyro_z_std_higher_in_turn_windows(training_npz, training_labels):
    """Turn windows must have higher gyro_z variance than non-turn windows.

    This tests the core signal assumption behind the XGBoost turn classifier.
    If preprocessing ever corrupts or inverts the gyro_z signal, this fails
    before bad predictions reach the model.
    """
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
        gz_std = float(np.std(signal[i0:i1, 2]))   # column 2 = gyro_z
        (turn_stds if is_turn else none_stds).append(gz_std)

    assert turn_stds, "No turn-labeled windows found — check label file"
    assert none_stds, "No non-turn windows found"

    ratio = np.mean(turn_stds) / np.mean(none_stds)
    assert ratio > 1.2, (
        f"Turn windows gyro_z std is {ratio:.2f}x non-turn (expected >1.2x). "
        "Preprocessing may have destroyed the turn signal."
    )