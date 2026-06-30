import numpy as np
import pytest

from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    lowpass_filter,
    preprocess_sensor_data,
    resample_sensors_to_common_grid,
    smooth_signal,
)


# ── smooth_signal ─────────────────────────────────────────────────────────────

def test_smooth_flat_signal_unchanged():
    sig = np.ones(50) * 3.7
    np.testing.assert_allclose(smooth_signal(sig, window_size=5), sig, rtol=1e-5)


def test_smooth_window_1_is_identity():
    sig = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    np.testing.assert_array_equal(smooth_signal(sig, window_size=1), sig)


def test_smooth_known_output_values():
    # [0, 0, 6, 0, 0] with window=3: each output is average of 3 neighbours
    sig = np.array([0.0, 0.0, 6.0, 0.0, 0.0])
    result = smooth_signal(sig, window_size=3)
    assert result[2] == pytest.approx(2.0, rel=1e-5)
    assert result[0] < 6.0


def test_smooth_reduces_spike():
    sig = np.zeros(20)
    sig[10] = 100.0
    assert smooth_signal(sig, window_size=5)[10] < 100.0


def test_smooth_output_length_matches_input():
    sig = np.random.default_rng(0).random(100)
    assert len(smooth_signal(sig, window_size=7)) == len(sig)


# ── lowpass_filter ────────────────────────────────────────────────────────────

def test_lowpass_raises_on_nonpositive_fs():
    with pytest.raises(ValueError):
        lowpass_filter(np.ones(100), fs=0, cutoff=5.0)


def test_lowpass_raises_on_nonpositive_cutoff():
    with pytest.raises(ValueError):
        lowpass_filter(np.ones(100), fs=50, cutoff=0.0)


def test_lowpass_raises_when_cutoff_exceeds_nyquist():
    with pytest.raises(ValueError):
        lowpass_filter(np.ones(100), fs=50, cutoff=25.0)


def test_lowpass_attenuates_high_frequency():
    fs = 50.0
    t = np.linspace(0, 4, int(fs * 4), endpoint=False)
    low  = np.sin(2 * np.pi * 1.0  * t)   # 1 Hz — passes through
    high = np.sin(2 * np.pi * 20.0 * t)   # 20 Hz — blocked
    filtered = lowpass_filter(low + high, fs=fs, cutoff=5.0)
    np.testing.assert_allclose(filtered, low, atol=0.1)


def test_lowpass_preserves_dc():
    sig = np.ones(200) * 5.0
    result = lowpass_filter(sig, fs=50, cutoff=5.0)
    # Skip the filter transient at the start
    np.testing.assert_allclose(result[50:], 5.0, atol=0.01)


# ── resample_sensors_to_common_grid ──────────────────────────────────────────

def _make_sensor(ts, amp=1.0):
    return {
        "ts": np.array(ts, dtype=float),
        "x":  np.ones(len(ts)) * amp,
        "y":  np.ones(len(ts)) * amp,
        "z":  np.ones(len(ts)) * amp,
    }


def test_resample_all_sensors_share_same_timestamps():
    sensors = {
        "gyro":  _make_sensor(np.linspace(0, 2, 200)),
        "accel": _make_sensor(np.linspace(0, 2, 50)),
        "mag":   _make_sensor(np.linspace(0, 2, 20)),
    }
    result = resample_sensors_to_common_grid(sensors, target_fs=50)
    np.testing.assert_array_equal(result["gyro"]["ts"], result["accel"]["ts"])
    np.testing.assert_array_equal(result["gyro"]["ts"], result["mag"]["ts"])


def test_resample_timestamps_start_at_zero():
    sensors = {"gyro": _make_sensor(np.linspace(0, 3, 300))}
    result = resample_sensors_to_common_grid(sensors, target_fs=50)
    assert result["gyro"]["ts"][0] == pytest.approx(0.0, abs=1e-6)


def test_resample_converts_millisecond_timestamps():
    # Timestamps spaced ~10 ms apart — detected as ms, converted to s.
    # Duration should be ~2 s after conversion, not ~2000 s.
    sensors = {"gyro": _make_sensor(np.linspace(0, 2000, 200))}
    result = resample_sensors_to_common_grid(sensors, target_fs=50)
    duration = result["gyro"]["ts"][-1]
    assert duration < 10.0, f"Duration {duration:.1f}s — ms→s conversion likely failed"


def test_resample_flat_signal_stays_flat():
    sensors = {"gyro": _make_sensor(np.linspace(0, 2, 200), amp=5.0)}
    result = resample_sensors_to_common_grid(sensors, target_fs=50)
    np.testing.assert_allclose(result["gyro"]["x"], 5.0, atol=1e-5)


# ── preprocess_sensor_data ────────────────────────────────────────────────────

def test_preprocess_does_not_normalize_axes():
    # Axes must NOT be independently normalised — the gravity vector
    # relationship between axes must survive preprocessing.
    # This invariant was explicitly fixed after a past regression (see source comment).
    n, fs = 500, 50
    t = np.arange(n) / fs
    sensors = {
        "gyro": {
            "ts": t,
            "x": np.sin(2 * np.pi * t) * 10.0,
            "y": np.sin(2 * np.pi * t) * 1.0,
            "z": np.sin(2 * np.pi * t) * 0.1,
        }
    }
    result = preprocess_sensor_data(sensors, fs=fs)
    std_x = np.std(result["gyro"]["x"])
    std_y = np.std(result["gyro"]["y"])
    std_z = np.std(result["gyro"]["z"])
    assert std_x > std_y * 5, "Axis ratios not preserved — per-axis normalisation may have been re-introduced"
    assert std_y > std_z * 5, "Axis ratios not preserved — per-axis normalisation may have been re-introduced"


def test_preprocess_timestamps_unchanged():
    ts = np.arange(500) / 50
    rng = np.random.default_rng(0)
    sensors = {"gyro": {"ts": ts, "x": rng.random(500), "y": rng.random(500), "z": rng.random(500)}}
    result = preprocess_sensor_data(sensors, fs=50)
    np.testing.assert_array_equal(result["gyro"]["ts"], ts)


# ── Real data tests (committed training data — runs in CI) ────────────────────

def test_real_npz_resamples_to_50hz(training_npz):
    data = load_sensor_data(str(training_npz))
    resampled = resample_sensors_to_common_grid(data, target_fs=50)
    step = np.median(np.diff(resampled["gyro"]["ts"]))
    assert step == pytest.approx(0.020, abs=0.001), \
        f"Resampled step {step:.4f}s — expected 0.020s (50 Hz)"


def test_real_npz_duration_retained_after_resampling(training_npz):
    data = load_sensor_data(str(training_npz))
    ts_raw = data["gyro"]["ts"]
    raw_diffs = np.diff(ts_raw)
    raw_diffs = raw_diffs[raw_diffs > 0]
    dur_raw = (ts_raw[-1] - ts_raw[0]) / 1000.0 if np.median(raw_diffs) > 10 else (ts_raw[-1] - ts_raw[0])

    resampled = resample_sensors_to_common_grid(data, target_fs=50)
    dur_resampled = resampled["gyro"]["ts"][-1]

    loss_pct = abs(dur_raw - dur_resampled) / dur_raw * 100
    assert loss_pct < 2.0, f"Duration loss {loss_pct:.1f}% after resampling — expected < 2%"


def test_real_npz_all_finite_after_preprocessing(training_npz):
    data = load_sensor_data(str(training_npz))
    resampled = resample_sensors_to_common_grid(data, target_fs=50)
    processed = preprocess_sensor_data(resampled, fs=50)
    for sensor, axes in processed.items():
        for axis, values in axes.items():
            if axis == "ts":
                continue
            assert np.all(np.isfinite(values)), \
                f"{sensor}.{axis} contains NaN or inf after preprocessing"


def test_real_npz_gyro_has_nonzero_variance(training_npz):
    data = load_sensor_data(str(training_npz))
    resampled = resample_sensors_to_common_grid(data, target_fs=50)
    processed = preprocess_sensor_data(resampled, fs=50)
    assert np.std(processed["gyro"]["z"]) > 0.0, \
        "Gyro Z std is zero — preprocessing may have killed the signal"