"""Reusable preprocessing functions for sensor data stored in .npz files."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def load_sensor_data(filepath):
    """Load .npz sensor data into a common dictionary format."""
    sensors = {}

    with np.load(filepath, allow_pickle=True) as raw:
        for sensor_name in raw.files:
            data = raw[sensor_name]
            
            # divide accel x/y/z by 16 to correct left-shift alignment
            if sensor_name == "accel":
                data = data.copy()
                data[:, 1:] = data[:, 1:] / 16.0
                
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item() if data.shape == () else data[0]

            if isinstance(data, dict):
                ts = np.asarray(data["ts"], dtype=float)
                x = np.asarray(data["x"], dtype=float)
                y = np.asarray(data["y"], dtype=float)
                z = np.asarray(data["z"], dtype=float)
            else:
                data = np.asarray(data, dtype=float)
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError(
                        f"{sensor_name} must have columns: timestamp, x, y, z"
                    )
                    
                order = np.argsort(data[:, 0])
                data = data[order]
                ts = data[:, 0]
                x = data[:, 1]
                y = data[:, 2]
                z = data[:, 3]

            sensors[sensor_name] = {
                "ts": ts.astype(float),
                "x": x.astype(float),
                "y": y.astype(float),
                "z": z.astype(float),
            }

    return sensors


def smooth_signal(signal, window_size=5):
    """Smooth signal with a moving average."""
    signal = np.asarray(signal, dtype=float)

    if window_size <= 1 or len(signal) < 2:
        return signal.copy()

    window_size = int(window_size)
    kernel = np.ones(window_size, dtype=float) / window_size

    left_pad = window_size // 2
    right_pad = window_size - 1 - left_pad
    padded = np.pad(signal, (left_pad, right_pad), mode="edge")

    return np.convolve(padded, kernel, mode="valid")


def lowpass_filter(signal, fs, cutoff, order=4):
    """Apply a Butterworth low-pass filter."""
    signal = np.asarray(signal, dtype=float)

    if fs <= 0:
        raise ValueError("fs must be positive")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    nyquist = fs / 2.0
    if cutoff >= nyquist:
        raise ValueError("cutoff must be lower than fs / 2")

    min_len = 3 * (order + 1)
    if len(signal) <= min_len:
        return signal.copy()

    b, a = butter(order, cutoff / nyquist, btype="low")
    return filtfilt(b, a, signal)


def normalize_signal(signal):
    """Normalize signal to range [-1, 1]."""
    signal = np.asarray(signal, dtype=float)

    max_value = np.max(np.abs(signal))
    if max_value == 0:
        return signal.copy()

    return signal / max_value


def preprocess_sensor_data(sensors, fs, cutoff=5.0, window_size=5, filter_order=4):
    """Apply smoothing and low-pass filtering to all sensors (WITHOUT independent axis normalization)."""
    processed = {}
    effective_cutoff = min(float(cutoff), 0.45 * float(fs))

    if effective_cutoff <= 0:
        effective_cutoff = float(cutoff)

    for sensor_name, sensor_data in sensors.items():
        processed[sensor_name] = {"ts": sensor_data["ts"].copy()}

        for axis in ["x", "y", "z"]:
            signal = sensor_data[axis]
            # Pipeline: glajenje -> low pass filter
            signal = smooth_signal(signal, window_size=window_size)
            signal = lowpass_filter(signal, fs, effective_cutoff, order=filter_order)
            
            # POPRAVEK: Tukaj NE izvajamo več normalize_signal(signal), 
            # saj bi s tem uničili medsebojno razmerje osi (vektor gravitacije).
            processed[sensor_name][axis] = signal

    return processed


def plot_before_after(ts, raw, processed, title):
    """Plot raw and processed signal."""
    ts = np.asarray(ts, dtype=float)
    ts = ts - ts[0]
    positive_diffs = np.diff(np.asarray(ts, dtype=float))
    positive_diffs = positive_diffs[positive_diffs > 0]
    if len(positive_diffs) > 0 and np.median(positive_diffs) > 10:
        ts = ts / 1000.0

    raw_plot = normalize_signal(raw)

    plt.figure(figsize=(12, 4))
    plt.plot(ts, raw_plot, label="Raw", alpha=0.5)
    plt.plot(ts, processed, label="Processed", linewidth=2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def estimate_sampling_rate(ts):
    """Estimate sampling rate from timestamps."""
    ts = np.asarray(ts, dtype=float)
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 1.0

    step = float(np.median(diffs))
    if step <= 0:
        return 1.0

    if step > 10:
        step = step / 1000.0

    return 1.0 / step


def save_preprocessed_data(processed, filepath):
    """Save processed sensor data to .npz with columns: ts, x, y, z."""
    arrays = {}

    for sensor_name, sensor_data in processed.items():
        arrays[sensor_name] = np.column_stack(
            [sensor_data["ts"], sensor_data["x"], sensor_data["y"], sensor_data["z"]]
        ).astype(np.float32)

    np.savez(filepath, **arrays)
    
    
def _timestamps_to_seconds(ts):
    """Convert timestamps to seconds"""
    ts = np.asarray(ts, dtype=float)

    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    
    return ts / 1000.0

    """if len(diffs) > 0 and np.median(diffs) > 10:
        return ts / 1000.0

    return ts"""


def resample_sensors_to_common_grid(sensors, target_fs=None):
    """Resample all sensors to the same timestamp grid."""
    converted = {}

    for sensor_name, sensor_data in sensors.items():
        converted[sensor_name] = {
            "ts": _timestamps_to_seconds(sensor_data["ts"]),
            "x": np.asarray(sensor_data["x"], dtype=float),
            "y": np.asarray(sensor_data["y"], dtype=float),
            "z": np.asarray(sensor_data["z"], dtype=float),
        }

    if target_fs is None:
        target_fs = min(
            estimate_sampling_rate(sensor_data["ts"])
            for sensor_data in converted.values()
        )

    start = max(sensor_data["ts"][0] for sensor_data in converted.values())
    end = min(sensor_data["ts"][-1] for sensor_data in converted.values())

    common_ts = np.arange(start, end, 1.0 / target_fs)

    resampled = {}

    for sensor_name, sensor_data in converted.items():
        resampled[sensor_name] = {
            "ts": common_ts - common_ts[0],
        }

        for axis in ["x", "y", "z"]:
            resampled[sensor_name][axis] = np.interp(
                common_ts,
                sensor_data["ts"],
                sensor_data[axis],
            )

    return resampled