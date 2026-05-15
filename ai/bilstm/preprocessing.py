"""Reusable preprocessing functions for sensor data stored in .npz files."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def load_sensor_data(filepath):
    """Load .npz sensor data into a common dictionary format."""
    sensors = {}

    with np.load(filepath, allow_pickle=True) as raw:
        #podpora za vec senzorjev iz istega posnetka
        for sensor_name in raw.files:
            data = raw[sensor_name]

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
            #enoten format ts/x/y/z da je pipeline isti za celo ekipo

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

    #glajenje signala za zmanjsanje suma pred filtriranjem
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

    #low pass za odstranitev hitrega noise dela
    b, a = butter(order, cutoff / nyquist, btype="low")
    return filtfilt(b, a, signal)


def normalize_signal(signal):
    """Normalize signal to range [-1, 1]."""
    signal = np.asarray(signal, dtype=float)

    max_value = np.max(np.abs(signal))
    if max_value == 0:
        return signal.copy()

    #normalizacija amplitude za bolj stabilen ai input
    return signal / max_value


def preprocess_sensor_data(sensors, fs, cutoff=5.0, window_size=5, filter_order=4):
    """Apply smoothing, low-pass filtering and normalization to all sensors."""
    processed = {}
    #zascita da cutoff ostane pod nyquist mejo
    effective_cutoff = min(float(cutoff), 0.45 * float(fs))

    if effective_cutoff <= 0:
        effective_cutoff = float(cutoff)

    #isti preprocessing koraki za vse senzorje
    for sensor_name, sensor_data in sensors.items():
        processed[sensor_name] = {"ts": sensor_data["ts"].copy()}

        #podpora za x/y/z kanale vsakega senzorja
        for axis in ["x", "y", "z"]:
            signal = sensor_data[axis]
            #pipeline: glajenje -> low pass -> normalizacija
            signal = smooth_signal(signal, window_size=window_size)
            signal = lowpass_filter(signal, fs, effective_cutoff, order=filter_order)
            processed[sensor_name][axis] = normalize_signal(signal)

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

    #raw vs processed vizualizacija za dokaz efekta preprocessinga
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
        #ocena fs ostane pravilna tudi ce so timestampi v ms
        step = step / 1000.0

    return 1.0 / step


def save_preprocessed_data(processed, filepath):
    """Save processed sensor data to .npz with columns: ts, x, y, z."""
    arrays = {}

    #shrani preprocessirane signale za naslednji ai korak
    for sensor_name, sensor_data in processed.items():
        arrays[sensor_name] = np.column_stack(
            [sensor_data["ts"], sensor_data["x"], sensor_data["y"], sensor_data["z"]]
        ).astype(np.float32)

    np.savez(filepath, **arrays)
    #todo dodat opcijski resampling vseh senzorjev na skupni fs