"""Reusable spectrogram conversion functions for preprocessed sensor data."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def load_sensor_data(filepath):
    """Load .npz sensor data into a common dictionary format."""
    sensors = {}

    with np.load(filepath, allow_pickle=True) as raw:
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

    return sensors


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
        #pretvorba ms v s ce so timestampi v milisekundah
        step = step / 1000.0

    return 1.0 / step


def compute_stft_spectrogram(signal, fs, nperseg, noverlap, nfft):
    """Compute STFT spectrogram in decibels (dB)."""
    signal = np.asarray(signal, dtype=float)

    #stft pretvorba 1d signala v spektrogram
    #segmentacija preko okna in overlap parametrov
    frequencies, times, zxx = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        nfft=int(nfft),
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(zxx)
    #pretvorba v db poudari razlike med dogodki
    spectrogram_db = 20.0 * np.log10(magnitude + 1e-8)

    return frequencies, times, spectrogram_db


def normalize_spectrogram_to_uint8(spectrogram):
    """Normalize spectrogram values to 8-bit range [0, 255]."""
    spectrogram = np.asarray(spectrogram, dtype=float)
    finite_mask = np.isfinite(spectrogram)

    if not np.any(finite_mask):
        return np.zeros_like(spectrogram, dtype=np.uint8)

    min_value = float(np.min(spectrogram[finite_mask]))
    max_value = float(np.max(spectrogram[finite_mask]))

    if max_value <= min_value:
        return np.zeros_like(spectrogram, dtype=np.uint8)

    normalized = (spectrogram - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite_mask] = 0.0

    #normalizacija amplitude na [0,255] za vhod v cnn
    return (normalized * 255.0).astype(np.uint8)


def create_rgb_spectrogram(sensor_data, fs, nperseg, noverlap, nfft):
    """Create one RGB spectrogram image from x/y/z channels."""
    required_axes = ["x", "y", "z"]
    for axis in required_axes:
        if axis not in sensor_data:
            raise ValueError(f"Missing axis '{axis}' in sensor data")

    spectrograms = {}
    frequencies = None
    times = None

    for axis in required_axes:
        frequencies, times, spectrogram = compute_stft_spectrogram(
            sensor_data[axis], fs, nperseg, noverlap, nfft
        )
        spectrograms[axis] = spectrogram

    #vsi kanali dobijo isto casovno in frekvencno mrezo
    #rgb pakiranje osi: x->r, y->g, z->b
    r = normalize_spectrogram_to_uint8(spectrograms["x"])
    g = normalize_spectrogram_to_uint8(spectrograms["y"])
    b = normalize_spectrogram_to_uint8(spectrograms["z"])

    rgb_image = np.stack([r, g, b], axis=-1)
    return frequencies, times, spectrograms, rgb_image


def save_spectrogram_image(image, filepath):
    """Save spectrogram image to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    image = np.asarray(image)
    #moznost shranjevanja rezultatov kot slike
    if image.ndim == 2:
        plt.imsave(filepath, image, cmap="viridis")
    else:
        plt.imsave(filepath, image)


def plot_spectrogram(frequencies, times, spectrogram, title):
    """Plot a single spectrogram with proper axis labels."""
    plt.figure(figsize=(12, 5))
    mesh = plt.pcolormesh(times, frequencies, spectrogram, shading="gouraud", cmap="viridis")
    plt.title(title)
    #pravilno oznacene osi za obrambo in analizo
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    cbar = plt.colorbar(mesh)
    cbar.set_label("Amplitude (dB)")
    plt.tight_layout()
    plt.show()


def convert_all_sensors_to_spectrograms(sensors, fs, nperseg, noverlap, nfft):
    """Convert all sensors in dictionary to spectrogram representations."""
    converted = {}

    #podpora za vec senzorjev (npr accel gyro mag)
    for sensor_name, sensor_data in sensors.items():
        #enaki parametri za vse clane ekipe in vse posnetke
        frequencies, times, spectrograms, rgb_image = create_rgb_spectrogram(
            sensor_data, fs, nperseg, noverlap, nfft
        )
        converted[sensor_name] = {
            "frequencies": frequencies,
            "times": times,
            "spectrograms": spectrograms,
            "rgb_image": rgb_image,
        }

    #reusable funkcije vrnejo direktne vhode za nevronske mreze
    #todo razbit dolg signal na vec manjsih training segmentov
    return converted
