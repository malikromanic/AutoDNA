"""
Add this function to build_training_data.py (or import it from here).

Produces report figures for one log file:
1. Raw vs. preprocessed signal (per axis, per sensor) — shows the effect of
   smoothing + low-pass filtering + normalization.
2. RGB spectrogram images for accel / gyro / mag after preprocessing.

Usage (inside build_training_data.py main, or a separate script):

    from AutoDNA.ai.preprocessing import (
        load_sensor_data, estimate_sampling_rate,
        preprocess_sensor_data, resample_sensors_to_common_grid,
        normalize_signal,
    )
    from AutoDNA.ai.spectrograms import convert_all_sensors_to_spectrograms
    from AutoDNA.ai.demo_spectrograms import choose_stft_parameters

    plot_preprocessing_and_spectrograms(PARSED_DIR, log_num=10)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    estimate_sampling_rate,
    preprocess_sensor_data,
    resample_sensors_to_common_grid,
    normalize_signal,
)
from AutoDNA.ai.spectrograms import convert_all_sensors_to_spectrograms
from AutoDNA.ai.demo_spectrograms import choose_stft_parameters

# import axis-correction helpers + file sets from build_training_data
from raw_to_inputs import (
    FILES_90_RIGHT_FLIP,
    FILES_90_LEFT_FLIP,
    swap_axes_90_right_flip,
    swap_axes_90_left_flip,
)


def plot_preprocessing_and_spectrograms(parsed_dir, log_num, output_dir='report_figures'):
    """
    Generate raw-vs-processed plots and RGB spectrogram images for one log file.

    :param parsed_dir: Path to directory containing raw .npz recordings.
    :param log_num: Numeric log index (e.g. 10 for LOG010).
    :param output_dir: Directory where figures are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- locate file ---
    matches = list(Path(parsed_dir).glob(f'*{log_num:03d}*.npz'))
    if not matches:
        matches = list(Path(parsed_dir).glob(f'*{log_num}*.npz'))
    if not matches:
        print(f"File LOG{log_num:03d} not found")
        return
    npz_path = matches[0]

    # --- load + resample (same as build_training_data pipeline) ---
    sensors = load_sensor_data(npz_path)
    sensors = resample_sensors_to_common_grid(sensors)

    # --- axis correction, same logic as main() ---
    if log_num in FILES_90_RIGHT_FLIP:
        sensors = swap_axes_90_right_flip(sensors)
    elif log_num in FILES_90_LEFT_FLIP:
        sensors = swap_axes_90_left_flip(sensors)

    # keep a copy of raw (post axis-correction, pre-preprocessing) for comparison
    raw_sensors = {name: {k: v.copy() for k, v in data.items()}
                   for name, data in sensors.items()}

    # --- preprocessing ---
    first = sensors[next(iter(sensors))]
    fs = estimate_sampling_rate(first['ts'])
    processed = preprocess_sensor_data(sensors, fs, cutoff=5.0)

    # --- raw vs processed plots, per sensor, all 3 axes ---
    ts = first['ts']
    ts = ts - ts[0]

    for sensor_name in sensors:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"LOG{log_num:03d} — {sensor_name}: raw vs preprocessed", fontsize=14)

        for axis_plot, axis_name in zip(axes, ['x', 'y', 'z']):
            raw_axis = normalize_signal(raw_sensors[sensor_name][axis_name])
            processed_axis = processed[sensor_name][axis_name]

            axis_plot.plot(ts, raw_axis, label='Raw (normalized)', alpha=0.5)
            axis_plot.plot(ts, processed_axis, label='Processed', linewidth=2)
            axis_plot.set_ylabel(axis_name.upper())
            axis_plot.grid(True)
            axis_plot.legend()

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        fig_path = output_dir / f'LOG{log_num:03d}_{sensor_name}_preprocessing.png'
        plt.savefig(fig_path, dpi=150)
        plt.show()
        print(f"Saved: {fig_path.name}")

    # --- spectrograms from preprocessed data ---
    sample_count = len(first['x'])
    nperseg, noverlap, nfft = choose_stft_parameters(fs, sample_count)
    converted = convert_all_sensors_to_spectrograms(processed, fs, nperseg, noverlap, nfft)

    for sensor_name, result in converted.items():
        frequencies = result['frequencies']
        times = result['times']
        rgb_image = result['rgb_image']

        plt.figure(figsize=(11, 5))
        plt.imshow(
            rgb_image,
            origin='lower',
            aspect='auto',
            extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
        )
        plt.title(f"LOG{log_num:03d} — {sensor_name}: RGB spectrogram (X->R, Y->G, Z->B)")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()

        fig_path = output_dir / f'LOG{log_num:03d}_{sensor_name}_rgb_spectrogram.png'
        plt.savefig(fig_path, dpi=150)
        plt.show()
        print(f"Saved: {fig_path.name}")


if __name__ == "__main__":
    from raw_to_inputs import PARSED_DIR
    plot_preprocessing_and_spectrograms(PARSED_DIR, log_num=10)