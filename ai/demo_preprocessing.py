"""Demo script for preprocessing telemetry sensor signals."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from preprocessing import (
    load_sensor_data,
    estimate_sampling_rate,
    preprocess_sensor_data,
    normalize_signal,
    save_preprocessed_data,
    resample_sensors_to_common_grid,
)


WINDOW_SIZE = 5
CUTOFF = 5.0
FILTER_ORDER = 4

DATA_FILE = Path(__file__).with_name("LOG009.npz")
OUTPUT_FILE = Path(__file__).with_name("LOG009_preprocessed.npz")


def _to_seconds(ts):
    """Convert timestamps to seconds when they are likely milliseconds."""
    ts = np.asarray(ts, dtype=float)
    ts = ts - ts[0]

    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) > 0 and np.median(diffs) > 10:
        #pretvorba v sekunde za pravilne osi grafa
        ts = ts / 1000.0

    return ts


def plot_sensor_all_axes(sensor_name, raw_sensor, processed_sensor):
    """Show raw vs processed signals for x/y/z axes of one sensor."""
    ts = _to_seconds(raw_sensor["ts"])

    #demonstracija preprocessinga na vseh 3 oseh
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{sensor_name}: raw vs processed", fontsize=14)

    #primerjava raw vs processed za x/y/z kanale
    for axis_plot, axis_name in zip(axes, ["x", "y", "z"]):
        raw_axis = normalize_signal(raw_sensor[axis_name])
        processed_axis = processed_sensor[axis_name]

        axis_plot.plot(ts, raw_axis, label="Raw", alpha=0.5)
        axis_plot.plot(ts, processed_axis, label="Processed", linewidth=2)
        axis_plot.set_ylabel(axis_name.upper())
        axis_plot.grid(True)
        axis_plot.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def main():
    #load podatkov za enoten preprocessing pipeline
    sensors = load_sensor_data(DATA_FILE)
    
    #resampling vseh senzorjev na skupno casovno mrezo za enostavnejso obdelavo
    sensors = resample_sensors_to_common_grid(sensors)

    #ocena frekvence vzorcenja iz timestampov
    first_sensor = sensors[next(iter(sensors))]
    fs = estimate_sampling_rate(first_sensor["ts"])

    #zadrzimo cutoff pod nyquist mejo
    effective_cutoff = min(CUTOFF, 0.45 * fs)

    print("Loaded sensors:")
    for sensor_name, sensor_data in sensors.items():
        print(f"- {sensor_name}: {len(sensor_data['ts'])} samples")

    print(f"Estimated sampling rate: {fs:.2f} Hz")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Filter order: {FILTER_ORDER}")
    print(f"Low-pass cutoff requested: {CUTOFF:.2f} Hz")
    print(f"Low-pass cutoff used: {effective_cutoff:.2f} Hz")

    #reusable funkcija naredi ciscenje vseh senzorjev
    processed = preprocess_sensor_data(
        sensors=sensors,
        fs=fs,
        cutoff=CUTOFF,
        window_size=WINDOW_SIZE,
        filter_order=FILTER_ORDER,
    )

    #moznost shranjevanja preprocessiranih podatkov
    save_preprocessed_data(processed, OUTPUT_FILE)
    print(f"Saved processed data: {OUTPUT_FILE.name}")

    #vizualna potrditev ucinka preprocessinga za zagovor
    for sensor_name in sensors:
        plot_sensor_all_axes(
            sensor_name=sensor_name,
            raw_sensor=sensors[sensor_name],
            processed_sensor=processed[sensor_name],
        )

    #todo dodat avtomatsko metriko izboljsanja noise nivoja


if __name__ == "__main__":
    main()
