"""Demo script for converting preprocessed sensor signals to spectrograms."""

from pathlib import Path

import matplotlib.pyplot as plt

from AutoDNA.ai.spectrograms import (
    load_sensor_data,
    estimate_sampling_rate,
    plot_spectrogram,
    convert_all_sensors_to_spectrograms,
)


DATA_FILE = Path(__file__).with_name("LOG010_preprocessed.npz")
OUTPUT_DIR = Path(__file__).with_name("spectrogram_examples")
AXIS_FIGSIZE = (12, 5)
RGB_FIGSIZE = (11, 5)


def choose_stft_parameters(fs, sample_count):
    """Choose simple STFT parameters suitable for slower motion events."""
    #nastavitev velikosti okna za stft segmente
    nperseg = int(round(fs * 2.0))
    #omejevanje okna da ne preseze dolzine signala
    nperseg = max(32, min(sample_count, nperseg))

    if nperseg < 2:
        nperseg = 2

    #50% overlap da je casovni potek bolj gladek
    noverlap = nperseg // 2

    #fft dolzina kot potenca 2 za stabilen izracun
    nfft = 1
    while nfft < nperseg:
        nfft *= 2

    return nperseg, noverlap, nfft


def find_matching_sensors(sensors):
    """Prefer accel/gyro/mag sensors if available, otherwise use all."""
    #podpora za vec tipov senzorjev iz naloge
    preferred = ["accel", "gyro", "mag"]
    selected = []

    lower_to_original = {name.lower(): name for name in sensors.keys()}

    for key in preferred:
        exact = lower_to_original.get(key)
        if exact is not None and exact not in selected:
            selected.append(exact)
            continue

        for name in sensors.keys():
            if key in name.lower() and name not in selected:
                selected.append(name)
                break

    if selected:
        return selected

    return list(sensors.keys())


def show_rgb_image(sensor_name, frequencies, times, rgb_image):
    """Display RGB packed spectrogram image with proper axes."""
    plt.figure(figsize=RGB_FIGSIZE)
    plt.imshow(
        rgb_image,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
    )
    plt.title(f"{sensor_name}: RGB spectrogram (X->R, Y->G, Z->B)")
    #osi ostanejo iste tudi pri rgb pogledu
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


def save_axis_spectrogram_image(sensor_name, axis, frequencies, times, spectrogram, output_dir):
    """Save one axis spectrogram image with labeled axes."""
    plt.figure(figsize=AXIS_FIGSIZE)
    mesh = plt.pcolormesh(times, frequencies, spectrogram, shading="gouraud", cmap="viridis")
    plt.title(f"{sensor_name} - {axis.upper()} axis spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    cbar = plt.colorbar(mesh)
    cbar.set_label("Amplitude (dB)")
    plt.tight_layout()

    output_file = output_dir / f"{sensor_name}_{axis}_spectrogram.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    return output_file


def main():
    """Load a preprocessed recording, build spectrograms, and display them per sensor."""
    #load preprocessed podatkov za pretvorbo v 2d
    sensors = load_sensor_data(DATA_FILE)

    first_sensor_name = next(iter(sensors))
    first_sensor = sensors[first_sensor_name]
    #ocena frekvence vzorcenja iz timestampov
    fs = estimate_sampling_rate(first_sensor["ts"])

    sample_count = len(first_sensor["x"])
    nperseg, noverlap, nfft = choose_stft_parameters(fs, sample_count)

    print("Loaded sensors:")
    for sensor_name, sensor_data in sensors.items():
        print(f"- {sensor_name}: {len(sensor_data['ts'])} samples")

    print(f"Estimated sampling rate: {fs:.2f} Hz")
    print("STFT parameters:")
    print(f"- window size (nperseg): {nperseg}")
    print(f"- overlap (noverlap): {noverlap}")
    print(f"- FFT length (nfft): {nfft}")

    converted = convert_all_sensors_to_spectrograms(
        sensors=sensors,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    selected_sensors = find_matching_sensors(sensors)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sensor_name in selected_sensors:
        #za vsak senzor pripravimo svoj 2d vhod
        result = converted[sensor_name]
        frequencies = result["frequencies"]
        times = result["times"]
        spectrograms = result["spectrograms"]
        rgb_image = result["rgb_image"]

        #prikaz osi spektrogramov kjer se vidi dogodek
        for axis in ["x", "y", "z"]:
            plot_spectrogram(
                frequencies,
                times,
                spectrograms[axis],
                f"{sensor_name} - {axis.upper()} axis spectrogram",
            )

            #dodaten primer na disk za vsako os
            """axis_output = save_axis_spectrogram_image(
                sensor_name=sensor_name,
                axis=axis,
                frequencies=frequencies,
                times=times,
                spectrogram=spectrograms[axis],
                output_dir=OUTPUT_DIR,
            )
            print(f"Saved: {axis_output.name}")"""

        show_rgb_image(sensor_name, frequencies, times, rgb_image)

        #shrani primer slike za porocilo ali ucenje mreze
        """output_file = OUTPUT_DIR / f"{sensor_name}_rgb_spectrogram.png"
        save_spectrogram_image(rgb_image, output_file)
        print(f"Saved: {output_file.name}")"""

    #todo dodati avtomatski izbor najboljsih odsekov z dogodkom


if __name__ == "__main__":
    main()
