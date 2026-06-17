# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:07:08 2026
 
@author: mihal
 
build_training_data.py
======================
Batch pipeline that converts raw sensor recordings (.npz) into compressed
training samples for the AutoDNA driving-event classifier.
 
For every recording found in ``PARSED_DIR`` that has a matching label file in
``LABELED_DIR``, the pipeline:
 
1. Loads and resamples multi-axis IMU data onto a common time grid.
2. Pre-processes (low-pass filters) the signals.
3. Selects STFT parameters suited to the recording length and sampling rate.
4. Converts every sensor axis to an RGB spectrogram image.
5. Computes per-window statistics (mean) for gyro Z, gyro Y, and accel X —
   these preserve signal sign and magnitude for downstream direction / tilt
   regression heads.
6. Aligns the JSON event labels to the spectrogram time axis, producing a
   dense (T × 7) label matrix.
7. Writes everything to a single compressed ``.npz`` file under ``input_data/``.
 
Label column layout (axis 1 of the label matrix)
-------------------------------------------------
0 — ``is_turn``        binary flag: 1 if a turn event is active
1 — ``turn_dir``       binary flag: 1 = right, 0 = left  (valid when col 0 = 1)
2 — ``turn_angle``     normalised angle in [0, 1]  (clipped at MAX_ANGLE_TURN)
3 — ``is_hill``        binary flag: 1 if a hill event is active
4 — ``hill_dir``       binary flag: 1 = uphill, 0 = downhill (valid when col 3 = 1)
5 — ``hill_angle``     normalised angle in [0, 1]  (clipped at MAX_ANGLE_HILL)
6 — ``is_straight``    binary flag: 1 if the road segment is straight
"""
 
from AutoDNA.ai.spectrograms import convert_all_sensors_to_spectrograms
from AutoDNA.ai.demo_spectrograms import choose_stft_parameters
from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    estimate_sampling_rate,
    preprocess_sensor_data,
    resample_sensors_to_common_grid,
)
 
from pathlib import Path
import numpy as np
import json

BASE_DATA_DIR = Path('../data/training_data')
PARSED_DIR = BASE_DATA_DIR / 'parsed_data'
LABELED_DIR = BASE_DATA_DIR / 'labeled_data_json'
LOG_FILES = list(PARSED_DIR.glob('*.npz'))


MAX_ANGLE_TURN = 100.0
#Turn angles larger than this value are clipped to 1.0 in the label matrix
 
MAX_ANGLE_HILL = 15.0
#Hill angles larger than this value are clipped to 1.0 in the label matrix


FILES_90_RIGHT_FLIP = set(range(5, 12)) | set(range(15, 30))        # 5-11, 15-29
FILES_90_LEFT_FLIP = set(range(1, 5)) | set(range(12, 15)) | set(range(30, 46))  # 1-4, 12-14, 30-45


def get_log_number(npz_path):
    """Extract the numeric index from a log filename e.g. LOG005 -> 5."""
    stem = Path(npz_path).stem  #   "LOG005"
    digits = ''.join(c for c in stem if c.isdigit())
    return int(digits) if digits else -1


def swap_axes_90_right_flip(sensors):
    """
    Correct axis orientation for files where sensor points to right window.
    Sensor X points right, Y points forward — swap to standard (X=forward, Y=left).
    
    standard_x =  sensor_y
    standard_y = -sensor_x
    standard_z =  sensor_z  (unchanged)
    """
    corrected = {}
    for name, data in sensors.items():
        corrected[name] = {
            'ts':  data['ts'],
            'x':   data['y'].copy(),    # new X = old Y (forward)
            'y':  -data['x'].copy(),    # new Y = -old X (left = -right)
            'z':   data['z'].copy(),
        }
    return corrected


def swap_axes_90_left_flip(sensors):
    """
    Correct axis orientation for files where sensor points to left window.
    Sensor X points left, Y points backward — swap to standard (X=forward, Y=left).
    
    standard_x = -sensor_y
    standard_y =  sensor_x
    standard_z =  sensor_z  (unchanged)
    """
    corrected = {}
    for name, data in sensors.items():
        corrected[name] = {
            'ts':  data['ts'],
            'x':  -data['y'].copy(),    # new X = -old Y (forward = -backward)
            'y':   data['x'].copy(),    # new Y = old X (left)
            'z':   data['z'].copy(),
        }
    return corrected


def compute_pitch_angle_signal(accel_x, accel_z):
    """
    Compute pitch angle with mounting offset removed.
    Uses median of first 5 seconds as baseline (assumes car starts on flat road).
    """
    raw_pitch = np.degrees(np.arctan2(-accel_x, accel_z))
    
    # estimate mounting offset from first 2% of recording
    # assumes recording starts on approximately flat road
    n_baseline = max(100, len(raw_pitch) // 50)
    baseline = np.median(raw_pitch[:n_baseline])
    
    print(f"  Mounting offset: {baseline:.2f}°")
    return raw_pitch - baseline


def save_training_sample(npz_path, converted, labels, processed, nperseg, noverlap, output_dir):
    """Save one recording's spectrogram, per-window statistics, and labels as a
    single compressed ``.npz`` file.
 
    The file written to *output_dir* contains the following arrays:
 
    * ``<sensor>_rgb``   — uint8 RGB spectrogram image for each sensor axis,
      shape ``(H, W, 3)``.
    * ``<sensor>_times`` — float32 centre times (s) of each STFT column,
      shape ``(T,)``.
    * ``<sensor>_freqs`` — float32 frequency bin centres (Hz),
      shape ``(F,)``.
    * ``gyro_z_mean_per_window``  — float32 raw gyro-Z mean per window,
      shape ``(T,)``.  Sign is preserved so the model can infer turn direction.
    * ``gyro_y_mean_per_window``  — float32 raw gyro-Y mean per window,
      shape ``(T,)``.  Sign preserved for pitch direction.
    * ``accel_x_mean_per_window`` — float32 raw accel-X mean per window,
      shape ``(T,)``.  Sign preserved for forward/backward acceleration.
    * ``labels`` — float32 dense label matrix, shape ``(T, 7)``.  See module
      docstring for column definitions.
 
    :param npz_path: Path to the source recording ``.npz`` file.  Used only to
        derive the output filename.
    :type npz_path: str or pathlib.Path
    :param converted: Mapping returned by
        :func:`~AutoDNA.ai.spectrograms.convert_all_sensors_to_spectrograms`.
        Keys are sensor names; values are dicts with keys
        ``'rgb_image'``, ``'times'``, and ``'frequencies'``.
    :type converted: dict[str, dict]
    :param labels: List of label segment dicts loaded from the JSON label file.
        Each dict must contain at minimum ``'t_start'`` and ``'t_end'`` keys,
        plus optional ``'straight'``, ``'turn'``, and ``'hill'`` sub-dicts.
    :type labels: list[dict]
    :param processed: Pre-processed sensor data as returned by
        :func:`~AutoDNA.ai.preprocessing.preprocess_sensor_data`.
        Expected structure: ``{'gyro': {'x', 'y', 'z'}, 'accel': {'x', 'y', 'z'}}``.
    :type processed: dict[str, dict[str, numpy.ndarray]]
    :param nperseg: STFT window length in samples (must match the value used
        to produce *converted*).
    :type nperseg: int
    :param noverlap: Number of samples of overlap between consecutive STFT
        windows (must match the value used to produce *converted*).
    :type noverlap: int
    :param output_dir: Directory where the ``.npz`` training file is written.
        Created automatically if it does not exist.
    :type output_dir: str or pathlib.Path
    :returns: Path of the newly written ``.npz`` file.
    :rtype: pathlib.Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    arrays = {}
 
    for sensor_name, result in converted.items():
        arrays[f'{sensor_name}_rgb'] = result['rgb_image']
        arrays[f'{sensor_name}_times'] = result['times'].astype(np.float32)
        arrays[f'{sensor_name}_freqs'] = result['frequencies'].astype(np.float32)
        
        
    gyro_z_signal = processed['gyro']['z']  
    gyro_z_means = compute_mean_per_window(gyro_z_signal, nperseg, noverlap)
    arrays['gyro_z_mean_per_window'] = gyro_z_means 
 
    gyro_y_signal = processed['gyro']['y']
    gyro_y_means = compute_mean_per_window(gyro_y_signal, nperseg, noverlap)
    arrays['gyro_y_mean_per_window'] = gyro_y_means
 
    acc_x_signal = processed['accel']['x']  
    acc_x_means = compute_mean_per_window(acc_x_signal, nperseg, noverlap)
    arrays['accel_x_mean_per_window'] = acc_x_means  
    
    accel_x_raw = processed['accel']['x']
    accel_z_raw = processed['accel']['z']
    pitch_signal = compute_pitch_angle_signal(accel_x_raw, accel_z_raw)
    pitch_means = compute_mean_per_window(pitch_signal, nperseg, noverlap)
    arrays['pitch_angle_per_window'] = pitch_means.astype(np.float32)


    first_sensor = next(iter(converted.values()))
    times = first_sensor['times']
    y = labels_to_timeseries(labels, times)
    arrays['labels'] = y
 
    output_path = output_dir / (Path(npz_path).stem + '_training.npz')
    np.savez_compressed(output_path, **arrays)
    print(f"Saved: {output_path.name}")
    return output_path
 
 
def labels_to_timeseries(labels, times):
    """Convert a list of labelled event segments into a dense per-column label
    matrix aligned to the spectrogram time axis.
 
    Each segment in *labels* contributes to every spectrogram column whose
    centre time falls within ``[t_start, t_end]`` (inclusive).  Overlapping
    segments are supported: later segments in the list simply overwrite earlier
    ones for the overlapping columns.
 
    :param labels: List of segment dicts.  Recognised keys per segment:
 
        * ``t_start`` *(float, required)* — segment start time in seconds.
        * ``t_end``   *(float, required)* — segment end time in seconds.
        * ``straight`` *(any truthy value, optional)* — marks a straight
          road section; sets column 6 to ``1.0``.
        * ``turn`` *(dict, optional)* — turn event with sub-keys:
 
          * ``dir``      *(str)* — ``'right'`` or ``'left'``.
          * ``angleDeg`` *(float)* — unsigned turn angle in degrees.
 
        * ``hill`` *(dict, optional)* — hill event with sub-keys:
 
          * ``dir``      *(str)* — ``'up'`` or ``'down'``.
          * ``angleDeg`` *(float)* — unsigned hill angle in degrees.
 
    :type labels: list[dict]
    :param times: 1-D array of STFT column centre times (seconds), shape
        ``(T,)``.
    :type times: numpy.ndarray
    :returns: Float32 label matrix of shape ``(T, 7)``.  See module docstring
        for column definitions.
    :rtype: numpy.ndarray
    """
    T = len(times)
    y = np.zeros((T, 7), dtype=np.float32)
 
    for seg in labels:
        t0, t1 = seg['t_start'], seg['t_end']
        mask = (times >= t0) & (times <= t1)
 
        if seg.get('straight'):
            y[mask, 6] = 1.0
 
        if seg.get('turn'):
            y[mask, 0] = 1.0
            y[mask, 1] = 1.0 if seg['turn']['dir'] == 'right' else 0.0
            y[mask, 2] = min(seg['turn']['angleDeg'] / MAX_ANGLE_TURN, 1.0)
 
        if seg.get('hill'):
            y[mask, 3] = 1.0
            y[mask, 4] = 1.0 if seg['hill']['dir'] == 'up' else 0.0
            y[mask, 5] = min(seg['hill']['angleDeg'] / MAX_ANGLE_HILL, 1.0)
 
    return y
 
 
def compute_mean_per_window(gyro_z_signal, nperseg, noverlap):
    """Compute the arithmetic mean of a signal within each STFT window.
 
    The windowing scheme mirrors the one used by
    :func:`scipy.signal.stft` / the project's spectrogram conversion so that
    the returned array is index-aligned with the spectrogram time axis.
 
    Sign is deliberately *not* rectified: positive values indicate one
    physical direction and negative values the opposite, which is necessary
    for the model's direction regression heads (e.g. left vs. right turn,
    uphill vs. downhill).
 
    .. note::
        Trailing samples that do not fill a complete window of length
        *nperseg* are silently discarded, consistent with ``scipy.signal.stft``
        default behaviour (``boundary=None``).
 
    :param gyro_z_signal: 1-D array of pre-processed sensor samples.
        Although the parameter is named ``gyro_z_signal`` for historical
        reasons, the function is axis-agnostic and can be used with any
        1-D signal (gyro Y, accel X, etc.).
    :type gyro_z_signal: numpy.ndarray
    :param nperseg: STFT window length in samples.
    :type nperseg: int
    :param noverlap: Number of overlapping samples between consecutive
        windows.  The hop size is ``nperseg - noverlap``.
    :type noverlap: int
    :returns: Float32 array of per-window means, shape ``(T,)`` where *T* is
        the number of complete windows.
    :rtype: numpy.ndarray
    """
    step = nperseg - noverlap
    means = []
    for i in range(0, len(gyro_z_signal) - nperseg + 1, step):
        window = gyro_z_signal[i:i + nperseg]
        means.append(window.mean())
    return np.array(means, dtype=np.float32)


def plot_axis_correction(parsed_dir, my_file_num=10, teammate_file_num=3):
    """
    Plot gyro and accel X/Y/Z before and after axis correction
    for one file from each device to verify the swap is correct.
    """
    import matplotlib.pyplot as plt

    def load_and_swap(log_num):
        # find the file
        matches = list(parsed_dir.glob(f'*{log_num:03d}*.npz'))
        if not matches:
            matches = list(parsed_dir.glob(f'*{log_num}*.npz'))
        if not matches:
            print(f"File LOG{log_num:03d} not found")
            return None, None, None
        
        path = matches[0]
        sensors = load_sensor_data(path)
        sensors = resample_sensors_to_common_grid(sensors)
        raw = {name: {k: v.copy() for k, v in data.items()} 
               for name, data in sensors.items()}

        if log_num in FILES_90_RIGHT_FLIP:
            corrected = swap_axes_90_right_flip(sensors)
            label = f'LOG{log_num:03d} (mine)'
        elif log_num in FILES_90_LEFT_FLIP:
            corrected = swap_axes_90_left_flip(sensors)
            label = f'LOG{log_num:03d} (teammate)'
        else:
            corrected = sensors
            label = f'LOG{log_num:03d} (unknown)'

        return raw, corrected, label

    fig, axes = plt.subplots(6, 4, figsize=(20, 18))
    fig.suptitle('Axis correction verification — gyro and accel X/Y/Z', fontsize=14)

    sensor_pairs = [
        ('gyro',  'x', 0), ('gyro',  'y', 1), ('gyro',  'z', 2),
        ('accel', 'x', 3), ('accel', 'y', 4), ('accel', 'z', 5),
    ]

    for file_num, col_offset in [(my_file_num, 0), (teammate_file_num, 2)]:
        raw, corrected, file_label = load_and_swap(file_num)
        if raw is None:
            continue

        for sensor_name, axis, row in sensor_pairs:
            if sensor_name not in raw:
                continue

            ts = raw[sensor_name]['ts']
            ts = (ts - ts[0]) / 1000.0  # to seconds

            raw_signal = raw[sensor_name][axis]
            cor_signal = corrected[sensor_name][axis]

            # before
            ax = axes[row][col_offset]
            ax.plot(ts, raw_signal, linewidth=0.8)
            ax.set_title(f'{file_label}\n{sensor_name.upper()} {axis.upper()} — BEFORE')
            ax.set_ylabel('amplitude')
            ax.grid(True)

            # after
            ax = axes[row][col_offset + 1]
            ax.plot(ts, cor_signal, linewidth=0.8, color='orange')
            ax.set_title(f'{file_label}\n{sensor_name.upper()} {axis.upper()} — AFTER')
            ax.set_ylabel('amplitude')
            ax.grid(True)

    for ax in axes[-1]:
        ax.set_xlabel('time (s)')

    plt.tight_layout()
    plt.savefig('axis_correction_check.png', dpi=100)
    plt.show()
    print("Saved: axis_correction_check.png")


def plot_pitch_vs_labels(parsed_dir, label_dir, log_num):
    """
    Plot computed pitch angle alongside hill labels to verify correctness.
    Uphill regions should show positive pitch, downhill negative.
    """
    import matplotlib.pyplot as plt
    import json

    matches = list(parsed_dir.glob(f'*{log_num:03d}*.npz'))
    if not matches:
        print(f"File LOG{log_num:03d} not found")
        return

    sensors = load_sensor_data(matches[0])
    sensors = resample_sensors_to_common_grid(sensors)

    if log_num in FILES_90_RIGHT_FLIP:
        sensors = swap_axes_90_right_flip(sensors)
    elif log_num in FILES_90_LEFT_FLIP:
        sensors = swap_axes_90_left_flip(sensors)

    accel = sensors['accel']
    ts = (accel['ts'] - accel['ts'][0]) / 1000.0  # seconds

    pitch = compute_pitch_angle_signal(accel['x'], accel['z'])
    
    print(f"pitch stats: min={pitch.min():.3f}  max={pitch.max():.3f}  mean={pitch.mean():.3f}")
    print(f"accel_x stats: min={accel['x'].min():.1f}  max={accel['x'].max():.1f}  mean={accel['x'].mean():.1f}")
    print(f"accel_z stats: min={accel['z'].min():.1f}  max={accel['z'].max():.1f}  mean={accel['z'].mean():.1f}")

    label_path = label_dir / f'LOG{log_num:03d}_labels.json'
    if not label_path.exists():
        print(f"Labels not found for LOG{log_num:03d}")
        return

    with open(label_path) as f:
        labels = json.load(f)['labels']

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'LOG{log_num:03d} — pitch angle vs hill labels', fontsize=13)

    # plot 1: raw accel x and z for reference
    axes[0].plot(ts, accel['x'], linewidth=0.5, label='accel X (forward)', alpha=0.8)
    axes[0].plot(ts, accel['z'], linewidth=0.5, label='accel Z (vertical)', alpha=0.8)
    axes[0].axhline(0,    color='gray', linewidth=0.5, linestyle='--')
    axes[0].axhline(1000, color='green', linewidth=0.8, linestyle='--', label='expected 1g')
    axes[0].set_ylabel('mg (raw)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    # plot 2: computed pitch angle
    axes[1].plot(ts, pitch, linewidth=0.6, color='purple', label='pitch angle (°)')
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[1].axhline( 1.5, color='orange', linewidth=0.8, linestyle='--', label='threshold ±1.5°')
    axes[1].axhline(-1.5, color='orange', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('pitch (degrees)')
    axes[1].set_ylim(-20, 20)  # expected range for normal roads
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    # plot 3: pitch angle with hill label shading
    axes[2].plot(ts, pitch, linewidth=0.6, color='purple', label='pitch angle (°)')
    axes[2].axhline(0, color='gray', linewidth=0.5, linestyle='--')

    legend_added = set()
    for seg in labels:
        if seg.get('hill'):
            color = 'red' if seg['hill']['dir'] == 'up' else 'blue'
            label_str = f"hill {seg['hill']['dir']}"
            axes[2].axvspan(
                seg['t_start'], seg['t_end'],
                alpha=0.25, color=color,
                label=label_str if label_str not in legend_added else ''
            )
            legend_added.add(label_str)

    axes[2].set_ylabel('pitch (degrees)')
    axes[2].set_ylim(-20, 20)
    axes[2].set_xlabel('time (s)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'pitch_check_LOG{log_num:03d}.png', dpi=100)
    plt.show()
    print(f"Saved: pitch_check_LOG{log_num:03d}.png")

    # print statistics per labeled hill segment
    print("\nPitch statistics per hill segment:")
    for seg in labels:
        if seg.get('hill'):
            t0, t1 = seg['t_start'], seg['t_end']
            mask = (ts >= t0) & (ts <= t1)
            if mask.any():
                seg_pitch = pitch[mask]
                print(f"  t={t0:.1f}-{t1:.1f}s  dir={seg['hill']['dir']:4s}  "
                      f"angle={seg['hill']['angleDeg']}°  "
                      f"pitch_mean={seg_pitch.mean():.2f}°  "
                      f"pitch_std={seg_pitch.std():.2f}°  "
                      f"pitch_range=[{seg_pitch.min():.2f}, {seg_pitch.max():.2f}]")


def sanity_check_sensors(parsed_dir, log_num):
    """
    Print statistics for a file to verify sensor readings make sense.
    At rest: accel_z ≈ ±1000mg, accel_x/y ≈ 0, gyro_x/y/z ≈ 0
    """
    import matplotlib.pyplot as plt
    
    matches = list(parsed_dir.glob(f'*{log_num:03d}*.npz'))
    if not matches:
        print(f"File not found: LOG{log_num:03d}")
        return
    
    sensors = load_sensor_data(matches[0])
    sensors = resample_sensors_to_common_grid(sensors)
    

    if log_num in FILES_90_RIGHT_FLIP:
        sensors = swap_axes_90_right_flip(sensors)
    elif log_num in FILES_90_LEFT_FLIP:
        sensors = swap_axes_90_left_flip(sensors)

    for sensor_name in ['accel', 'gyro']:
        if sensor_name not in sensors:
            continue
        d = sensors[sensor_name]
        ts = (d['ts'] - d['ts'][0]) / 1000.0
        
        print(f"\n{sensor_name.upper()} statistics:")
        for axis in ['x', 'y', 'z']:
            sig = d[axis]
            print(f"  {axis}: mean={sig.mean():.1f}  std={sig.std():.1f}  "
                  f"min={sig.min():.1f}  max={sig.max():.1f}")
        
        # plot all 3 axes
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f'LOG{log_num:03d} — {sensor_name.upper()} (after axis correction)')
        for i, axis in enumerate(['x', 'y', 'z']):
            axes[i].plot(ts, d[axis], linewidth=0.6)
            axes[i].set_ylabel(f'{axis} (raw)')
            axes[i].axhline(0, color='red', linewidth=0.5, linestyle='--')
            if sensor_name == 'accel' and axis == 'z':
                axes[i].axhline(1000, color='green', linewidth=0.8, 
                               linestyle='--', label='expected 1g')
                axes[i].axhline(-1000, color='green', linewidth=0.8, linestyle='--')
                axes[i].legend()
        axes[-1].set_xlabel('time (s)')
        plt.tight_layout()
        plt.savefig(f'sanity_LOG{log_num:03d}_{sensor_name}.png', dpi=100)
        plt.show()



def main():
    plot_axis_correction(PARSED_DIR, 10, 3)
    sanity_check_sensors(PARSED_DIR, log_num=10)   # one of yours
    sanity_check_sensors(PARSED_DIR, log_num=2)    # teammate's

    """Process all recordings found in ``PARSED_DIR`` and write training
    samples to ``input_data/``.
 
    For each ``.npz`` file in ``LOG_FILES`` the function:
 
    1. Looks for a matching label file in ``LABELED_DIR``; skips the recording
       with a warning if none is found.
    2. Applies conversion into standard orientation depending on teammate A or teammate B recording
    3. Loads, resamples, and pre-processes the IMU sensor data.
    4. Selects STFT parameters via
       :func:`~AutoDNA.ai.demo_spectrograms.choose_stft_parameters`.
    5. Converts pre-processed data to spectrograms.
    6. Loads the JSON labels and calls :func:`save_training_sample`.
 
    :raises FileNotFoundError: Propagated from underlying I/O helpers if a
        ``.npz`` file listed in ``LOG_FILES`` cannot be read.
    :raises json.JSONDecodeError: If a label file contains malformed JSON.
    """
    #plot_pitch_vs_labels(PARSED_DIR, LABELED_DIR, 10)
    #plot_pitch_vs_labels(PARSED_DIR, LABELED_DIR, 3)

    for npz_path in LOG_FILES:
        label_path = LABELED_DIR / (npz_path.stem + '_labels.json')
            
        if not label_path.exists():
            print(f"Warning: Label file not found for {npz_path.name}, skipping...")
            continue
        
        sensors = load_sensor_data(npz_path)
        sensors = resample_sensors_to_common_grid(sensors)

        log_num = get_log_number(npz_path)
        if log_num in FILES_90_RIGHT_FLIP:
            sensors = swap_axes_90_right_flip(sensors)
        elif log_num in FILES_90_LEFT_FLIP:
            sensors = swap_axes_90_left_flip(sensors)
        else:
            print(f"  Warning: {npz_path.name} not in any known file set, no axis correction applied")

        first = sensors[next(iter(sensors))]
        fs = estimate_sampling_rate(first['ts'])
    
        processed = preprocess_sensor_data(sensors, fs, cutoff=5.0)
    
        sample_count = len(first['x'])
        nperseg, noverlap, nfft = choose_stft_parameters(fs, sample_count)
        converted = convert_all_sensors_to_spectrograms(processed, fs, nperseg, noverlap, nfft)
 
        with open(label_path) as f:
            labels = json.load(f)['labels']
                
        save_training_sample(npz_path, converted, labels, processed, nperseg, noverlap,
                             output_dir='input_data_flipped')
        
        
if __name__ == "__main__":
    main()