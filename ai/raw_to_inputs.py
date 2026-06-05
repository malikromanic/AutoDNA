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


def main():
    """Process all recordings found in ``PARSED_DIR`` and write training
    samples to ``input_data/``.
 
    For each ``.npz`` file in ``LOG_FILES`` the function:
 
    1. Looks for a matching label file in ``LABELED_DIR``; skips the recording
       with a warning if none is found.
    2. Loads, resamples, and pre-processes the IMU sensor data.
    3. Selects STFT parameters via
       :func:`~AutoDNA.ai.demo_spectrograms.choose_stft_parameters`.
    4. Converts pre-processed data to spectrograms.
    5. Loads the JSON labels and calls :func:`save_training_sample`.
 
    :raises FileNotFoundError: Propagated from underlying I/O helpers if a
        ``.npz`` file listed in ``LOG_FILES`` cannot be read.
    :raises json.JSONDecodeError: If a label file contains malformed JSON.
    """
    for npz_path in LOG_FILES:
        label_path = LABELED_DIR / (npz_path.stem + '_labels.json')
            
        if not label_path.exists():
            print(f"Warning: Label file not found for {npz_path.name}, skipping...")
            continue
        
        sensors = load_sensor_data(npz_path)
        sensors = resample_sensors_to_common_grid(sensors)
        first = sensors[next(iter(sensors))]
        fs = estimate_sampling_rate(first['ts'])
    
        processed = preprocess_sensor_data(sensors, fs, cutoff=5.0)
    
        sample_count = len(first['x'])
        nperseg, noverlap, nfft = choose_stft_parameters(fs, sample_count)
        converted = convert_all_sensors_to_spectrograms(processed, fs, nperseg, noverlap, nfft)
 
        with open(label_path) as f:
            labels = json.load(f)['labels']
                
        save_training_sample(npz_path, converted, labels, processed, nperseg, noverlap,
                             output_dir='input_data')
        
        
if __name__ == "__main__":
    main()