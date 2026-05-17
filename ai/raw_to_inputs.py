# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:07:08 2026

@author: mihal
"""
from AutoDNA.ai.spectrograms import convert_all_sensors_to_spectrograms
from AutoDNA.ai.demo_spectrograms import choose_stft_parameters
from AutoDNA.ai.preprocessing import load_sensor_data, estimate_sampling_rate, preprocess_sensor_data, resample_sensors_to_common_grid

from pathlib import Path
import numpy as np
import json

BASE_DATA_DIR = Path('../data/training_data')
PARSED_DIR = BASE_DATA_DIR / 'parsed_data'
LABELED_DIR = BASE_DATA_DIR / 'labeled_data_json'
LOG_FILES = list(PARSED_DIR.glob('*.npz'))


def save_training_sample(npz_path, converted, labels, output_dir):
    """
    Save one recording's spectrogram + labels as a single .npz.
    One file per recording, easy to load in NN training loop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}

    for sensor_name, result in converted.items():
        arrays[f'{sensor_name}_rgb'] = result['rgb_image']
        arrays[f'{sensor_name}_times'] = result['times'].astype(np.float32)
        arrays[f'{sensor_name}_freqs'] = result['frequencies'].astype(np.float32)

    #labele v timestep arr
    first_sensor = next(iter(converted.values()))
    times = first_sensor['times']
    y = labels_to_timeseries(labels, times)
    arrays['labels'] = y  

    output_path = output_dir / (Path(npz_path).stem + '_training.npz')
    np.savez_compressed(output_path, **arrays)
    print(f"Saved: {output_path.name}")
    return output_path


MAX_ANGLE_TURN = 180.0
MAX_ANGLE_HILL = 45.0

def labels_to_timeseries(labels, times):
    """
    Convert label list to per-spectrogram-column label matrix.
    Returns shape (T, 7) — one row per time column in spectrogram.

    col 0: turn_present
    col 1: turn_dir        (0=left, 1=right)
    col 2: turn_angle_norm (angleDeg / MAX_ANGLE_TURN)
    col 3: hill_present
    col 4: hill_dir        (0=down, 1=up)
    col 5: hill_angle_norm (angleDeg / MAX_ANGLE_HILL)
    col 6: straight
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


def main():
    for npz_path in LOG_FILES:
        label_path = LABELED_DIR / (npz_path.stem + '_labels.json')
            
        if not label_path.exists():
            print(f"Warning: Label file not found for {npz_path.name}, skipping...")
            continue
        
        sensors = load_sensor_data(npz_path)
        sensors = resample_sensors_to_common_grid(sensors)      #interpolacija, vsi senzorji na enak T
        first = sensors[next(iter(sensors))]
        fs = estimate_sampling_rate(first['ts'])
    
        processed = preprocess_sensor_data(sensors, fs, cutoff=5.0)     #predprocesiranje
    
        sample_count = len(first['x'])
        nperseg, noverlap, nfft = choose_stft_parameters(fs, sample_count)   #v 2d
        converted = convert_all_sensors_to_spectrograms(processed, fs, nperseg, noverlap, nfft)

        with open(label_path) as f:  #labele
                labels = json.load(f)['labels']
                
        save_training_sample(npz_path, converted, labels, output_dir='input_data')
        
        
if __name__ == "__main__":
    main()
