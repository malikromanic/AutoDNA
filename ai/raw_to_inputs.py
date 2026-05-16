# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:07:08 2026

@author: mihal
"""

from pathlib import Path
from preprocessing import load_sensor_data, estimate_sampling_rate, preprocess_sensor_data
from spectrograms import convert_all_sensors_to_spectrograms
from demo_spectrograms import choose_stft_parameters
import numpy as np
import json

BASE_DATA_DIR = Path('../data/training_data')
PARSED_DIR = BASE_DATA_DIR / 'parsed_data'
LABELED_DIR = BASE_DATA_DIR / 'labeled_data_json'

#print(f"Current Working Directory: {Path.cwd()}")
#print(f"Looking for NPZs in: {PARSED_DIR.resolve()}")

LOG_FILES = list(PARSED_DIR.glob('*.npz'))

#print(f"Found {len(LOG_FILES)} files.")


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


def labels_to_timeseries(labels, times):
    """
    Convert label list to per-spectrogram-column label matrix.
    Returns shape (T, 5) — one row per time column in spectrogram.

    Classes: [turn_left, turn_right, hill_up, hill_down, straight]
    """
    CLASS_IDX = {
        'turn_left': 0,
        'turn_right': 1,
        'hill_up': 2,
        'hill_down': 3,
        'straight': 4,
    }
    T = len(times)
    y = np.zeros((T, len(CLASS_IDX)), dtype=np.float32)

    for seg in labels:
        t0, t1 = seg['t_start'], seg['t_end']
        mask = (times >= t0) & (times <= t1)

        if seg.get('straight'):
            y[mask, CLASS_IDX['straight']] = 1.0
        if seg.get('turn'):
            key = 'turn_left' if seg['turn']['dir'] == 'left' else 'turn_right'
            y[mask, CLASS_IDX[key]] = 1.0
        if seg.get('hill'):
            key = 'hill_up' if seg['hill']['dir'] == 'up' else 'hill_down'
            y[mask, CLASS_IDX[key]] = 1.0

    return y


def main():
    for npz_path in LOG_FILES:
        label_path = LABELED_DIR / (npz_path.stem + '_labels.json')
            
        if not label_path.exists():
            print(f"Warning: Label file not found for {npz_path.name}, skipping...")
            continue
        
        sensors = load_sensor_data(npz_path)
        first = sensors[next(iter(sensors))]
        fs = estimate_sampling_rate(first['ts'])
    
        processed = preprocess_sensor_data(sensors, fs, cutoff=5.0)     #predprocesiranje
    
        sample_count = len(first['x'])
        nperseg, noverlap, nfft = choose_stft_parameters(fs, sample_count)   #v 2d
        converted = convert_all_sensors_to_spectrograms(processed, fs, nperseg, noverlap, nfft)

        with open(label_path) as f:  #labele
                labels = json.load(f)['labels']
                
        save_training_sample(npz_path, converted, labels, output_dir='training_data')
        
        
if __name__ == "__main__":
    main()
