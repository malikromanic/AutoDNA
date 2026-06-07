"""
Minimal Pipeline za pripravo training dataseta iz IMU .npz logov.
Kanali (4): gyro_z, accel_x, accel_y, accel_z
"""

import sys
import json
from pathlib import Path
import numpy as np

# ── Poti & Parametri ─────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'training_data'
NPZ_DIR       = BASE_DATA_DIR / 'parsed_data'
JSON_DIR      = BASE_DATA_DIR / 'labeled_data_json'
OUTPUT_DIR    = Path(__file__).parent / 'dataset_output'

TARGET_FS    = 50
WINDOW_SIZE_TURN = 100   # 2s
STRIDE_TURN      = 25
WINDOW_SIZE_HILL = 200   # 4s
STRIDE_HILL      = 50

FLIP_LOG_NUMBERS = set(list(range(1, 5)) + list(range(12, 15)) + list(range(30, 46)))

TURN_MAP = {'none': 0, 'left': 1, 'right': 2}
HILL_MAP = {'none': 0, 'up':   1, 'down':  2}

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from AutoDNA.ai.preprocessing import load_sensor_data, resample_sensors_to_common_grid, preprocess_sensor_data  # noqa: E402

# ── Helper Funkcije ──────────────────────────────────────────────────────────

def _log_number_from_path(npz_path):
    import re
    m = re.search(r'(\d+)', Path(npz_path).stem)
    return int(m.group(1)) if m else None

def tare_signal(signal, fs=50, init_seconds=2.0):
    """Odšteje začetno povprečje (mentor's trick) za odstranitev statičnega biasa."""
    init_samples = int(fs * init_seconds)
    if len(signal) > init_samples:
        baseline = np.mean(signal[:init_samples])
        return signal - baseline
    return signal

def get_dominant_label(t_start, t_end, labels, label_key):
    """Poenostavljen label assignment brez kompliciranih pragov."""
    best_label, best_overlap = 'none', 0.0
    for lbl in labels:
        if lbl.get(label_key) is None:
            continue
        overlap = min(t_end, lbl['t_end']) - max(t_start, lbl['t_start'])
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = lbl[label_key]['dir']
    
    return best_label if (best_overlap / (t_end - t_start)) >= 0.4 else 'none'

def make_windows(signal, t, labels, log_idx, window_size, stride, label_key):
    X, Y, log_ids = [], [], []
    label_map = TURN_MAP if label_key == 'turn' else HILL_MAP
    
    n_windows = (len(signal) - window_size) // stride + 1
    for i in range(n_windows):
        i0, i1 = i * stride, i * stride + window_size
        t_start, t_end = float(t[i0]), float(t[i1 - 1])
        
        lbl = get_dominant_label(t_start, t_end, labels, label_key)
        X.append(signal[i0:i1])
        Y.append(label_map[lbl])
        log_ids.append(log_idx)
        
    return np.array(X), np.array(Y), np.array(log_ids)

# ── Glavni Pipeline ──────────────────────────────────────────────────────────

def process_log(npz_path, json_path, log_idx):
    sensors   = load_sensor_data(npz_path)
    resampled = resample_sensors_to_common_grid(sensors, target_fs=TARGET_FS)
    processed = preprocess_sensor_data(resampled, fs=TARGET_FS)

    gz = processed['gyro']['z']
    ax = processed['accel']['x']
    ay = processed['accel']['y']
    az = processed['accel']['z']

    if _log_number_from_path(npz_path) in FLIP_LOG_NUMBERS:
        ax, ay = -ax, -ay  

    # 2. Tare signala deluje zdaj na pravih amplitudah
    gz = tare_signal(gz)
    ax = tare_signal(ax)
    ay = tare_signal(ay)
    az = tare_signal(az)

    signal = np.column_stack([gz, ax, ay, az]).astype(np.float32)
    t = processed['gyro']['ts']

    with open(json_path) as f:
        labels = json.load(f)['labels']

    X_t, Y_t, ids_t = make_windows(signal, t, labels, log_idx, WINDOW_SIZE_TURN, STRIDE_TURN, 'turn')
    X_h, Y_h, ids_h = make_windows(signal, t, labels, log_idx, WINDOW_SIZE_HILL, STRIDE_HILL, 'hill')

    return (X_t, Y_t, ids_t), (X_h, Y_h, ids_h)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(NPZ_DIR.glob('*.npz'))

    all_X_turn, all_Y_turn, all_ids_turn = [], [], []
    all_X_hill, all_Y_hill, all_ids_hill = [], [], []

    for log_idx, npz_path in enumerate(npz_files):
        json_path = JSON_DIR / (npz_path.stem + '_labels.json')
        if not json_path.exists():
            continue

        (X_t, Y_t, ids_t), (X_h, Y_h, ids_h) = process_log(npz_path, json_path, log_idx)

        all_X_turn.append(X_t)
        all_Y_turn.append(Y_t)
        all_ids_turn.append(ids_t)
        all_X_hill.append(X_h)
        all_Y_hill.append(Y_h)
        all_ids_hill.append(ids_h)

    np.save(OUTPUT_DIR / 'X_turn.npy', np.concatenate(all_X_turn))
    np.save(OUTPUT_DIR / 'Y_turn.npy', np.concatenate(all_Y_turn))
    np.save(OUTPUT_DIR / 'log_ids_turn.npy', np.concatenate(all_ids_turn))
    
    np.save(OUTPUT_DIR / 'X_hill.npy', np.concatenate(all_X_hill))
    np.save(OUTPUT_DIR / 'Y_hill.npy', np.concatenate(all_Y_hill))
    np.save(OUTPUT_DIR / 'log_ids_hill.npy', np.concatenate(all_ids_hill))

    print(f"Končano. Shranjeno v {OUTPUT_DIR}")

if __name__ == '__main__':
    main()