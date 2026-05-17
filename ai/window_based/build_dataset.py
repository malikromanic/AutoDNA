"""
Pipeline za pripravo training dataseta iz IMU .npz logov in JSON labelov za xgboost in CNN.

Izhod:
    X_all.npy       shape (N, 100, 9)  -- okna: 2s @ 50Hz, 9 kanalov
    Y_turn_all.npy  shape (N,)         -- 0=none  1=left   2=right
    Y_hill_all.npy  shape (N,)         -- 0=none  1=up     2=down
    log_ids.npy     shape (N,)         -- kateremu logu pripada vsako okno
"""

import sys
import json
from pathlib import Path

import numpy as np

# ── Poti ─────────────────────────────────────────────────────────────────────

BASE_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'training_data'
NPZ_DIR       = BASE_DATA_DIR / 'parsed_data'
JSON_DIR      = BASE_DATA_DIR / 'labeled_data_json'
OUTPUT_DIR    = Path(__file__).parent / 'dataset_output'

# ── Parametri ─────────────────────────────────────────────────────────────────

TARGET_FS    = 50
WINDOW_SIZE  = 100
STRIDE       = 25
THRESHOLD    = 0.5
CUTOFF_HZ    = 5.0
FILTER_ORDER = 4

TURN_MAP = {'none': 0, 'left': 1, 'right': 2}
HILL_MAP  = {'none': 0, 'up':   1, 'down':  2}

# ── Import preprocessing.py ekipe ────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    resample_sensors_to_common_grid,
    preprocess_sensor_data,
)

# ── Razrez na okna ───────────────────────────────────────────────────────────

def _label_window(t_start, t_end, labels, channel, threshold):
    window_dur   = t_end - t_start
    best_label   = 'none'
    best_overlap = 0.0
    for lbl in labels:
        if lbl.get(channel) is None:
            continue
        overlap = min(t_end, lbl['t_end']) - max(t_start, lbl['t_start'])
        if overlap > best_overlap:
            best_overlap = overlap
            best_label   = lbl[channel]['dir']
    return best_label if best_overlap / window_dur >= threshold else None


def make_windows(signal, t, labels, log_idx, window_size, stride, threshold):
    """
    Vrne:
        X       (N, window_size, 9)
        Y_turn  (N,)
        Y_hill  (N,)
        log_ids (N,)  -- int indeks loga za vsako okno
    """
    X, Y_turn, Y_hill, log_ids = [], [], [], []
    n_windows = (len(signal) - window_size) // stride + 1

    for i in range(n_windows):
        i0, i1  = i * stride, i * stride + window_size
        t_start = float(t[i0])
        t_end   = float(t[i1 - 1])

        turn_lbl = _label_window(t_start, t_end, labels, 'turn', threshold)
        hill_lbl = _label_window(t_start, t_end, labels, 'hill', threshold)

        if turn_lbl is None and hill_lbl is None:
            any_coverage = any(
                min(t_end, l['t_end']) - max(t_start, l['t_start']) > 0
                for l in labels
            )
            if not any_coverage:
                continue

        X.append(signal[i0:i1])
        Y_turn.append(TURN_MAP[turn_lbl or 'none'])
        Y_hill.append(HILL_MAP[hill_lbl or 'none'])
        log_ids.append(log_idx)

    return np.array(X), np.array(Y_turn), np.array(Y_hill), np.array(log_ids)

# ── Glavni pipeline ───────────────────────────────────────────────────────────

def process_log(npz_path, json_path, log_idx):
    sensors   = load_sensor_data(npz_path)
    resampled = resample_sensors_to_common_grid(sensors, target_fs=TARGET_FS)
    processed = preprocess_sensor_data(
        resampled, fs=TARGET_FS, cutoff=CUTOFF_HZ, filter_order=FILTER_ORDER
    )
    signal = np.column_stack([
        processed['gyro']['x'],  processed['gyro']['y'],  processed['gyro']['z'],
        processed['accel']['x'], processed['accel']['y'], processed['accel']['z'],
        processed['mag']['x'],   processed['mag']['y'],   processed['mag']['z'],
    ])
    t = processed['gyro']['ts']

    with open(json_path) as f:
        labels = json.load(f)['labels']

    return make_windows(signal, t, labels, log_idx, WINDOW_SIZE, STRIDE, THRESHOLD)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(NPZ_DIR.glob('*.npz'))
    if not npz_files:
        print(f"[NAPAKA] Ni .npz filov v: {NPZ_DIR}")
        import sys; sys.exit(1)

    all_X, all_Y_turn, all_Y_hill, all_log_ids = [], [], [], []

    for log_idx, npz_path in enumerate(npz_files):
        json_path = JSON_DIR / (npz_path.stem + '_labels.json')
        if not json_path.exists():
            print(f"[SKIP] {npz_path.name} -- ni labels.json")
            continue

        X, Y_turn, Y_hill, log_ids = process_log(npz_path, json_path, log_idx)
        all_X.append(X)
        all_Y_turn.append(Y_turn)
        all_Y_hill.append(Y_hill)
        all_log_ids.append(log_ids)

        print(
            f"[{log_idx:2d}] {npz_path.name}: {len(X):3d} oken | "
            f"turn  none={np.sum(Y_turn==0)} left={np.sum(Y_turn==1)} right={np.sum(Y_turn==2)} | "
            f"hill  none={np.sum(Y_hill==0)} up={np.sum(Y_hill==1)} down={np.sum(Y_hill==2)}"
        )

    X_all      = np.concatenate(all_X)
    Y_turn_all = np.concatenate(all_Y_turn)
    Y_hill_all = np.concatenate(all_Y_hill)
    log_ids    = np.concatenate(all_log_ids)

    print(f"\nSkupaj: {len(X_all)} oken iz {len(npz_files)} logov")
    print(f"Y_turn:  none={np.sum(Y_turn_all==0)}  left={np.sum(Y_turn_all==1)}  right={np.sum(Y_turn_all==2)}")
    print(f"Y_hill:  none={np.sum(Y_hill_all==0)}  up={np.sum(Y_hill_all==1)}    down={np.sum(Y_hill_all==2)}")

    np.save(OUTPUT_DIR / 'X_all.npy',      X_all)
    np.save(OUTPUT_DIR / 'Y_turn_all.npy', Y_turn_all)
    np.save(OUTPUT_DIR / 'Y_hill_all.npy', Y_hill_all)
    np.save(OUTPUT_DIR / 'log_ids.npy',    log_ids)

    print(f"\nDataset shranjen v: {OUTPUT_DIR}")
    print(f"  X_all.npy       {X_all.shape}")
    print(f"  Y_turn_all.npy  {Y_turn_all.shape}  (0=none 1=left 2=right)")
    print(f"  Y_hill_all.npy  {Y_hill_all.shape}  (0=none 1=up   2=down)")
    print(f"  log_ids.npy     {log_ids.shape}      (0-{log_ids.max()} = kateri log)")


if __name__ == '__main__':
    main()