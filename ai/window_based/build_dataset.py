"""
Pipeline za pripravo training dataseta za random forest/RNN iz IMU .npz logov in JSON labelov.

Izhod:
    X_all.npy       shape (N, 100, 9)  -- okna: 2s @ 50Hz, 9 kanalov
    Y_turn_all.npy  shape (N,)         -- 0=none  1=left   2=right
    Y_hill_all.npy  shape (N,)         -- 0=none  1=up     2=down

"""

import sys
import json
from pathlib import Path

import numpy as np

# ── Poti ──────────────────────────────────────────────────────────────────────

BASE_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'training_data'
NPZ_DIR       = BASE_DATA_DIR / 'parsed_data'
JSON_DIR      = BASE_DATA_DIR / 'labeled_data_json'
OUTPUT_DIR    = Path(__file__).parent / 'dataset_output'

# ── Parametri ─────────────────────────────────────────────────────────────────

TARGET_FS    = 50     # Hz -- skupna frekvenca
WINDOW_SIZE  = 100    #  št. vzorcev = 2s @ 50Hz
STRIDE       = 25     #  = 0.5s premik (75% overlap)
THRESHOLD    = 0.5    # minimalni delež okna pokrit z labelom
CUTOFF_HZ    = 5.0    # low-pass filter cutoff
FILTER_ORDER = 4      # Butterworth

TURN_MAP = {'none': 0, 'left': 1, 'right': 2}
HILL_MAP  = {'none': 0, 'up':   1, 'down':  2}

# ── Import preprocessing.py ekipe ────────────────────────────────────────────

from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    resample_sensors_to_common_grid,
    preprocess_sensor_data,
)

# ── Razrez na okna ───────────────────────────────────────────────────────────

def _label_window(t_start, t_end, labels, channel, threshold):
    """
    Poišče label za en kanal (turn/hill) v enem oknu.
    Vrne string razreda ali None če prekrivanje < threshold.
    """
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


def make_windows(signal, t, labels, window_size, stride, threshold):
    """
    Razreže signal na prekrivajoča se okna in vsakemu dodeli label.

    Parametri:
        signal      (T, 9) -- preprocessiran signal
        t           (T,)   -- časovna os v sekundah
        labels      list   -- labeli iz JSON['labels']
        window_size int    -- dolžina okna v vzorcih
        stride      int    -- premik med okni v vzorcih
        threshold   float  -- min. delež pokritosti za dodelitev labela

    Vrne:
        X      (N, window_size, 9)
        Y_turn (N,)  int  0=none 1=left 2=right
        Y_hill (N,)  int  0=none 1=up   2=down
    """
    X, Y_turn, Y_hill = [], [], []
    n_windows = (len(signal) - window_size) // stride + 1

    for i in range(n_windows):
        i0, i1  = i * stride, i * stride + window_size
        t_start = float(t[i0])
        t_end   = float(t[i1 - 1])

        turn_lbl = _label_window(t_start, t_end, labels, 'turn', threshold)
        hill_lbl = _label_window(t_start, t_end, labels, 'hill', threshold)

        # okno popolnoma izven labeliranega območja → izpusti
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

    return np.array(X), np.array(Y_turn), np.array(Y_hill)


# ── Glavni pipeline ──────────────────────────────────────────────────────────

def process_log(npz_path, json_path):
    """En log → (X, Y_turn, Y_hill)."""

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

    return make_windows(signal, t, labels, WINDOW_SIZE, STRIDE, THRESHOLD)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(NPZ_DIR.glob('*.npz'))
    if not npz_files:
        print(f"[NAPAKA] Ni .npz filov v: {NPZ_DIR.resolve()}")
        sys.exit(1)

    all_X, all_Y_turn, all_Y_hill = [], [], []

    for npz_path in npz_files:
        json_path = JSON_DIR / (npz_path.stem + '_labels.json')

        if not json_path.exists():
            print(f"[SKIP] {npz_path.name} -- ni labels.json")
            continue

        X, Y_turn, Y_hill = process_log(npz_path, json_path)
        all_X.append(X)
        all_Y_turn.append(Y_turn)
        all_Y_hill.append(Y_hill)

        print(
            f"{npz_path.name}: {len(X):3d} oken | "
            f"turn  none={np.sum(Y_turn==0)} left={np.sum(Y_turn==1)} right={np.sum(Y_turn==2)} | "
            f"hill  none={np.sum(Y_hill==0)} up={np.sum(Y_hill==1)} down={np.sum(Y_hill==2)}"
        )

    if not all_X:
        print("Ni nobenih logov za procesiranje.")
        sys.exit(1)

    X_all      = np.concatenate(all_X,      axis=0)
    Y_turn_all = np.concatenate(all_Y_turn, axis=0)
    Y_hill_all = np.concatenate(all_Y_hill, axis=0)

    print(f"\nSkupaj: {len(X_all)} oken, shape={X_all.shape}")
    print(f"Y_turn:  none={np.sum(Y_turn_all==0)}  left={np.sum(Y_turn_all==1)}  right={np.sum(Y_turn_all==2)}")
    print(f"Y_hill:  none={np.sum(Y_hill_all==0)}  up={np.sum(Y_hill_all==1)}    down={np.sum(Y_hill_all==2)}")

    np.save(OUTPUT_DIR / 'X_all.npy',      X_all)
    np.save(OUTPUT_DIR / 'Y_turn_all.npy', Y_turn_all)
    np.save(OUTPUT_DIR / 'Y_hill_all.npy', Y_hill_all)

    print(f"\nDataset shranjen v: {OUTPUT_DIR.resolve()}")
    print(f"  X_all.npy       {X_all.shape}  float64")
    print(f"  Y_turn_all.npy  {Y_turn_all.shape}  int  (0=none 1=left 2=right)")
    print(f"  Y_hill_all.npy  {Y_hill_all.shape}  int  (0=none 1=up   2=down)")


if __name__ == '__main__':
    main()