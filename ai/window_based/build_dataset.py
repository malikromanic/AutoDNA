"""
Pipeline za pripravo training dataseta iz IMU .npz logov in JSON labelov.

Izboljšave v2:
    - Filter "preblage" labele pod MIN_HILL_ANGLE in MIN_TURN_ANGLE -> 'none'.
    - Dva ločena dataseta: 2s okna za turn, 4s okna za hill.
    - **NOVO**: Komplementarni filter za pitch in roll (kanala 9 in 10).
    - **NOVO**: Dual-threshold logika za label assignment (anti-šum pri kratkih
      zaporednih segmentih z nasprotnimi smermi).
    - Ohranja log_ids za pravilen GroupKFold split.

Signal layout (11 kanalov):
    [0] gyro_x   [1] gyro_y   [2] gyro_z
    [3] accel_x  [4] accel_y  [5] accel_z
    [6] mag_x    [7] mag_y    [8] mag_z
    [9] pitch_cf [10] roll_cf      <-- komplementarni filter
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
CUTOFF_HZ    = 5.0
FILTER_ORDER = 4


#TODO: uravnotežit značilke (testno) - da se vidi slika

# Dva ločena okna
WINDOW_SIZE_TURN = 100   # 2s
STRIDE_TURN      = 25
WINDOW_SIZE_HILL = 200   # 4s
STRIDE_HILL      = 50

# Threshold za label assignment
# Turn: enojen prag 0.5 (2s okna so kratka, dogodek je hiter)
# Hill: dual-threshold (overlap >= 0.30 IN dominantnost >= 2x nad konkurenco)
THRESHOLD_TURN          = 0.5
THRESHOLD_HILL_MIN      = 0.30
THRESHOLD_HILL_DOMINANT = 2.0

# Filter za preblage labele
MIN_HILL_ANGLE = 1.0
MIN_TURN_ANGLE = 5.0

# Komplementarni filter
CF_ALPHA = 0.98              # alpha = tau/(tau+dt), tau~1s pri dt=0.02s
CF_INIT_SECONDS = 2.0        # koliko začetnih sekund uporabimo za bias init
CF_GYRO_IN_DEGREES = False   # gyro je že v rad/s (preverjeno: range ±1, ne ±50)

TURN_MAP = {'none': 0, 'left': 1, 'right': 2}
HILL_MAP = {'none': 0, 'up':   1, 'down':  2}

# ── Import preprocessing.py ────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from AutoDNA.ai.preprocessing import (
    load_sensor_data,
    resample_sensors_to_common_grid,
    preprocess_sensor_data,
)

# ── Komplementarni filter ─────────────────────────────────────────────────────

def complementary_filter(ax, ay, az, gx, gy, fs,
                        alpha=CF_ALPHA, init_seconds=CF_INIT_SECONDS,
                        gyro_in_degrees=CF_GYRO_IN_DEGREES):
    """
    Komplementarni filter za pitch in roll.

    Pitch (naklon naprej/nazaj) -- kombinacija integriranega gy in arctan2(ax,...).
    Roll  (nagib v stran)        -- kombinacija integriranega gx in arctan2(ay, az).

    Initial bias se izračuna kot povprečje prvih `init_seconds` sekund.
    To prepreči trajni offset, če log začne med pospeševanjem ali zaviranjem.

    Vrne pitch in roll v RADIANIH.
    """
    n  = len(ax)
    dt = 1.0 / fs

    if gyro_in_degrees:
        gx_rad = np.radians(gx)
        gy_rad = np.radians(gy)
    else:
        gx_rad = gx.astype(np.float64)
        gy_rad = gy.astype(np.float64)

    # Initial bias: povprečje prvih N sekund
    init_n = min(int(init_seconds * fs), n // 4)
    init_n = max(init_n, 1)

    ax_i = ax[:init_n].mean()
    ay_i = ay[:init_n].mean()
    az_i = az[:init_n].mean()

    pitch = np.zeros(n, dtype=np.float64)
    roll  = np.zeros(n, dtype=np.float64)

    pitch[0] = np.arctan2(ax_i, np.sqrt(ay_i**2 + az_i**2))
    roll[0]  = np.arctan2(ay_i, az_i)

    for i in range(1, n):
        # Accel-only ocena (šumna ob dinamiki, ampak brez drifta)
        p_acc = np.arctan2(ax[i], np.sqrt(ay[i]**2 + az[i]**2))
        r_acc = np.arctan2(ay[i], az[i])

        # Gyro integration + accel korekcija
        pitch[i] = alpha * (pitch[i-1] + gy_rad[i] * dt) + (1 - alpha) * p_acc
        roll[i]  = alpha * (roll[i-1]  + gx_rad[i] * dt) + (1 - alpha) * r_acc

    return pitch.astype(np.float32), roll.astype(np.float32)

# ── Čiščenje labelov ──────────────────────────────────────────────────────────

def filter_weak_labels(labels):
    """Vrne kopijo labelov; turn/hill pod min kotom -> None."""
    cleaned = []
    for lbl in labels:
        new_lbl = dict(lbl)
        if new_lbl.get('turn') is not None:
            if abs(new_lbl['turn'].get('angleDeg', 0)) < MIN_TURN_ANGLE:
                new_lbl['turn'] = None
        if new_lbl.get('hill') is not None:
            if abs(new_lbl['hill'].get('angleDeg', 0)) < MIN_HILL_ANGLE:
                new_lbl['hill'] = None
        cleaned.append(new_lbl)
    return cleaned

# ── Label assignment za okno ──────────────────────────────────────────────────

def _label_window_turn(t_start, t_end, labels, threshold):
    """Enojen prag za turn (kratka okna, hitra dinamika)."""
    window_dur   = t_end - t_start
    best_label   = 'none'
    best_overlap = 0.0
    for lbl in labels:
        if lbl.get('turn') is None:
            continue
        overlap = min(t_end, lbl['t_end']) - max(t_start, lbl['t_start'])
        if overlap > best_overlap:
            best_overlap = overlap
            best_label   = lbl['turn']['dir']
    return best_label if best_overlap / window_dur >= threshold else 'none'


def _label_window_hill(t_start, t_end, labels,
                       min_overlap_ratio=THRESHOLD_HILL_MIN,
                       dominance_factor=THRESHOLD_HILL_DOMINANT):
    """
    Dual-threshold za hill:
        - dominantna smer mora pokrivati >= min_overlap_ratio okna
        - in mora biti vsaj dominance_factor x daljša od druge smeri
    Sicer 'none'.
    """
    window_dur = t_end - t_start

    overlap_by_dir = {'up': 0.0, 'down': 0.0}
    for lbl in labels:
        if lbl.get('hill') is None:
            continue
        overlap = min(t_end, lbl['t_end']) - max(t_start, lbl['t_start'])
        if overlap <= 0:
            continue
        direction = lbl['hill']['dir']
        if direction in overlap_by_dir:
            overlap_by_dir[direction] += overlap

    best_dir, best_overlap = max(overlap_by_dir.items(), key=lambda x: x[1])
    other_overlap = sum(v for k, v in overlap_by_dir.items() if k != best_dir)

    if best_overlap / window_dur < min_overlap_ratio:
        return 'none'
    if other_overlap > 0 and best_overlap < dominance_factor * other_overlap:
        return 'none'
    return best_dir

# ── Razrez na okna ───────────────────────────────────────────────────────────

def make_windows(signal, t, labels, log_idx, window_size, stride, channel):
    X, Y, log_ids = [], [], []
    n_windows = (len(signal) - window_size) // stride + 1
    label_map = TURN_MAP if channel == 'turn' else HILL_MAP

    for i in range(n_windows):
        i0, i1  = i * stride, i * stride + window_size
        t_start = float(t[i0])
        t_end   = float(t[i1 - 1])

        if channel == 'turn':
            lbl = _label_window_turn(t_start, t_end, labels, THRESHOLD_TURN)
        else:
            lbl = _label_window_hill(t_start, t_end, labels)

        any_coverage = any(
            min(t_end, l['t_end']) - max(t_start, l['t_start']) > 0
            for l in labels
        )
        if not any_coverage:
            continue

        X.append(signal[i0:i1])
        Y.append(label_map[lbl])
        log_ids.append(log_idx)

    return np.array(X), np.array(Y), np.array(log_ids)

# ── Glavni pipeline ───────────────────────────────────────────────────────────

def process_log(npz_path, json_path, log_idx):
    sensors   = load_sensor_data(npz_path)
    resampled = resample_sensors_to_common_grid(sensors, target_fs=TARGET_FS)
    processed = preprocess_sensor_data(
        resampled, fs=TARGET_FS, cutoff=CUTOFF_HZ, filter_order=FILTER_ORDER
    )

    ax = processed['accel']['x']
    ay = processed['accel']['y']
    az = processed['accel']['z']
    gx = processed['gyro']['x']
    gy = processed['gyro']['y']
    gz = processed['gyro']['z']
    mx = processed['mag']['x']
    my = processed['mag']['y']
    mz = processed['mag']['z']

    # Komplementarni filter -> pitch_cf, roll_cf v RADIANIH
    pitch_cf, roll_cf = complementary_filter(
        ax, ay, az, gx, gy, fs=TARGET_FS
    )

    signal = np.column_stack([
        gx, gy, gz,
        ax, ay, az,
        mx, my, mz,
        pitch_cf, roll_cf,
    ]).astype(np.float32)

    t = processed['gyro']['ts']

    with open(json_path) as f:
        labels = json.load(f)['labels']
    labels = filter_weak_labels(labels)

    X_t, Y_t, ids_t = make_windows(signal, t, labels, log_idx,
                                    WINDOW_SIZE_TURN, STRIDE_TURN, 'turn')
    X_h, Y_h, ids_h = make_windows(signal, t, labels, log_idx,
                                    WINDOW_SIZE_HILL, STRIDE_HILL, 'hill')

    return (X_t, Y_t, ids_t), (X_h, Y_h, ids_h)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(NPZ_DIR.glob('*.npz'))
    if not npz_files:
        print(f"[NAPAKA] Ni .npz filov v: {NPZ_DIR}")
        sys.exit(1)

    all_X_turn, all_Y_turn, all_ids_turn = [], [], []
    all_X_hill, all_Y_hill, all_ids_hill = [], [], []

    for log_idx, npz_path in enumerate(npz_files):
        json_path = JSON_DIR / (npz_path.stem + '_labels.json')
        if not json_path.exists():
            print(f"[SKIP] {npz_path.name} -- ni labels.json")
            continue

        (X_t, Y_t, ids_t), (X_h, Y_h, ids_h) = process_log(npz_path, json_path, log_idx)

        all_X_turn.append(X_t); all_Y_turn.append(Y_t); all_ids_turn.append(ids_t)
        all_X_hill.append(X_h); all_Y_hill.append(Y_h); all_ids_hill.append(ids_h)

        print(
            f"[{log_idx:2d}] {npz_path.name}: "
            f"turn={len(X_t):3d} (n={np.sum(Y_t==0)} l={np.sum(Y_t==1)} r={np.sum(Y_t==2)}) | "
            f"hill={len(X_h):3d} (n={np.sum(Y_h==0)} u={np.sum(Y_h==1)} d={np.sum(Y_h==2)})"
        )

    X_turn   = np.concatenate(all_X_turn)
    Y_turn   = np.concatenate(all_Y_turn)
    ids_turn = np.concatenate(all_ids_turn)
    X_hill   = np.concatenate(all_X_hill)
    Y_hill   = np.concatenate(all_Y_hill)
    ids_hill = np.concatenate(all_ids_hill)

    print(f"\n=== TURN dataset ({WINDOW_SIZE_TURN/TARGET_FS:.1f}s okna) ===")
    print(f"  X: {X_turn.shape}  (11 kanalov)")
    print(f"  none={np.sum(Y_turn==0)}  left={np.sum(Y_turn==1)}  right={np.sum(Y_turn==2)}")

    print(f"\n=== HILL dataset ({WINDOW_SIZE_HILL/TARGET_FS:.1f}s okna) ===")
    print(f"  X: {X_hill.shape}")
    print(f"  none={np.sum(Y_hill==0)}  up={np.sum(Y_hill==1)}  down={np.sum(Y_hill==2)}")

    np.save(OUTPUT_DIR / 'X_turn.npy',        X_turn)
    np.save(OUTPUT_DIR / 'Y_turn.npy',        Y_turn)
    np.save(OUTPUT_DIR / 'log_ids_turn.npy',  ids_turn)
    np.save(OUTPUT_DIR / 'X_hill.npy',        X_hill)
    np.save(OUTPUT_DIR / 'Y_hill.npy',        Y_hill)
    np.save(OUTPUT_DIR / 'log_ids_hill.npy',  ids_hill)

    print(f"\nDataset shranjen v: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()