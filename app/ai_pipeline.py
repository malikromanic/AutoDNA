# ============================================================================
# AutoDNA Real AI Pipeline
#
# Data flow:
#   BIN file ──► stm32 parser ──► preprocessing ──► windows ──► XGBoost
#
# Uses existing:  stm32/bin_parser/stm_utils.py
#                 ai/preprocessing.py
#                 ai/window_based/dataset_output/model_*.json
# ============================================================================
"""Window-based IMU turn/hill classifier (XGBoost), not wired into the live app.

An earlier architecture used this to detect turns and hills directly
from IMU sensor windows. The current app detects both from GPS/DEM data
instead (see :mod:`app.gps_analysis`), so nothing in the live
application imports this module — it's exercised only by
``tests/test_pipeline.py`` and kept for reference / possible future
reintroduction of a learned detector.
"""

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# dashboard/main.py adds GITHUB/AutoDNA to sys.path; stm_utils.py internally
# uses "from AutoDNA.stm32..." so GITHUB (parent) must also be on path.
_ROOT   = Path(__file__).resolve().parent.parent   # …/GITHUB/AutoDNA
_GITHUB = _ROOT.parent                              # …/GITHUB
for _p in (str(_ROOT), str(_GITHUB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AutoDNA.stm32.bin_parser.stm_utils import read_packets_from_file  # noqa: E402
from AutoDNA.ai.preprocessing import (  # noqa: E402
    load_sensor_data,
    resample_sensors_to_common_grid,
    preprocess_sensor_data,
)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 100   # samples per prediction window  (2 s @ 50 Hz)
STRIDE      = 25    # samples between consecutive windows (matches build_dataset.py)
TARGET_FS   = 50    # Hz after resampling
CUTOFF_HZ   = 5.0  # Butterworth low-pass cutoff

# Column indices in the (N, 9) IMU signal matrix
GYRO_X, GYRO_Y, GYRO_Z    = 0, 1, 2
ACCEL_X, ACCEL_Y, ACCEL_Z = 3, 4, 5
MAG_X,  MAG_Y,  MAG_Z     = 6, 7, 8

# Model output classes
TURN_LABELS = {0: 'none', 1: 'left', 2: 'right'}
HILL_LABELS = {0: 'none', 1: 'up',   2: 'down'}

# Paths
MODEL_DIR       = _ROOT / 'ai' / 'window_based' / 'dataset_output'
PARSED_DATA_DIR = _ROOT / 'data' / 'training_data' / 'parsed_data'
LABELS_DIR      = _ROOT / 'data' / 'training_data' / 'labeled_data_json'


# ── Feature extraction (must be identical to train_xgboost.py) ───────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """38 hand-crafted features from a (100, 9) IMU window."""
    feats: list[float] = []
    gz = window[:, GYRO_Z]
    feats += [gz.max(), gz.min(), gz.mean(), gz.std(),
              float(np.sum(gz)) / len(gz), float(np.abs(gz).max()),
              float(np.sum(gz >  0.1)) / len(gz),
              float(np.sum(gz < -0.1)) / len(gz)]
    for ch in (GYRO_X, GYRO_Y):
        g = window[:, ch]
        feats += [float(g.mean()), float(g.std()), float(np.abs(g).max())]
    ax = window[:, ACCEL_X]
    feats += [float(ax.mean()), float(ax.std()), float(ax.max()), float(ax.min()),
              float(np.sum(ax)) / len(ax)]
    ay = window[:, ACCEL_Y]
    feats += [float(ay.mean()), float(ay.std()), float(np.abs(ay).max()),
              float(np.sum(ay < -0.2)) / len(ay),
              float(np.sum(ay >  0.2)) / len(ay)]
    az = window[:, ACCEL_Z]
    feats += [float(az.mean()), float(az.std())]
    for ch in (MAG_X, MAG_Y, MAG_Z):
        m = window[:, ch]
        feats += [float(m.mean()), float(m.max() - m.min())]
    corr = (float(np.corrcoef(gz, ay)[0, 1])
            if gz.std() > 0 and ay.std() > 0 else 0.0)
    feats += [corr,
              float(gz.std()) / (float(ax.std()) + 1e-6),
              float(np.mean(gz ** 2)),
              float(np.mean(ax ** 2))]
    return np.array(feats, dtype=np.float32)


# ── BIN parsing + preprocessing ───────────────────────────────────────────────
def parse_and_preprocess_bin(bin_path: Path):
    """
    Parse a .BIN file through the full AutoDNA preprocessing pipeline.

    Returns
    -------
    signal     : (N, 9) float32  — [gyro_xyz, accel_xyz, mag_xyz], normalized
    timestamps : (N,)  float64  — seconds from recording start (0-indexed)
    """
    from collections import defaultdict

    # 1. Parse BIN → {sensor_name: [[ts_ms, x, y, z], ...]}
    packets = read_packets_from_file(str(bin_path))
    rows: dict[str, list] = defaultdict(list)
    for pkt in packets:
        for x, y, z in pkt.data:
            rows[pkt.sensor].append([float(pkt.ts), float(x), float(y), float(z)])

    if not rows:
        raise ValueError(f"No sensor packets parsed from {bin_path.name}")

    raw_arrays = {name: np.array(r, dtype=np.float32) for name, r in rows.items()}

    # 2. Write to a temp NPZ so load_sensor_data() can consume it
    fd, tmp = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        np.savez(tmp, **raw_arrays)
        sensors = load_sensor_data(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    # 3. Resample all sensors to a common 50 Hz grid
    resampled = resample_sensors_to_common_grid(sensors, target_fs=TARGET_FS)

    # 4. Smooth → lowpass → normalize
    processed = preprocess_sensor_data(resampled, fs=TARGET_FS, cutoff=CUTOFF_HZ)

    # 5. Stack into (N, 9): gyro_xyz | accel_xyz | mag_xyz
    signal = np.column_stack([
        processed['gyro']['x'],  processed['gyro']['y'],  processed['gyro']['z'],
        processed['accel']['x'], processed['accel']['y'], processed['accel']['z'],
        processed['mag']['x'],   processed['mag']['y'],   processed['mag']['z'],
    ]).astype(np.float32)

    timestamps = processed['gyro']['ts'].astype(np.float64)
    return signal, timestamps


# ── Windowing ─────────────────────────────────────────────────────────────────
def make_windows(signal: np.ndarray):
    """
    Overlapping windows from (N, 9) using STRIDE=25 (matches build_dataset.py).

    Returns
    -------
    windows       : (M, 100, 9)
    start_samples : (M,) int  — sample index in `signal` where each window starts
    """
    n_windows = (len(signal) - WINDOW_SIZE) // STRIDE + 1
    windows = np.stack([
        signal[i * STRIDE : i * STRIDE + WINDOW_SIZE]
        for i in range(n_windows)
    ])
    start_samples = np.arange(n_windows, dtype=np.int64) * STRIDE
    return windows, start_samples


# ── XGBoost inference ─────────────────────────────────────────────────────────
def models_exist() -> bool:
    """Whether both trained model files are present in ``MODEL_DIR``."""
    return ((MODEL_DIR / 'model_turn.json').exists() and
            (MODEL_DIR / 'model_hill.json').exists())


def load_xgb_models():
    """Load the turn and hill XGBoost classifiers from ``MODEL_DIR``.

    Raises:
        FileNotFoundError: Either model file is missing — train them
            first via :func:`train_models`.
    """
    from xgboost import XGBClassifier
    if not models_exist():
        raise FileNotFoundError(
            f"XGBoost models not found in {MODEL_DIR}. "
            "Train them first via the app Setup dialog."
        )
    m_turn = XGBClassifier()
    m_turn.load_model(str(MODEL_DIR / 'model_turn.json'))
    m_hill = XGBClassifier()
    m_hill.load_model(str(MODEL_DIR / 'model_hill.json'))
    return m_turn, m_hill


def predict_windows(windows: np.ndarray):
    """
    Feature-extract and infer on windows (M, 100, 9).

    Returns
    -------
    turn_preds : (M,) int32   — 0=none  1=left  2=right
    hill_preds : (M,) int32   — 0=none  1=up    2=down
    turn_proba : (M, 3) f32   — class probabilities
    hill_proba : (M, 3) f32
    """
    m_turn, m_hill = load_xgb_models()
    X = np.array([extract_features(windows[i]) for i in range(len(windows))])
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    return (
        m_turn.predict(X).astype(np.int32),
        m_hill.predict(X).astype(np.int32),
        m_turn.predict_proba(X).astype(np.float32),
        m_hill.predict_proba(X).astype(np.float32),
    )


# ── Model training ────────────────────────────────────────────────────────────
def train_models(progress_cb=None):
    """
    Train XGBoost turn + hill classifiers on all labeled training logs.
    Saves model_turn.json and model_hill.json to MODEL_DIR.

    progress_cb : callable(msg: str, pct: int) | None  — for UI progress bar
    """
    from xgboost import XGBClassifier

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(
        f for f in PARSED_DATA_DIR.glob('LOG*.npz')
        if 'preprocessed' not in f.stem
    )
    total = len(npz_files)
    all_X, all_Yt, all_Yh = [], [], []

    for i, npz_path in enumerate(npz_files):
        json_path = LABELS_DIR / (npz_path.stem + '_labels.json')
        if not json_path.exists():
            continue
        if progress_cb:
            progress_cb(f"Loading {npz_path.name}  ({i+1}/{total})", int(i / total * 75))
        try:
            sensors   = load_sensor_data(str(npz_path))
            resampled = resample_sensors_to_common_grid(sensors, target_fs=TARGET_FS)
            processed = preprocess_sensor_data(resampled, fs=TARGET_FS, cutoff=CUTOFF_HZ)
        except Exception as exc:
            print(f"[SKIP] {npz_path.name}: {exc}")
            continue

        signal = np.column_stack([
            processed['gyro']['x'],  processed['gyro']['y'],  processed['gyro']['z'],
            processed['accel']['x'], processed['accel']['y'], processed['accel']['z'],
            processed['mag']['x'],   processed['mag']['y'],   processed['mag']['z'],
        ]).astype(np.float32)
        ts = processed['gyro']['ts']

        with open(json_path) as f:
            labels = json.load(f)['labels']

        n_w = (len(signal) - WINDOW_SIZE) // STRIDE + 1
        for w in range(n_w):
            i0, i1 = w * STRIDE, w * STRIDE + WINDOW_SIZE
            all_X.append(signal[i0:i1])
            t_lbl, h_lbl = _label_window(float(ts[i0]), float(ts[i1 - 1]), labels)
            all_Yt.append(t_lbl)
            all_Yh.append(h_lbl)

    if not all_X:
        raise ValueError("No training windows found. Check parsed_data and labels dirs.")

    if progress_cb:
        progress_cb("Extracting 38-feature vectors…", 76)

    X  = np.array(all_X)
    Yt = np.array(all_Yt)
    Yh = np.array(all_Yh)

    X_feat = np.array([extract_features(X[i]) for i in range(len(X))])
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1.0, neginf=-1.0)

    params = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric='mlogloss', verbosity=0)

    if progress_cb:
        progress_cb(f"Training turn classifier  ({len(X)} windows)…", 80)
    m_turn = XGBClassifier(**params)
    m_turn.fit(X_feat, Yt)

    if progress_cb:
        progress_cb("Training hill classifier…", 92)
    m_hill = XGBClassifier(**params)
    m_hill.fit(X_feat, Yh)

    m_turn.save_model(str(MODEL_DIR / 'model_turn.json'))
    m_hill.save_model(str(MODEL_DIR / 'model_hill.json'))

    if progress_cb:
        progress_cb(f"Done — {len(X)} windows from {total} logs.", 100)

    return len(X), int(Yt.sum()), int(Yh.sum())


def _label_window(t_start: float, t_end: float, labels: list):
    """Assign a window its turn/hill class by majority time-overlap with the labeled intervals.

    A class is only assigned if the best-overlapping label covers at
    least 30% of the window's duration; otherwise the window is
    labeled 'none' for that task.
    """
    TURN_MAP = {'none': 0, 'left': 1, 'right': 2}
    HILL_MAP  = {'none': 0, 'up': 1, 'down': 2}
    dur = max(t_end - t_start, 1e-9)
    bt = ('none', 0.0)
    bh = ('none', 0.0)
    for lbl in labels:
        ov = min(t_end, lbl['t_end']) - max(t_start, lbl['t_start'])
        if ov <= 0:
            continue
        if lbl.get('turn') and ov > bt[1]:
            bt = (lbl['turn']['dir'], ov)
        if lbl.get('hill') and ov > bh[1]:
            bh = (lbl['hill']['dir'], ov)
    return (
        TURN_MAP[bt[0]] if bt[1] / dur >= 0.3 else 0,
        HILL_MAP[bh[0]] if bh[1] / dur >= 0.3 else 0,
    )
