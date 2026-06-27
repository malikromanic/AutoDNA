# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 21:53:17 2026

@author: mihal
"""

"""
fuel_features.py
=================
Per-drive aggregate feature extraction + persistent storage for the
fuel-regression model. Hooks into the live app AFTER DriveDataLoader has
loaded the GPS/OBD CSV, since that's where distance/fuel ground truth
and segment data come from.
"""

import json
import datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

from stm32.bin_parser.stm_utils import read_packets_from_file, save_to_npz
from ai.preprocessing import load_sensor_data, resample_sensors_to_common_grid

STORE_PATH = Path(__file__).resolve().parent.parent / 'data' / 'fuel_features_store.json'

HARSH_JERK_THRESHOLD = 855.0

LIVE_DEVICE_NEEDS_RIGHT_FLIP_CORRECTION = True


def find_bin_in_drive_dir(drive_dir: Path) -> Path:
    """Find the STM .bin recording in a drive folder (alongside the GPS CSV)."""
    bins = sorted(Path(drive_dir).glob('*.bin')) + sorted(Path(drive_dir).glob('*.BIN'))
    if not bins:
        raise FileNotFoundError(f"No .bin (STM IMU) file found in {drive_dir}")
    if len(bins) > 1:
        print(f"[AutoDNA] Multiple .bin files found — using first: {bins[0].name}")
    return bins[0]


def convert_bin_to_npz(bin_path: Path) -> Path:
    """
    Convert a .bin recording to .npz, caching the result next to the .bin
    so repeated drive loads don't re-parse the same file.
    """
    npz_path = bin_path.with_suffix('.npz')

    if npz_path.exists() and npz_path.stat().st_mtime >= bin_path.stat().st_mtime:
        print(f"[AutoDNA] Using cached npz: {npz_path.name}")
        return npz_path

    print(f"[AutoDNA] Converting {bin_path.name} -> {npz_path.name}")
    packets = read_packets_from_file(str(bin_path))
    if not packets:
        raise ValueError(f"No packets parsed from {bin_path.name}")
    save_to_npz(packets, str(npz_path))
    return npz_path


def swap_axes_right_flip(sensors):
    """Axis correction for the live recording device's mounting orientation."""
    corrected = {}
    for name, data in sensors.items():
        corrected[name] = {
            'ts':  data['ts'],
            'x':   data['y'].copy(),
            'y':  -data['x'].copy(),
            'z':   data['z'].copy(),
        }
    return corrected


# ── feature computation ──────────────────────────────────────────────────────

"""def compute_drive_features(sensors, distance_km: float, duration_min: float) -> dict:
    if LIVE_DEVICE_NEEDS_RIGHT_FLIP_CORRECTION:
        sensors = swap_axes_right_flip(sensors)

    sensors_common = resample_sensors_to_common_grid(sensors)

    accel_x = sensors_common['accel']['x']
    ts      = sensors_common['accel']['ts']
    
    fs = 1.0 / np.median(np.diff(ts))
    
    from AutoDNA.ai.preprocessing import _timestamps_to_seconds, estimate_sampling_rate
    for name, data in sensors.items():
        ts_sec = _timestamps_to_seconds(data['ts'])
        fs_est = estimate_sampling_rate(ts_sec)
          
    window = max(3, int(fs * 0.1))
    accel_x_smooth = np.convolve(accel_x, np.ones(window) / window, mode='same')

    jerk = np.diff(accel_x_smooth) * fs
    n_harsh_accel   = int(np.sum(jerk >  HARSH_JERK_THRESHOLD))
    n_harsh_braking = int(np.sum(jerk < -HARSH_JERK_THRESHOLD))
    
    features = {}
    if distance_km and distance_km > 0:
        features['harsh_accel_rate']   = n_harsh_accel   / distance_km
        features['harsh_braking_rate'] = n_harsh_braking / distance_km
    else:
        features['harsh_accel_rate']   = 0.0
        features['harsh_braking_rate'] = 0.0

    features['accel_x_std'] = float(accel_x.std())
    features['rms_jerk']    = float(np.sqrt(np.mean(jerk**2)))

    features['distance_km']     = distance_km if distance_km else 0.0
    features['inv_distance_km'] = 1.0 / (distance_km + 1.0) if distance_km else 0.0
    features['avg_speed_kmh']   = (distance_km / (duration_min / 60.0)
                                    if distance_km and duration_min > 0 else 0.0)

    return features"""


def compute_drive_features(sensors_raw: dict, drive_data) -> dict:
    """
    Compute the full 7-feature dict for one drive.
    Combines IMU features (from STM32) and GPS features (from OBD CSV).
    """
    if LIVE_DEVICE_NEEDS_RIGHT_FLIP_CORRECTION:
        sensors_raw = swap_axes_right_flip(sensors_raw)

    sensors_common = resample_sensors_to_common_grid(sensors_raw)

    accel_x = sensors_common['accel']['x']
    ts      = sensors_common['accel']['ts']
    fs      = 1.0 / np.median(np.diff(ts))

    window         = max(3, int(fs * 0.1))
    accel_x_smooth = np.convolve(accel_x, np.ones(window) / window, mode='same')
    jerk           = np.diff(accel_x_smooth) * fs

    n_harsh_accel   = int(np.sum(jerk >  HARSH_JERK_THRESHOLD))
    n_harsh_braking = int(np.sum(jerk < -HARSH_JERK_THRESHOLD))

    distance_km = drive_data.drive_distance_km
    gps_speed   = drive_data.gps_speed   # km/h array

    features = {}

    # ── IMU features ──────────────────────────────────────────────────────
    if distance_km and distance_km > 0:
        features['harsh_accel_rate']   = n_harsh_accel   / distance_km
        features['harsh_braking_rate'] = n_harsh_braking / distance_km
    else:
        features['harsh_accel_rate']   = 0.0
        features['harsh_braking_rate'] = 0.0

    features['rms_jerk'] = float(np.sqrt(np.mean(jerk ** 2)))

    # ── GPS features ──────────────────────────────────────────────────────
    if len(gps_speed) > 0:
        features['avg_speed_kmh']     = float(np.mean(gps_speed))
        features['speed_variability'] = float(np.std(gps_speed))
        features['idle_time_pct']     = float((gps_speed < 2.0).mean() * 100)
    else:
        features['avg_speed_kmh']     = 0.0
        features['speed_variability'] = 0.0
        features['idle_time_pct']     = 0.0

    # ── metadata ──────────────────────────────────────────────────────────
    features['distance_km'] = distance_km if distance_km else 0.0

    print(f"[AutoDNA] Features: {', '.join(f'{k}={v:.2f}' for k, v in features.items())}")
    return features


# ── persistent storage ──────────────────────────────────────────────────────

@dataclass
class DriveRecord:
    drive_name:  str
    drive_path:  str        # path to the drive folder — needed to reload the drive from the Drives tab
    features:    dict
    fuel_l100km: float
    timestamp:   str
 
 
def load_store() -> list[dict]:
    if not STORE_PATH.exists():
        return []
    return json.loads(STORE_PATH.read_text())
 
 
def save_store(records: list[dict]):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(records, indent=2))
 
 
def append_drive_record(drive_name: str, drive_path: str,
                        features: dict, fuel_l100km: float):
    """Append one drive record to the persistent store, skipping duplicates."""
    records = load_store()
 
    # avoid duplicating the same drive if reloaded
    records = [r for r in records if r.get('drive_name') != drive_name]
 
    records.append(asdict(DriveRecord(
        drive_name=drive_name,
        drive_path=drive_path,
        features=features,
        fuel_l100km=fuel_l100km,
        timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
    )))
    save_store(records)
    return records
 
 
def store_to_xy(records: list[dict]):
    """Convert stored records into (X, y, feature_names) for model fitting."""
    if not records:
        return None, None, None
    feature_names = list(records[0]['features'].keys())
    X = np.array([[r['features'][name] for name in feature_names] for r in records])
    y = np.array([r['fuel_l100km'] for r in records])
    return X, y, feature_names
 
 
# ── full hook for the live app ─────────────────────────────────────────────
 
def process_drive_fuel_features(drive_dir: Path, drive_data) -> dict:
    """
    Full hook: find .bin in the drive folder, convert to .npz (cached),
    compute fuel-regression features, append to persistent store.
 
    Call AFTER DriveDataLoader.load_drive() — drive_data must already
    have drive_distance_km, drive_duration_sec, avg_fuel_l100km populated.
    """
    bin_path = find_bin_in_drive_dir(drive_dir)
    npz_path = convert_bin_to_npz(bin_path)
 
    sensors_raw = load_sensor_data(npz_path)  
 
    features = compute_drive_features(sensors_raw, drive_data)
 
    records = append_drive_record(
        drive_name=drive_data.drive_name,
        drive_path=str(drive_dir),      
        features=features,
        fuel_l100km=drive_data.avg_fuel_l100km,
    )
 
    print(f"[AutoDNA] Fuel features stored for {drive_data.drive_name} "
          f"(total drives in store: {len(records)})")
 
    return features
