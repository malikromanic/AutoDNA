import math
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from app.ai_pipeline import (
    parse_and_preprocess_bin,
    make_windows,
    predict_windows,
    WINDOW_SIZE,
    STRIDE,
    TARGET_FS,
)

# Temporal smoothing: majority vote over ±N windows  (0 = disabled)
SMOOTH_HALF_WINDOW = 2


@dataclass
class DriveData:
    """Complete drive session — real IMU predictions mapped to real GPS."""

    # GPS track (deduplicated, movement-filtered)
    gps_timestamps: np.ndarray      # (N,)  absolute seconds
    gps_lat:        np.ndarray      # (N,)
    gps_lon:        np.ndarray      # (N,)
    gps_speed:      np.ndarray      # (N,)  km/h
    gps_heading:    np.ndarray      # (N,)  degrees 0-360

    # IMU signal preprocessed at 50 Hz, timestamps relative (0 = recording start)
    imu_timestamps: np.ndarray      # (K,)
    imu_signal:     np.ndarray      # (K, 9)  [gyro_xyz | accel_xyz | mag_xyz]

    # XGBoost predictions (smoothed) — one element per 100-sample / 2-second window
    xgb_turn_preds:     np.ndarray  # (M,)  int32   0=none 1=left 2=right  (smoothed)
    xgb_hill_preds:     np.ndarray  # (M,)  int32   0=none 1=up   2=down   (smoothed)
    xgb_turn_proba:     np.ndarray  # (M, 3) float32  class probabilities
    xgb_hill_proba:     np.ndarray  # (M, 3) float32

    # Raw (unsmoothed) predictions for diagnostics
    xgb_turn_preds_raw: np.ndarray  # (M,)  int32  before smoothing
    xgb_hill_preds_raw: np.ndarray  # (M,)  int32  before smoothing

    # GPS index range covered by each prediction window
    window_gps_start_idx:  np.ndarray  # (M,) int32
    window_gps_end_idx:    np.ndarray  # (M,) int32
    window_gps_center_idx: np.ndarray  # (M,) int32  — center-time GPS position

    # Metadata
    bin_file:          str
    gps_file:          str
    drive_name:        str
    drive_duration_sec: float
    drive_distance_km:  float
    window_size:       int           # 100
    window_stride:     int           # 25  (overlapping)
    target_fs:         int           # 50
    smooth_half:       int           # smoothing half-window used


class DriveDataLoader:
    """
    Open a drive directory containing exactly one .BIN and one .csv file.
    Runs the real AutoDNA pipeline and returns a DriveData instance.
    """

    def __init__(self, drive_dir: Path):
        self.drive_dir = Path(drive_dir)

    def load_drive(self) -> DriveData:
        bin_path, csv_path = self._find_files()

        # ── Step 1: IMU pipeline ─────────────────────────────────────────────
        print(f"[AutoDNA] Parsing BIN: {bin_path.name}")
        signal, imu_ts = parse_and_preprocess_bin(bin_path)
        print(f"          signal {signal.shape}  duration {imu_ts[-1]:.1f}s")

        windows, start_samples = make_windows(signal)
        M = len(windows)
        print(f"          {M} windows x {WINDOW_SIZE} samples")

        print("[AutoDNA] Running XGBoost inference...")
        turn_preds_raw, hill_preds_raw, turn_proba, hill_proba = predict_windows(windows)

        print(f"          raw turn: {np.bincount(turn_preds_raw, minlength=3).tolist()}")
        print(f"          raw hill: {np.bincount(hill_preds_raw, minlength=3).tolist()}")

        # ── Step 2: Temporal smoothing ───────────────────────────────────────
        turn_preds = _smooth_preds(turn_preds_raw, SMOOTH_HALF_WINDOW)
        hill_preds = _smooth_preds(hill_preds_raw, SMOOTH_HALF_WINDOW)
        print(f"          smoothed (+-{SMOOTH_HALF_WINDOW}) turn: {np.bincount(turn_preds, minlength=3).tolist()}")
        print(f"          smoothed (+-{SMOOTH_HALF_WINDOW}) hill: {np.bincount(hill_preds, minlength=3).tolist()}")

        # ── Step 3: GPS track ────────────────────────────────────────────────
        gps_ts, gps_lat, gps_lon, gps_speed, gps_heading = self._load_gps(csv_path)
        print(f"[AutoDNA] GPS {len(gps_lat)} points  span {gps_ts[-1]-gps_ts[0]:.0f}s")

        # ── Step 4: Align windows → GPS ──────────────────────────────────────
        win_start_idx, win_end_idx, win_center_idx = self._align_to_gps(
            imu_ts, start_samples, gps_ts
        )

        # ── Debug: print alignment for first 10 windows ──────────────────────
        print(f"\n[AutoDNA] {M} windows (stride={STRIDE})  — alignment (first 10):")
        print(f"  {'W':>4}  {'t0':>6}  {'t1':>6}  {'ic':>5}  {'lat':>10}  {'lon':>11}  turn  hill")
        for w in range(min(10, M)):
            s   = int(start_samples[w])
            e   = min(s + WINDOW_SIZE - 1, len(imu_ts) - 1)
            ic  = int(win_center_idx[w])
            lat = gps_lat[ic] if ic < len(gps_lat) else float('nan')
            lon = gps_lon[ic] if ic < len(gps_lon) else float('nan')
            tl  = ['N', 'L', 'R'][int(turn_preds[w])]
            hl  = ['N', 'U', 'D'][int(hill_preds[w])]
            print(f"  {w:>4}  {imu_ts[s]:>6.1f}  {imu_ts[e]:>6.1f}  {ic:>5}  {lat:>10.5f}  {lon:>11.5f}   {tl}     {hl}")

        print(f"\n[AutoDNA] turn dist (smo): {np.bincount(turn_preds, minlength=3).tolist()}")
        print(f"          hill dist (smo): {np.bincount(hill_preds, minlength=3).tolist()}")

        return DriveData(
            gps_timestamps=gps_ts,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            gps_speed=gps_speed,
            gps_heading=gps_heading,
            imu_timestamps=imu_ts,
            imu_signal=signal,
            xgb_turn_preds=turn_preds,
            xgb_hill_preds=hill_preds,
            xgb_turn_proba=turn_proba,
            xgb_hill_proba=hill_proba,
            xgb_turn_preds_raw=turn_preds_raw,
            xgb_hill_preds_raw=hill_preds_raw,
            window_gps_start_idx=win_start_idx,
            window_gps_end_idx=win_end_idx,
            window_gps_center_idx=win_center_idx,
            bin_file=str(bin_path),
            gps_file=str(csv_path),
            drive_name=bin_path.stem,
            drive_duration_sec=float(imu_ts[-1]),
            drive_distance_km=_haversine_total(gps_lat, gps_lon),
            window_size=WINDOW_SIZE,
            window_stride=STRIDE,
            target_fs=TARGET_FS,
            smooth_half=SMOOTH_HALF_WINDOW,
        )

    # ── File discovery ────────────────────────────────────────────────────────
    def _find_files(self):
        bins = sorted(
            list(self.drive_dir.glob('*.BIN')) +
            list(self.drive_dir.glob('*.bin'))
        )
        csvs = sorted(self.drive_dir.glob('*.csv'))
        if not bins:
            raise FileNotFoundError(
                f"No .BIN file found in {self.drive_dir}.\n"
                "The folder must contain a .BIN file (STM32 IMU recording)."
            )
        if not csvs:
            raise FileNotFoundError(
                f"No .csv file found in {self.drive_dir}.\n"
                "The folder must contain a .csv file (GPS/OBD2 log)."
            )
        return bins[0], csvs[0]

    # ── GPS loading ───────────────────────────────────────────────────────────
    def _load_gps(self, csv_path: Path):
        df = pd.read_csv(csv_path, sep=';', quotechar='"')
        df.columns = df.columns.str.lower().str.strip()

        df['latitude']   = pd.to_numeric(df['latitude'],   errors='coerce')
        df['longtitude'] = pd.to_numeric(df['longtitude'], errors='coerce')
        df['seconds']    = pd.to_numeric(df['seconds'],    errors='coerce')

        df = df.dropna(subset=['seconds', 'latitude', 'longtitude'])
        df = df[(df['latitude'] != 0) & (df['longtitude'] != 0)]
        df = df.sort_values('seconds').reset_index(drop=True)

        # One GPS position per timestamp (OBD2 CSV has 31 PIDs per timestamp)
        gps_df = (
            df.groupby('seconds', sort=True)
              .first()
              .reset_index()[['seconds', 'latitude', 'longtitude']]
        )

        # Drop stationary clusters
        lat_r = gps_df['latitude'].round(6)
        lon_r = gps_df['longtitude'].round(6)
        gps_df = gps_df[(lat_r != lat_r.shift()) | (lon_r != lon_r.shift())].reset_index(drop=True)

        if len(gps_df) < 2:
            raise ValueError("Not enough GPS movement data in the CSV.")

        ts  = gps_df['seconds'].values.astype(np.float64)
        lat = gps_df['latitude'].values.astype(np.float64)
        lon = gps_df['longtitude'].values.astype(np.float64)

        speed_df = df[df['pid'].str.strip().isin(['Speed (GPS)', 'Vehicle speed'])].copy()
        speed    = _align_speed(speed_df, ts)
        heading  = _compute_heading(lat, lon)

        return ts, lat, lon, speed, heading

    # ── Window → GPS alignment ────────────────────────────────────────────────
    def _align_to_gps(self, imu_ts, start_samples, gps_ts):
        """
        Map each IMU prediction window to GPS indices.

        Windows are overlapping (stride=25, size=100). Each window's center
        sample time is used as the representative GPS position so that
        adjacent overlapping windows map to incrementally different GPS points.

        Linear mapping: IMU t=0 -> GPS t[0], IMU t=imu_dur -> GPS t[-1].
        """
        imu_dur = float(imu_ts[-1])
        gps_dur = float(gps_ts[-1] - gps_ts[0])
        n_gps   = len(gps_ts)
        M       = len(start_samples)

        win_start  = np.zeros(M, dtype=np.int32)
        win_end    = np.zeros(M, dtype=np.int32)
        win_center = np.zeros(M, dtype=np.int32)

        for w in range(M):
            s      = int(start_samples[w])
            e      = min(s + WINDOW_SIZE - 1, len(imu_ts) - 1)
            mid    = (s + e) // 2

            t0  = float(imu_ts[s])
            t1  = float(imu_ts[e])
            t_c = float(imu_ts[mid])

            if imu_dur > 0 and gps_dur > 0:
                g0  = gps_ts[0] + (t0  / imu_dur) * gps_dur
                g1  = gps_ts[0] + (t1  / imu_dur) * gps_dur
                g_c = gps_ts[0] + (t_c / imu_dur) * gps_dur
            else:
                g0 = g1 = g_c = gps_ts[0]

            i0 = max(0, min(int(np.searchsorted(gps_ts, g0, side='left')),  n_gps - 1))
            i1 = max(0, min(int(np.searchsorted(gps_ts, g1, side='right')) - 1, n_gps - 1))
            ic = max(0, min(int(np.searchsorted(gps_ts, g_c, side='left')), n_gps - 1))

            i1 = max(i0, i1)

            win_start[w]  = i0
            win_end[w]    = i1
            win_center[w] = ic

        return win_start, win_end, win_center


# ── Smoothing ─────────────────────────────────────────────────────────────────

def _smooth_preds(preds: np.ndarray, half: int) -> np.ndarray:
    """
    Temporal majority vote over a sliding window of size (2*half + 1).
    Each output label is the most common label among the ±half neighbors.
    half=0 returns a copy of the input unchanged.
    """
    if half <= 0:
        return preds.copy()
    M = len(preds)
    out = np.empty_like(preds)
    for i in range(M):
        lo = max(0, i - half)
        hi = min(M, i + half + 1)
        out[i] = np.bincount(preds[lo:hi], minlength=3).argmax()
    return out.astype(np.int32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _align_speed(speed_df: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    if speed_df.empty:
        return np.zeros(len(timestamps), dtype=np.float32)
    speed_df = speed_df.sort_values('seconds')
    sp_ts  = speed_df['seconds'].values.astype(np.float64)
    sp_val = pd.to_numeric(speed_df['value'], errors='coerce').fillna(0).values.astype(np.float64)
    return np.interp(timestamps, sp_ts, sp_val).astype(np.float32)


def _compute_heading(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    heading = np.zeros(len(lat))
    for i in range(1, len(lat)):
        dlon = math.radians(lon[i] - lon[i - 1])
        lat1 = math.radians(lat[i - 1])
        lat2 = math.radians(lat[i])
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        heading[i] = (math.degrees(math.atan2(x, y)) + 360) % 360
    heading[0] = heading[1] if len(heading) > 1 else 0.0
    return heading


def _haversine_total(lat: np.ndarray, lon: np.ndarray) -> float:
    R = 6371.0
    total = 0.0
    for i in range(len(lat) - 1):
        phi1 = math.radians(lat[i])
        phi2 = math.radians(lat[i + 1])
        dphi = math.radians(lat[i + 1] - lat[i])
        dlam = math.radians(lon[i + 1] - lon[i])
        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
        total += R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return total
