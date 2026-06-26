# ============================================================================
# AutoDNA — GPS drive loader
#
# Everything is derived from the GPS/OBD2 CSV. No IMU, no ML:
#   Turns  ← GPS heading change            (app/gps_analysis.compute_turns)
#   Hills  ← DEM ground-elevation grade    (app/elevation + compute_hills)
#
# The device's own GPS altitude column is intentionally ignored — elevation is
# looked up from a DEM (online EU-DEM 25 m, Copernicus GLO-90 fallback).
# ============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from AutoDNA.app.elevation import ElevationProvider, get_default_provider
from AutoDNA.app.gps_analysis import (
    cumulative_distance,
    compute_turns,
    compute_hills,
    elevation_gain_loss,
)


@dataclass
class DriveData:
    """A complete drive session, fully derived from GPS + DEM."""

    # GPS track (deduplicated, movement-filtered)
    gps_timestamps: np.ndarray   # (N,)  seconds from start (0-indexed)
    gps_lat:        np.ndarray   # (N,)
    gps_lon:        np.ndarray   # (N,)
    gps_speed:      np.ndarray   # (N,)  km/h
    gps_heading:    np.ndarray   # (N,)  degrees 0-360
    cum_distance_m: np.ndarray   # (N,)  cumulative metres

    # Elevation / grade (from DEM, NOT device altitude)
    elevation_m:    np.ndarray   # (N,)  raw DEM elevation
    elevation_sm:   np.ndarray   # (N,)  smoothed elevation used for grades
    grade_pct:      np.ndarray   # (N,)  signed road grade %

    # Per-point event classes (one element per GPS point)
    turn_preds:     np.ndarray   # (N,) int32  0=straight 1=left 2=right
    hill_preds:     np.ndarray   # (N,) int32  0=flat 1=uphill 2=downhill
    turn_conf:      np.ndarray   # (N,) float  0..1
    hill_conf:      np.ndarray   # (N,) float  0..1
    turn_rate:      np.ndarray   # (N,) float  signed heading change (deg)

    # Metadata
    gps_file:           str
    drive_name:         str
    drive_duration_sec: float
    drive_distance_km:  float
    elevation_gain_m:   float
    elevation_loss_m:   float
    elevation_source:   str
    avg_fuel_l100km: float
    fuel_consumption_l: float   

    @property
    def n_points(self) -> int:
        return len(self.gps_lat)


class DriveDataLoader:
    """
    Open a drive directory containing a GPS/OBD2 `.csv` and produce DriveData.

    A `.BIN` (IMU) file may also be present but is ignored — the IMU pipeline
    now lives outside the app.
    """

    def __init__(self, drive_dir: Path, provider: ElevationProvider | None = None):
        self.drive_dir = Path(drive_dir)
        self.provider = provider or get_default_provider()

    def load_drive(self, progress_cb=None) -> DriveData:
        def _log(msg):
            print(f"[AutoDNA] {msg}")
            if progress_cb:
                progress_cb(msg)

        csv_path = self._find_csv()

        # ── GPS track ────────────────────────────────────────────────────────
        _log(f"Loading GPS: {csv_path.name}")
        ts, lat, lon, speed, heading = self._load_gps(csv_path)
        cum = cumulative_distance(lat, lon)
        _log(f"{len(lat)} GPS points, {cum[-1]:.0f} m")

        # ── Turns (heading) ──────────────────────────────────────────────────
        turn_preds, turn_conf, turn_rate = compute_turns(heading, cum)
        _log(f"Turns straight/left/right: {np.bincount(turn_preds, minlength=3).tolist()}")

        # ── Hills (DEM grade) ────────────────────────────────────────────────
        _log(f"Looking up elevation via {self.provider.name}…")
        elev = self.provider.elevations(lat, lon)
        n_missing = int(np.isnan(elev).sum())
        if n_missing == len(elev):
            _log("Elevation unavailable (offline?) — hills disabled for this drive.")
            source = f"{self.provider.name} (unavailable)"
        else:
            source = self.provider.name
            if n_missing:
                _log(f"{n_missing}/{len(elev)} points missing elevation (interpolated).")
        hill_preds, hill_conf, grade, elev_sm = compute_hills(elev, cum)
        gain, loss = elevation_gain_loss(elev_sm)
        _log(f"Hills flat/up/down: {np.bincount(hill_preds, minlength=3).tolist()}  "
             f"(+{gain:.0f}/-{loss:.0f} m)")

        df_full = pd.read_csv(csv_path, sep=";", quotechar='"')
        df_full.columns = df_full.columns.str.lower().str.strip()
        df_full['seconds'] = pd.to_numeric(df_full['seconds'], errors='coerce')

        dist_km  = cum[-1] / 1000.0
        fuel_l   = _extract_fuel_l(df_full, dist_km)
        avg_l100 = (fuel_l / dist_km * 100.0) if dist_km > _MIN_DIST_KM_FOR_AVG else 0.0

        return DriveData(
            gps_timestamps=ts,
            gps_lat=lat,
            gps_lon=lon,
            gps_speed=speed,
            gps_heading=heading,
            cum_distance_m=cum,
            elevation_m=elev,
            elevation_sm=elev_sm,
            grade_pct=grade,
            turn_preds=turn_preds,
            hill_preds=hill_preds,
            turn_conf=turn_conf,
            hill_conf=hill_conf,
            turn_rate=turn_rate,
            gps_file=str(csv_path),
            drive_name=csv_path.stem,
            drive_duration_sec=float(ts[-1] - ts[0]),
            drive_distance_km=cum[-1] / 1000.0,
            elevation_gain_m=gain,
            elevation_loss_m=loss,
            elevation_source=source,
            avg_fuel_l100km=avg_l100,
            fuel_consumption_l=fuel_l,
        )

    # ── File discovery ──────────────────────────────────────────────────────
    def _find_csv(self) -> Path:
        csvs = sorted(self.drive_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No .csv file found in {self.drive_dir}.\n"
                "The folder must contain a GPS/OBD2 .csv log."
            )
        return csvs[0]

    # ── GPS loading ───────────────────────────────────────────────────────--
    def _load_gps(self, csv_path: Path):
        df = pd.read_csv(csv_path, sep=";", quotechar='"')
        df.columns = df.columns.str.lower().str.strip()

        df["latitude"]   = pd.to_numeric(df["latitude"],   errors="coerce")
        df["longtitude"] = pd.to_numeric(df["longtitude"], errors="coerce")
        df["seconds"]    = pd.to_numeric(df["seconds"],    errors="coerce")

        df = df.dropna(subset=["seconds", "latitude", "longtitude"])
        df = df[(df["latitude"] != 0) & (df["longtitude"] != 0)]
        df = df.sort_values("seconds").reset_index(drop=True)

        # One GPS position per timestamp (OBD2 CSV repeats many PIDs per second)
        gps_df = (
            df.groupby("seconds", sort=True)
              .first()
              .reset_index()[["seconds", "latitude", "longtitude"]]
        )

        # Drop stationary clusters
        lat_r = gps_df["latitude"].round(6)
        lon_r = gps_df["longtitude"].round(6)
        gps_df = gps_df[(lat_r != lat_r.shift()) | (lon_r != lon_r.shift())].reset_index(drop=True)

        if len(gps_df) < 2:
            raise ValueError("Not enough GPS movement data in the CSV.")

        ts  = gps_df["seconds"].values.astype(np.float64)
        lat = gps_df["latitude"].values.astype(np.float64)
        lon = gps_df["longtitude"].values.astype(np.float64)

        speed_df = df[df["pid"].str.strip().isin(["Speed (GPS)", "Vehicle speed"])].copy()
        speed    = _align_speed(speed_df, ts)
        heading  = _compute_heading(lat, lon)

        return ts, lat, lon, speed, heading


# ── Helpers ───────────────────────────────────────────────────────────────--
def _align_speed(speed_df: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    if speed_df.empty:
        return np.zeros(len(timestamps), dtype=np.float32)
    speed_df = speed_df.sort_values("seconds")
    sp_ts  = speed_df["seconds"].values.astype(np.float64)
    sp_val = pd.to_numeric(speed_df["value"], errors="coerce").fillna(0).values.astype(np.float64)
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


_FALLBACK_FUEL_L100KM = 8.0
_MIN_DIST_KM_FOR_AVG  = 0.1


def _extract_fuel_l(df: pd.DataFrame, distance_km: float) -> float:
    if 'pid' in df.columns:
        pid_col = df['pid'].str.strip()

        # 1. priority: 'Fuel used' PID — direct litre measurement
        fuel_used_df = df[pid_col == 'Fuel used'].copy()
        if not fuel_used_df.empty:
            vals = pd.to_numeric(fuel_used_df['value'], errors='coerce').dropna()
            if len(vals) >= 2:
                total = float(vals.iloc[-1]) - float(vals.iloc[0])
                if total > 0:
                    print(f"[AutoDNA] Fuel: read from 'Fuel used' PID: {total:.4f} L")
                    return total

        # 2. priority: instant fuel rate — trapezoid integration
        for pid_name in ('Calculated instant fuel rate', 'engine fuel rate'):
            rate_df = df[pid_col.str.lower() == pid_name.lower()].copy()
            if not rate_df.empty:
                rate_df = rate_df.sort_values('seconds')
                ts  = rate_df['seconds'].values.astype(np.float64)
                val = pd.to_numeric(rate_df['value'], errors='coerce').fillna(0).values.astype(np.float64)
                dt_h  = np.diff(ts) / 3600.0
                total = float(np.sum(((val[:-1] + val[1:]) / 2.0) * dt_h))
                if total > 0:
                    print(f"[AutoDNA] Fuel: trapz integration of {pid_name!r}: {total:.4f} L")
                    return total

    # 3. fallback estimate
    print(f"[AutoDNA] Fuel: no PID data — estimating at {_FALLBACK_FUEL_L100KM} L/100km")
    return distance_km * _FALLBACK_FUEL_L100KM / 100.0
