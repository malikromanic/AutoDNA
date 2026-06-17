import math
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

<<<<<<< Updated upstream
#ocena porabe goriva ce obd2 pid 'engine fuel rate' ni na voljo
_FALLBACK_FUEL_L100KM = 8.0
=======
<<<<<<< HEAD
from app.ai_pipeline import (
    parse_and_preprocess_bin,
    make_windows,
    predict_windows,
    WINDOW_SIZE,
    STRIDE,
    TARGET_FS,
)
>>>>>>> Stashed changes

#prag gps vrzeli: manjse vrzeli se linearno interpolirajo, vecje ponavljajo zadnjo pozicijo
_GPS_GAP_THRESHOLD_S = 10.0

<<<<<<< Updated upstream
=======
# A GPS sample counts as "moving" above this speed (km/h). Used to trim the
# stationary prefix/suffix the GPS logger records before/after the STM is running.
SPEED_MOVING_KMH = 1.0
=======
#ocena porabe goriva ce obd2 pid 'engine fuel rate' ni na voljo
_FALLBACK_FUEL_L100KM = 8.0

#prag gps vrzeli: manjse vrzeli se linearno interpolirajo, vecje ponavljajo zadnjo pozicijo
_GPS_GAP_THRESHOLD_S = 10.0

>>>>>>> Stashed changes
#korak pri vstavljanju vmesnih tock v vrzeli (1 hz)
_GPS_INTERP_STEP_S = 1.0

#minimalna razdalja v km za izracun povprecne porabe (izogib deljenju z niclo)
_MIN_DIST_KM_FOR_AVG = 0.1
<<<<<<< Updated upstream
=======
>>>>>>> 15643d88c1a87fd432878e46626916791f4397d6
>>>>>>> Stashed changes


@dataclass
class DriveData:
    """Complete drive session from GPS/OBD2 CSV."""

<<<<<<< Updated upstream
    #gps sled - deduplicirane tocke med katerimi se vozilo premika
    gps_timestamps: np.ndarray  # (N,) absolutni casi v sekundah
    gps_lat: np.ndarray  # (N,)
    gps_lon: np.ndarray  # (N,)
    gps_speed: np.ndarray  # (N,) km/h
    gps_heading: np.ndarray  # (N,) smerni kot 0-360 stopinj
    gps_altitude: np.ndarray  # (N,) metri nadmorske visine; nicle ce pid ni na voljo
=======
<<<<<<< HEAD
    # GPS track (deduplicated, movement-filtered)
    gps_timestamps: np.ndarray      # (N,)  absolute seconds
    gps_lat:        np.ndarray      # (N,)
    gps_lon:        np.ndarray      # (N,)
    gps_speed:      np.ndarray      # (N,)  km/h
    gps_heading:    np.ndarray      # (N,)  degrees 0-360
>>>>>>> Stashed changes

    #poraba goriva - iz pid 'engine fuel rate' ali ocena 8 l/100km
    fuel_consumption_l: float
    avg_fuel_l100km: float

    #metapodatki o voznji
    gps_file: str
    drive_name: str
    drive_duration_sec: float
<<<<<<< Updated upstream
    drive_distance_km: float
=======
    drive_distance_km:  float
    fuel_consumption_l: float        # total litres burned this trip
    avg_fuel_l100km:    float        # trip average L/100km
    window_size:       int           # 100
    window_stride:     int           # 25  (overlapping)
    target_fs:         int           # 50
    smooth_half:       int           # smoothing half-window used
=======
    #gps sled - deduplicirane tocke med katerimi se vozilo premika
    gps_timestamps: np.ndarray  # (N,) absolutni casi v sekundah
    gps_lat: np.ndarray  # (N,)
    gps_lon: np.ndarray  # (N,)
    gps_speed: np.ndarray  # (N,) km/h
    gps_heading: np.ndarray  # (N,) smerni kot 0-360 stopinj
    gps_altitude: np.ndarray  # (N,) metri nadmorske visine; nicle ce pid ni na voljo

    #poraba goriva - iz pid 'engine fuel rate' ali ocena 8 l/100km
    fuel_consumption_l: float
    avg_fuel_l100km: float

    #metapodatki o voznji
    gps_file: str
    drive_name: str
    drive_duration_sec: float
    drive_distance_km: float
>>>>>>> 15643d88c1a87fd432878e46626916791f4397d6
>>>>>>> Stashed changes


class DriveDataLoader:
    """
    Open a drive directory containing a GPS/OBD2 .csv file.
    Returns a DriveData instance built purely from GPS data.
    """

    def __init__(self, drive_dir: Path):
        self.drive_dir = Path(drive_dir)

    def load_drive(self) -> DriveData:
        csv_path = self._find_csv()

<<<<<<< Updated upstream
        print(f"[AutoDNA] GPS : {csv_path}")
        gps_ts, gps_lat, gps_lon, gps_speed, gps_heading, gps_altitude, fuel_l = self._load_gps(csv_path)
        print(f"[AutoDNA] GPS : {len(gps_lat)} points  span {float(gps_ts[-1] - gps_ts[0]):.1f}s")
=======
<<<<<<< HEAD
        # ── Step 1: IMU pipeline ─────────────────────────────────────────────
        print(f"[AutoDNA] Parsing BIN: {bin_path.name}")
        signal, imu_ts = parse_and_preprocess_bin(bin_path)
        print(f"          signal {signal.shape}  duration {imu_ts[-1]:.1f}s")
>>>>>>> Stashed changes

        dist_km  = _haversine_total(gps_lat, gps_lon)
        avg_l100 = (fuel_l / dist_km * 100.0) if dist_km > _MIN_DIST_KM_FOR_AVG else 0.0

=======
        print(f"[AutoDNA] GPS : {csv_path}")
        gps_ts, gps_lat, gps_lon, gps_speed, gps_heading, gps_altitude, fuel_l = self._load_gps(csv_path)
        print(f"[AutoDNA] GPS : {len(gps_lat)} points  span {float(gps_ts[-1] - gps_ts[0]):.1f}s")

        dist_km  = _haversine_total(gps_lat, gps_lon)
        avg_l100 = (fuel_l / dist_km * 100.0) if dist_km > _MIN_DIST_KM_FOR_AVG else 0.0

>>>>>>> 15643d88c1a87fd432878e46626916791f4397d6
        return DriveData(
            gps_timestamps=gps_ts,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            gps_speed=gps_speed,
            gps_heading=gps_heading,
<<<<<<< Updated upstream
            gps_altitude=gps_altitude,
=======
<<<<<<< HEAD
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
>>>>>>> Stashed changes
            fuel_consumption_l=fuel_l,
            avg_fuel_l100km=avg_l100,
            gps_file=str(csv_path),
            drive_name=csv_path.stem,
            drive_duration_sec=float(gps_ts[-1] - gps_ts[0]),
            drive_distance_km=dist_km,
        )

    def _find_csv(self) -> Path:
        #poisci csv datoteko gps/obd2 posnetka v mapi voznje
        csvs = sorted(self.drive_dir.glob('*.csv'))
<<<<<<< Updated upstream
=======
        if not bins:
            raise FileNotFoundError(
                f"No .BIN file found in {self.drive_dir}.\n"
                "The folder must contain a .BIN file (STM32 IMU recording)."
            )
=======
            gps_altitude=gps_altitude,
            fuel_consumption_l=fuel_l,
            avg_fuel_l100km=avg_l100,
            gps_file=str(csv_path),
            drive_name=csv_path.stem,
            drive_duration_sec=float(gps_ts[-1] - gps_ts[0]),
            drive_distance_km=dist_km,
        )

    def _find_csv(self) -> Path:
        #poisci csv datoteko gps/obd2 posnetka v mapi voznje
        csvs = sorted(self.drive_dir.glob('*.csv'))
>>>>>>> 15643d88c1a87fd432878e46626916791f4397d6
>>>>>>> Stashed changes
        if not csvs:
            raise FileNotFoundError(
                f"No .csv file found in {self.drive_dir}.\n"
                "The folder must contain a .csv file (GPS/OBD2 log)."
            )
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
        return bins[0], csvs[0]
=======
>>>>>>> Stashed changes
        if len(csvs) > 1:
            print(f"[AutoDNA] Multiple CSV files found — using first: {csvs[0].name}")
            for c in csvs[1:]:
                print(f"          skipping: {c.name}")
        return csvs[0]
<<<<<<< Updated upstream
=======
>>>>>>> 15643d88c1a87fd432878e46626916791f4397d6
>>>>>>> Stashed changes

    # ── GPS loading ───────────────────────────────────────────────────────────
    def _load_gps(self, csv_path: Path):
        #najprej poskusi s podpicjem (obd2 privzeto), ce ne uspe uporabi vejico
        try:
            df = pd.read_csv(csv_path, sep=';', quotechar='"')
            if df.shape[1] < 3:
                raise ValueError("too few columns with ';' separator")
        except Exception:
            df = pd.read_csv(csv_path, sep=',', quotechar='"')

        #normaliziraj imena stolpcev - odstrani presledke in velika zacetnice
        df.columns = df.columns.str.lower().str.strip()

        #sprejmi obe obliki: longitude (pravilno) in longtitude (napaka obd2)
        lon_col = None
        for candidate in ('longitude', 'longtitude', 'lon'):
            if candidate in df.columns:
                lon_col = candidate
                break
        if lon_col is None:
            raise ValueError(
                f"No longitude column found in {csv_path.name}. "
                f"Columns present: {list(df.columns)}"
            )
        if lon_col != 'longtitude':
            df = df.rename(columns={lon_col: 'longtitude'})

        #pretvori stolpce v stevila - napacne vrednosti postanejo nan
        df['latitude']   = pd.to_numeric(df['latitude'],   errors='coerce')
        df['longtitude'] = pd.to_numeric(df['longtitude'], errors='coerce')
        df['seconds']    = pd.to_numeric(df['seconds'],    errors='coerce')

        #pocisti vrstice brez koordinat ali z niclo (neveljaven gps fix)
        df = df.dropna(subset=['seconds', 'latitude', 'longtitude'])
        df = df[(df['latitude'] != 0) & (df['longtitude'] != 0)]
        df = df.sort_values('seconds').reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No valid GPS rows in {csv_path.name}. "
                "Check that latitude/longitude/seconds columns contain real values."
            )

        #ena gps tocka na casovno oznako - obd2 csv ima vec pid vrstic na cas
        gps_df = (
            df.groupby('seconds', sort=True)
              .first()
              .reset_index()[['seconds', 'latitude', 'longtitude']]
        )

        #odstrani stacionarne tocke - enaka pozicija na 6 decimalnih mest (~0.1m)
        lat_r = gps_df['latitude'].round(6)
        lon_r = gps_df['longtitude'].round(6)
        gps_df = gps_df[(lat_r != lat_r.shift()) | (lon_r != lon_r.shift())].reset_index(drop=True)

        if len(gps_df) < 2:
            raise ValueError("Not enough GPS movement data in the CSV.")

        #pretvori v numpy polja za hitro vektorizirano racunanje
        ts  = gps_df['seconds'].values.astype(np.float64)
        lat = gps_df['latitude'].values.astype(np.float64)
        lon = gps_df['longtitude'].values.astype(np.float64)

<<<<<<< Updated upstream
        #hitrost je neobvezna - prazna ce csv nima pid stolpca ali ustreznih vrednosti
        speed_df = pd.DataFrame()
        altitude_df = pd.DataFrame()
        if 'pid' in df.columns:
            speed_df    = df[df['pid'].str.strip().isin(['Speed (GPS)', 'Vehicle speed'])].copy()
            altitude_df = df[df['pid'].str.strip() == 'Altitude (GPS)'].copy()
=======
<<<<<<< HEAD
        speed_df = df[df['pid'].str.strip().isin(['Speed (GPS)', 'Vehicle speed'])].copy()
>>>>>>> Stashed changes
        speed    = _align_speed(speed_df, ts)
        altitude = _align_speed(altitude_df, ts)   # same interpolation, reuse helper
        #smerni kot med zaporednimi gps tockami - potreben za preverjanje l/r oznak
        heading  = _compute_heading(lat, lon)

        gps_dur_s = float(ts[-1] - ts[0])
        if gps_dur_s > 0 and len(ts) < gps_dur_s:
            #gps je redek - vstavi vmesne tocke pri 1 hz
            ts_i: list[float] = []
            lat_i: list[float] = []
            lon_i: list[float] = []
            for k in range(len(ts) - 1):
                gap = float(ts[k + 1] - ts[k])
                ts_i.append(float(ts[k]))
                lat_i.append(float(lat[k]))
                lon_i.append(float(lon[k]))
                if _GPS_INTERP_STEP_S < gap <= _GPS_GAP_THRESHOLD_S:
                    #majhna vrzel: linearna interpolacija (gps se pocasi posodablja)
                    for sub_t in np.arange(ts[k] + _GPS_INTERP_STEP_S, ts[k + 1], _GPS_INTERP_STEP_S):
                        frac = (sub_t - ts[k]) / gap
                        ts_i.append(float(sub_t))
                        lat_i.append(float(lat[k]) + frac * (float(lat[k + 1]) - float(lat[k])))
                        lon_i.append(float(lon[k]) + frac * (float(lon[k + 1]) - float(lon[k])))
                elif gap > _GPS_GAP_THRESHOLD_S:
                    #velika vrzel (stojisce, rdeca luc): ponovi zadnjo pozicijo
                    for sub_t in np.arange(ts[k] + _GPS_INTERP_STEP_S, ts[k + 1], _GPS_INTERP_STEP_S):
                        ts_i.append(float(sub_t))
                        lat_i.append(float(lat[k]))
                        lon_i.append(float(lon[k]))
            ts_i.append(float(ts[-1]))
            lat_i.append(float(lat[-1]))
            lon_i.append(float(lon[-1]))
            ts_new = np.array(ts_i, dtype=np.float64)
            lat_new = np.array(lat_i, dtype=np.float64)
            lon_new = np.array(lon_i, dtype=np.float64)
            speed    = _align_speed(speed_df,    ts_new)
            altitude = _align_speed(altitude_df, ts_new)
            heading  = _compute_heading(lat_new, lon_new)
            print(f"[AutoDNA] GPS interpolated: {len(ts)} → {len(ts_new)} points"
                  f"  ({gps_dur_s:.0f}s span  sparse={len(ts)/gps_dur_s:.2f} pts/s)")
            ts, lat, lon = ts_new, lat_new, lon_new

        if altitude.max() > 1.0:
            print(f"[AutoDNA] GPS altitude: min={altitude.min():.1f}m  max={altitude.max():.1f}m"
                  f"  range={altitude.max()-altitude.min():.1f}m")
        else:
            print("[AutoDNA] GPS altitude: not available in CSV")

<<<<<<< Updated upstream
        #poraba goriva - iz obd2 pid ali ocena na osnovi razdalje
        fuel_l = _extract_fuel_l(df, _haversine_total(lat, lon))
        return ts, lat, lon, speed, heading, altitude, fuel_l
=======
        Both clocks tick real seconds but have independent, unknown offsets, so
        we anchor on the one event both recordings share: motion. The IMU
        driving span [imu_t0, imu_t1] is mapped linearly onto the GPS driving
        span [gps_t0, gps_t1]; the stationary GPS prefix/suffix is excluded.
        Each window's center-sample time is used as its representative position
        so adjacent overlapping windows resolve to incrementally different GPS
        points.
        """
        n_gps     = len(gps_ts)
        M         = len(start_samples)
        span_imu  = max(float(imu_t1 - imu_t0), 1e-6)
        span_gps  = float(gps_t1 - gps_t0)

        def map_t(t):
            frac = (float(t) - imu_t0) / span_imu
            frac = min(1.0, max(0.0, frac))          # clip windows outside the driving span
            return gps_t0 + frac * span_gps

        win_start  = np.zeros(M, dtype=np.int32)
        win_end    = np.zeros(M, dtype=np.int32)
        win_center = np.zeros(M, dtype=np.int32)

        for w in range(M):
            s   = int(start_samples[w])
            e   = min(s + WINDOW_SIZE - 1, len(imu_ts) - 1)
            mid = (s + e) // 2

            g0  = map_t(imu_ts[s])
            g1  = map_t(imu_ts[e])
            g_c = map_t(imu_ts[mid])

            i0 = max(0, min(int(np.searchsorted(gps_ts, g0, side='left')),  n_gps - 1))
            i1 = max(0, min(int(np.searchsorted(gps_ts, g1, side='right')) - 1, n_gps - 1))
            ic = max(0, min(int(np.searchsorted(gps_ts, g_c, side='left')), n_gps - 1))
            i1 = max(i0, i1)

            win_start[w]  = i0
            win_end[w]    = i1
            win_center[w] = ic

        return win_start, win_end, win_center


# ── Fuel ────────────────────────────────────────────────────────────────────

def _total_fuel_liters(df: pd.DataFrame) -> float:
    """Total litres burned this trip from the OBD2 log. Prefers the vehicle's
    own trip 'Fuel used' counter (litres, starts at 0); falls back to
    integrating 'Calculated instant fuel rate' (L/h) over time. Returns 0 if
    neither is present."""
    pid = df['pid'].astype(str).str.strip()

    fu = pd.to_numeric(df.loc[pid == 'Fuel used', 'value'], errors='coerce').dropna()
    if len(fu):
        used = float(fu.max() - fu.min())          # monotonic trip counter
        if used > 0:
            return used

    fr = df.loc[pid == 'Calculated instant fuel rate', ['seconds', 'value']].copy()
    fr['value'] = pd.to_numeric(fr['value'], errors='coerce')
    fr = fr.dropna(subset=['seconds', 'value']).sort_values('seconds')
    if len(fr) > 1:
        t = fr['seconds'].values.astype(float)
        r = fr['value'].values.astype(float)       # L/h
        return float(np.sum((r[:-1] + r[1:]) / 2.0 * np.diff(t)) / 3600.0)

    return 0.0


# ── Driving-window detection ────────────────────────────────────────────────

def _gps_driving_window(gps_ts: np.ndarray, gps_speed: np.ndarray):
    """GPS time span where the vehicle is actually moving (trims the parked
    prefix/suffix the logger records). Falls back to the full span if no speed
    data crosses the threshold."""
    moving = np.asarray(gps_speed) > SPEED_MOVING_KMH
    if moving.any():
        idx = np.flatnonzero(moving)
        return float(gps_ts[idx[0]]), float(gps_ts[idx[-1]])
    return float(gps_ts[0]), float(gps_ts[-1])


def _imu_driving_window(signal: np.ndarray, imu_ts: np.ndarray):
    """IMU time span that contains driving, from gyro activity (rolling std of
    the gyro-vector magnitude). A parked car reads ~0 angular rate; driving adds
    turn + road-induced angular motion. Conservative: if the detected span is
    implausibly short (<60% of the recording) the detection is distrusted and
    the full recording is used — typical recordings are started/stopped at the
    drive boundaries, so the whole thing is driving."""
    t = np.asarray(imu_ts, float)
    if len(t) < 3:
        return float(t[0]), float(t[-1])

    gmag = np.sqrt((np.asarray(signal[:, 0:3], float) ** 2).sum(axis=1))
    w = max(5, int(2 * TARGET_FS))                      # ~2 s window
    rstd = pd.Series(gmag).rolling(w, min_periods=1, center=True).std().fillna(0.0).values

    base = float(np.percentile(rstd, 10))               # quietest moments
    hi   = float(np.percentile(rstd, 90))
    thr  = base + 0.15 * (hi - base)
    active = rstd > thr

    full = (float(t[0]), float(t[-1]))
    if not active.any():
        return full
    idx = np.flatnonzero(active)
    t0, t1 = float(t[idx[0]]), float(t[idx[-1]])
    if (t1 - t0) < 0.60 * (t[-1] - t[0]):               # detection chopped too much → distrust
        return full
    return t0, t1


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
=======
        #hitrost je neobvezna - prazna ce csv nima pid stolpca ali ustreznih vrednosti
        speed_df = pd.DataFrame()
        altitude_df = pd.DataFrame()
        if 'pid' in df.columns:
            speed_df    = df[df['pid'].str.strip().isin(['Speed (GPS)', 'Vehicle speed'])].copy()
            altitude_df = df[df['pid'].str.strip() == 'Altitude (GPS)'].copy()
        speed    = _align_speed(speed_df, ts)
        altitude = _align_speed(altitude_df, ts)   # same interpolation, reuse helper
        #smerni kot med zaporednimi gps tockami - potreben za preverjanje l/r oznak
        heading  = _compute_heading(lat, lon)

        gps_dur_s = float(ts[-1] - ts[0])
        if gps_dur_s > 0 and len(ts) < gps_dur_s:
            #gps je redek - vstavi vmesne tocke pri 1 hz
            ts_i: list[float] = []
            lat_i: list[float] = []
            lon_i: list[float] = []
            for k in range(len(ts) - 1):
                gap = float(ts[k + 1] - ts[k])
                ts_i.append(float(ts[k]))
                lat_i.append(float(lat[k]))
                lon_i.append(float(lon[k]))
                if _GPS_INTERP_STEP_S < gap <= _GPS_GAP_THRESHOLD_S:
                    #majhna vrzel: linearna interpolacija (gps se pocasi posodablja)
                    for sub_t in np.arange(ts[k] + _GPS_INTERP_STEP_S, ts[k + 1], _GPS_INTERP_STEP_S):
                        frac = (sub_t - ts[k]) / gap
                        ts_i.append(float(sub_t))
                        lat_i.append(float(lat[k]) + frac * (float(lat[k + 1]) - float(lat[k])))
                        lon_i.append(float(lon[k]) + frac * (float(lon[k + 1]) - float(lon[k])))
                elif gap > _GPS_GAP_THRESHOLD_S:
                    #velika vrzel (stojisce, rdeca luc): ponovi zadnjo pozicijo
                    for sub_t in np.arange(ts[k] + _GPS_INTERP_STEP_S, ts[k + 1], _GPS_INTERP_STEP_S):
                        ts_i.append(float(sub_t))
                        lat_i.append(float(lat[k]))
                        lon_i.append(float(lon[k]))
            ts_i.append(float(ts[-1]))
            lat_i.append(float(lat[-1]))
            lon_i.append(float(lon[-1]))
            ts_new = np.array(ts_i, dtype=np.float64)
            lat_new = np.array(lat_i, dtype=np.float64)
            lon_new = np.array(lon_i, dtype=np.float64)
            speed    = _align_speed(speed_df,    ts_new)
            altitude = _align_speed(altitude_df, ts_new)
            heading  = _compute_heading(lat_new, lon_new)
            print(f"[AutoDNA] GPS interpolated: {len(ts)} → {len(ts_new)} points"
                  f"  ({gps_dur_s:.0f}s span  sparse={len(ts)/gps_dur_s:.2f} pts/s)")
            ts, lat, lon = ts_new, lat_new, lon_new

        if altitude.max() > 1.0:
            print(f"[AutoDNA] GPS altitude: min={altitude.min():.1f}m  max={altitude.max():.1f}m"
                  f"  range={altitude.max()-altitude.min():.1f}m")
        else:
            print("[AutoDNA] GPS altitude: not available in CSV")

        #poraba goriva - iz obd2 pid ali ocena na osnovi razdalje
        fuel_l = _extract_fuel_l(df, _haversine_total(lat, lon))
        return ts, lat, lon, speed, heading, altitude, fuel_l
>>>>>>> 15643d88c1a87fd432878e46626916791f4397d6
>>>>>>> Stashed changes


# ── Helpers ───────────────────────────────────────────────────────────────────

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
=======
>>>>>>> Stashed changes
def _extract_fuel_l(df: pd.DataFrame, distance_km: float) -> float:
    #izvleci skupno porabo goriva iz obd2 csv datoteke
    #poskusi vire po prednostnem vrstnem redu: neposredno merjeni litri, nato pretok, nato ocena

    if 'pid' in df.columns:
        pid_col = df['pid'].str.strip()

        #1. prioriteta: 'fuel used' v litrih - direktna meritev (zadnja - prva vrednost)
        fuel_used_df = df[pid_col == 'Fuel used'].copy()
        if not fuel_used_df.empty:
            vals = pd.to_numeric(fuel_used_df['value'], errors='coerce').dropna()
            if len(vals) >= 2:
                total = float(vals.iloc[-1]) - float(vals.iloc[0])
                if total > 0:
                    print(f"[AutoDNA] Fuel: read from 'Fuel used' PID: {total:.4f} L")
                    return total

        #2. prioriteta: 'calculated instant fuel rate' ali 'engine fuel rate' (l/h) - trapezna integracija
        for pid_name in ('Calculated instant fuel rate', 'engine fuel rate'):
            rate_df = df[pid_col.str.lower() == pid_name.lower()].copy()
            if not rate_df.empty:
                rate_df = rate_df.sort_values('seconds')
                ts  = rate_df['seconds'].values.astype(np.float64)
                val = pd.to_numeric(rate_df['value'], errors='coerce').fillna(0).values.astype(np.float64)
                dt_h = np.diff(ts) / 3600.0
                total = float(np.sum(((val[:-1] + val[1:]) / 2.0) * dt_h))
                if total > 0:
                    print(f"[AutoDNA] Fuel: trapz integration of {pid_name!r}: {total:.4f} L")
                    return total

    #3. ocena: _FALLBACK_FUEL_L100KM l/100km ce ni pid podatkov
    print(f"[AutoDNA] Fuel: no PID data — estimating at {_FALLBACK_FUEL_L100KM} L/100km")
    return distance_km * _FALLBACK_FUEL_L100KM / 100.0


<<<<<<< Updated upstream
=======
>>>>>>> 15643d88c1a87fd432878e46626916791f4397d6
>>>>>>> Stashed changes
def _align_speed(speed_df: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    #interpoliraj hitrost na gps casovne oznake z linearno interpolacijo
    #vrne nicle ce obd2 csv ne vsebuje podatkov o hitrosti
    if speed_df.empty:
        return np.zeros(len(timestamps), dtype=np.float32)
    speed_df = speed_df.sort_values('seconds')
    sp_ts  = speed_df['seconds'].values.astype(np.float64)
    sp_val = pd.to_numeric(speed_df['value'], errors='coerce').fillna(0).values.astype(np.float64)
    return np.interp(timestamps, sp_ts, sp_val).astype(np.float32)


def _compute_heading(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    #izracunaj smerni kot (bearing) med zaporednimi gps tockami v stopinjah [0,360]
    #prva tocka dobi isti kot kot druga da se izognemo nanima
    heading = np.zeros(len(lat))
    for i in range(1, len(lat)):
        dlon = math.radians(lon[i] - lon[i - 1])
        lat1 = math.radians(lat[i - 1])
        lat2 = math.radians(lat[i])
        #sfericna formula za smerni kot (forward azimuth)
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        heading[i] = (math.degrees(math.atan2(x, y)) + 360) % 360
    heading[0] = heading[1] if len(heading) > 1 else 0.0
    return heading


def _haversine_total(lat: np.ndarray, lon: np.ndarray) -> float:
    #skupna razdalja gps sledi v kilometrih - vsota haversine razdalj med zaporednimi tockami
    R = 6371.0  #polmer zemlje v kilometrih
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
