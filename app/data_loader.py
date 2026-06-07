import math
import hashlib
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

#casovno glajenje: vecinska napoved v oknu +-n (0 = izklopljeno)
SMOOTH_HALF_WINDOW = 2

#ocena porabe goriva ce obd2 pid 'engine fuel rate' ni na voljo
_FALLBACK_FUEL_L100KM = 8.0

#prag gps vrzeli: manjse vrzeli se linearno interpolirajo, vecje ponavljajo zadnjo pozicijo
_GPS_GAP_THRESHOLD_S = 10.0

#korak pri vstavljanju vmesnih tock v vrzeli (1 hz)
_GPS_INTERP_STEP_S = 1.0

#minimalna razdalja v km za izracun povprecne porabe (izogib deljenju z niclo)
_MIN_DIST_KM_FOR_AVG = 0.1


@dataclass
class DriveData:
    """Complete drive session — real IMU predictions mapped to real GPS."""

    #gps sled - deduplicirane tocke med katerimi se vozilo premika
    gps_timestamps: np.ndarray  # (N,) absolutni casi v sekundah
    gps_lat: np.ndarray  # (N,)
    gps_lon: np.ndarray  # (N,)
    gps_speed: np.ndarray  # (N,) km/h
    gps_heading: np.ndarray  # (N,) smerni kot 0-360 stopinj

    #imu signal predprocesiran na 50 hz, cas relativen od zacetka posnetka
    imu_timestamps: np.ndarray  # (K,)
    imu_signal: np.ndarray  # (K, 11) [gyro_xyz | accel_xyz | mag_xyz | pitch_cf | roll_cf]

    #xgboost napovedi po casovnem glajenju - en element na 2-sekundno okno
    xgb_turn_preds: np.ndarray  # (M,) int32  0=brez 1=levo 2=desno
    xgb_hill_preds: np.ndarray  # (M,) int32  0=brez 1=gor 2=dol
    xgb_turn_proba: np.ndarray  # (M, 3) float32 verjetnosti razredov
    xgb_hill_proba: np.ndarray  # (M, 3) float32

    #surove napovedi pred casovnim glajanjem - za diagnostiko
    xgb_turn_preds_raw: np.ndarray  # (M,) int32
    xgb_hill_preds_raw: np.ndarray  # (M,) int32

    #gps indeksi ki jih pokriva vsako napovedno okno
    window_gps_start_idx: np.ndarray  # (M,) int32
    window_gps_end_idx: np.ndarray  # (M,) int32
    window_gps_center_idx: np.ndarray  # (M,) int32 gps tocka v sredini okna

    #poraba goriva - iz pid 'engine fuel rate' ali ocena 8 l/100km
    fuel_consumption_l: float   # skupna poraba v litrih
    avg_fuel_l100km: float      # povprecna poraba l/100km

    #metapodatki o voznji
    bin_file: str
    gps_file: str
    drive_name: str
    drive_duration_sec: float
    drive_distance_km: float
    window_size: int  # 100 vzorcev
    window_stride: int  # 25 vzorcev (75% prekrivanje)
    target_fs: int  # 50 hz
    smooth_half: int  # polsirina okna za glajanje


class DriveDataLoader:
    """
    Open a drive directory containing exactly one .BIN and one .csv file.
    Runs the real AutoDNA pipeline and returns a DriveData instance.
    """

    def __init__(self, drive_dir: Path):
        self.drive_dir = Path(drive_dir)

    def load_drive(self) -> DriveData:
        bin_path, csv_path = self._find_files()

        #korak 1: predprocesiranje imu signala iz bin datoteke
        print(f"[AutoDNA] BIN : {bin_path}")
        print(f"[AutoDNA] GPS : {csv_path}")
        signal, imu_ts = parse_and_preprocess_bin(bin_path)
        print(f"[AutoDNA] IMU : {signal.shape[0]} samples  duration {imu_ts[-1]:.1f}s")

        #samo zacetni vzorci za poravnavo z gps - okna gradi predict_windows() sam
        _, start_samples = make_windows(signal)
        M = len(start_samples)
        print(f"          {M} turn-windows (size={WINDOW_SIZE} stride={STRIDE})")

        print("[AutoDNA] Running XGBoost inference...")
        turn_preds_raw, hill_preds_raw, turn_proba, hill_proba = predict_windows(signal)

        print(f"          raw turn: {np.bincount(turn_preds_raw, minlength=3).tolist()}")
        print(f"          raw hill: {np.bincount(hill_preds_raw, minlength=3).tolist()}")

        #korak 2: casovno glajanje z vecinskim glasovanjem v oknu +-smooth_half_window
        turn_preds = _smooth_preds(turn_preds_raw, SMOOTH_HALF_WINDOW)
        hill_preds = _smooth_preds(hill_preds_raw, SMOOTH_HALF_WINDOW)
        print(f"          smoothed (+-{SMOOTH_HALF_WINDOW}) turn: {np.bincount(turn_preds, minlength=3).tolist()}")
        print(f"          smoothed (+-{SMOOTH_HALF_WINDOW}) hill: {np.bincount(hill_preds, minlength=3).tolist()}")

        #korak 3: nalozi in pocisti gps sled iz csv datoteke
        gps_ts, gps_lat, gps_lon, gps_speed, gps_heading, fuel_l = self._load_gps(csv_path)
        gps_dur = float(gps_ts[-1] - gps_ts[0])  #trajanje gps posnetka v sekundah
        imu_dur = float(imu_ts[-1])               #trajanje imu posnetka (ts zacne pri 0)
        print(f"[AutoDNA] GPS : {len(gps_lat)} points  span {gps_dur:.1f}s")
        print(f"[AutoDNA] IMU duration {imu_dur:.1f}s  GPS duration {gps_dur:.1f}s")

        #opozori ko se trajanja posnetkov mocno razlikujeta
        #poravnava uporablja sorazmerno preslikavo - velik razkorak zamakne napovedi na karti
        if imu_dur > 0 and gps_dur > 0:
            ratio = imu_dur / gps_dur
            if ratio < 0.80 or ratio > 1.25:
                print(
                    f"[AutoDNA] WARNING: IMU/GPS duration ratio = {ratio:.2f} "
                    f"(IMU {imu_dur:.0f}s vs GPS {gps_dur:.0f}s). "
                    "If the recordings did not start at the same moment, map predictions "
                    "will appear at shifted GPS positions."
                )

        #korak 4: poravnaj imu okna z gps sledjo
        #stm32 snema od vklopa - gps pa se zacne neodvisno
        #brez poravnave bi napovedi prikazoval na napacnem gps mestu

        #imu: kdaj se vozilo zacne premikati (adaptivni prag - ni vec fiksen 0.06)
        imu_motion_t = _find_imu_motion_onset(signal, imu_ts)
        print(f"[AutoDNA] IMU motion onset: {imu_motion_t:.1f}s")

        #gps: kdaj se vozilo prvic premakne za vsaj 10 m (popravi parkirni prefix)
        gps_motion_onset_t = _find_gps_motion_onset(gps_lat, gps_lon, gps_ts)
        gps_parked_prefix_s = gps_motion_onset_t - float(gps_ts[0])
        print(f"[AutoDNA] GPS motion onset: {gps_motion_onset_t:.1f}  "
              f"(gps_ts[0]={float(gps_ts[0]):.1f}  "
              f"parked_prefix={gps_parked_prefix_s:.1f}s)")

        win_start_idx, win_end_idx, win_center_idx = self._align_to_gps(
            imu_ts, start_samples, gps_ts, imu_motion_t, gps_motion_onset_t
        )

        #izpisi tabelo za vsa okna z dogodki in vzorcna okna za diagnostiko
        print(f"\n[AutoDNA] {M} windows (stride={STRIDE})")
        print(f"  Alignment scale: {imu_motion_t:.1f}s onset  "
              f"active={float(imu_ts[-1]) - imu_motion_t:.1f}s  "
              f"GPS={gps_dur:.1f}s")
        print(f"  {'W':>4}  {'IMU_t0':>7}  {'IMU_t1':>7}  "
              f"{'GPS_i0':>6}  {'GPS_i1':>6}  {'ic':>5}  "
              f"{'lat':>10}  {'lon':>11}  {'hdg_d':>6}  "
              f"{'turn':>5}  {'t_conf':>6}  {'hill':>5}  {'h_conf':>6}")
        n_gps = len(gps_lat)
        for w in range(M):
            tp = int(turn_preds[w])
            hp = int(hill_preds[w])
            #izpisi vsa okna z dogodki ter prvih 5 in zadnji 2 za diagnostiko
            is_event = (tp != 0 or hp != 0)
            is_sample = (w < 5 or w >= M - 2)
            if not (is_event or is_sample):
                continue
            s  = int(start_samples[w])
            e  = min(s + WINDOW_SIZE - 1, len(imu_ts) - 1)
            i0 = int(win_start_idx[w])
            i1 = int(win_end_idx[w])
            ic = int(win_center_idx[w])
            lat_c = gps_lat[min(ic, n_gps - 1)]
            lon_c = gps_lon[min(ic, n_gps - 1)]
            #sprememba smernega kota med zacetno in koncno gps tocko okna
            if i0 < i1 and i1 < n_gps:
                hdg_d = float(gps_heading[i1]) - float(gps_heading[i0])
                hdg_d = (hdg_d + 180) % 360 - 180
            else:
                hdg_d = float('nan')
            tl = ['N', 'L', 'R'][tp]
            hl = ['N', 'U', 'D'][hp]
            tc = float(turn_proba[w, tp])
            hc = float(hill_proba[w, hp])
            marker = ' ◄' if is_event else ''
            print(f"  {w:>4}  {imu_ts[s]:>7.2f}  {imu_ts[e]:>7.2f}  "
                  f"{i0:>6}  {i1:>6}  {ic:>5}  "
                  f"{lat_c:>10.5f}  {lon_c:>11.5f}  {hdg_d:>6.1f}  "
                  f"{tl:>5}  {tc:>6.1%}  {hl:>5}  {hc:>6.1%}{marker}")

        print(f"\n[AutoDNA] turn dist (smo): {np.bincount(turn_preds, minlength=3).tolist()}")
        print(f"          turn dist (raw): {np.bincount(turn_preds_raw, minlength=3).tolist()}")
        print(f"          hill dist (smo): {np.bincount(hill_preds, minlength=3).tolist()}")
        print(f"          hill dist (raw): {np.bincount(hill_preds_raw, minlength=3).tolist()}")

        #preveri skladnost l/r napovedi z gps smernim kotom
        #padajoci smerni kot = levi zavoj, narasca = desni zavoj
        #razmerje blizu 0% pomeni da sta l/r zamenjana
        event_wins = np.where(turn_preds != 0)[0]
        if len(event_wins) >= 4:
            n_consistent = 0
            for w in event_wins:
                i0 = int(win_start_idx[w])
                i1 = int(win_end_idx[w])
                if i0 < i1 and i1 < len(gps_heading):
                    delta = float(gps_heading[i1]) - float(gps_heading[i0])
                    delta = (delta + 180) % 360 - 180  # normalise to [-180, 180]
                    pred_left = (int(turn_preds[w]) == 1)
                    gps_left  = (delta < 0)             # heading decreasing = left
                    if pred_left == gps_left:
                        n_consistent += 1
            ratio = n_consistent / len(event_wins)
            if ratio >= 0.65:
                note = "✓ L/R labels match GPS heading — correct"
            elif ratio <= 0.35:
                note = "⚠ L/R labels OPPOSITE to GPS heading — sensor may be mounted reversed"
            else:
                note = "inconclusive (GPS heading too noisy or alignment offset)"
            print(
                f"[AutoDNA] L/R GPS-heading consistency: "
                f"{n_consistent}/{len(event_wins)} ({ratio:.0%}) — {note}"
            )

        #prstni odtis voznje - md5 zgoscenka imu signala in napovedi
        #razlicne voznje morajo imeti razlicni zgoscenki; enaki bi pomenili napako v predprocesiranju
        _sig_hash  = hashlib.md5(signal.tobytes()).hexdigest()[:12]
        _pred_hash = hashlib.md5(
            turn_preds_raw.tobytes() + hill_preds_raw.tobytes() + turn_proba.tobytes()
        ).hexdigest()[:12]
        print(f"\n[AutoDNA] ── DRIVE FINGERPRINT ─────────────────────────────────────")
        print(f"  File              : {bin_path.name}")
        print(f"  Signal hash       : {_sig_hash}  shape={signal.shape}"
              f"  min={signal.min():.3f}  max={signal.max():.3f}")
        print(f"  Prediction hash   : {_pred_hash}")
        print(f"  IMU samples       : {signal.shape[0]}"
              f"  ({signal.shape[0]/TARGET_FS:.1f}s @ {TARGET_FS}Hz)")
        print(f"  GPS points        : {len(gps_lat)}"
              f"  span {float(gps_ts[-1]-gps_ts[0]):.1f}s")
        print(f"  Turn windows (M)  : {len(turn_preds_raw)}")
        print(f"  Turn counts       : none={int((turn_preds_raw==0).sum())}"
              f"  left={int((turn_preds_raw==1).sum())}"
              f"  right={int((turn_preds_raw==2).sum())}")
        print(f"  Hill counts       : none={int((hill_preds_raw==0).sum())}"
              f"  up={int((hill_preds_raw==1).sum())}"
              f"  down={int((hill_preds_raw==2).sum())}")
        print(f"  Turn proba (mean) : none={turn_proba[:,0].mean():.3f}"
              f"  left={turn_proba[:,1].mean():.3f}"
              f"  right={turn_proba[:,2].mean():.3f}")
        print(f"  Hill proba (mean) : none={hill_proba[:,0].mean():.3f}"
              f"  up={hill_proba[:,1].mean():.3f}"
              f"  down={hill_proba[:,2].mean():.3f}")
        print(f"  Turn proba range  : min={turn_proba.min():.3f}"
              f"  max={turn_proba.max():.3f}"
              f"  std={turn_proba.std():.3f}")
        print(f"[AutoDNA] ── END FINGERPRINT ─────────────────────────────────────\n")

        dist_km = _haversine_total(gps_lat, gps_lon)  #skupna razdalja poti v km (vsota haversine segmentov)
        avg_l100 = (fuel_l / dist_km * 100.0) if dist_km > _MIN_DIST_KM_FOR_AVG else 0.0  #fuel_l / dist_km * 100 = l/100km
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
            fuel_consumption_l=fuel_l,
            avg_fuel_l100km=avg_l100,
            bin_file=str(bin_path),
            gps_file=str(csv_path),
            drive_name=bin_path.stem,
            drive_duration_sec=float(imu_ts[-1]),
            drive_distance_km=dist_km,
            window_size=WINDOW_SIZE,
            window_stride=STRIDE,
            target_fs=TARGET_FS,
            smooth_half=SMOOTH_HALF_WINDOW,
        )

    def _find_files(self):
        #dedupliciraj bin datoteke - windows je case-insensitive, glob vrne dvojnike
        seen: dict = {}
        for p in sorted(
            list(self.drive_dir.glob('*.BIN')) +
            list(self.drive_dir.glob('*.bin'))
        ):
            seen[p.resolve()] = p
        bins = list(seen.values())

        #poisci vse csv datoteke v mapi voznje
        csvs = sorted(self.drive_dir.glob('*.csv'))

        #zagotovi da sta obe datoteki prisotni - brez katerekoli nalozen ne gre
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
        #ce je vec datotek vzemi prvo in opozori
        if len(bins) > 1:
            print(f"[AutoDNA] Multiple BIN files found — using first: {bins[0].name}")
            for b in bins[1:]:
                print(f"          skipping: {b.name}")
        if len(csvs) > 1:
            print(f"[AutoDNA] Multiple CSV files found — using first: {csvs[0].name}")
            for c in csvs[1:]:
                print(f"          skipping: {c.name}")
        return bins[0], csvs[0]

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

        #hitrost je neobvezna - prazna ce csv nima pid stolpca ali ustreznih vrednosti
        speed_df = pd.DataFrame()
        if 'pid' in df.columns:
            speed_df = df[df['pid'].str.strip().isin(['Speed (GPS)', 'Vehicle speed'])].copy()
        speed   = _align_speed(speed_df, ts)
        #smerni kot med zaporednimi gps tockami - potreben za preverjanje l/r oznak
        heading = _compute_heading(lat, lon)
        
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
                    #velika vrzel (stojisce, rdeca luč): ponovi zadnjo pozicijo
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
            speed = _align_speed(speed_df, ts_new)
            heading = _compute_heading(lat_new, lon_new)
            print(f"[AutoDNA] GPS interpolated: {len(ts)} → {len(ts_new)} points"
                  f"  ({gps_dur_s:.0f}s span  sparse={len(ts)/gps_dur_s:.2f} pts/s)")
            ts, lat, lon = ts_new, lat_new, lon_new

        #poraba goriva - iz obd2 pid ali ocena na osnovi razdalje
        fuel_l = _extract_fuel_l(df, _haversine_total(lat, lon))
        return ts, lat, lon, speed, heading, fuel_l

    # ── Window → GPS alignment ────────────────────────────────────────────────
    def _align_to_gps(self, imu_ts, start_samples, gps_ts,
                      imu_motion_t=0.0, gps_motion_onset_t=None):
        """
        Map each IMU prediction window to GPS indices (fully vectorised).

        imu_motion_t      — IMU timestamp of first sustained vehicle motion.
                            Windows before this are clamped to GPS motion onset.
        gps_motion_onset_t — GPS timestamp of first vehicle movement (10 m
                            cumulative distance).  Used as the GPS zero-point so
                            that GPS parked-prefix time is excluded from the
                            scale computation.  Defaults to gps_ts[0].

        The active driving portion [imu_motion_t … imu_ts[-1]] is mapped
        proportionally onto [gps_motion_onset_t … gps_ts[-1]].
        Both sides therefore start at "vehicle first moved" rather than
        "recorder first switched on", which is the critical fix for drives
        where GPS or IMU was already running while parked.
        """
        #preslika vsako imu okno na gps indekse s sorazmernim scalingom aktivnih delov
        imu_dur   = float(imu_ts[-1])
        n_gps     = len(gps_ts)
        N_imu     = len(imu_ts)
        start     = start_samples.astype(np.int64)

        #referencna gps tocka: prvi gps cas ko se vozilo zacne premikati
        gps_ref = float(gps_motion_onset_t) if gps_motion_onset_t is not None else float(gps_ts[0])
        #aktivno trajanje vsakega signala: od prvega gibanja do konca
        imu_active  = max(imu_dur - imu_motion_t, 1.0)
        gps_active  = max(float(gps_ts[-1]) - gps_ref, 1.0)

        #vzorcni indeksi zacetka, konca in sredisca vsakega okna v imu signalu
        s_idx = start
        e_idx = np.minimum(start + WINDOW_SIZE - 1, N_imu - 1)
        c_idx = (s_idx + e_idx) // 2

        t0_all  = imu_ts[s_idx]
        t1_all  = imu_ts[e_idx]
        t_c_all = imu_ts[c_idx]

        #cas okna relativen glede na zacetek gibanja - okna pred gibanjem se stisnejo na 0
        t0_rel  = np.maximum(t0_all  - imu_motion_t, 0.0)
        t1_rel  = np.maximum(t1_all  - imu_motion_t, 0.0)
        t_c_rel = np.maximum(t_c_all - imu_motion_t, 0.0)

        #lestveni faktor: koliko gps sekund ustreza eni imu sekundi
        scale = gps_active / imu_active

        #pretvori cas okna v gps cas (absolutna referenca)
        g0_all  = gps_ref + t0_rel  * scale
        g1_all  = gps_ref + t1_rel  * scale
        g_c_all = gps_ref + t_c_rel * scale

        #pretvori gps case v indekse z binarnim iskanjem; clip zadrzi znotraj veljavnega obsega
        i0 = np.clip(np.searchsorted(gps_ts, g0_all,  side='left'),      0, n_gps - 1)
        i1 = np.clip(np.searchsorted(gps_ts, g1_all,  side='right') - 1, 0, n_gps - 1)
        ic = np.clip(np.searchsorted(gps_ts, g_c_all, side='left'),       0, n_gps - 1)
        i1 = np.maximum(i0, i1)  #zagotovi i1 >= i0 (okna z niclo ali enim vzorcem)

        #diagnostika poravnave
        unique_gps = len(np.unique(ic))
        gps_util   = unique_gps / max(n_gps, 1)
        max_share  = int(np.bincount(ic.astype(np.intp), minlength=n_gps).max())
        print(
            f"[AutoDNA] Alignment — IMU onset {imu_motion_t:.1f}s  "
            f"GPS onset {gps_ref:.1f}  scale={scale:.3f}  "
            f"active IMU={imu_active:.1f}s  GPS={gps_active:.1f}s"
        )
        print(
            f"[AutoDNA] GPS utilization: {unique_gps}/{n_gps} unique pts "
            f"({gps_util:.0%})  max windows per GPS pt = {max_share}"
        )
        if gps_util < 0.30:
            print(
                f"[AutoDNA] WARNING: Low GPS utilization ({gps_util:.0%}) — "
                "predictions are clustered; check IMU/GPS duration mismatch"
            )
        if scale < 0.70:
            print(
                f"[AutoDNA] WARNING: scale={scale:.3f} < 0.70 — IMU active period "
                f"({imu_active:.0f}s) is much longer than GPS active ({gps_active:.0f}s). "
                "Predictions may be compressed to the first portion of the GPS track."
            )
        elif scale > 1.40:
            print(
                f"[AutoDNA] WARNING: scale={scale:.3f} > 1.40 — GPS active period "
                f"({gps_active:.0f}s) is much longer than IMU active ({imu_active:.0f}s). "
                "GPS may include time before/after IMU was recording."
            )
        print(f"[AutoDNA] First 3 window centers: ic={ic[:3].tolist()}")

        return i0.astype(np.int32), i1.astype(np.int32), ic.astype(np.int32)


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
        #vecinska napoved med sosednjimi okni
        out[i] = np.bincount(preds[lo:hi], minlength=3).argmax()
    return out.astype(np.int32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_imu_motion_onset(
    signal: np.ndarray,
    imu_ts: np.ndarray,
    min_dur_s: float = 1.5,
) -> float:
    """
    Return the IMU timestamp (seconds) of the first sustained vehicle motion.

    Uses the gyro magnitude (channels 0-2 of the normalised signal).
    A rolling window of min_dur_s must exceed an adaptive threshold to confirm
    motion onset.  Falls back to 0.0 (no offset) if no clear onset is found.

    Adaptive threshold — measured as 3× the 95th-percentile gyro magnitude
    during the first 10 % of the recording.  This handles drives where
    per-axis normalisation amplifies idle noise (e.g. highway drives with a
    small gyro_z range), which would cause a fixed threshold to fire too early.
    """
    gyro_mag = np.sqrt(signal[:, 0]**2 + signal[:, 1]**2 + signal[:, 2]**2)
    min_samples = max(1, int(min_dur_s * TARGET_FS))

    #adaptivni prag: 3× suma v prvi desetini posnetka (prilagodi se na raven suma)
    n_baseline = max(min_samples * 2, len(signal) // 10)
    noise_level = float(np.percentile(gyro_mag[:n_baseline], 95))
    threshold = max(noise_level * 3.0, 0.02)
    print(f"[AutoDNA] IMU onset  baseline_noise={noise_level:.4f}  "
          f"adaptive_threshold={threshold:.4f}")

    #drsece povprecje z kumulativno vsoto - o(n) casovna zahtevnost
    cs = np.concatenate([[0.0], np.cumsum(gyro_mag)])
    wins = (cs[min_samples:] - cs[:-min_samples]) / min_samples

    hits = np.where(wins > threshold)[0]
    if len(hits) == 0:
        #ni zaznanega gibanja - zacni poravnavo od t=0
        return 0.0

    onset_idx = int(hits[0])
    return float(imu_ts[min(onset_idx, len(imu_ts) - 1)])


def _find_gps_motion_onset(lat: np.ndarray, lon: np.ndarray,
                           ts: np.ndarray, min_dist_m: float = 10.0) -> float:
    """
    Return the GPS timestamp when cumulative travel distance first exceeds
    min_dist_m metres.  Falls back to ts[0] when no clear motion is detected.

    This is used as the GPS alignment reference point so that the IMU-to-GPS
    mapping starts from the first actual vehicle movement on both sides, even
    when the GPS logger was already running while the car was parked.
    """
    R = 6_371_000.0
    cum_m = 0.0
    for i in range(1, len(lat)):
        phi1 = math.radians(lat[i - 1])
        phi2 = math.radians(lat[i])
        dphi = math.radians(lat[i] - lat[i - 1])
        dlam = math.radians(lon[i] - lon[i - 1])
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        cum_m += R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        if cum_m >= min_dist_m:
            return float(ts[i])
    return float(ts[0])


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
