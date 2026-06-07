import os
import sys
import json
import tempfile
import numpy as np
from numpy.fft import rfft
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parent.parent   
_GITHUB = _ROOT.parent                              
for _p in (str(_ROOT), str(_GITHUB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stm32.bin_parser.stm_utils import read_packets_from_file
from ai.preprocessing import (
    load_sensor_data,
    resample_sensors_to_common_grid,
    preprocess_sensor_data,
    normalize_signal,
)

#frekvenca vzorcenja po ponovnem vzorcevanju
TARGET_FS = 50  # Hz after resampling
#meja nizkopasovnega butterworth filtra
CUTOFF_HZ = 5.0  # Butterworth low-pass cutoff

#okna za zavoje - mora se ujemati z build_dataset.py in train_xgboost.py
WINDOW_SIZE_TURN = 100  # 2 s @ 50 Hz
STRIDE_TURN = 25

#okna za klance - mora se ujemati z build_dataset.py in train_xgboost.py
WINDOW_SIZE_HILL = 200  # 4 s @ 50 Hz
STRIDE_HILL = 50

#ohrani stare aliases za module ki uvazajo WINDOW_SIZE / STRIDE
WINDOW_SIZE = WINDOW_SIZE_TURN
STRIDE = STRIDE_TURN

#indeksi kanalov v imu signalu oblike (n, 11)
GYRO_X, GYRO_Y, GYRO_Z = 0, 1, 2
ACCEL_X, ACCEL_Y, ACCEL_Z = 3, 4, 5
MAG_X, MAG_Y, MAG_Z = 6, 7, 8
PITCH_CF, ROLL_CF = 9, 10

#parametri komplementarnega filtra - mora se ujemati z build_dataset.py
_CF_ALPHA = 0.98
_CF_INIT_SECONDS = 2.0

#pragi za ciscenje oznak - mora se ujemati z build_dataset.py
_MIN_TURN_ANGLE = 5.0
_MIN_HILL_ANGLE = 1.0
_THRESHOLD_TURN = 0.5
_THRESHOLD_HILL_MIN = 0.30
_THRESHOLD_HILL_DOM = 2.0

#pretvorba med stevilko razreda in besednim imenom
TURN_MAP = {0: 'none', 1: 'left', 2: 'right'}
HILL_MAP = {0: 'none', 1: 'up', 2: 'down'}
_TURN_INT = {'none': 0, 'left': 1, 'right': 2}
_HILL_INT = {'none': 0, 'up': 1, 'down': 2}

#poti do modelov in ucnih podatkov
MODEL_DIR = _ROOT / 'ai' / 'window_based' / 'dataset_output'
PARSED_DATA_DIR = _ROOT / 'data' / 'training_data' / 'parsed_data'
LABELS_DIR = _ROOT / 'data' / 'training_data' / 'labeled_data_json'

#hiperparametri xgboost - ena definicija za ucenje in metriko (brez duplikatov)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss',
    verbosity=0,
    tree_method='hist',
)

#stevilo znacilk za vsak model - mora se ujemati z extract_features_*
N_FEATURES_TURN = 43
N_FEATURES_HILL = 41


# ── Complementary filter ──────────────────────────────────────────────────────

def _complementary_filter(ax, ay, az, gx, gy, fs):
    """
    Pitch/roll complementary filter (matches build_dataset.py exactly).
    Gyro is expected in rad/s.  Returns pitch, roll in radians.
    """
    n = len(ax)
    dt = 1.0 / fs

    #inicializacija kota iz prvih 2 sekund - predpostavlja mirovanje senzorja
    init_n = max(1, min(int(_CF_INIT_SECONDS * fs), n // 4))
    ax_i = ax[:init_n].mean()
    ay_i = ay[:init_n].mean()
    az_i = az[:init_n].mean()

    pitch = np.zeros(n, dtype=np.float64)
    roll = np.zeros(n, dtype=np.float64)

    #zacetna kota iz samega pospeska (arctan2)
    pitch[0] = np.arctan2(ax_i, np.sqrt(ay_i**2 + az_i**2))
    roll[0] = np.arctan2(ay_i, az_i)

    #glavna zanka: 98% gyro integracija + 2% popravek iz pospeska
    for i in range(1, n):
        p_acc = np.arctan2(ax[i], np.sqrt(ay[i]**2 + az[i]**2))
        r_acc = np.arctan2(ay[i], az[i])
        pitch[i] = _CF_ALPHA * (pitch[i-1] + gy[i] * dt) + (1 - _CF_ALPHA) * p_acc
        roll[i] = _CF_ALPHA * (roll[i-1] + gx[i] * dt) + (1 - _CF_ALPHA) * r_acc

    return pitch.astype(np.float32), roll.astype(np.float32)


_FS = float(TARGET_FS)


def extract_features_turn(window: np.ndarray) -> np.ndarray:
    """43 features from a (100, 11) window. Mirrors train_xgboost.py exactly."""
    #izvleci 43 znacilk iz okna (100, 11) za detekcijo zavojev
    #mora biti identicno train_xgboost.py - vsaka razlika kvari napovedi
    feats = []

    #razvezi kanale signala - vsak kanal je 1d polje dolzine 100
    gx = window[:, GYRO_X]
    gy = window[:, GYRO_Y]
    gz = window[:, GYRO_Z]
    ax = window[:, ACCEL_X]
    ay = window[:, ACCEL_Y]
    az = window[:, ACCEL_Z]
    roll_cf = window[:, ROLL_CF]

    #znacilke gyro_z - glavni signal za smer in intenziteto zavoja
    feats += [
        gz.max(), gz.min(), gz.mean(), gz.std(),
        np.trapezoid(gz),
        abs(np.trapezoid(gz)),
        np.abs(gz).max(),
        np.sum(gz > 0.1) / len(gz),
        np.sum(gz < -0.1) / len(gz),
        np.median(gz),
        np.argmax(np.abs(gz)) / len(gz),
        np.mean(gz**3) / (gz.std()**3 + 1e-6),
    ]

    #znacilke gyro_x in gyro_y
    for g in (gx, gy):
        feats += [g.mean(), g.std(), np.abs(g).max()]

    #znacilke pospeska - vzdolzni in bocni pospesek med zavojem
    feats += [ax.mean(), ax.std(), ax.max(), ax.min()]
    feats += [
        ay.mean(), ay.std(), np.abs(ay).max(),
        np.sum(ay < -0.2) / len(ay),
        np.sum(ay > 0.2) / len(ay),
        np.trapezoid(ay),
    ]
    feats += [az.mean(), az.std()]

    #korelacija med kotno hitrostjo in bocnim pospesekom
    corr_gz_ay = (float(np.corrcoef(gz, ay)[0, 1])
                  if gz.std() > 0 and ay.std() > 0 else 0.0)
    corr_gz_ax = (float(np.corrcoef(gz, ax)[0, 1])
                  if gz.std() > 0 and ax.std() > 0 else 0.0)
    feats += [corr_gz_ay, corr_gz_ax]

    #energija signalov
    feats += [np.mean(gz**2), np.mean(ay**2), np.mean(ax**2)]

    #frekvencan analiza kotne hitrosti
    fft_gz = np.abs(rfft(gz - gz.mean()))
    feats += [fft_gz[:3].sum(), fft_gz[3:10].sum(), float(np.argmax(fft_gz))]

    #znacilke nagiba iz komplementarnega filtra
    feats += [
        roll_cf.mean(),
        roll_cf.std(),
        roll_cf[-1] - roll_cf[0],
        np.abs(roll_cf).max(),
        roll_cf.max() - roll_cf.min(),
    ]

    return np.array(feats, dtype=np.float32)


def extract_features_hill(window: np.ndarray) -> np.ndarray:
    """41 features from a (200, 11) window. Mirrors train_xgboost.py exactly."""
    #izvleci 41 znacilk iz okna (200, 11) za detekcijo klancev
    #mora biti identicno train_xgboost.py - vsaka razlika kvari napovedi
    feats = []

    #razvezi kanale signala - vsak kanal je 1d polje dolzine 200
    gx = window[:, GYRO_X]
    gy = window[:, GYRO_Y]
    gz = window[:, GYRO_Z]
    ax = window[:, ACCEL_X]
    ay = window[:, ACCEL_Y]
    az = window[:, ACCEL_Z]
    pitch_cf = window[:, PITCH_CF]
    roll_cf = window[:, ROLL_CF]

    #znacilke naklona iz komplementarnega filtra - kljucne za detekcijo klanca
    p_mean = pitch_cf.mean()
    p_std = pitch_cf.std()
    p_delta = float(pitch_cf[-1]) - float(pitch_cf[0])

    #trend naklona skozi 4 cetrtine okna - zazna enakomeren vzpon ali spust
    n_q = max(1, len(pitch_cf) // 4)
    quarters_p = [pitch_cf[i*n_q:(i+1)*n_q].mean() for i in range(4)]
    pitch_trend_q = quarters_p[3] - quarters_p[0]

    t_axis = np.arange(len(pitch_cf), dtype=np.float32)
    pitch_slope = np.polyfit(t_axis, pitch_cf, 1)[0]

    feats += [
        p_mean, p_std, p_delta, pitch_trend_q, pitch_slope,
        pitch_cf.min(), pitch_cf.max(),
        pitch_cf.max() - pitch_cf.min(),
    ]
    feats += quarters_p

    #fizikalne znacilke - odstopanje od gravitacije locuje surove in normirane enote
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    a_norm_mean = a_norm.mean()
    expected_g = 9.81 if a_norm_mean > 5.0 else 1.0
    g_dev_mean = np.abs(a_norm - expected_g).mean()
    g_dev_max = np.abs(a_norm - expected_g).max()

    dt = 1.0 / _FS
    pitch_delta_gyro = np.trapezoid(gy) * dt

    feats += [
        a_norm_mean, a_norm.std(),
        g_dev_mean, g_dev_max,
        pitch_delta_gyro, abs(pitch_delta_gyro),
        np.abs(pitch_delta_gyro - p_delta),
    ]

    #surovi pospesek in linearni trend vzdolz okna
    ax_slope = np.polyfit(t_axis, ax, 1)[0]
    az_slope = np.polyfit(t_axis, az, 1)[0]
    feats += [
        ax.mean(), ax.std(),
        ay.mean(), ay.std(),
        az.mean(), az.std(),
        ax.max() - ax.min(),
        az.max() - az.min(),
        ax_slope, az_slope,
    ]

    #znacilke ziroskopa
    feats += [
        gy.mean(), gy.std(), np.abs(gy).max(),
        gx.mean(), gx.std(),
        gz.mean(), gz.std(),
    ]

    #znacilke nagiba
    feats += [roll_cf.mean(), roll_cf.std()]

    #frekvencan analiza navpicnega pospeska - zazna vibracije klanca
    fft_az = np.abs(rfft(az - az.mean()))
    total_e = fft_az.sum() + 1e-6
    feats += [
        fft_az[:3].sum() / total_e,
        fft_az[3:10].sum() / total_e,
        fft_az[10:].sum() / total_e,
    ]

    return np.array(feats, dtype=np.float32)



def check_model_compatibility() -> tuple[bool, str]:
    """
    Return (ok, message).  ok=False means models were trained by the old
    build_dataset.py pipeline which uses a different preprocessing order.
    The app's own train_models() produces compatible models.
    """
    #preveri ali so shranjeni modeli zdrzljivi s trenutnim predprocesiranjem

    #preberi shranjene metrike - ce ne obstajajo so modeli netrenirani
    metrics = load_metrics()
    if metrics is None:
        return False, "No model_metrics.json found — models have never been trained via the app."

    #stara pipeline je ucila cf na ze normiranih podatkih (napacen vrstni red)
    #nova pipeline najprej izracuna cf, sele potem normira - metrika razlikuje versiji
    method = metrics.get('eval_method', '')
    if '5-fold' in method:
        return False, (
            "Models were trained by the OLD pipeline (5-fold CV). "
            "Preprocessing mismatch: old training used CF on post-normalization data "
            "while inference uses CF on raw data. "
            "Retrain via File → Train Models (Ctrl+T)."
        )
    return True, f"Models OK ({method})."


def parse_and_preprocess_bin(bin_path: Path):
    """
    Parse a .BIN file through the full AutoDNA preprocessing pipeline.

    Returns
    -------
    signal     : (N, 11) float32 — [gyro_xyz, accel_xyz, mag_xyz, pitch_cf, roll_cf]
    timestamps : (N,) float64   — seconds from recording start
    """
    #razcleni bin datoteko in pozeni celoten predprocesirni pipeline

    from collections import defaultdict

    #razcleni vse pakete iz bin datoteke v slovar po imenu senzorja
    packets = read_packets_from_file(str(bin_path))
    rows: dict[str, list] = defaultdict(list)
    for pkt in packets:
        for x, y, z in pkt.data:
            rows[pkt.sensor].append([float(pkt.ts), float(x), float(y), float(z)])

    if not rows:
        raise ValueError(f"No sensor packets parsed from {bin_path.name}")

    #pretvori vrstice v numpy matrike - ena matrika na senzor
    raw_arrays = {name: np.array(r, dtype=np.float32) for name, r in rows.items()}

    #zacasna npz datoteka za load_sensor_data - takoj zbrisana po branju
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

    print(f"[AutoDNA] ── PIPELINE STAGE 1: BIN PARSING ─────────────────────────")
    print(f"  Total packets parsed: {sum(len(v) for v in rows.values())}"
          f"  sensors: {list(rows.keys())}")

    # ── Debug: raw sensor ranges ──────────────────────────────────────────────
    print("[AutoDNA] Raw sensor packet counts:")
    for sname, arr in raw_arrays.items():
        print(f"  {sname}: {len(arr)} packets  "
              f"x=[{arr[:,1].min():.3f}, {arr[:,1].max():.3f}]  "
              f"y=[{arr[:,2].min():.3f}, {arr[:,2].max():.3f}]  "
              f"z=[{arr[:,3].min():.3f}, {arr[:,3].max():.3f}]")

    print(f"[AutoDNA] ── PIPELINE STAGE 2: RESAMPLING → {TARGET_FS}Hz ──────────────")
    resampled = resample_sensors_to_common_grid(sensors, target_fs=TARGET_FS)
    _n_resampled = len(resampled.get('gyro', {}).get('x', []))
    _ts_resampled = resampled.get('gyro', {}).get('ts', np.array([0.0, 0.0]))
    print(f"  Resampled to {_n_resampled} samples"
          f"  span={float(_ts_resampled[-1] - _ts_resampled[0]):.2f}s"
          f"  @ {TARGET_FS}Hz")

    print("[AutoDNA] Resampled sensor ranges (raw, before normalization):")
    for sname in ('gyro', 'accel', 'mag'):
        if sname in resampled:
            d = resampled[sname]
            print(f"  {sname}  x=[{d['x'].min():.3f},{d['x'].max():.3f}]"
                  f"  y=[{d['y'].min():.3f},{d['y'].max():.3f}]"
                  f"  z=[{d['z'].min():.3f},{d['z'].max():.3f}]")

    print(f"[AutoDNA] ── PIPELINE STAGE 3: COMPLEMENTARY FILTER ─────────────────")
    #cf se izracuna na surovih (adc) podatkih - pred normalizacijo po osi
    #normalizacija po osi unici razmerje ax/sqrt(ay²+az²) ki ga arctan2 potrebuje
    #za pravilen izracun naklona - vrstni red je bistvenega pomena
    pitch_cf_raw, roll_cf_raw = _complementary_filter(
        resampled['accel']['x'].astype(np.float64),
        resampled['accel']['y'].astype(np.float64),
        resampled['accel']['z'].astype(np.float64),
        resampled['gyro']['x'].astype(np.float64),
        resampled['gyro']['y'].astype(np.float64),
        fs=TARGET_FS,
    )
    print(f"[AutoDNA] CF (raw) pitch=[{pitch_cf_raw.min():.3f},{pitch_cf_raw.max():.3f}] rad"
          f"  roll=[{roll_cf_raw.min():.3f},{roll_cf_raw.max():.3f}] rad")

    pitch_cf = normalize_signal(pitch_cf_raw).astype(np.float32)
    roll_cf  = normalize_signal(roll_cf_raw).astype(np.float32)

    print(f"[AutoDNA] ── PIPELINE STAGE 4: SMOOTH + LOWPASS + NORMALIZE ─────────")
    processed = preprocess_sensor_data(resampled, fs=TARGET_FS, cutoff=CUTOFF_HZ)

    print("[AutoDNA] Processed (normalized) sensor ranges:")
    for sname in ('gyro', 'accel', 'mag'):
        if sname in processed:
            d = processed[sname]
            print(f"  {sname}  x=[{d['x'].min():.3f},{d['x'].max():.3f}]"
                  f"  y=[{d['y'].min():.3f},{d['y'].max():.3f}]"
                  f"  z=[{d['z'].min():.3f},{d['z'].max():.3f}]")

    signal = np.column_stack([
        processed['gyro']['x'],  processed['gyro']['y'],  processed['gyro']['z'],
        processed['accel']['x'], processed['accel']['y'], processed['accel']['z'],
        processed['mag']['x'],   processed['mag']['y'],   processed['mag']['z'],
        pitch_cf, roll_cf,
    ]).astype(np.float32)

    print(f"[AutoDNA] Signal matrix assembled: shape={signal.shape}"
          f"  channels=gyro_xyz|accel_xyz|mag_xyz|pitch_cf|roll_cf")
    ch_names = ['gx','gy','gz','ax','ay','az','mx','my','mz','pitch','roll']
    for ci, cn in enumerate(ch_names):
        col = signal[:, ci]
        print(f"  ch{ci:02d} {cn:5s}: min={col.min():7.3f}  max={col.max():7.3f}"
              f"  mean={col.mean():7.3f}  std={col.std():6.3f}")

    #odstrani staticni odmik gravitacije iz accel_y
    #senzor je fizicno nagnjen - to povzroci velik negativni dc odmik ki zakrijee centripetalni pospesek
    #odstevamo povprecje prvih 2 sekund da ay odraza samo dinamicne sile
    init_n = min(int(_CF_INIT_SECONDS * TARGET_FS), len(signal))
    ay_offset = float(signal[:init_n, ACCEL_Y].mean())
    signal[:, ACCEL_Y] -= ay_offset
    print(f"[AutoDNA] accel_y gravity offset removed: {ay_offset:.4f}"
          f"  ay range after: [{signal[:,ACCEL_Y].min():.3f},{signal[:,ACCEL_Y].max():.3f}]")

    print(f"[AutoDNA] ── PIPELINE STAGE 5: FINAL SIGNAL READY ───────────────────")
    print(f"  signal.shape = {signal.shape}  dtype={signal.dtype}")
    print(f"  global range: min={signal.min():.3f}  max={signal.max():.3f}")

    timestamps = processed['gyro']['ts'].astype(np.float64)
    print(f"  timestamps: 0.00 … {float(timestamps[-1]):.2f}s"
          f"  (span={float(timestamps[-1]-timestamps[0]):.2f}s)")

    #preveri ali so shranjeni modeli uceni z istim predprocesiranjem
    ok, msg = check_model_compatibility()
    if not ok:
        print(f"\n[AutoDNA] ⚠ MODEL COMPATIBILITY WARNING: {msg}\n")
    else:
        print(f"[AutoDNA] {msg}")

    return signal, timestamps


# ── Windowing ─────────────────────────────────────────────────────────────────

def make_windows(signal: np.ndarray):
    """
    Overlapping turn windows from (N, 11) using STRIDE_TURN=25.
    Used for GPS alignment (finer granularity).

    Returns
    -------
    windows       : (M, 100, 11)
    start_samples : (M,) int
    """
    #ustvari prekrivajoca se okna (75% prekrivanje) za fino casovno poravnavo z gps

    #izracunaj stevilo oken - (dolzina - velikost_okna) // korak + 1
    n_windows = (len(signal) - WINDOW_SIZE_TURN) // STRIDE_TURN + 1
    #sestavi 3d matriko oken z rezanjem brez kopiranja (pogled na originalen signal)
    windows = np.stack([
        signal[i * STRIDE_TURN : i * STRIDE_TURN + WINDOW_SIZE_TURN]
        for i in range(n_windows)
    ])
    #zacetni vzorec vsakega okna - potreben za casovno poravnavo z gps sledjo
    start_samples = np.arange(n_windows, dtype=np.int64) * STRIDE_TURN
    return windows, start_samples


# ── XGBoost inference ─────────────────────────────────────────────────────────

#preveri obstoj obeh json datotek - potrebno pred vsakim nalaganjem voznje
def models_exist() -> bool:
    return ((MODEL_DIR / 'model_turn.json').exists() and
            (MODEL_DIR / 'model_hill.json').exists())


def load_xgb_models():
    #nalozi oba shranjena xgboost modela iz json datotek
    from xgboost import XGBClassifier
    if not models_exist():
        raise FileNotFoundError(
            f"XGBoost models not found in {MODEL_DIR}. "
            "Train them first via the app Setup dialog."
        )
    #model za zavoje: 43 znacilk, 3 razredi (brez/levo/desno)
    m_turn = XGBClassifier()
    m_turn.load_model(str(MODEL_DIR / 'model_turn.json'))
    #model za klance: 41 znacilk, 3 razredi (brez/gor/dol)
    m_hill = XGBClassifier()
    m_hill.load_model(str(MODEL_DIR / 'model_hill.json'))
    return m_turn, m_hill


def _load_hill_multipliers() -> np.ndarray:
    #multiplikatorji verjetnosti klancev - nauceni med treningom za boljso tocnost
    #ce datoteka ne obstaja vrni enote (brez ucinka)
    path = MODEL_DIR / 'hill_prob_multipliers.npy'
    if path.exists():
        return np.load(str(path)).astype(np.float32)
    return np.ones(3, dtype=np.float32)


def predict_windows(signal: np.ndarray):
    """
    Run turn + hill inference on the full (N, 11) signal.

    Internally creates:
      • Turn windows : (M_turn, 100, 11) — stride 25
      • Hill windows : (M_hill, 200, 11) — stride 50

    Hill predictions are mapped back to the turn window grid so all
    output arrays have length M_turn (= number of turn windows).

    Returns
    -------
    turn_preds : (M_turn,) int32   — 0=none  1=left  2=right
    hill_preds : (M_turn,) int32   — 0=none  1=up    2=down
    turn_proba : (M_turn, 3) f32
    hill_proba : (M_turn, 3) f32   — raw hill proba resampled to turn grid
    """
    #pozeni napovedi zavojev in klancev - hill okna se preslikajo na zavorno mrezo

    #nalozi oba modela in multiplikatorje verjetnosti klancev
    m_turn, m_hill = load_xgb_models()
    hill_mult = _load_hill_multipliers()

    print(f"[AutoDNA] ── PIPELINE STAGE 6: FEATURE EXTRACTION + INFERENCE ────────")
    print(f"  Input signal shape: {signal.shape}")
    print(f"  Hill prob multipliers: {hill_mult.tolist()}")

    N = len(signal)

    #okna za zavoje: 100 vzorcev, korak 25
    M_turn = max(0, (N - WINDOW_SIZE_TURN) // STRIDE_TURN + 1)
    if M_turn == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    X_turn = np.array([
        extract_features_turn(signal[i * STRIDE_TURN:i * STRIDE_TURN + WINDOW_SIZE_TURN])
        for i in range(M_turn)
    ], dtype=np.float32)
    #zamenjaj nan/inf ki nastanejo pri enic korelacijah ali fft
    X_turn = np.nan_to_num(X_turn, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"[AutoDNA] Turn features ({M_turn} windows × {X_turn.shape[1]} feats):"
          f"  min={X_turn.min():.4f}  max={X_turn.max():.4f}"
          f"  mean={X_turn.mean():.4f}  std={X_turn.std():.4f}")
    nan_count = int(np.isnan(X_turn).sum())
    if nan_count:
        print(f"[AutoDNA] WARNING: {nan_count} NaN values in turn feature matrix!")

    turn_preds = m_turn.predict(X_turn).astype(np.int32)
    turn_proba = m_turn.predict_proba(X_turn).astype(np.float32)

    #izpisi 5 oken z najvisjim zaupanjem za zaznavo zavoja
    turn_event_prob = 1.0 - turn_proba[:, 0]
    top5 = np.argsort(turn_event_prob)[::-1][:5]
    print(f"[AutoDNA] Turn top-5 event windows: "
          + "  ".join(f"w{i}={turn_event_prob[i]:.2f}({['N','L','R'][turn_preds[i]]})"
                      for i in top5))

    #okna za klance: 200 vzorcev, korak 50
    M_hill = max(0, (N - WINDOW_SIZE_HILL) // STRIDE_HILL + 1)
    if M_hill == 0:
        return turn_preds, np.zeros(M_turn, np.int32), turn_proba, np.zeros((M_turn, 3), np.float32)

    X_hill = np.array([
        extract_features_hill(signal[i * STRIDE_HILL:i * STRIDE_HILL + WINDOW_SIZE_HILL])
        for i in range(M_hill)
    ], dtype=np.float32)
    X_hill = np.nan_to_num(X_hill, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"[AutoDNA] Hill features ({M_hill} windows × {X_hill.shape[1]} feats):"
          f"  min={X_hill.min():.4f}  max={X_hill.max():.4f}"
          f"  mean={X_hill.mean():.4f}  std={X_hill.std():.4f}")

    hill_proba_raw = m_hill.predict_proba(X_hill).astype(np.float32)
    #prilagoditev verjetnosti z naученimi multiplikatorji
    hill_proba_tuned = hill_proba_raw * hill_mult
    hill_preds_raw = np.argmax(hill_proba_tuned, axis=1).astype(np.int32)

    #preslikaj napovedi klancev na mrezo zavojev z najblizjim sosedom
    turn_centers = np.arange(M_turn) * STRIDE_TURN + WINDOW_SIZE_TURN // 2
    hill_centers = np.arange(M_hill) * STRIDE_HILL + WINDOW_SIZE_HILL // 2

    idx = np.searchsorted(hill_centers, turn_centers)
    idx_r = np.clip(idx, 0, M_hill - 1)
    idx_l = np.clip(idx - 1, 0, M_hill - 1)
    dist_r = np.abs(hill_centers[idx_r] - turn_centers)
    dist_l = np.abs(hill_centers[idx_l] - turn_centers)
    nearest = np.where(dist_l < dist_r, idx_l, idx_r)

    hill_preds = hill_preds_raw[nearest].astype(np.int32)
    hill_proba = hill_proba_raw[nearest]

    return turn_preds, hill_preds, turn_proba, hill_proba


# ── Model training ────────────────────────────────────────────────────────────

def train_models(progress_cb=None):
    """
    Train XGBoost turn + hill classifiers.
    Matches build_dataset.py + train_xgboost.py pipeline:
      • 11-channel signal (with pitch_cf, roll_cf)
      • Separate feature extractors for turn (43-feat) and hill (41-feat)
      • Separate window sizes: turn=100, hill=200
      • Label filtering: min turn angle 5°, min hill angle 1°
      • Single-threshold for turn (0.5), dual-threshold for hill
      • Sample weights (balanced)
      • Hill probability-multiplier tuning
    """
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    #ustvari izhodisno mapo ce ne obstaja
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    #poiscí vse npz datoteke z ucnimi podatki - brez ze predprocesiranih
    npz_files = sorted(
        f for f in PARSED_DATA_DIR.glob('LOG*.npz')
        if 'preprocessed' not in f.stem
    )
    total = len(npz_files)
    if total == 0:
        raise ValueError(f"No LOG*.npz files found in {PARSED_DATA_DIR}.")

    #seznami za zbiranje znacilk in oznak vseh posnetkov
    all_Xt, all_Yt, all_Xh, all_Yh = [], [], [], []
    #id posnetka za vsako okno - prepreci pusc podatkov med logi v krizni validaciji
    all_ids_t, all_ids_h = [], []

    for i, npz_path in enumerate(npz_files):
        #vsak npz mora imeti ustrezno json datoteko z oznakami
        json_path = LABELS_DIR / (npz_path.stem + '_labels.json')
        if not json_path.exists():
            continue
        if progress_cb:
            progress_cb(f"Loading {npz_path.name}  ({i+1}/{total})", int(i / total * 70))
        try:
            #enako predprocesiranje kot pri napovedi - mora se ujemati do pikice
            sensors   = load_sensor_data(str(npz_path))
            resampled = resample_sensors_to_common_grid(sensors, target_fs=TARGET_FS)
            #cf se izracuna na surovih podatkih pred normiranjem - to je kljucno
            pitch_cf_raw, roll_cf_raw = _complementary_filter(
                resampled['accel']['x'].astype(np.float64),
                resampled['accel']['y'].astype(np.float64),
                resampled['accel']['z'].astype(np.float64),
                resampled['gyro']['x'].astype(np.float64),
                resampled['gyro']['y'].astype(np.float64),
                fs=TARGET_FS,
            )
            pitch_cf = normalize_signal(pitch_cf_raw).astype(np.float32)
            roll_cf  = normalize_signal(roll_cf_raw).astype(np.float32)
            processed = preprocess_sensor_data(resampled, fs=TARGET_FS, cutoff=CUTOFF_HZ)
        except Exception as exc:
            print(f"[SKIP] {npz_path.name}: {exc}")
            continue

        #sestavi 11-kanalni signal: gyro | accel | mag | pitch | roll
        signal = np.column_stack([
            processed['gyro']['x'],  processed['gyro']['y'],  processed['gyro']['z'],
            processed['accel']['x'], processed['accel']['y'], processed['accel']['z'],
            processed['mag']['x'],   processed['mag']['y'],   processed['mag']['z'],
            pitch_cf, roll_cf,
        ]).astype(np.float32)

        #isto odstranjevanje odmika gravitacije kot pri napovedi
        init_n = min(int(_CF_INIT_SECONDS * TARGET_FS), len(signal))
        signal[:, ACCEL_Y] -= signal[:init_n, ACCEL_Y].mean()

        ts = processed['gyro']['ts']

        #nalozi oznake in pocisti sibke (premajhni koti)
        with open(json_path) as f:
            labels = json.load(f)['labels']
        labels = _filter_weak_labels(labels)

        #okna za zavoje (100 vzorcev): pridobi znacilke in dodeli oznako
        n_turn = max(0, (len(signal) - WINDOW_SIZE_TURN) // STRIDE_TURN + 1)
        for w in range(n_turn):
            i0, i1 = w * STRIDE_TURN, w * STRIDE_TURN + WINDOW_SIZE_TURN
            t_start, t_end = float(ts[i0]), float(ts[i1 - 1])
            lbl = _label_window_turn(t_start, t_end, labels)
            feat = extract_features_turn(signal[i0:i1])
            all_Xt.append(feat)
            all_Yt.append(_TURN_INT[lbl])
            all_ids_t.append(i)

        #okna za klance (200 vzorcev): pridobi znacilke in dodeli oznako
        n_hill = max(0, (len(signal) - WINDOW_SIZE_HILL) // STRIDE_HILL + 1)
        for w in range(n_hill):
            i0, i1 = w * STRIDE_HILL, w * STRIDE_HILL + WINDOW_SIZE_HILL
            t_start, t_end = float(ts[i0]), float(ts[i1 - 1])
            lbl = _label_window_hill(t_start, t_end, labels)
            feat = extract_features_hill(signal[i0:i1])
            all_Xh.append(feat)
            all_Yh.append(_HILL_INT[lbl])
            all_ids_h.append(i)

    if not all_Xt:
        raise ValueError("No training windows found.")

    if progress_cb:
        progress_cb("Building feature matrices…", 72)

    #pretvori sezname v numpy matrike in pocisti morebitne nan/inf vrednosti
    Xt        = np.array(all_Xt,   dtype=np.float32)
    Yt        = np.array(all_Yt,   dtype=np.int32)
    log_ids_t = np.array(all_ids_t, dtype=np.int32)
    Xh        = np.array(all_Xh,   dtype=np.float32)
    Yh        = np.array(all_Yh,   dtype=np.int32)
    log_ids_h = np.array(all_ids_h, dtype=np.int32)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=1e6, neginf=-1e6)
    Xh = np.nan_to_num(Xh, nan=0.0, posinf=1e6, neginf=-1e6)

    #uravnotezene teze vzorcev: razredi z manj primeri dobijo vecjo teza
    if progress_cb:
        progress_cb(f"Training turn classifier  ({len(Xt)} windows)…", 75)
    sw_turn = compute_sample_weight('balanced', Yt)
    m_turn = XGBClassifier(**XGB_PARAMS)
    m_turn.fit(Xt, Yt, sample_weight=sw_turn)

    if progress_cb:
        progress_cb(f"Training hill classifier  ({len(Xh)} windows)…", 88)
    sw_hill = compute_sample_weight('balanced', Yh)
    m_hill = XGBClassifier(**XGB_PARAMS)
    m_hill.fit(Xh, Yh, sample_weight=sw_hill)

    #poisci optimalne multiplikatorje verjetnosti klancev z iskanjem po mrezi
    if progress_cb:
        progress_cb("Tuning hill probability multipliers…", 94)
    hill_mult = _search_hill_multipliers(m_hill.predict_proba(Xh), Yh)
    np.save(str(MODEL_DIR / 'hill_prob_multipliers.npy'), hill_mult)

    #shrani oba modela v json format
    m_turn.save_model(str(MODEL_DIR / 'model_turn.json'))
    m_hill.save_model(str(MODEL_DIR / 'model_hill.json'))

    #izracunaj in shrani metrike krizne validacije
    if progress_cb:
        progress_cb("Computing metrics…", 97)
    _save_metrics(m_turn, m_hill, Xt, Yt, Xh, Yh, hill_mult, log_ids_t, log_ids_h)

    if progress_cb:
        progress_cb(f"Done — {len(Xt)} turn / {len(Xh)} hill windows.", 100)

    return len(Xt), int((Yt != 0).sum()), int((Yh != 0).sum())


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _save_metrics(m_turn, m_hill, Xt, Yt, Xh, Yh, hill_mult, log_ids_t, log_ids_h):
    """
    Computes generalisation metrics using 3-fold stratified group cross-validation.
    Each fold trains a new model on 2/3 of the logs and evaluates on the held-out 1/3.
    This gives honest out-of-sample accuracy (no data leakage between logs).
    """
    #3-kratna krizna validacija po skupinah (skupina = posnetek) - brez puscanja podatkov
    import json as _json
    import datetime
    from xgboost import XGBClassifier as _XGB
    from sklearn.metrics import (accuracy_score,
                                 precision_recall_fscore_support, f1_score)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.utils.class_weight import compute_sample_weight

    def _cv_preds(X, Y, ids, mult=None):
        #3-kratna stratificirana grupna krizna validacija - ena skupina = en posnetek
        #prepreci da bi okna istega posnetka bila v ucni in testni mnozici hkrati
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        all_true, all_pred = [], []
        for tr_idx, val_idx in sgkf.split(X, Y, groups=ids):
            Xtr, Ytr = X[tr_idx], Y[tr_idx]
            Xv, Yv   = X[val_idx], Y[val_idx]
            sw  = compute_sample_weight('balanced', Ytr)
            mdl = _XGB(**XGB_PARAMS)
            mdl.fit(Xtr, Ytr, sample_weight=sw)
            #ce so na voljo multiplikatorji jih upostevaj pri napovedi
            if mult is not None:
                proba = mdl.predict_proba(Xv).astype(np.float32)
                pred  = np.argmax(proba * mult, axis=1).astype(np.int32)
            else:
                pred = mdl.predict(Xv).astype(np.int32)
            all_true.extend(Yv.tolist())
            all_pred.extend(pred.tolist())
        return np.array(all_true), np.array(all_pred)

    #zberi napovedi krizne validacije za oba modela
    Yt_true, yt_pred = _cv_preds(Xt, Yt, log_ids_t)
    Yh_true, yh_pred = _cv_preds(Xh, Yh, log_ids_h, hill_mult)

    def _per_class(true, pred, names):
        #izracunaj preciznost, priklic in f1 za vsak razred posebej
        p, r, f, _ = precision_recall_fscore_support(
            true, pred, labels=list(range(len(names))),
            average=None, zero_division=0,
        )
        return {names[i]: {'precision': round(float(p[i]), 3),
                            'recall':    round(float(r[i]), 3),
                            'f1':        round(float(f[i]), 3)}
                for i in range(len(names))}

    #sestavi json slovar metrik za oba modela
    metrics = {
        'trained_at':   datetime.datetime.now().isoformat(timespec='seconds'),
        'eval_method':  '3-fold stratified group cross-validation (group = log)',
        'turn': {
            'n_windows': int(len(Yt)),
            'accuracy':  round(float(accuracy_score(Yt_true, yt_pred)), 4),
            'f1_macro':  round(float(f1_score(Yt_true, yt_pred, average='macro',
                                              zero_division=0)), 4),
            'per_class': _per_class(Yt_true, yt_pred, ['none', 'left', 'right']),
        },
        'hill': {
            'n_windows': int(len(Yh)),
            'accuracy':  round(float(accuracy_score(Yh_true, yh_pred)), 4),
            'f1_macro':  round(float(f1_score(Yh_true, yh_pred, average='macro',
                                              zero_division=0)), 4),
            'per_class': _per_class(Yh_true, yh_pred, ['none', 'up', 'down']),
        },
    }
    #shrani metrike v json datoteko za kasnejsi pregled ali diagnostiko
    (MODEL_DIR / 'model_metrics.json').write_text(_json.dumps(metrics, indent=2))


def load_metrics() -> dict | None:
    """Load saved training metrics, or None if not yet trained."""
    #preberi model_metrics.json - vrni none ce se modeli niso bili trenirani
    path = MODEL_DIR / 'model_metrics.json'
    if not path.exists():
        return None
    try:
        import json as _json
        return _json.loads(path.read_text())
    except Exception:
        return None


# ── Label helpers ─────────────────────────────────────────────────────────────

def _filter_weak_labels(labels):
    #odstrani oznake z majhnim kotom - preslabi signali za zanesljivo ucenje
    cleaned = []
    for lbl in labels:
        new = dict(lbl)
        #zavoj z manj kot 5 stopinj ni zanesljiv
        if new.get('turn') is not None:
            if abs(new['turn'].get('angleDeg', 0)) < _MIN_TURN_ANGLE:
                new['turn'] = None
        #klanec z manj kot 1 stopinjo ni zanesljiv
        if new.get('hill') is not None:
            if abs(new['hill'].get('angleDeg', 0)) < _MIN_HILL_ANGLE:
                new['hill'] = None
        cleaned.append(new)
    return cleaned


def _label_window_turn(t_start, t_end, labels):
    #poisci oznako z najvecjim casovnim prekrivanjem z oknom
    #okno oznacimo kot zavoj ce se >50% casa prekriva z oznacenim zavojem
    window_dur   = t_end - t_start
    best_label   = 'none'
    best_overlap = 0.0
    for lbl in labels:
        if lbl.get('turn') is None:
            continue
        ov = min(t_end, lbl['t_end']) - max(t_start, lbl['t_start'])
        if ov > best_overlap:
            best_overlap = ov
            best_label   = lbl['turn']['dir']
    return best_label if (window_dur > 0 and best_overlap / window_dur >= _THRESHOLD_TURN) else 'none'


def _label_window_hill(t_start, t_end, labels):
    #seStej prekrivanje za vsako smer klanca posebej
    #pogoj 1: vsaj 30% okna mora biti klanec
    #pogoj 2: dominantna smer mora biti vsaj 2x boljsa od nasprotne
    window_dur = t_end - t_start
    overlap_by_dir = {'up': 0.0, 'down': 0.0}
    for lbl in labels:
        if lbl.get('hill') is None:
            continue
        ov = min(t_end, lbl['t_end']) - max(t_start, lbl['t_start'])
        if ov <= 0:
            continue
        d = lbl['hill']['dir']
        if d in overlap_by_dir:
            overlap_by_dir[d] += ov

    best_dir, best_ov = max(overlap_by_dir.items(), key=lambda x: x[1])
    other_ov = sum(v for k, v in overlap_by_dir.items() if k != best_dir)

    #preveri minimalno pokritost
    if window_dur <= 0 or best_ov / window_dur < _THRESHOLD_HILL_MIN:
        return 'none'
    #preveri da ena smer mocno prevladuje nad drugo
    if other_ov > 0 and best_ov < _THRESHOLD_HILL_DOM * other_ov:
        return 'none'
    return best_dir


def _search_hill_multipliers(probs_val, Y_val):
    #iskanje po mrezi za optimalne multiplikatorje verjetnosti gor/dol
    #none ostane 1.0, za gor in dol preizkusi kombinacije iz mreze
    from sklearn.metrics import f1_score as _f1
    grid = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    best_f1, best_m = -1.0, (1.0, 1.0, 1.0)
    for m1 in grid:
        for m2 in grid:
            mult  = np.array([1.0, m1, m2])
            y_hat = np.argmax(probs_val * mult, axis=1)
            f1    = _f1(Y_val, y_hat, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_m = f1, (1.0, m1, m2)
    return np.array(best_m, dtype=np.float32)
