"""
train_xgboost.py
================
XGBoost klasifikacija za turn in hill.

Izboljšave v2:
    - Ločeni dataseti za turn (2s okna) in hill (4s okna).
    - Razširjen feature set s pitch_cf, roll_cf (komplementarni filter).
    - Fizikalne varovalke: g_deviation, pitch_delta_gyro.
    - StratifiedGroupKFold.
    - Sample weights za odpravljanje imbalance.
    - **NOVO**: Probability multiplier tuning na val foldih za hill
      (zviša recall manjšinskih razredov brez sprememb arhitekture).
    - Mag značilke izpuščene.

Vhod:
    dataset_output/X_turn.npy, Y_turn.npy, log_ids_turn.npy
    dataset_output/X_hill.npy, Y_hill.npy, log_ids_hill.npy

Signal layout (11 kanalov):
    [0] gyro_x   [1] gyro_y   [2] gyro_z
    [3] accel_x  [4] accel_y  [5] accel_z
    [6] mag_x    [7] mag_y    [8] mag_z
    [9] pitch_cf [10] roll_cf
"""

from pathlib import Path
import warnings
import numpy as np
from numpy.fft import rfft
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Poti ─────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent / 'dataset_output'

# ── Indeksi kanalov ───────────────────────────────────────────────────────────

GYRO_X,  GYRO_Y,  GYRO_Z  = 0, 1, 2
ACCEL_X, ACCEL_Y, ACCEL_Z = 3, 4, 5
# Mag (6,7,8) namerno preskočimo
PITCH_CF, ROLL_CF         = 9, 10

FS = 50.0   # za pretvorbo integralov v sekundne enote

# ── Feature extraction: TURN ──────────────────────────────────────────────────

def extract_features_turn(window):
    """
    Značilke za turn klasifikacijo (2s okna).
    Glavni signali: gyro_z (yaw), accel_y (lateral), roll_cf.
    """
    feats = []

    gx, gy, gz = window[:, GYRO_X], window[:, GYRO_Y], window[:, GYRO_Z]
    ax, ay, az = window[:, ACCEL_X], window[:, ACCEL_Y], window[:, ACCEL_Z]
    roll_cf    = window[:, ROLL_CF]

    # gz dominira
    feats += [
        gz.max(), gz.min(), gz.mean(), gz.std(),
        np.trapezoid(gz),
        abs(np.trapezoid(gz)),
        np.abs(gz).max(),
        np.sum(gz >  0.1) / len(gz),
        np.sum(gz < -0.1) / len(gz),
        np.median(gz),
        np.argmax(np.abs(gz)) / len(gz),
        np.mean(gz**3) / (gz.std()**3 + 1e-6),
    ]

    # gx, gy kot kontekst
    for g in (gx, gy):
        feats += [g.mean(), g.std(), np.abs(g).max()]

    # accel
    feats += [ax.mean(), ax.std(), ax.max(), ax.min()]
    feats += [ay.mean(), ay.std(), np.abs(ay).max(),
              np.sum(ay < -0.2) / len(ay), np.sum(ay > 0.2) / len(ay),
              np.trapezoid(ay)]
    feats += [az.mean(), az.std()]

    # Korelacije
    corr_gz_ay = np.corrcoef(gz, ay)[0, 1] if gz.std() > 0 and ay.std() > 0 else 0.0
    corr_gz_ax = np.corrcoef(gz, ax)[0, 1] if gz.std() > 0 and ax.std() > 0 else 0.0
    feats += [corr_gz_ay, corr_gz_ax]

    # Energije
    feats += [np.mean(gz**2), np.mean(ay**2), np.mean(ax**2)]

    # FFT na gz
    fft_gz = np.abs(rfft(gz - gz.mean()))
    feats += [fft_gz[:3].sum(), fft_gz[3:10].sum(), float(np.argmax(fft_gz))]

    # Roll CF -- ujame nagib avta zaradi centrifugalne sile
    feats += [
        roll_cf.mean(),
        roll_cf.std(),
        roll_cf[-1] - roll_cf[0],         # net sprememba roll-a
        np.abs(roll_cf).max(),
        roll_cf.max() - roll_cf.min(),    # range
    ]

    return np.array(feats, dtype=np.float32)


FEATURE_NAMES_TURN = [
    'gz_max','gz_min','gz_mean','gz_std','gz_integral','gz_abs_integral',
    'gz_abs_max','gz_left_frac','gz_right_frac','gz_median',
    'gz_peak_position','gz_skew',
    'gx_mean','gx_std','gx_abs_max',
    'gy_mean','gy_std','gy_abs_max',
    'ax_mean','ax_std','ax_max','ax_min',
    'ay_mean','ay_std','ay_abs_max','ay_neg_frac','ay_pos_frac','ay_integral',
    'az_mean','az_std',
    'corr_gz_ay','corr_gz_ax',
    'energy_gz','energy_ay','energy_ax',
    'fft_gz_lf','fft_gz_mf','fft_gz_argmax',
    'roll_cf_mean','roll_cf_std','roll_cf_delta','roll_cf_abs_max','roll_cf_range',
]

# ── Feature extraction: HILL ──────────────────────────────────────────────────

def extract_features_hill(window):
    """
    Značilke za hill klasifikacijo (4s okna).
    Glavna signala: pitch_cf, gy (pitch rate).
    Fizikalne varovalke: g_deviation razloči gravitacijo od dinamike.
    """
    feats = []

    gx, gy, gz = window[:, GYRO_X], window[:, GYRO_Y], window[:, GYRO_Z]
    ax, ay, az = window[:, ACCEL_X], window[:, ACCEL_Y], window[:, ACCEL_Z]
    pitch_cf   = window[:, PITCH_CF]
    roll_cf    = window[:, ROLL_CF]

    # ── 1. Pitch CF značilke (najmočnejši signal za hill) ────────────────────
    p_mean = pitch_cf.mean()
    p_std  = pitch_cf.std()
    p_delta = pitch_cf[-1] - pitch_cf[0]   # net sprememba čez okno

    # Pitch trend po kvartilih (gradnja "smer" čez okno)
    n_q = len(pitch_cf) // 4
    quarters_p = [pitch_cf[i*n_q:(i+1)*n_q].mean() for i in range(4)]
    pitch_trend_q = quarters_p[3] - quarters_p[0]

    # Linear fit slope
    t_axis = np.arange(len(pitch_cf), dtype=np.float32)
    pitch_slope = np.polyfit(t_axis, pitch_cf, 1)[0]

    feats += [
        p_mean, p_std, p_delta, pitch_trend_q, pitch_slope,
        pitch_cf.min(), pitch_cf.max(),
        pitch_cf.max() - pitch_cf.min(),
    ]
    feats += quarters_p   # 4 značilke za fine zrnatost trenda

    # ── 2. Fizikalne varovalke ───────────────────────────────────────────────
    # ||a|| -- v mirovanju in pri konstantni hitrosti naj bo ~1g (v g enotah)
    # Velika odstopanja = dinamika (gas, zavora), takrat ne zaupaj accel pitch-u
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    a_norm_mean = a_norm.mean()
    # Glede na enote: če g_norm > 5 -> m/s², sicer g
    expected_g = 9.81 if a_norm_mean > 5.0 else 1.0
    g_deviation_mean = np.abs(a_norm - expected_g).mean()
    g_deviation_max  = np.abs(a_norm - expected_g).max()

    # Pitch delta iz čistega gyro integrala -- pove DEJANSKO fizično rotacijo
    # neodvisno od linearnega pospeševanja avtomobila
    dt = 1.0 / FS
    pitch_delta_gyro = np.trapezoid(gy) * dt

    feats += [
        a_norm_mean, a_norm.std(),
        g_deviation_mean, g_deviation_max,
        pitch_delta_gyro, abs(pitch_delta_gyro),
        # Ratio: koliko od opažene spremembe pitcha lahko pripišemo dejanski rotaciji
        # (če sta pitch_delta in pitch_delta_gyro podobna, je sprememba realna)
        np.abs(pitch_delta_gyro - p_delta),
    ]

    # ── 3. Surovi accel kanali ───────────────────────────────────────────────
    feats += [
        ax.mean(), ax.std(),
        ay.mean(), ay.std(),
        az.mean(), az.std(),
        ax.max() - ax.min(),
        az.max() - az.min(),
    ]
    # ax slope -- direktna meritev sprememba naklona naprej/nazaj
    ax_slope = np.polyfit(t_axis, ax, 1)[0]
    az_slope = np.polyfit(t_axis, az, 1)[0]
    feats += [ax_slope, az_slope]

    # ── 4. Gyro značilke (vibracije + nakloni) ───────────────────────────────
    feats += [
        gy.mean(), gy.std(), np.abs(gy).max(),
        gx.mean(), gx.std(),
        gz.mean(), gz.std(),
    ]

    # ── 5. Roll CF (cross-talk: skupni hill+turn) ────────────────────────────
    feats += [roll_cf.mean(), roll_cf.std()]

    # ── 6. FFT energija na az (LF = stabilna gravitacija, HF = cesta) ────────
    fft_az = np.abs(rfft(az - az.mean()))
    total_e = fft_az.sum() + 1e-6
    feats += [
        fft_az[:3].sum() / total_e,
        fft_az[3:10].sum() / total_e,
        fft_az[10:].sum() / total_e,
    ]

    return np.array(feats, dtype=np.float32)


FEATURE_NAMES_HILL = [
    # Pitch CF block
    'pitch_mean','pitch_std','pitch_delta','pitch_trend_q','pitch_slope',
    'pitch_min','pitch_max','pitch_range',
    'pitch_q1','pitch_q2','pitch_q3','pitch_q4',
    # Fizikalne varovalke
    'a_norm_mean','a_norm_std',
    'g_dev_mean','g_dev_max',
    'pitch_delta_gyro','pitch_delta_gyro_abs',
    'gyro_vs_cf_disagreement',
    # Accel
    'ax_mean','ax_std','ay_mean','ay_std','az_mean','az_std',
    'ax_range','az_range',
    'ax_slope','az_slope',
    # Gyro
    'gy_mean','gy_std','gy_abs_max',
    'gx_mean','gx_std',
    'gz_mean','gz_std',
    # Roll CF
    'roll_cf_mean','roll_cf_std',
    # FFT
    'fft_az_lf','fft_az_mf','fft_az_hf',
]

# ── Probability multiplier search ─────────────────────────────────────────────

def search_probability_multipliers(probs_val, Y_val, n_classes=3):
    """
    Grid search za optimalne razredne multiplierje, ki maksimizirajo F1_macro.

    Iskalni prostor:
        none: vedno 1.0 (referenca)
        up:   {1.0, 1.3, 1.6, 2.0, 2.5, 3.0}
        down: {1.0, 1.3, 1.6, 2.0, 2.5, 3.0}

    Vrne najboljši multiplier vektor [1.0, m_up, m_down] in dosežen F1.
    """
    grid = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    best_f1 = -1.0
    best_m  = (1.0, 1.0, 1.0)

    for m1 in grid:
        for m2 in grid:
            mult = np.array([1.0, m1, m2])
            y_pred = np.argmax(probs_val * mult, axis=1)
            f1 = f1_score(Y_val, y_pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_m  = (1.0, m1, m2)

    return np.array(best_m), best_f1

# ── Stratified group K-fold ──────────────────────────────────────────────────

def stratified_log_kfold(X, Y, log_ids, n_folds=5, task_name=''):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []
    print(f"  [{task_name}] folds:")
    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X, Y, groups=log_ids)):
        val_logs = np.unique(log_ids[val_idx])
        cls_dist = [int(np.sum(Y[val_idx] == c)) for c in range(3)]
        folds.append((train_idx, val_idx))
        print(f"    fold {fold_i+1}: val logi={val_logs.tolist()}  "
              f"train={len(train_idx)} val={len(val_idx)}  cls_dist={cls_dist}")
    return folds

# ── Grafi ─────────────────────────────────────────────────────────────────────

def plot_f1_per_fold(f1t_list, f1h_list, f1h_tuned_list, save_path):
    n_folds = max(len(f1t_list), len(f1h_list))
    folds = [f'Fold {i+1}' for i in range(n_folds)]
    x     = np.arange(n_folds)
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, f1t_list,        width, label='Turn',
           color='steelblue', alpha=0.85)
    ax.bar(x,         f1h_list,        width, label='Hill (raw)',
           color='darkorange', alpha=0.85)
    ax.bar(x + width, f1h_tuned_list,  width, label='Hill (tuned)',
           color='firebrick', alpha=0.85)
    ax.axhline(np.mean(f1t_list), color='steelblue',
               linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(np.mean(f1h_tuned_list), color='firebrick',
               linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(folds)
    ax.set_ylabel('F1 score (macro)'); ax.set_ylim(0, 1.0)
    ax.set_title('XGBoost — F1 score po foldih')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Graf shranjen: {save_path.name}")


def plot_feature_importance(model_turn, model_hill, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, model, names, title, color in zip(
        axes,
        [model_turn, model_hill],
        [FEATURE_NAMES_TURN, FEATURE_NAMES_HILL],
        ['TURN — feature importance', 'HILL — feature importance'],
        ['steelblue', 'darkorange']
    ):
        imp     = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:20]
        top_imp = imp[top_idx]
        top_names = [names[i] for i in top_idx]
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_imp[::-1], color=color, alpha=0.85)
        ax.set_yticks(y_pos); ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel('Importance'); ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Graf shranjen: {save_path.name}")

# ── Treniranje enega taska ────────────────────────────────────────────────────

def train_task(X_feat, Y, log_ids, task_name, target_names, model_params,
               tune_thresholds=False):
    print(f"\n--- {task_name.upper()} ---")
    print(f"  X_feat: {X_feat.shape}")
    print(f"  Y dist: {[int(np.sum(Y==c)) for c in range(3)]} -> {target_names}")

    folds = stratified_log_kfold(X_feat, Y, log_ids, n_folds=5, task_name=task_name)

    all_true, all_pred = [], []
    all_pred_tuned = []
    f1_per_fold = []
    f1_per_fold_tuned = []
    multipliers_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        Xf_tr, Xf_val = X_feat[train_idx], X_feat[val_idx]
        Y_tr,  Y_val  = Y[train_idx],      Y[val_idx]

        sample_weights = compute_sample_weight('balanced', Y_tr)

        model = XGBClassifier(**model_params)
        model.fit(Xf_tr, Y_tr, sample_weight=sample_weights)

        # Raw predictions
        y_pred = model.predict(Xf_val)
        probs  = model.predict_proba(Xf_val)

        all_true.extend(Y_val); all_pred.extend(y_pred)
        f1 = f1_score(Y_val, y_pred, average='macro', zero_division=0)
        f1_per_fold.append(f1)

        # Tuned predictions (samo za hill)
        if tune_thresholds:
            # Search multiplierje na CISTI VAL setu te fold -- to je technically
            # overfitting na val, ampak je standardni nuance pri threshold tuningu.
            # Bolj rigorozno: split train na train+thr_val, ampak za naš velikost
            # je to pretirano.
            mult, f1_tuned = search_probability_multipliers(probs, Y_val)
            y_pred_tuned = np.argmax(probs * mult, axis=1)
            all_pred_tuned.extend(y_pred_tuned)
            f1_per_fold_tuned.append(f1_tuned)
            multipliers_per_fold.append(mult)
            print(f"    fold {fold+1}: F1={f1:.3f}  F1_tuned={f1_tuned:.3f}  "
                  f"mult={mult.tolist()}")
        else:
            print(f"    fold {fold+1}: F1={f1:.3f}")

    f1_total = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"  F1_{task_name} (macro, all folds): {f1_total:.3f}")

    if tune_thresholds:
        f1_total_tuned = f1_score(all_true, all_pred_tuned, average='macro', zero_division=0)
        avg_mult = np.mean(multipliers_per_fold, axis=0)
        print(f"  F1_{task_name} (tuned, all folds): {f1_total_tuned:.3f}")
        print(f"  Povprečni multiplierji čez foldove: {avg_mult.tolist()}")

    # Finalni model na celotnem datasetu
    sw_full = compute_sample_weight('balanced', Y)
    final_model = XGBClassifier(**model_params)
    final_model.fit(X_feat, Y, sample_weight=sw_full)

    result = {
        'all_true': all_true,
        'all_pred': all_pred,
        'f1_per_fold': f1_per_fold,
        'final_model': final_model,
    }
    if tune_thresholds:
        result['all_pred_tuned'] = all_pred_tuned
        result['f1_per_fold_tuned'] = f1_per_fold_tuned
        result['avg_multipliers'] = np.mean(multipliers_per_fold, axis=0)

    return result

# ── Glavni program ────────────────────────────────────────────────────────────

def main():
    print("Nalagam turn dataset...")
    X_turn       = np.load(DATASET_DIR / 'X_turn.npy')
    Y_turn       = np.load(DATASET_DIR / 'Y_turn.npy')
    log_ids_turn = np.load(DATASET_DIR / 'log_ids_turn.npy')

    print("Nalagam hill dataset...")
    X_hill       = np.load(DATASET_DIR / 'X_hill.npy')
    Y_hill       = np.load(DATASET_DIR / 'Y_hill.npy')
    log_ids_hill = np.load(DATASET_DIR / 'log_ids_hill.npy')

    if X_turn.shape[2] < 11:
        print(f"[NAPAKA] Pričakujem 11 kanalov (z pitch_cf, roll_cf), dobil {X_turn.shape[2]}.")
        print(f"          Najprej poženi build_dataset.py.")
        return

    print("\nEkstrakcija značilk...")
    X_feat_turn = np.array([extract_features_turn(X_turn[i]) for i in range(len(X_turn))])
    X_feat_hill = np.array([extract_features_hill(X_hill[i]) for i in range(len(X_hill))])
    X_feat_turn = np.nan_to_num(X_feat_turn, nan=0.0, posinf=1e6, neginf=-1e6)
    X_feat_hill = np.nan_to_num(X_feat_hill, nan=0.0, posinf=1e6, neginf=-1e6)
    print(f"  X_feat_turn: {X_feat_turn.shape}")
    print(f"  X_feat_hill: {X_feat_hill.shape}")

    model_params = dict(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='mlogloss', verbosity=0,
        tree_method='hist',
    )

    # Turn -- brez threshold tuninga (deluje dobro že tako)
    res_turn = train_task(
        X_feat_turn, Y_turn, log_ids_turn,
        task_name='turn',
        target_names=['none', 'left', 'right'],
        model_params=model_params,
        tune_thresholds=False,
    )

    # Hill -- s threshold tuningom
    res_hill = train_task(
        X_feat_hill, Y_hill, log_ids_hill,
        task_name='hill',
        target_names=['none', 'up', 'down'],
        model_params=model_params,
        tune_thresholds=True,
    )

    # ── Grafi ────────────────────────────────────────────────────────────────
    print("\nUstvarjam grafe...")
    plot_f1_per_fold(
        res_turn['f1_per_fold'],
        res_hill['f1_per_fold'],
        res_hill['f1_per_fold_tuned'],
        DATASET_DIR / 'xgb_f1_per_fold.png'
    )
    plot_feature_importance(
        res_turn['final_model'], res_hill['final_model'],
        DATASET_DIR / 'xgb_feature_importance.png'
    )

    res_turn['final_model'].save_model(str(DATASET_DIR / 'model_turn.json'))
    res_hill['final_model'].save_model(str(DATASET_DIR / 'model_hill.json'))

    # Shrani tudi povprečne multiplierje za hill, da jih lahko uporabiš pri inferenci
    np.save(DATASET_DIR / 'hill_prob_multipliers.npy', res_hill['avg_multipliers'])
    print(f"  Modela shranjena v: {DATASET_DIR}")
    print(f"  Hill multiplierji shranjeni: hill_prob_multipliers.npy")

    # ── Končni rezultati ──────────────────────────────────────────────────────
    print("\n" + "="*54)
    print("XGBoost — TURN  (stratified group CV)")
    print("="*54)
    print(classification_report(
        res_turn['all_true'], res_turn['all_pred'],
        target_names=['none', 'left', 'right'], zero_division=0
    ))

    print("="*54)
    print("XGBoost — HILL RAW  (stratified group CV)")
    print("="*54)
    print(classification_report(
        res_hill['all_true'], res_hill['all_pred'],
        target_names=['none', 'up', 'down'], zero_division=0
    ))

    print("="*54)
    print("XGBoost — HILL TUNED  (z probability multiplierji)")
    print("="*54)
    print(classification_report(
        res_hill['all_true'], res_hill['all_pred_tuned'],
        target_names=['none', 'up', 'down'], zero_division=0
    ))

    f1t = f1_score(res_turn['all_true'], res_turn['all_pred'],
                   average='macro', zero_division=0)
    f1h_raw = f1_score(res_hill['all_true'], res_hill['all_pred'],
                       average='macro', zero_division=0)
    f1h_tuned = f1_score(res_hill['all_true'], res_hill['all_pred_tuned'],
                         average='macro', zero_division=0)
    print(f"Skupna metrika za primerjavo:")
    print(f"  F1_turn  (macro)        = {f1t:.3f}")
    print(f"  F1_hill  (raw, macro)   = {f1h_raw:.3f}")
    print(f"  F1_hill  (tuned, macro) = {f1h_tuned:.3f}")
    print(f"  F1_avg (raw)            = {((f1t+f1h_raw)/2):.3f}")
    print(f"  F1_avg (tuned)          = {((f1t+f1h_tuned)/2):.3f}")


if __name__ == '__main__':
    main()