"""
train_xgboost.py
================
XGBoost klasifikacija za turn in hill.
Split po logih -- model nikoli ne vidi okna iz istega loga v train in val.

Vhod:
    dataset_output/X_all.npy
    dataset_output/Y_turn_all.npy
    dataset_output/Y_hill_all.npy
    dataset_output/log_ids.npy

Izhod:
    dataset_output/model_turn.json
    dataset_output/model_hill.json
    dataset_output/xgb_f1_per_fold.png
    dataset_output/xgb_feature_importance.png
"""

from pathlib import Path
import warnings
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Poti ─────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent / 'dataset_output'

# ── Indeksi kanalov ───────────────────────────────────────────────────────────

GYRO_X, GYRO_Y, GYRO_Z    = 0, 1, 2
ACCEL_X, ACCEL_Y, ACCEL_Z = 3, 4, 5
MAG_X, MAG_Y, MAG_Z       = 6, 7, 8

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(window):
    feats = []
    gz = window[:, GYRO_Z]
    feats += [gz.max(), gz.min(), gz.mean(), gz.std(),
              np.sum(gz) / len(gz), np.abs(gz).max(),
              np.sum(gz >  0.1) / len(gz),
              np.sum(gz < -0.1) / len(gz)]
    for ch in [GYRO_X, GYRO_Y]:
        g = window[:, ch]
        feats += [g.mean(), g.std(), np.abs(g).max()]
    ax = window[:, ACCEL_X]
    feats += [ax.mean(), ax.std(), ax.max(), ax.min(), np.sum(ax) / len(ax)]
    ay = window[:, ACCEL_Y]
    feats += [ay.mean(), ay.std(), np.abs(ay).max(),
              np.sum(ay < -0.2) / len(ay), np.sum(ay > 0.2) / len(ay)]
    az = window[:, ACCEL_Z]
    feats += [az.mean(), az.std()]
    for ch in [MAG_X, MAG_Y, MAG_Z]:
        m = window[:, ch]
        feats += [m.mean(), m.max() - m.min()]
    corr = np.corrcoef(gz, ay)[0, 1] if gz.std() > 0 and ay.std() > 0 else 0.0
    feats += [corr, gz.std() / (ax.std() + 1e-6),
              np.mean(gz ** 2), np.mean(ax ** 2)]
    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = [
    'gz_max','gz_min','gz_mean','gz_std','gz_integral','gz_abs_max',
    'gz_left_frac','gz_right_frac',
    'gx_mean','gx_std','gx_abs_max',
    'gy_mean','gy_std','gy_abs_max',
    'ax_mean','ax_std','ax_max','ax_min','ax_integral',
    'ay_mean','ay_std','ay_abs_max','ay_left_frac','ay_right_frac',
    'az_mean','az_std',
    'mx_mean','mx_range','my_mean','my_range','mz_mean','mz_range',
    'corr_gz_ay','ratio_gz_ax','energy_gz','energy_ax',
]

# ── Log-based cross-validation ────────────────────────────────────────────────

def log_kfold(log_ids, n_folds=5):
    unique_logs = np.unique(log_ids)
    log_groups  = np.array_split(unique_logs, n_folds)
    folds = []
    for val_logs in log_groups:
        val_mask  = np.isin(log_ids, val_logs)
        folds.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
        print(f"  Val logi: {val_logs.tolist()}  "
              f"(train={np.sum(~val_mask)} oken, val={np.sum(val_mask)} oken)")
    return folds

# ── Grafi ─────────────────────────────────────────────────────────────────────

def plot_f1_per_fold(f1t_list, f1h_list, save_path):
    """F1 score za turn in hill po vsakem foldu."""
    folds = [f'Fold {i+1}' for i in range(len(f1t_list))]
    x     = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, f1t_list, width, label='Turn',
           color='steelblue', alpha=0.85)
    ax.bar(x + width/2, f1h_list, width, label='Hill',
           color='darkorange', alpha=0.85)

    # povprečna črta
    ax.axhline(np.mean(f1t_list), color='steelblue',
               linestyle='--', linewidth=1, alpha=0.6, label='Turn avg')
    ax.axhline(np.mean(f1h_list), color='darkorange',
               linestyle='--', linewidth=1, alpha=0.6, label='Hill avg')

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel('F1 score (macro)')
    ax.set_ylim(0, 1.0)
    ax.set_title('XGBoost — F1 score po foldih (log-based CV)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Graf shranjen: {save_path.name}")


def plot_feature_importance(model_turn, model_hill, save_path):
    """Feature importance za turn in hill -- top 15 značilk vsak."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model, title, color in zip(
        axes,
        [model_turn, model_hill],
        ['TURN — feature importance', 'HILL — feature importance'],
        ['steelblue', 'darkorange']
    ):
        imp     = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:15]
        top_imp = imp[top_idx]
        top_names = [FEATURE_NAMES[i] for i in top_idx]

        # horizontalni bar chart, najpomembnejša zgoraj
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_imp[::-1], color=color, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Graf shranjen: {save_path.name}")

# ── Glavni program ────────────────────────────────────────────────────────────

def main():
    print("Nalagam dataset...")
    X       = np.load(DATASET_DIR / 'X_all.npy')
    Y_turn  = np.load(DATASET_DIR / 'Y_turn_all.npy')
    Y_hill  = np.load(DATASET_DIR / 'Y_hill_all.npy')
    log_ids = np.load(DATASET_DIR / 'log_ids.npy')
    print(f"  X: {X.shape},  logi: {np.unique(log_ids).tolist()}")
    print(f"  Y_turn: none={np.sum(Y_turn==0)}  left={np.sum(Y_turn==1)}  right={np.sum(Y_turn==2)}")
    print(f"  Y_hill: none={np.sum(Y_hill==0)}  up={np.sum(Y_hill==1)}    down={np.sum(Y_hill==2)}")

    print("\nEkstrakcija značilk...")
    X_feat = np.array([extract_features(X[i]) for i in range(len(X))])
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1.0, neginf=-1.0)
    print(f"  X_feat: {X_feat.shape}")

    model_params = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='mlogloss', verbosity=0)

    print("\nLog-based cross-validation split:")
    folds = log_kfold(log_ids, n_folds=5)

    all_yt_true, all_yt_pred = [], []
    all_yh_true, all_yh_pred = [], []
    f1t_per_fold, f1h_per_fold = [], []

    for fold, (train_idx, val_idx) in enumerate(folds):
        Xf_tr, Xf_val = X_feat[train_idx], X_feat[val_idx]
        Yt_tr, Yt_val = Y_turn[train_idx],  Y_turn[val_idx]
        Yh_tr, Yh_val = Y_hill[train_idx],  Y_hill[val_idx]

        m_turn = XGBClassifier(**model_params)
        m_hill = XGBClassifier(**model_params)
        m_turn.fit(Xf_tr, Yt_tr)
        m_hill.fit(Xf_tr, Yh_tr)

        yt_pred = m_turn.predict(Xf_val)
        yh_pred = m_hill.predict(Xf_val)

        all_yt_true.extend(Yt_val);  all_yt_pred.extend(yt_pred)
        all_yh_true.extend(Yh_val);  all_yh_pred.extend(yh_pred)

        f1t = f1_score(Yt_val, yt_pred, average='macro', zero_division=0)
        f1h = f1_score(Yh_val, yh_pred, average='macro', zero_division=0)
        f1t_per_fold.append(f1t)
        f1h_per_fold.append(f1h)
        print(f"  Fold {fold+1}: F1_turn={f1t:.3f}  F1_hill={f1h:.3f}  "
              f"F1_avg={((f1t+f1h)/2):.3f}")

    # ── Grafi ────────────────────────────────────────────────────────────────
    print("\nUstvarjam grafe...")
    plot_f1_per_fold(f1t_per_fold, f1h_per_fold,
                     DATASET_DIR / 'xgb_f1_per_fold.png')

    # Finalni modeli na celotnem datasetu
    print("\nTreniram finale modele na celotnem datasetu...")
    m_turn_full = XGBClassifier(**model_params)
    m_hill_full = XGBClassifier(**model_params)
    m_turn_full.fit(X_feat, Y_turn)
    m_hill_full.fit(X_feat, Y_hill)

    plot_feature_importance(m_turn_full, m_hill_full,
                            DATASET_DIR / 'xgb_feature_importance.png')

    m_turn_full.save_model(str(DATASET_DIR / 'model_turn.json'))
    m_hill_full.save_model(str(DATASET_DIR / 'model_hill.json'))
    print(f"  Modela shranjena v: {DATASET_DIR}")

    # ── Končni rezultati ──────────────────────────────────────────────────────
    print("\n" + "="*54)
    print("XGBoost — TURN  (log-based cross-validation)")
    print("="*54)
    print(classification_report(
        all_yt_true, all_yt_pred,
        target_names=['none', 'left', 'right'], zero_division=0
    ))

    print("="*54)
    print("XGBoost — HILL  (log-based cross-validation)")
    print("="*54)
    print(classification_report(
        all_yh_true, all_yh_pred,
        target_names=['none', 'up', 'down'], zero_division=0
    ))

    f1t = f1_score(all_yt_true, all_yt_pred, average='macro', zero_division=0)
    f1h = f1_score(all_yh_true, all_yh_pred, average='macro', zero_division=0)
    print(f"Skupna metrika za primerjavo z BiLSTM/GRU:")
    print(f"  F1_turn (macro) = {f1t:.3f}")
    print(f"  F1_hill (macro) = {f1h:.3f}")
    print(f"  F1_avg          = {((f1t+f1h)/2):.3f}  ← primerjaj s kolegi")


if __name__ == '__main__':
    main()