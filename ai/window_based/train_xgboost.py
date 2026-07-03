"""
Minimalni XGBoost pipeline s 5-kratno navzkrižno validacijo (GroupKFold CV).
Vhod: Prilagodljivo število čustih kanalov.
"""

from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold

DATASET_DIR = Path(__file__).parent / 'dataset_output'

def extract_basic_features(window):
    """Izračun preprostih statistik za vse kanale dinamično."""
    feats = []
    # POPRAVEK: Namesto fiksnega range(4) sedaj dinamično preberemo število kanalov,
    # kar prepreči hude hrošče, ko se dimenzije matrike X spremenijo.
    n_channels = window.shape[1]
    for ch in range(n_channels):
        signal = window[:, ch]
        feats.extend([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.trapezoid(signal)  # Integriranje signala za zaznavanje premika/spremembe
        ])
    return np.array(feats, dtype=np.float32)

def train_xgb_task(X, Y, log_ids, task_name):
    """Train XGBoost on hand-crafted window features for one task with 5-fold GroupKFold CV."""
    print("\n==================================================")
    print(f"Treniram XGBoost (5-Fold CV + SMOOTH POPRAVKI): {task_name.upper()}")
    print("==================================================")
    
    X_feat = np.array([extract_basic_features(x) for x in X])
    
    gkf = GroupKFold(n_splits=5)
    oof_predictions = np.zeros(len(Y), dtype=np.int64)
    target_names = ['none', 'left', 'right'] if task_name == 'turn' else ['none', 'up', 'down']

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_feat, Y, groups=log_ids)):
        print(f"-> Fold {fold + 1}/5...")
        
        class_counts = np.bincount(Y[train_idx])
        smooth_weights = 1.0 / np.sqrt(class_counts)
        smooth_weights = smooth_weights / smooth_weights.sum() * 3.0
        
        sample_weights = np.array([smooth_weights[c] for c in Y[train_idx]])
        
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            eval_metric='mlogloss'
        )
        
        model.fit(X_feat[train_idx], Y[train_idx], sample_weight=sample_weights)
        oof_predictions[val_idx] = model.predict(X_feat[val_idx])

    print(f"\n=== NOVO POPRAVLJENO POROČILO ZA XGBOOST: {task_name.upper()} ===")
    print(classification_report(Y, oof_predictions, target_names=target_names, zero_division=0))

def main():
    """Train and evaluate the XGBoost classifier on both the turn and hill datasets."""
    X_turn = np.load(DATASET_DIR / 'X_turn.npy')
    Y_turn = np.load(DATASET_DIR / 'Y_turn.npy')
    ids_turn = np.load(DATASET_DIR / 'log_ids_turn.npy')
    train_xgb_task(X_turn, Y_turn, ids_turn, 'turn')

    X_hill = np.load(DATASET_DIR / 'X_hill.npy')
    Y_hill = np.load(DATASET_DIR / 'Y_hill.npy')
    ids_hill = np.load(DATASET_DIR / 'log_ids_hill.npy')
    train_xgb_task(X_hill, Y_hill, ids_hill, 'hill')

if __name__ == '__main__':
    main()