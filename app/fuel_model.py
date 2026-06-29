# -*- coding: utf-8 -*-
"""
fuel_model.py
=============
Ridge regression for predicting fuel consumption (L/100km) from
per-drive aggregate IMU features. Fits on all stored drives and
returns per-feature coefficients expressed in real-world units
(L/100km per +1 unit of each feature) for direct user feedback.

Typical call sequence after a new drive is loaded:
    records = load_store()
    result  = fit_and_explain(records, new_drive_features)
    # result contains global importance ranking + this-drive breakdown
"""

import numpy as np
import json as _json
import datetime as _datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from AutoDNA.app.get_path import get_data_dir

# ── model settings ──────────────────────────────────────────────────────────

# Ridge regularisation strength — higher = more shrinkage toward zero,
# useful when n_drives is small and features may be correlated.
# Reasonable starting range: 0.1 (weak) to 10.0 (strong).
RIDGE_ALPHA = 1.0

# Minimum drives needed before fitting makes sense.
# Below this the coefficients are too noisy to be informative.
MIN_DRIVES_FOR_FIT = 1

# Features excluded from the model even if present in the store.
# inv_distance_km is kept as a modelling input but we don't show it
# to the user since "1 / (km+1)" is not self-explanatory.
_HIDDEN_FROM_DISPLAY = {'inv_distance_km', '_duration_min'}

# Human-readable labels and units for each feature key.
# Any feature not listed here will still be shown, just without a label.
FEATURE_LABELS = {
    'harsh_accel_rate':   ('Harsh acceleration',  'events/km'),
    'harsh_braking_rate': ('Harsh braking',        'events/km'),
    'accel_x_std':        ('Acceleration roughness','mg'),
    'rms_jerk':           ('Jerk (RMS)',            'mg/s'),
    'distance_km':        ('Trip distance',         'km'),
    'duration_min':       ('Trip duration',         'min'),
    'avg_speed_kmh':      ('Average speed',         'km/h'),
    'speed_variability': ('Speed variability', 'km/h'),
    'idle_time_pct':     ('Idle time',         '%'),
}


# ── data helpers ────────────────────────────────────────────────────────────

def records_to_xy(records: list[dict]) -> tuple:
    """
    Convert the persistent store records into (X, y, feature_names).

    :param records: List of dicts from load_store().
    :returns: Tuple (X, y, feature_names) where X is (n_drives, n_features)
              float32, y is (n_drives,) float32, feature_names is list[str].
              Returns (None, None, None) if records is empty.
    """
    if not records:
        return None, None, None

    feature_names = [
        k for k in records[0]['features'].keys()
        if k not in _HIDDEN_FROM_DISPLAY
    ]

    X = np.array(
        [[r['features'].get(name, 0.0) for name in feature_names]
         for r in records],
        dtype=np.float32,
    )
    y = np.array([r['fuel_l100km'] for r in records], dtype=np.float32)

    return X, y, feature_names


# ── model ───────────────────────────────────────────────────────────────────

class FuelModel:
    """
    Thin wrapper around Ridge + StandardScaler.

    fit()     — train on all stored drives
    predict() — predict L/100km for a single drive's feature dict
    explain() — break down a drive's predicted fuel into per-feature contributions
    global_importance() — ranked list of features by absolute coefficient
    """

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha   = alpha
        self.model   = Ridge(alpha=alpha)
        self.scaler  = StandardScaler()
        self.feature_names: list[str] = []
        self._fitted = False

    # ── training ────────────────────────────────────────────────────────────

    def fit(self, records: list[dict]) -> bool:
        """
        Fit the model on all stored drives.

        :param records: List of dicts from load_store().
        :returns: True if fit succeeded, False if not enough drives.
        """
        X, y, feature_names = records_to_xy(records)

        if X is None or len(X) < MIN_DRIVES_FOR_FIT:
            self._fitted = False
            return False

        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return True

    # ── inference ───────────────────────────────────────────────────────────

    def predict(self, drive_features: dict) -> float | None:
        """
        Predict L/100km for a single drive.

        :param drive_features: Feature dict from compute_drive_features().
        :returns: Predicted L/100km, or None if model is not fitted.
        """
        if not self._fitted:
            return None
        x = self._features_to_row(drive_features)
        x_scaled = self.scaler.transform(x)
        return float(self.model.predict(x_scaled)[0])

    def explain(self, drive_features: dict) -> list[dict] | None:
        """
        Break down this drive's predicted fuel into per-feature contributions.

        Each contribution is: coefficient (std units) × scaled feature value.
        Positive = adds to predicted fuel, negative = reduces it.
        The sum of all contributions + intercept equals the total prediction.

        :param drive_features: Feature dict from compute_drive_features().
        :returns: List of dicts with keys:
                  'key', 'label', 'unit', 'value' (raw),
                  'contribution_l100km' (how much this feature adds to prediction),
                  'std_coef' (standardised importance, for ranking).
                  Sorted by abs(contribution_l100km), largest first.
                  Returns None if model is not fitted.
        """
        if not self._fitted:
            return None

        x      = self._features_to_row(drive_features)
        x_sc   = self.scaler.transform(x)[0]           # (n_features,)
        coeffs = self.model.coef_                       # (n_features,) in std units

        # raw-unit coefficients: how many L/100km per +1 unit of the feature
        raw_coeffs = coeffs / (self.scaler.scale_ + 1e-8)

        results = []
        for i, key in enumerate(self.feature_names):
            label, unit = FEATURE_LABELS.get(key, (key, ''))
            results.append({
                'key':                key,
                'label':              label,
                'unit':               unit,
                'value':              float(drive_features.get(key, 0.0)),
                'contribution_l100km': float(coeffs[i] * x_sc[i]),
                'raw_coef':           float(raw_coeffs[i]),
                'std_coef':           float(coeffs[i]),
            })

        results.sort(key=lambda r: abs(r['contribution_l100km']), reverse=True)
        return results

    def global_importance(self) -> list[dict] | None:
        """
        Ranked list of features by absolute standardised coefficient.
        Represents which features have the strongest *general* effect on
        fuel consumption across all drives, regardless of any single drive.

        :returns: List of dicts sorted by abs(std_coef), or None if not fitted.
        """
        if not self._fitted:
            return None

        raw_coeffs = self.model.coef_ / (self.scaler.scale_ + 1e-8)

        results = []
        for i, key in enumerate(self.feature_names):
            label, unit = FEATURE_LABELS.get(key, (key, ''))
            results.append({
                'key':       key,
                'label':     label,
                'unit':      unit,
                'std_coef':  float(self.model.coef_[i]),
                'raw_coef':  float(raw_coeffs[i]),
            })

        results.sort(key=lambda r: abs(r['std_coef']), reverse=True)
        return results

    # ── helpers ─────────────────────────────────────────────────────────────

    def _features_to_row(self, drive_features: dict) -> np.ndarray:
        """Convert a feature dict to a (1, n_features) array matching the scaler."""
        return np.array(
            [[drive_features.get(k, 0.0) for k in self.feature_names]],
            dtype=np.float32,
        )

    @property
    def n_drives(self) -> int:
        """Number of drives the model was last fitted on."""
        return int(getattr(self.scaler, 'n_samples_seen_', 0))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def intercept(self) -> float | None:
        """Baseline predicted fuel (L/100km) when all features are at their mean."""
        return float(self.model.intercept_) if self._fitted else None


# ── convenience wrapper ──────────────────────────────────────────────────────

def fit_and_explain(records: list[dict], current_drive_features: dict) -> dict:
    """
    Fit the model on all stored drives and explain the current drive.

    :param records: All records from load_store() (should include the current drive).
    :param current_drive_features: Feature dict of the drive just loaded.
    :returns: Dict with keys:
              'success'           — bool, False if not enough drives
              'n_drives'          — int, number of drives used for fitting
              'predicted_l100km'  — float, model prediction for this drive
              'actual_l100km'     — float, OBD ground truth from the current drive
              'error_l100km'      — float, actual - predicted (+ = worse than predicted)
              'global_importance' — list[dict], ranked features by global effect
              'drive_explanation' — list[dict], per-feature contributions for this drive
              'intercept'         — float, baseline prediction at mean feature values
    """
    model = FuelModel()
    ok    = model.fit(records)

    if not ok:
        return {
            'success': False,
            'n_drives': len(records),
            'message': (
                f"Need at least {MIN_DRIVES_FOR_FIT} drives to fit the model "
                f"(currently have {len(records)})."
            ),
        }

    predicted  = model.predict(current_drive_features)
    actual     = records[-1]['fuel_l100km']   # most recently appended = current drive
    
    confidence = bootstrap_coefficients(records)
    
    return {
        'success':            True,
        'n_drives':           model.n_drives,
        'predicted_l100km':   predicted,
        'actual_l100km':      actual,
        'error_l100km':       actual - predicted,   # positive = used more than expected
        'global_importance':  model.global_importance(),
        'drive_explanation':  model.explain(current_drive_features),
        'intercept':          model.intercept,
        'confidence':         confidence,
    }


def compute_confidence(records):
    """
    For each feature, check sign consistency across LOOCV folds.
    Returns fraction of folds where sign matches the full-model sign.
    1.0 = always same sign = reliable
    0.5 = random = unreliable
    """
    X, y, feature_names = records_to_xy(records)
    full_model = FuelModel()
    full_model.fit(records)
    full_signs = np.sign(full_model.model.coef_)

    sign_matches = np.zeros(len(feature_names))
    n = len(records)

    for i in range(n):
        # leave one out
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]

        fold_model = Ridge(alpha=RIDGE_ALPHA)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        fold_model.fit(X_scaled, y_train)

        sign_matches += (np.sign(fold_model.coef_) == full_signs)

    return {
        name: float(sign_matches[i] / (n))   # fraction of folds with consistent sign
        for i, name in enumerate(feature_names)
    }


def bootstrap_coefficients(records: list[dict],
                           n_bootstrap: int = 500,
                           ci_level: float = 0.95) -> dict:
    """
    Bootstrap confidence intervals for Ridge coefficients.
    Treats the full pipeline (StandardScaler + Ridge) as a black box.
    
    Returns dict: feature_name -> {
        'mean': float,
        'lower': float,   # lower bound of CI
        'upper': float,   # upper bound of CI
        'ci_contains_zero': bool,
        'confidence': str  # 'high', 'low', or 'negligible'
    }
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X, y, feature_names = records_to_xy(records)
    if X is None or len(X) < 3:
        return {}

    n = len(X)
    coef_samples = []

    for _ in range(n_bootstrap):
        # resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_boot)

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_scaled, y_boot)
        coef_samples.append(model.coef_)

    coef_samples = np.array(coef_samples)  # (n_bootstrap, n_features)

    alpha = 1.0 - ci_level
    results = {}
    for i, name in enumerate(feature_names):
        samples = coef_samples[:, i]
        lower = float(np.percentile(samples, 100 * alpha / 2))
        upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        mean  = float(samples.mean())

        ci_contains_zero = lower <= 0 <= upper

        if not ci_contains_zero:
            confidence = 'high'
        elif abs(lower) < 0.05 and abs(upper) < 0.05:
            confidence = 'negligible'
        else:
            confidence = 'low'

        results[name] = {
            'mean':             mean,
            'lower':            lower,
            'upper':            upper,
            'ci_contains_zero': ci_contains_zero,
            'confidence':       confidence,
        }

    return results


def compute_and_plot_learning_curve(records: list[dict]):
    """
    Compute LOOCV MAE at each dataset size and plot the learning curve.
    Shows how model accuracy improves as more drives are added.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    import matplotlib.pyplot as plt

    X, y, feature_names = records_to_xy(records)
    if X is None or len(records) < 3:
        print("Not enough drives to compute learning curve (need at least 3).")
        return []

    results = []
    min_drives = 3

    for n in range(min_drives, len(records) + 1):
        X_n, y_n = X[:n], y[:n]

        loo  = LeaveOneOut()
        maes = []
        for train_idx, val_idx in loo.split(X_n):
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_n[train_idx])
            X_val  = scaler.transform(X_n[val_idx])
            model  = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X_tr, y_n[train_idx])
            pred   = model.predict(X_val)[0]
            maes.append(abs(pred - y_n[val_idx][0]))

        results.append({
            'n_drives': n,
            'loocv_mae': float(np.mean(maes)),
        })
        print(f"  n={n:3d}  LOOCV MAE = {np.mean(maes):.3f} L/100km")

    # ── plot ──────────────────────────────────────────────────────────────
    ns   = [r['n_drives']  for r in results]
    maes = [r['loocv_mae'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, maes, marker='o', linewidth=2,
            color='#4e79a7', markerfacecolor='#f28e2b', markersize=7)

    # shade the "unreliable" region (fewer than 8 drives)
    ax.axvspan(min_drives, min(7.5, max(ns)),
               alpha=0.12, color='#e74c3c', label='Unreliable (<8 drives)')

    # horizontal reference line at the final MAE
    final_mae = maes[-1]
    ax.axhline(final_mae, color='#27ae60', linestyle='--', linewidth=1,
               label=f'Current MAE = {final_mae:.3f} L/100km')

    ax.set_xlabel('Number of drives')
    ax.set_ylabel('LOOCV MAE (L/100km)')
    ax.set_title('Model Learning Curve — Prediction Error vs Dataset Size')
    ax.set_xticks(ns)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig('learning_curve.png', dpi=120)
    plt.show()
    print(f"\nSaved: learning_curve.png")

    return results


STATS_CACHE_PATH = get_data_dir() / 'stats_cache.json'
 
def save_stats_cache(result: dict):
    """
    Persist the model result so the Stats view survives app restarts.
    Saves global_importance and n_drives — enough to fully rebuild the view.
    """
    if not result or not result.get('success'):
        return
    cache = {
        'success':           True,
        'global_importance': result.get('global_importance', []),
        'confidence':        result.get('confidence', {}),
        'n_drives':          result.get('n_drives', 0),
        'last_updated':      _datetime.datetime.now().strftime('%d. %m. %Y  %H:%M'),
    }
    STATS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_CACHE_PATH.write_text(_json.dumps(cache, indent=2))
 
 
def load_stats_cache() -> dict | None:
    """Load the last saved stats result. Returns None if no cache exists."""
    if not STATS_CACHE_PATH.exists():
        return None
    try:
        return _json.loads(STATS_CACHE_PATH.read_text())
    except Exception:
        return None