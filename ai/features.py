"""Hand-crafted IMU feature extraction for the turn/hill window classifier.

Collapses a fixed ``(100, 9)`` window of preprocessed IMU samples
(``[gyro_xyz | accel_xyz | mag_xyz]``) into a fixed-length vector of
:data:`N_FEATURES` statistics — per-axis mean/std/extremes, threshold-crossing
fractions, a gyro-Z / accel-Y correlation, and energy terms. The width is fixed
so it can feed a model with a constant input size.
"""

import numpy as np

# Column indices into a (100, 9) window.
GYRO_X, GYRO_Y, GYRO_Z    = 0, 1, 2
ACCEL_X, ACCEL_Y, ACCEL_Z = 3, 4, 5
MAG_X,  MAG_Y,  MAG_Z     = 6, 7, 8

#: Length of the vector returned by :func:`extract_features`.
N_FEATURES = 36


def extract_features(window: np.ndarray) -> np.ndarray:
    """Compute the :data:`N_FEATURES`-length feature vector for one IMU window.

    :param window: ``(100, 9)`` float array, columns ``[gyro_xyz|accel_xyz|mag_xyz]``.
    :returns: ``(N_FEATURES,)`` ``float32`` vector. Zero-variance inputs (e.g. a
        stationary vehicle) are guarded so the correlation / ratio terms never
        return ``NaN``.
    """
    feats: list[float] = []
    gz = window[:, GYRO_Z]
    feats += [gz.max(), gz.min(), gz.mean(), gz.std(),
              float(np.sum(gz)) / len(gz), float(np.abs(gz).max()),
              float(np.sum(gz >  0.1)) / len(gz),
              float(np.sum(gz < -0.1)) / len(gz)]
    for ch in (GYRO_X, GYRO_Y):
        g = window[:, ch]
        feats += [float(g.mean()), float(g.std()), float(np.abs(g).max())]
    ax = window[:, ACCEL_X]
    feats += [float(ax.mean()), float(ax.std()), float(ax.max()), float(ax.min()),
              float(np.sum(ax)) / len(ax)]
    ay = window[:, ACCEL_Y]
    feats += [float(ay.mean()), float(ay.std()), float(np.abs(ay).max()),
              float(np.sum(ay < -0.2)) / len(ay),
              float(np.sum(ay >  0.2)) / len(ay)]
    az = window[:, ACCEL_Z]
    feats += [float(az.mean()), float(az.std())]
    for ch in (MAG_X, MAG_Y, MAG_Z):
        m = window[:, ch]
        feats += [float(m.mean()), float(m.max() - m.min())]
    corr = (float(np.corrcoef(gz, ay)[0, 1])
            if gz.std() > 0 and ay.std() > 0 else 0.0)
    feats += [corr,
              float(gz.std()) / (float(ax.std()) + 1e-6),
              float(np.mean(gz ** 2)),
              float(np.mean(ax ** 2))]
    return np.array(feats, dtype=np.float32)