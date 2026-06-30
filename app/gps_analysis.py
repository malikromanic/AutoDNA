"""GPS/DEM-only turn and hill detection.

Everything here is derived purely from the GPS track:

* Turns — net heading change over a short look-around distance window.
* Hills — road grade (delta elevation / delta distance) from a DEM
  elevation profile.

No IMU, no ML. Class codes match the map widget colour tables: turn
``0=straight 1=left 2=right``, hill ``0=flat 1=uphill 2=downhill``.
"""

from __future__ import annotations

import math

import numpy as np

# ── Tunables ────────────────────────────────────────────────────────────────
# Turns — a turn spans from where the road starts curving (exit threshold),
# through the apex (enter threshold), to where it straightens again, so segments
# are long and continuous rather than just the apex points.
TURN_WINDOW_M = 30.0       # window for net heading change (deg) at each point
TURN_ENTER_DEG = 10.0      # apex: net change that confirms a turn
TURN_EXIT_DEG = 4.0        # extend the turn while still curving at least this much
TURN_MERGE_GAP_M = 18.0    # bridge same-direction turns across short straights
TURN_MIN_SEG_M = 8.0       # discard turn blips shorter than this

# Hills
HILL_SMOOTH_M = 60.0      # elevation smoothing window (kills DEM/GPS noise)
HILL_GRADE_WINDOW_M = 40.0  # distance span over which grade is measured
HILL_GRADE_THRESHOLD = 3.0  # |grade| %% ⇒ uphill / downhill


# ── Geometry helpers ────────────────────────────────────────────────────────
def cumulative_distance(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance (metres) along the track, length N."""
    R = 6_371_000.0
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    cum = np.zeros(len(lat), dtype=float)
    for i in range(1, len(lat)):
        p1, p2 = math.radians(lat[i - 1]), math.radians(lat[i])
        dphi = math.radians(lat[i] - lat[i - 1])
        dlam = math.radians(lon[i] - lon[i - 1])
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
        cum[i] = cum[i - 1] + R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return cum


def _wrap180(deg: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-180, 180]."""
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


# ── Turns ─────────────────────────────────────────────────────────────────--
def _iter_segments(preds):
    """Yield (start, end, value) for each run of equal values."""
    n = len(preds)
    i = 0
    while i < n:
        v = int(preds[i])
        s = i
        while i < n and int(preds[i]) == v:
            i += 1
        yield s, i - 1, v


def _bridge_gaps(preds: np.ndarray, cum: np.ndarray, gap_m: float) -> np.ndarray:
    """Fill short straight gaps flanked by the same turn direction on both sides."""
    out = preds.copy()
    segs = list(_iter_segments(out))
    for k in range(1, len(segs) - 1):
        s, e, v = segs[k]
        if v == 0 and (cum[e] - cum[s]) < gap_m:
            vl, vr = segs[k - 1][2], segs[k + 1][2]
            if vl != 0 and vl == vr:
                out[s:e + 1] = vl
    return out


def _drop_short_turns(preds: np.ndarray, cum: np.ndarray, min_m: float) -> np.ndarray:
    """Reset turn segments shorter than min_m back to straight."""
    out = preds.copy()
    for s, e, v in _iter_segments(out):
        if v != 0 and (cum[e] - cum[s]) < min_m:
            out[s:e + 1] = 0
    return out


def compute_turns(heading: np.ndarray, cum_dist: np.ndarray,
                  window_m: float = TURN_WINDOW_M,
                  enter_deg: float = TURN_ENTER_DEG,
                  exit_deg: float = TURN_EXIT_DEG,
                  merge_gap_m: float = TURN_MERGE_GAP_M,
                  min_seg_m: float = TURN_MIN_SEG_M):
    """
    Detect turns as contiguous regions, not just apex points.

    For each point, ``rate`` is the net signed heading change over a ±window_m/2
    span (bearing is clockwise from north, so positive = right, negative = left).
    A turn region is a run of points that are *at least* curving (abs(rate) >=
    exit_deg) and contains at least one apex point (abs(rate) >= enter_deg). This
    captures the whole curve — entry, apex and exit — as one segment. Same-
    direction regions separated by a short straight are then bridged, and blips
    shorter than min_seg_m are dropped.

    Returns
    -------
    preds : (N,) int32  0=straight 1=left 2=right
    conf  : (N,) float64 0..1   (abs(rate) scaled, saturating at 90°)
    rate  : (N,) float64        signed net heading change over the window (deg)
    """
    heading = np.asarray(heading, dtype=float)
    cum = np.asarray(cum_dist, dtype=float)
    n = len(heading)
    if n == 0:
        z = np.zeros(0)
        return z.astype(np.int32), z, z

    step = _wrap180(np.diff(heading, prepend=heading[0]))  # per-point Δheading
    rate = np.zeros(n, dtype=float)
    half = window_m / 2.0
    for i in range(n):
        lo = np.searchsorted(cum, cum[i] - half, side="left")
        hi = np.searchsorted(cum, cum[i] + half, side="right")
        rate[i] = float(step[lo:hi].sum())

    strong = np.abs(rate) >= enter_deg
    weak = np.abs(rate) >= exit_deg

    preds = np.zeros(n, dtype=np.int32)
    i = 0
    while i < n:
        if weak[i]:
            s = i
            while i < n and weak[i]:
                i += 1
            e = i - 1
            if strong[s:e + 1].any():               # the run has a real apex
                net = float(rate[s:e + 1].sum())
                preds[s:e + 1] = 2 if net > 0 else 1
        else:
            i += 1

    preds = _bridge_gaps(preds, cum, merge_gap_m)
    preds = _drop_short_turns(preds, cum, min_seg_m)
    preds = _bridge_gaps(preds, cum, merge_gap_m)

    conf = np.clip(np.abs(rate) / 90.0, 0.0, 1.0)
    flat = preds == 0
    conf[flat] = 1.0 - np.clip(np.abs(rate[flat]) / enter_deg, 0.0, 1.0)
    return preds, conf, rate


# ── Hills ─────────────────────────────────────────────────────────────────--
def _smooth_over_distance(values: np.ndarray, cum: np.ndarray, window_m: float) -> np.ndarray:
    """Distance-weighted moving average (handles uneven GPS spacing)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.copy(values)
    half = window_m / 2.0
    for i in range(n):
        lo = np.searchsorted(cum, cum[i] - half, side="left")
        hi = np.searchsorted(cum, cum[i] + half, side="right")
        seg = values[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def compute_hills(elevation: np.ndarray, cum_dist: np.ndarray,
                  smooth_m: float = HILL_SMOOTH_M,
                  grade_window_m: float = HILL_GRADE_WINDOW_M,
                  threshold_pct: float = HILL_GRADE_THRESHOLD):
    """
    Road grade from a smoothed DEM elevation profile.

    Returns
    -------
    preds     : (N,) int32   0=flat 1=uphill 2=downhill
    conf      : (N,) float64 0..1
    grade_pct : (N,) float64 signed grade (%)
    elev_sm   : (N,) float64 smoothed elevation used for the calculation
    """
    elevation = np.asarray(elevation, dtype=float)
    cum = np.asarray(cum_dist, dtype=float)
    n = len(elevation)

    # No usable elevation → everything flat (hills disabled gracefully).
    if n == 0 or np.all(np.isnan(elevation)):
        z = np.zeros(n)
        return z.astype(np.int32), z.copy(), z.copy(), elevation

    # Fill NaNs by interpolation so a few missing samples don't break grades.
    elev = elevation.copy()
    nan = np.isnan(elev)
    if nan.any() and (~nan).sum() >= 2:
        elev[nan] = np.interp(cum[nan], cum[~nan], elev[~nan])

    elev_sm = _smooth_over_distance(elev, cum, smooth_m)

    grade = np.zeros(n, dtype=float)
    half = grade_window_m / 2.0
    for i in range(n):
        lo = np.searchsorted(cum, cum[i] - half, side="left")
        hi = min(np.searchsorted(cum, cum[i] + half, side="right") - 1, n - 1)
        ds = cum[hi] - cum[lo]
        if ds > 1.0:  # need a metre of run to define a slope
            grade[i] = (elev_sm[hi] - elev_sm[lo]) / ds * 100.0

    preds = np.zeros(n, dtype=np.int32)
    preds[grade >= threshold_pct] = 1   # uphill
    preds[grade <= -threshold_pct] = 2  # downhill
    conf = np.clip(np.abs(grade) / max(threshold_pct, 1e-6), 0.0, 1.0)
    conf[preds == 0] = 1.0 - conf[preds == 0]  # confidence it's flat
    return preds, conf, grade, elev_sm


def elevation_gain_loss(elev_sm: np.ndarray) -> tuple[float, float]:
    """Total ascent / descent (metres) over the smoothed profile."""
    elev_sm = np.asarray(elev_sm, dtype=float)
    if len(elev_sm) < 2 or np.all(np.isnan(elev_sm)):
        return 0.0, 0.0
    d = np.diff(elev_sm)
    return float(d[d > 0].sum()), float(-d[d < 0].sum())
