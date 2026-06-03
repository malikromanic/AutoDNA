# ============================================================================
# Map Widget — GPS Route Colored by XGBoost Predictions
#
# Visualization modes (orthogonal):
#   Turns / Hills / Combined  — which model prediction to color
#   Per Window / Merged       — one segment per window vs merged consecutive runs
#
# Per Window (default): every prediction window is drawn as its own polyline.
#   Shows all M windows with individual colors.
#
# Merged: consecutive windows with the same prediction are merged into one
#   larger segment. Fewer, cleaner segments.
#
# Combined draws two overlapping layers:
#   thick semi-transparent background = hill prediction color
#   thinner opaque foreground         = turn prediction color
# ============================================================================

import os
import tempfile
import numpy as np
import folium
from collections import Counter

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
)

# ── Color / label tables ──────────────────────────────────────────────────────
_TURN_COLOR  = {0: '#9e9e9e', 1: '#1565c0', 2: '#e65100'}
_TURN_LABEL  = {0: 'Straight', 1: 'Left Turn', 2: 'Right Turn'}
_TURN_WEIGHT = {0: 3, 1: 6, 2: 6}

_HILL_COLOR  = {0: '#9e9e9e', 1: '#2e7d32', 2: '#6a1b9a'}
_HILL_LABEL  = {0: 'Flat', 1: 'Uphill', 2: 'Downhill'}
_HILL_WEIGHT = {0: 3, 1: 6, 2: 6}


# ── Popup builders ────────────────────────────────────────────────────────────

def _window_popup(w_idx, turn, hill, t0, t1, turn_conf, hill_conf,
                  gps_pts, turn_raw, hill_raw, mode):
    dur = t1 - t0
    t_col = _TURN_COLOR[turn]
    h_col = _HILL_COLOR[hill]
    header_col = h_col if mode == 'hills' else t_col
    header_lbl = _HILL_LABEL[hill] if mode == 'hills' else _TURN_LABEL[turn]
    return (
        f"<div style='font-family:sans-serif;font-size:12px;min-width:240px'>"
        f"<b style='color:{header_col};font-size:14px'>{header_lbl}</b>"
        f" <span style='color:#aaa;font-size:11px'>W#{w_idx+1}</span><br>"
        f"<hr style='margin:4px 0;border-color:#eee'>"
        f"<table style='border-collapse:collapse;width:100%'>"
        f"<tr><td colspan='2' style='padding:2px 0 4px'>"
        f"  <span style='background:{t_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Turn: {_TURN_LABEL[turn]}</span>&nbsp;"
        f"  <span style='background:{h_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Hill: {_HILL_LABEL[hill]}</span>"
        f"</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>IMU time</td>"
        f"    <td>{t0:.1f}s – {t1:.1f}s ({dur:.1f}s)</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Turn conf</td>"
        f"    <td><b style='color:{t_col}'>{turn_conf:.1%}</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Hill conf</td>"
        f"    <td><b style='color:{h_col}'>{hill_conf:.1%}</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>GPS points</td>"
        f"    <td>{gps_pts}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Turn (raw)</td>"
        f"    <td>N:{turn_raw[0]} L:{turn_raw[1]} R:{turn_raw[2]}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Hill (raw)</td>"
        f"    <td>Flat:{hill_raw[0]} Up:{hill_raw[1]} Dn:{hill_raw[2]}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Model</td>"
        f"    <td>XGBoost (turn + hill)</td></tr>"
        f"</table></div>"
    )


def _segment_popup(seg, i0, i1, mode):
    turn = seg['turn']; hill = seg['hill']
    t_col = _TURN_COLOR[turn]; h_col = _HILL_COLOR[hill]
    dur = seg['t1'] - seg['t0']
    header_col = h_col if mode == 'hills' else t_col
    header_lbl = _HILL_LABEL[hill] if mode == 'hills' else _TURN_LABEL[turn]
    tr = seg['turn_raw']; hr = seg['hill_raw']
    sparse = (f"<span style='color:#e67e22'>⚠ {seg['n_sparse']} sparse windows</span><br>"
              if seg['n_sparse'] else "")
    return (
        f"<div style='font-family:sans-serif;font-size:12px;min-width:250px'>"
        f"<b style='color:{header_col};font-size:14px'>{header_lbl}</b><br>"
        f"<hr style='margin:4px 0;border-color:#eee'>"
        f"<table style='border-collapse:collapse;width:100%'>"
        f"<tr><td colspan='2' style='padding:2px 0 4px'>"
        f"  <span style='background:{t_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Turn: {_TURN_LABEL[turn]}</span>&nbsp;"
        f"  <span style='background:{h_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Hill: {_HILL_LABEL[hill]}</span>"
        f"</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Windows</td>"
        f"    <td><b>{seg['start_w']+1}–{seg['end_w']+1}</b> ({seg['n_win']}×2s)</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>IMU time</td>"
        f"    <td>{seg['t0']:.1f}s – {seg['t1']:.1f}s ({dur:.1f}s)</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Turn conf</td>"
        f"    <td><b style='color:{t_col}'>{seg['avg_turn_conf']:.1%}</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Hill conf</td>"
        f"    <td><b style='color:{h_col}'>{seg['avg_hill_conf']:.1%}</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>GPS points</td>"
        f"    <td>{i1 - i0}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Turn (raw)</td>"
        f"    <td>N:{tr[0]} L:{tr[1]} R:{tr[2]}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Hill (raw)</td>"
        f"    <td>Flat:{hr[0]} Up:{hr[1]} Dn:{hr[2]}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Model</td>"
        f"    <td>XGBoost (turn + hill)</td></tr>"
        f"</table>{sparse}</div>"
    )


# ── Segment builder (for merged view) ────────────────────────────────────────

def _build_segments(label_seq, turn_preds, hill_preds,
                    turn_preds_raw, hill_preds_raw,
                    win_start, win_end, imu_ts, ws,
                    turn_proba, hill_proba):
    M = len(turn_preds)
    segments = []
    w = 0
    while w < M:
        key = label_seq[w]
        start_w = w
        while w < M and label_seq[w] == key:
            w += 1
        end_w = w - 1

        turn_seg = [int(turn_preds[ww]) for ww in range(start_w, end_w + 1)]
        hill_seg = [int(hill_preds[ww]) for ww in range(start_w, end_w + 1)]
        turn_raw_flat = [int(turn_preds_raw[ww]) for ww in range(start_w, end_w + 1)]
        hill_raw_flat = [int(hill_preds_raw[ww]) for ww in range(start_w, end_w + 1)]

        dom_turn = Counter(turn_seg).most_common(1)[0][0]
        dom_hill = Counter(hill_seg).most_common(1)[0][0]

        pts_per_win = [
            int(win_end[ww]) - int(win_start[ww]) + 1
            for ww in range(start_w, end_w + 1)
        ]
        n_sparse = sum(1 for p in pts_per_win if p < 3)

        avg_turn_conf = float(np.mean([
            float(turn_proba[ww, int(turn_preds[ww])])
            for ww in range(start_w, end_w + 1)
        ]))
        avg_hill_conf = float(np.mean([
            float(hill_proba[ww, int(hill_preds[ww])])
            for ww in range(start_w, end_w + 1)
        ]))

        t0 = float(imu_ts[min(start_w * ws, len(imu_ts) - 1)])
        t1 = float(imu_ts[min((end_w + 1) * ws - 1, len(imu_ts) - 1)])

        segments.append({
            'key':           key,
            'turn':          dom_turn,
            'hill':          dom_hill,
            'start_w':       start_w,
            'end_w':         end_w,
            'n_win':         end_w - start_w + 1,
            'gps_i0':        int(win_start[start_w]),
            'gps_i1':        int(win_end[end_w]),
            'n_sparse':      n_sparse,
            'avg_turn_conf': avg_turn_conf,
            'avg_hill_conf': avg_hill_conf,
            't0':            t0,
            't1':            t1,
            # raw vote tallies [none, left/up, right/down]
            'turn_raw': [turn_raw_flat.count(0), turn_raw_flat.count(1), turn_raw_flat.count(2)],
            'hill_raw': [hill_raw_flat.count(0), hill_raw_flat.count(1), hill_raw_flat.count(2)],
        })
    return segments


# ── Per-window renderer ───────────────────────────────────────────────────────

def _draw_per_window(fmap, turn_preds, hill_preds,
                     turn_preds_raw, hill_preds_raw,
                     win_start, win_end, win_center, imu_ts, ws,
                     turn_proba, hill_proba,
                     lat, lon, N, mode,
                     color_fn, radius_fn, opacity=0.9, add_popup=True):
    """
    Draw one CircleMarker per window at the window's center GPS position.

    Windows are overlapping (stride=25, size=100), so many share GPS points.
    Circle markers at center-time position are the correct visualization —
    each of the M windows gets its own visible dot on the route.
    """
    M   = len(turn_preds)
    K   = len(imu_ts)
    for w in range(M):
        ic   = min(int(win_center[w]), N - 1)
        turn = int(turn_preds[w])
        hill = int(hill_preds[w])
        color  = color_fn(turn, hill)
        radius = radius_fn(turn, hill)

        s0 = min(w * 25, K - 1)
        s1 = min(s0 + ws - 1, K - 1)
        t0 = float(imu_ts[s0])
        t1 = float(imu_ts[s1])
        turn_conf = float(turn_proba[w, turn])
        hill_conf = float(hill_proba[w, hill])

        kw: dict = dict(
            location=[float(lat[ic]), float(lon[ic])],
            radius=radius,
            color=color,
            weight=1.5,
            fill=True,
            fillColor=color,
            fillOpacity=opacity,
        )
        if add_popup:
            raw_t = [0, 0, 0]; raw_t[int(turn_preds_raw[w])] = 1
            raw_h = [0, 0, 0]; raw_h[int(hill_preds_raw[w])] = 1
            kw['tooltip'] = (
                f"W#{w+1} | {_TURN_LABEL[turn]} / {_HILL_LABEL[hill]} "
                f"| T:{turn_conf:.0%} H:{hill_conf:.0%}"
            )
            kw['popup'] = folium.Popup(
                _window_popup(w, turn, hill, t0, t1, turn_conf, hill_conf,
                              1, raw_t, raw_h, mode),
                max_width=280,
            )
        folium.CircleMarker(**kw).add_to(fmap)


# ── Merged segment renderer ───────────────────────────────────────────────────

def _draw_segment_list(fmap, segments, lat, lon, N, color_fn, weight_fn,
                       mode, opacity=0.92, add_popup=True):
    for i, seg in enumerate(segments):
        i0 = max(0, seg['gps_i0'])
        i1 = segments[i + 1]['gps_i0'] + 1 if i < len(segments) - 1 else N
        i1 = min(i1, N)
        color  = color_fn(seg['turn'], seg['hill'])
        weight = weight_fn(seg['turn'], seg['hill'])

        if i1 - i0 < 2:
            if i0 < N and add_popup:
                folium.CircleMarker(
                    location=[float(lat[i0]), float(lon[i0])],
                    radius=4, color=color, fill=True,
                    fillColor=color, fillOpacity=0.9,
                    tooltip=f"{_TURN_LABEL[seg['turn']]} / {_HILL_LABEL[seg['hill']]}",
                ).add_to(fmap)
            continue

        coords = [(float(lat[k]), float(lon[k])) for k in range(i0, i1)]
        kw = dict(color=color, weight=weight, opacity=opacity)
        if add_popup:
            kw['tooltip'] = (
                f"{_TURN_LABEL[seg['turn']]} / {_HILL_LABEL[seg['hill']]}  "
                f"({seg['n_win']}×2s)"
            )
            kw['popup'] = folium.Popup(
                _segment_popup(seg, i0, i1, mode), max_width=290
            )
        folium.PolyLine(coords, **kw).add_to(fmap)


# ── Main widget ───────────────────────────────────────────────────────────────

class MapWidget(QWidget):
    """Folium map with Turns/Hills/Combined × Per Window/Merged selectors."""

    def __init__(self):
        super().__init__()
        self.drive_data = None
        self._tmp_path: str | None = None
        self._mode   = 'turns'   # 'turns' | 'hills' | 'combined'
        self._merged = False     # False = per-window (default), True = merged

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Selector bar ──────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 4)
        bar.setSpacing(6)

        lbl = QLabel("View:")
        lbl.setStyleSheet("font-size: 11px; color: #666; padding: 0;")
        bar.addWidget(lbl)

        self._mode_btns: dict[str, QPushButton] = {}
        for mid, mlbl, col in [
            ('turns',    'Turns',    '#1565c0'),
            ('hills',    'Hills',    '#2e7d32'),
            ('combined', 'Combined', '#6a1b9a'),
        ]:
            btn = QPushButton(mlbl)
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:white; border:1px solid #ccc; border-radius:4px;
                    padding:2px 12px; font-size:11px; color:#333;
                }}
                QPushButton:checked {{
                    background:{col}; border-color:{col};
                    color:white; font-weight:bold;
                }}
                QPushButton:hover:!checked {{
                    background:#f5f5f5; border-color:{col}; color:{col};
                }}
            """)
            btn.clicked.connect(lambda checked, m=mid: self._set_mode(m))
            self._mode_btns[mid] = btn
            bar.addWidget(btn)

        # Separator
        sep = QLabel("  |  ")
        sep.setStyleSheet("color:#ccc; font-size:12px;")
        bar.addWidget(sep)

        # Merge toggle
        self._merge_cb = QCheckBox("Merge segments")
        self._merge_cb.setChecked(False)
        self._merge_cb.setStyleSheet("font-size: 11px; color: #555;")
        self._merge_cb.stateChanged.connect(self._on_merge_changed)
        bar.addWidget(self._merge_cb)

        bar.addStretch()
        self._mode_btns['turns'].setChecked(True)
        layout.addLayout(bar)

        # ── Web view ──────────────────────────────────────────────────────────
        self.webview = QWebEngineView()
        s = self.webview.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        layout.addWidget(self.webview)

        self._show_placeholder()

    # ── public ────────────────────────────────────────────────────────────────
    def set_drive_data(self, drive_data):
        self.drive_data = drive_data
        self._render_map()

    # ── internal ──────────────────────────────────────────────────────────────
    def _set_mode(self, mode: str):
        self._mode = mode
        for m, btn in self._mode_btns.items():
            btn.setChecked(m == mode)
        if self.drive_data:
            self._render_map()

    def _on_merge_changed(self, state):
        self._merged = bool(state)
        if self.drive_data:
            self._render_map()

    def _show_placeholder(self):
        self.webview.setHtml("""
            <html><body style="background:#f8f9fa;display:flex;align-items:center;
                justify-content:center;height:100vh;margin:0;
                font-family:sans-serif;color:#999;">
              <div style="text-align:center">
                <div style="font-size:48px">&#x1f5fa;</div>
                <div style="font-size:14px;margin-top:8px">
                  Open a drive folder (BIN + GPS CSV) to see the route
                </div>
              </div>
            </body></html>
        """)

    def _render_map(self):
        d = self.drive_data
        if d is None or len(d.gps_lat) < 2:
            self._show_placeholder()
            return

        lat            = d.gps_lat
        lon            = d.gps_lon
        imu_ts         = d.imu_timestamps
        turn_preds     = d.xgb_turn_preds
        hill_preds     = d.xgb_hill_preds
        turn_preds_raw = getattr(d, 'xgb_turn_preds_raw', turn_preds)
        hill_preds_raw = getattr(d, 'xgb_hill_preds_raw', hill_preds)
        turn_proba     = d.xgb_turn_proba
        hill_proba     = d.xgb_hill_proba
        win_start      = d.window_gps_start_idx
        win_end        = d.window_gps_end_idx
        win_center     = getattr(d, 'window_gps_center_idx',
                                 (win_start + win_end) // 2)
        ws             = d.window_size
        M              = len(turn_preds)
        N              = len(lat)
        smooth_h       = getattr(d, 'smooth_half', 2)
        mode           = self._mode
        merged         = self._merged

        fmap = folium.Map(
            location=[float(lat.mean()), float(lon.mean())],
            zoom_start=15,
            tiles='OpenStreetMap',
        )

        # ── Color/weight functions ────────────────────────────────────────────
        def turn_color(t, h):  return _TURN_COLOR[t]
        def turn_weight(t, h): return _TURN_WEIGHT[t]
        def hill_color(t, h):  return _HILL_COLOR[h]
        def hill_weight(t, h): return _HILL_WEIGHT[h]

        # ── GPS head (before first window) ────────────────────────────────────
        head_end = max(0, int(win_start[0]))
        if head_end > 0:
            coords = [(float(lat[k]), float(lon[k])) for k in range(0, head_end + 1)]
            if len(coords) >= 2:
                folium.PolyLine(
                    coords, color='#9e9e9e', weight=3, opacity=0.7,
                    tooltip='Pre-recording GPS',
                ).add_to(fmap)

        # ── Draw ──────────────────────────────────────────────────────────────
        if merged:
            # ── Merged mode ───────────────────────────────────────────────────
            def _segs(label_seq):
                return _build_segments(
                    label_seq, turn_preds, hill_preds,
                    turn_preds_raw, hill_preds_raw,
                    win_start, win_end, imu_ts, ws,
                    turn_proba, hill_proba,
                )

            if mode == 'turns':
                segs = _segs([int(x) for x in turn_preds])
                _draw_segment_list(fmap, segs, lat, lon, N,
                                   turn_color, turn_weight, mode)
            elif mode == 'hills':
                segs = _segs([int(x) for x in hill_preds])
                _draw_segment_list(fmap, segs, lat, lon, N,
                                   hill_color, hill_weight, mode)
            else:  # combined
                hsegs = _segs([int(x) for x in hill_preds])
                tsegs = _segs([int(x) for x in turn_preds])
                _draw_segment_list(fmap, hsegs, lat, lon, N,
                                   hill_color, lambda t, h: 9,
                                   mode, opacity=0.35, add_popup=False)
                _draw_segment_list(fmap, tsegs, lat, lon, N,
                                   turn_color, lambda t, h: 4,
                                   mode, opacity=0.95, add_popup=True)
            # Transition dots
            active = segs if mode != 'combined' else tsegs
            for i in range(1, len(active)):
                bi = active[i]['gps_i0']
                if bi < N:
                    prev = active[i - 1]; curr = active[i]
                    c = turn_color(curr['turn'], curr['hill']) if mode != 'hills' else hill_color(curr['turn'], curr['hill'])
                    pn = _TURN_LABEL[prev['turn']] if mode != 'hills' else _HILL_LABEL[prev['hill']]
                    cn = _TURN_LABEL[curr['turn']] if mode != 'hills' else _HILL_LABEL[curr['hill']]
                    folium.CircleMarker(
                        location=[float(lat[bi]), float(lon[bi])],
                        radius=5, color='white', weight=1.5,
                        fill=True, fillColor=c, fillOpacity=1.0,
                        tooltip=f"{pn} → {cn}",
                    ).add_to(fmap)
            n_vis = len(active)

        else:
            # ── Per-window mode — one circle per window ───────────────────────
            # Windows overlap (stride=25, size=100), so we use circle markers
            # at each window's center-time GPS position rather than polylines.
            # Radius: 5 for predicted event, 3 for "none/flat".
            def turn_radius(t, h): return 5 if t != 0 else 3
            def hill_radius(t, h): return 5 if h != 0 else 3
            def comb_radius(t, h): return 6 if (t != 0 or h != 0) else 3

            def _pw(color_fn, radius_fn, opacity, add_popup):
                _draw_per_window(
                    fmap, turn_preds, hill_preds,
                    turn_preds_raw, hill_preds_raw,
                    win_start, win_end, win_center, imu_ts, ws,
                    turn_proba, hill_proba,
                    lat, lon, N, mode,
                    color_fn=color_fn, radius_fn=radius_fn,
                    opacity=opacity, add_popup=add_popup,
                )

            if mode == 'turns':
                _pw(turn_color, turn_radius, 0.85, True)
            elif mode == 'hills':
                _pw(hill_color, hill_radius, 0.85, True)
            else:  # combined
                # Outer ring: hill color (larger, semi-transparent)
                _pw(hill_color, lambda t, h: (7 if (h != 0) else 4),
                    0.30, False)
                # Inner dot: turn color
                _pw(turn_color, lambda t, h: (4 if (t != 0) else 2.5),
                    0.90, True)
            n_vis = M

        # ── Start / End markers ───────────────────────────────────────────────
        folium.Marker(
            [float(lat[0]), float(lon[0])],
            tooltip='Start',
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
        ).add_to(fmap)
        folium.Marker(
            [float(lat[-1]), float(lon[-1])],
            tooltip='End',
            icon=folium.Icon(color='red', icon='stop', prefix='fa'),
        ).add_to(fmap)

        # ── Legend ────────────────────────────────────────────────────────────
        def _sw(color):
            return (f"<span style='display:inline-block;width:22px;height:4px;"
                    f"background:{color};vertical-align:middle;margin-right:5px'></span>")

        if mode == 'turns':
            rows = (f"{_sw('#9e9e9e')}Straight<br>"
                    f"{_sw('#1565c0')}Left Turn<br>"
                    f"{_sw('#e65100')}Right Turn")
            title = "AutoDNA — Turns"
        elif mode == 'hills':
            rows = (f"{_sw('#9e9e9e')}Flat<br>"
                    f"{_sw('#2e7d32')}Uphill<br>"
                    f"{_sw('#6a1b9a')}Downhill")
            title = "AutoDNA — Hills"
        else:
            rows = (f"<b style='font-size:10px;color:#555'>Background = Hill</b><br>"
                    f"{_sw('#9e9e9e')}Flat&nbsp;{_sw('#2e7d32')}Uphill&nbsp;"
                    f"{_sw('#6a1b9a')}Downhill<br>"
                    f"<b style='font-size:10px;color:#555'>Foreground = Turn</b><br>"
                    f"{_sw('#9e9e9e')}Straight&nbsp;{_sw('#1565c0')}Left&nbsp;"
                    f"{_sw('#e65100')}Right")
            title = "AutoDNA — Combined"

        view_lbl = "merged" if merged else "per-window"
        legend = f"""
        <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                    background:white;padding:10px 14px;border-radius:6px;
                    border:1px solid #ccc;font-family:sans-serif;font-size:12px;
                    box-shadow:0 2px 6px rgba(0,0,0,.2)">
          <b style="font-size:13px">{title}</b><br>
          {rows}<br>
          <hr style="margin:6px 0">
          <span style="font-size:10px;color:#888">
            {n_vis} {'segments' if merged else 'windows'} &bull; {view_lbl}<br>
            smoothed &plusmn;{smooth_h} &bull; click for details
          </span>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend))

        # ── Save → load ───────────────────────────────────────────────────────
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
        fd, tmp = tempfile.mkstemp(suffix='.html')
        os.close(fd)
        fmap.save(tmp)
        self._tmp_path = tmp
        self.webview.setUrl(QUrl.fromLocalFile(tmp))
