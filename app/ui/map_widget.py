# ============================================================================
# Map Widget — GPS route coloured by GPS-derived turns / DEM-derived hills
#
# Visualization modes (orthogonal):
#   Turns / Hills / Combined  — which classification to colour
#   Per Point / Merged        — one dot per GPS point vs merged runs
#
# Turns come from GPS heading change; hills come from DEM road grade.
# Combined draws two overlapping layers: thick background = hill, thin = turn.
# ============================================================================

import os
import tempfile

import folium
import numpy as np

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
)

# ── Colour / label tables ───────────────────────────────────────────────────
_TURN_COLOR = {0: '#9e9e9e', 1: '#1565c0', 2: '#e65100'}
_TURN_LABEL = {0: 'Straight', 1: 'Left Turn', 2: 'Right Turn'}

_HILL_COLOR = {0: '#9e9e9e', 1: '#2e7d32', 2: '#6a1b9a'}
_HILL_LABEL = {0: 'Flat', 1: 'Uphill', 2: 'Downhill'}


# ── Popups ──────────────────────────────────────────────────────────────────
def _point_popup(d, i, mode):
    turn = int(d.turn_preds[i])
    hill = int(d.hill_preds[i])
    t_col, h_col = _TURN_COLOR[turn], _HILL_COLOR[hill]
    header_col = h_col if mode == 'hills' else t_col
    header_lbl = _HILL_LABEL[hill] if mode == 'hills' else _TURN_LABEL[turn]
    return (
        f"<div style='font-family:sans-serif;font-size:12px;min-width:230px'>"
        f"<b style='color:{header_col};font-size:14px'>{header_lbl}</b>"
        f" <span style='color:#aaa;font-size:11px'>pt {i+1}/{d.n_points}</span>"
        f"<hr style='margin:4px 0;border-color:#eee'>"
        f"<table style='border-collapse:collapse;width:100%'>"
        f"<tr><td colspan='2' style='padding:2px 0 4px'>"
        f"  <span style='background:{t_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Turn: {_TURN_LABEL[turn]}</span>&nbsp;"
        f"  <span style='background:{h_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Hill: {_HILL_LABEL[hill]}</span>"
        f"</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Grade</td>"
        f"    <td><b style='color:{h_col}'>{d.grade_pct[i]:+.1f}%</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Elevation</td>"
        f"    <td>{d.elevation_sm[i]:.0f} m</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Speed</td>"
        f"    <td>{d.gps_speed[i]:.0f} km/h</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Heading</td>"
        f"    <td>{d.gps_heading[i]:.0f}°</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Distance</td>"
        f"    <td>{d.cum_distance_m[i]/1000:.2f} km</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Turn conf</td>"
        f"    <td><b style='color:{t_col}'>{d.turn_conf[i]:.0%}</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Hill conf</td>"
        f"    <td><b style='color:{h_col}'>{d.hill_conf[i]:.0%}</b></td></tr>"
        f"</table></div>"
    )


def _segment_popup(d, seg, mode):
    turn, hill = seg['turn'], seg['hill']
    t_col, h_col = _TURN_COLOR[turn], _HILL_COLOR[hill]
    header_col = h_col if mode == 'hills' else t_col
    header_lbl = _HILL_LABEL[hill] if mode == 'hills' else _TURN_LABEL[turn]
    return (
        f"<div style='font-family:sans-serif;font-size:12px;min-width:230px'>"
        f"<b style='color:{header_col};font-size:14px'>{header_lbl}</b>"
        f"<hr style='margin:4px 0;border-color:#eee'>"
        f"<table style='border-collapse:collapse;width:100%'>"
        f"<tr><td colspan='2' style='padding:2px 0 4px'>"
        f"  <span style='background:{t_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Turn: {_TURN_LABEL[turn]}</span>&nbsp;"
        f"  <span style='background:{h_col};color:#fff;padding:2px 5px;"
        f"border-radius:3px;font-size:10px'>Hill: {_HILL_LABEL[hill]}</span>"
        f"</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Length</td>"
        f"    <td><b>{seg['length_m']:.0f} m</b> ({seg['n_pts']} pts)</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Elev change</td>"
        f"    <td>{seg['delev']:+.0f} m</td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Avg grade</td>"
        f"    <td><b style='color:{h_col}'>{seg['avg_grade']:+.1f}%</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Turn conf</td>"
        f"    <td><b style='color:{t_col}'>{seg['avg_turn_conf']:.0%}</b></td></tr>"
        f"<tr><td style='color:#777;padding:2px 5px'>Hill conf</td>"
        f"    <td><b style='color:{h_col}'>{seg['avg_hill_conf']:.0%}</b></td></tr>"
        f"</table></div>"
    )


# ── Segment builder (for merged view) ───────────────────────────────────────
def _build_segments(d, class_seq):
    """Group consecutive GPS points sharing the same class into segments."""
    N = len(class_seq)
    segments = []
    i = 0
    while i < N:
        key = int(class_seq[i])
        start = i
        while i < N and int(class_seq[i]) == key:
            i += 1
        end = i - 1
        rng = range(start, end + 1)
        segments.append({
            'key':           key,
            'turn':          int(np.bincount(d.turn_preds[start:end + 1], minlength=3).argmax()),
            'hill':          int(np.bincount(d.hill_preds[start:end + 1], minlength=3).argmax()),
            'start':         start,
            'end':           end,
            'n_pts':         end - start + 1,
            'length_m':      float(d.cum_distance_m[end] - d.cum_distance_m[start]),
            'delev':         float(d.elevation_sm[end] - d.elevation_sm[start]),
            'avg_grade':     float(np.mean([d.grade_pct[k] for k in rng])),
            'avg_turn_conf': float(np.mean([d.turn_conf[k] for k in rng])),
            'avg_hill_conf': float(np.mean([d.hill_conf[k] for k in rng])),
        })
    return segments


# ── Renderers ────────────────────────────────────────────────────────────--
def _draw_per_point(fmap, d, mode, color_fn, radius_fn, opacity=0.85, add_popup=True):
    lat, lon = d.gps_lat, d.gps_lon
    for i in range(d.n_points):
        turn, hill = int(d.turn_preds[i]), int(d.hill_preds[i])
        color = color_fn(turn, hill)
        kw = dict(
            location=[float(lat[i]), float(lon[i])],
            radius=radius_fn(turn, hill),
            color=color, weight=1.5,
            fill=True, fillColor=color, fillOpacity=opacity,
        )
        if add_popup:
            kw['tooltip'] = (
                f"pt {i+1} | {_TURN_LABEL[turn]} / {_HILL_LABEL[hill]} "
                f"| {d.grade_pct[i]:+.0f}%"
            )
            kw['popup'] = folium.Popup(_point_popup(d, i, mode), max_width=280)
        folium.CircleMarker(**kw).add_to(fmap)


def _draw_segments(fmap, d, segments, color_fn, weight_fn, mode,
                   opacity=0.92, add_popup=True):
    lat, lon, N = d.gps_lat, d.gps_lon, d.n_points
    for si, seg in enumerate(segments):
        i0 = seg['start']
        # extend to the next segment's start so the polyline is continuous
        i1 = segments[si + 1]['start'] + 1 if si < len(segments) - 1 else N
        i1 = min(i1, N)
        color = color_fn(seg['turn'], seg['hill'])
        if i1 - i0 < 2:
            continue
        coords = [(float(lat[k]), float(lon[k])) for k in range(i0, i1)]
        kw = dict(color=color, weight=weight_fn(seg['turn'], seg['hill']), opacity=opacity)
        if add_popup:
            kw['tooltip'] = (
                f"{_TURN_LABEL[seg['turn']]} / {_HILL_LABEL[seg['hill']]}  "
                f"({seg['length_m']:.0f} m)"
            )
            kw['popup'] = folium.Popup(_segment_popup(d, seg, mode), max_width=280)
        folium.PolyLine(coords, **kw).add_to(fmap)


# ── Main widget ──────────────────────────────────────────────────────────--
class MapWidget(QWidget):
    """Folium map with Turns/Hills/Combined × Per Point/Merged selectors."""

    def __init__(self):
        super().__init__()
        self.drive_data = None
        self._tmp_path: str | None = None
        self._mode = 'turns'      # 'turns' | 'hills' | 'combined'
        self._merged = True       # continuous polylines (default); off = per-point dots

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Selector bar ────────────────────────────────────────────────────
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

        sep = QLabel("  |  ")
        sep.setStyleSheet("color:#ccc; font-size:12px;")
        bar.addWidget(sep)

        self._merge_cb = QCheckBox("Merge segments")
        self._merge_cb.setChecked(True)
        self._merge_cb.setStyleSheet("font-size: 11px; color: #555;")
        self._merge_cb.stateChanged.connect(self._on_merge_changed)
        bar.addWidget(self._merge_cb)

        bar.addStretch()
        self._mode_btns['turns'].setChecked(True)
        layout.addLayout(bar)

        # ── Web view ────────────────────────────────────────────────────────
        self.webview = QWebEngineView()
        s = self.webview.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        layout.addWidget(self.webview)

        self._show_placeholder()

    # ── public ──────────────────────────────────────────────────────────────
    def set_drive_data(self, drive_data):
        self.drive_data = drive_data
        self._render_map()

    # ── internal ──────────────────────────────────────────────────────────--
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
                  Open a drive folder (GPS CSV) to see the route
                </div>
              </div>
            </body></html>
        """)

    def _render_map(self):
        d = self.drive_data
        if d is None or d.n_points < 2:
            self._show_placeholder()
            return

        lat, lon = d.gps_lat, d.gps_lon
        mode, merged = self._mode, self._merged

        fmap = folium.Map(
            location=[float(lat.mean()), float(lon.mean())],
            zoom_start=15,
            tiles='OpenStreetMap',
        )

        def turn_color(t, h):  return _TURN_COLOR[t]
        def turn_weight(t, h): return 6 if t != 0 else 3
        def hill_color(t, h):  return _HILL_COLOR[h]
        def hill_weight(t, h): return 6 if h != 0 else 3

        if merged:
            if mode == 'turns':
                segs = _build_segments(d, d.turn_preds)
                _draw_segments(fmap, d, segs, turn_color, turn_weight, mode)
            elif mode == 'hills':
                segs = _build_segments(d, d.hill_preds)
                _draw_segments(fmap, d, segs, hill_color, hill_weight, mode)
            else:  # combined
                hsegs = _build_segments(d, d.hill_preds)
                tsegs = _build_segments(d, d.turn_preds)
                _draw_segments(fmap, d, hsegs, hill_color, lambda t, h: 9,
                               mode, opacity=0.35, add_popup=False)
                _draw_segments(fmap, d, tsegs, turn_color, lambda t, h: 4,
                               mode, opacity=0.95, add_popup=True)
            n_vis = len(segs) if mode != 'combined' else len(tsegs)
        else:
            def turn_radius(t, h): return 5 if t != 0 else 3
            def hill_radius(t, h): return 5 if h != 0 else 3
            if mode == 'turns':
                _draw_per_point(fmap, d, mode, turn_color, turn_radius, 0.85, True)
            elif mode == 'hills':
                _draw_per_point(fmap, d, mode, hill_color, hill_radius, 0.85, True)
            else:  # combined
                _draw_per_point(fmap, d, mode, hill_color,
                                lambda t, h: 7 if h != 0 else 4, 0.30, False)
                _draw_per_point(fmap, d, mode, turn_color,
                                lambda t, h: 4 if t != 0 else 2.5, 0.90, True)
            n_vis = d.n_points

        # ── Start / End markers ──────────────────────────────────────────────
        folium.Marker(
            [float(lat[0]), float(lon[0])], tooltip='Start',
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
        ).add_to(fmap)
        folium.Marker(
            [float(lat[-1]), float(lon[-1])], tooltip='End',
            icon=folium.Icon(color='red', icon='stop', prefix='fa'),
        ).add_to(fmap)

        # ── Legend ───────────────────────────────────────────────────────────
        def _sw(color):
            return (f"<span style='display:inline-block;width:22px;height:4px;"
                    f"background:{color};vertical-align:middle;margin-right:5px'></span>")

        if mode == 'turns':
            rows = (f"{_sw('#9e9e9e')}Straight<br>{_sw('#1565c0')}Left Turn<br>"
                    f"{_sw('#e65100')}Right Turn")
            title = "AutoDNA — Turns (GPS heading)"
        elif mode == 'hills':
            rows = (f"{_sw('#9e9e9e')}Flat<br>{_sw('#2e7d32')}Uphill<br>"
                    f"{_sw('#6a1b9a')}Downhill")
            title = "AutoDNA — Hills (DEM grade)"
        else:
            rows = (f"<b style='font-size:10px;color:#555'>Background = Hill</b><br>"
                    f"{_sw('#9e9e9e')}Flat&nbsp;{_sw('#2e7d32')}Uphill&nbsp;"
                    f"{_sw('#6a1b9a')}Downhill<br>"
                    f"<b style='font-size:10px;color:#555'>Foreground = Turn</b><br>"
                    f"{_sw('#9e9e9e')}Straight&nbsp;{_sw('#1565c0')}Left&nbsp;"
                    f"{_sw('#e65100')}Right")
            title = "AutoDNA — Combined"

        view_lbl = "merged" if merged else "per-point"
        legend = f"""
        <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                    background:white;padding:10px 14px;border-radius:6px;
                    border:1px solid #ccc;font-family:sans-serif;font-size:12px;
                    box-shadow:0 2px 6px rgba(0,0,0,.2)">
          <b style="font-size:13px">{title}</b><br>
          {rows}<br>
          <hr style="margin:6px 0">
          <span style="font-size:10px;color:#888">
            {n_vis} {'segments' if merged else 'points'} &bull; {view_lbl}<br>
            elevation: {d.elevation_source} &bull; click for details
          </span>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend))

        # ── Save → load ──────────────────────────────────────────────────────
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
