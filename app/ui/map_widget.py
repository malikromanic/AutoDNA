# ============================================================================
# Map Widget — GPS Route Colored by XGBoost Predictions
#
# Approach: for each GPS point find the nearest IMU window and color that
# route segment with the window's prediction.  This gives a continuous
# colored polyline without any overlapping circle-marker clutter.
#
# Visualization modes (orthogonal):
#   Turns / Hills / Combined  — which model prediction to color
#   Per Window / Merged       — raw per-GPS coloring vs short-event filtering
# ============================================================================

import os
import tempfile
import numpy as np
import folium

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
)

#barve in oznake za zavoje in klance
_TURN_COLOR = {0: '#9e9e9e', 1: '#1565c0', 2: '#e65100'}
_TURN_LABEL = {0: 'Straight', 1: 'Left Turn', 2: 'Right Turn'}
_TURN_WBASE = 3  # debelina crte za odsek brez zavoja
_TURN_WEVENT = 6  # debelina crte za zaznani zavoj

_HILL_COLOR = {0: '#9e9e9e', 1: '#2e7d32', 2: '#6a1b9a'}
_HILL_LABEL = {0: 'Flat', 1: 'Uphill', 2: 'Downhill'}
_HILL_WBASE = 3
_HILL_WEVENT = 6

#minimalno stevilo gps tock da se dogodek prikaze v filtriranem nacinu
_MIN_EVENT_GPS_PTS = 4


# ── GPS-to-window mapping ─────────────────────────────────────────────────────

def _nearest_window_per_gps(win_center_gps: np.ndarray, M: int, N: int) -> np.ndarray:
    """
    For every GPS point index k (0..N-1) return the index of the nearest
    prediction window (by GPS-center-index proximity).

    win_center_gps : (M,) array of GPS indices, one per window (non-decreasing)
    Returns        : (N,) int32 array in [0, M-1]

    On a tie (GPS point is equidistant from two window centers), the later
    window (ir) is preferred — it covers the current GPS position and beyond.
    """
    #vrni nicle ce ni oken
    if M == 0:
        return np.zeros(N, dtype=np.int32)
    #c = gps indeksi centrov oken; gps = [0,1,...,N-1]
    c = win_center_gps.astype(np.float64)
    gps = np.arange(N, dtype=np.float64)
    #za vsako gps tocko najdi indeks desnega in levega okna v c
    ir = np.clip(np.searchsorted(c, gps, side='left'), 0, M - 1)
    il = np.clip(ir - 1, 0, M - 1)
    #pri enaki razdalji preferiramo kasnejse okno (ir) - pokriva trenutno pozicijo
    nearest = np.where(np.abs(c[il] - gps) < np.abs(c[ir] - gps), il, ir)
    return nearest.astype(np.int32)


def _route_segments(arr: np.ndarray):
    """Return list of (i0, i1, pred) for consecutive equal-value runs."""
    #razstavi polje na odseke z enako vrednostjo - uporablja se za barvanje poti
    segs, i, N = [], 0, len(arr)
    while i < N:
        p = int(arr[i]); j = i
        while j < N and int(arr[j]) == p:
            j += 1
        segs.append((i, j, p))
        i = j
    return segs


def _suppress_short_events(arr: np.ndarray, min_pts: int = _MIN_EVENT_GPS_PTS) -> np.ndarray:
    """Reclassify non-zero runs shorter than min_pts GPS points to 0 (none)."""
    #kratki izoliran odseki so verjetno napake - prerazvrsti jih v razred 0 (brez)
    out = arr.copy()
    for (i0, i1, p) in _route_segments(arr):
        if p != 0 and (i1 - i0) < min_pts:
            out[i0:i1] = 0
    return out


# ── Event markers ────────────────────────────────────────────────────────────

def _draw_segment_dots(fmap, lat, lon,
                       turn_preds: np.ndarray, turn_proba: np.ndarray,
                       hill_preds: np.ndarray, hill_proba: np.ndarray,
                       win_center_gps: np.ndarray, mode: str):
    """
    Place a small filled circle at the GPS center of every prediction window.
    None windows get a tiny grey dot; event windows get a colored dot.
    Turn colors: grey=none, blue=left, orange=right.
    Hill colors: grey=none, green=up, purple=down.
    In combined mode both turn and hill dots are drawn.
    """
    N = len(lat)

    #barve in oznake za tocke zavojev in klancev (svetlejse kot barvanje poti)
    turn_color = {0: '#bdbdbd', 1: '#1565c0', 2: '#e65100'}
    turn_label = {0: 'Straight', 1: 'Left Turn', 2: 'Right Turn'}
    hill_color = {0: '#bdbdbd', 1: '#2e7d32',   2: '#6a1b9a'}
    hill_label = {0: 'Flat',    1: 'Uphill',     2: 'Downhill'}

    def _dot(gps_idx, color, tooltip_text, popup_html, is_event):
        #narisi krog na gps poziciji okna - vecji in bolj viden za dogodke
        gps_idx = int(min(gps_idx, N - 1))
        folium.CircleMarker(
            location=[float(lat[gps_idx]), float(lon[gps_idx])],
            radius=4 if is_event else 3,
            color='white' if is_event else '#aaa',
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.95 if is_event else 0.5,
            tooltip=tooltip_text,
            popup=folium.Popup(popup_html, max_width=200) if is_event else None,
        ).add_to(fmap)

    #en krog na okno - za vsak model posebej (ali oba v kombiniranem nacinu)
    for i in range(len(turn_preds)):
        tp      = int(turn_preds[i])
        hp      = int(hill_preds[i])
        gps_idx = win_center_gps[i]

        if mode in ('turns', 'combined'):
            conf  = float(turn_proba[i, tp])
            popup = (
                f"<div style='font-family:sans-serif;font-size:12px'>"
                f"<b style='color:{turn_color[tp]}'>{turn_label[tp]}</b><br>"
                f"Window #{i} &bull; conf {conf:.1%}</div>"
            )
            _dot(gps_idx, turn_color[tp],
                 f"{turn_label[tp]}  (win #{i}  {conf:.0%})",
                 popup, is_event=(tp != 0))

        if mode in ('hills', 'combined'):
            conf  = float(hill_proba[i, hp])
            popup = (
                f"<div style='font-family:sans-serif;font-size:12px'>"
                f"<b style='color:{hill_color[hp]}'>{hill_label[hp]}</b><br>"
                f"Window #{i} &bull; conf {conf:.1%}</div>"
            )
            _dot(gps_idx, hill_color[hp],
                 f"{hill_label[hp]}  (win #{i}  {conf:.0%})",
                 popup, is_event=(hp != 0))


# ── Route drawing ─────────────────────────────────────────────────────────────

def _draw_colored_route(fmap, lat, lon,
                        pred_at_gps: np.ndarray,
                        conf_at_gps: np.ndarray,
                        color_map: dict, label_map: dict,
                        weight_base: int, weight_event: int,
                        opacity_base: float, opacity_event: float,
                        task_name: str,
                        add_popup: bool = True):
    """
    Draw the GPS route as a sequence of colored PolyLine segments, one segment
    per consecutive run of the same prediction.  Event segments get a heavier
    line; 'none' segments get a thin grey line.

    Also places a small filled circle at the START of each event segment so
    transitions are easy to spot.
    """
    N    = len(lat)
    #razdeli polje napovedi na odseke z enako vrednostjo
    segs = _route_segments(pred_at_gps)

    for (i0, i1, pred) in segs:
        #podaljsaj za eno tocko na vsak konec da sosednji odseki delijo oglisce
        #brez tega nastanejo vizualne vrzeli med barvnimi prehodi na karti
        start  = max(0, i0)
        end    = min(N, i1 + 1)
        coords = [(float(lat[k]), float(lon[k])) for k in range(start, end)]
        if len(coords) < 2:
            continue

        #dogodki so debelejsi in bolj neprosojni od navadnih odsekov
        color   = color_map[pred]
        weight  = weight_event if pred != 0 else weight_base
        opacity = opacity_event if pred != 0 else opacity_base

        kw: dict = dict(color=color, weight=weight, opacity=opacity)

        #dodaj tooltip in popup samo za odseke z zaznamo (ne za ravne/ravninske)
        if add_popup and pred != 0:
            n_pts = i1 - i0
            avg_c = float(conf_at_gps[i0:i1].mean()) if n_pts > 0 else 0.0
            kw['tooltip'] = (
                f"{label_map[pred]}  "
                f"({n_pts} GPS pts · {avg_c:.0%} conf)"
            )
            kw['popup'] = folium.Popup(
                f"<div style='font-family:sans-serif;font-size:12px'>"
                f"<b style='color:{color};font-size:14px'>{label_map[pred]}</b><br>"
                f"<hr style='margin:4px 0'>"
                f"GPS pts&nbsp; {i0}–{i1-1} ({n_pts} pts)<br>"
                f"Avg confidence&nbsp; <b>{avg_c:.1%}</b><br>"
                f"Model&nbsp; XGBoost ({task_name})"
                f"</div>",
                max_width=220,
            )
        elif add_popup:
            kw['tooltip'] = label_map[pred]

        folium.PolyLine(coords, **kw).add_to(fmap)

    #oznaci zacetek vsakega dogodka z belim robom krogom
    if add_popup:
        for (i0, i1, pred) in segs:
            if pred == 0 or i0 >= N:
                continue
            n_pts = i1 - i0
            avg_c = float(conf_at_gps[i0:i1].mean()) if n_pts > 0 else 0.0
            folium.CircleMarker(
                location=[float(lat[i0]), float(lon[i0])],
                radius=5,
                color='white', weight=1.5,
                fill=True, fillColor=color_map[pred], fillOpacity=1.0,
                tooltip=(
                    f"▶ {label_map[pred]} starts here  "
                    f"({n_pts} GPS pts · {avg_c:.0%})"
                ),
            ).add_to(fmap)


# ── Main widget ───────────────────────────────────────────────────────────────

class MapWidget(QWidget):
    """Folium map with Turns/Hills/Combined × Per Window/Merged selectors."""

    def __init__(self):
        super().__init__()
        #podatki voznje - nastavljeni z set_drive_data
        self.drive_data = None
        #pot do zacasne html datoteke folium karte
        self._tmp_path: str | None = None
        #trenutni nacin prikaza: turns, hills ali combined
        self._mode         = 'turns'
        #ali filtriramo kratke odseke (manj kot 4 gps tocke)
        self._merged       = False
        #ali prikazujemo tocke za vsako okno posebej
        self._show_markers = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        #prva vrstica: gumbi za preklop med zavoji, klanci in kombiniranim prikazom
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 2)
        bar.setSpacing(6)

        lbl = QLabel("View:")
        lbl.setStyleSheet("font-size: 11px; color: #666; padding: 0;")
        bar.addWidget(lbl)

        #ustvari tri gumbe - vsak ima svojo barvo ko je aktiven
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

        bar.addStretch()
        #ob zagonu prikazi samo zavoje
        self._mode_btns['turns'].setChecked(True)
        layout.addLayout(bar)

        #druga vrstica: moznosti filtriranja in prikaza oznacevalnikov
        bar2 = QHBoxLayout()
        bar2.setContentsMargins(8, 2, 8, 4)
        bar2.setSpacing(6)

        #filtriraj odseke krajse od 4 gps tock - zmanjsa sum napacnih napovedi
        self._merge_cb = QCheckBox("Filter short events")
        self._merge_cb.setChecked(False)
        self._merge_cb.setStyleSheet("font-size: 11px; color: #555;")
        self._merge_cb.stateChanged.connect(self._on_merge_changed)
        bar2.addWidget(self._merge_cb)

        sep = QLabel(" | ")
        sep.setStyleSheet("color:#ccc; font-size:12px;")
        bar2.addWidget(sep)

        #prikazi kroge za vsako napovedno okno (en krog = eno okno)
        self._markers_cb = QCheckBox("Show all segments")
        self._markers_cb.setChecked(True)
        self._markers_cb.setStyleSheet("font-size: 11px; color: #555;")
        self._markers_cb.stateChanged.connect(self._on_markers_changed)
        bar2.addWidget(self._markers_cb)

        bar2.addStretch()
        layout.addLayout(bar2)

        #qwebengineview za prikaz folium html karte
        #lokalni dostop do oddaljenih virov je potreben za openstreetmap plosce
        self.webview = QWebEngineView()
        s = self.webview.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        layout.addWidget(self.webview)

        #prikazi nadomestno stran dokler ni nalozena voznja
        self._show_placeholder()

    # ── Public ────────────────────────────────────────────────────────────────

    def set_drive_data(self, drive_data):
        #shrani podatke in takoj izrisi karto
        self.drive_data = drive_data
        self._render_map()

    def _set_mode(self, mode: str):
        #preklopi nacin prikaza in odznaci vse ostale gumbe
        self._mode = mode
        for m, btn in self._mode_btns.items():
            btn.setChecked(m == mode)
        if self.drive_data:
            self._render_map()

    def _on_merge_changed(self, state):
        #vklopi ali izklopi filtriranje kratkih odsekov
        self._merged = bool(state)
        if self.drive_data:
            self._render_map()

    def _on_markers_changed(self, state):
        #vklopi ali izklopi prikaz tocke za vsako napovedno okno
        self._show_markers = bool(state)
        if self.drive_data:
            self._render_map()

    def _show_placeholder(self):
        #prikazi html stran z navodilom ko ni nalozene voznje
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
        #preveri da so podatki nalozeni in da ima gps sled vsaj 2 tocki
        if d is None or len(d.gps_lat) < 2:
            self._show_placeholder()
            return

        lat        = d.gps_lat
        lon        = d.gps_lon
        N          = len(lat)
        M          = len(d.xgb_turn_preds)
        mode       = self._mode
        merged     = self._merged
        win_center = d.window_gps_center_idx

        #ce ni nobenih napovednih oken (preveč kratek posnetek) prikazi samo pot brez barvanja
        if M == 0:
            turn_at_gps      = np.zeros(N, dtype=np.int32)
            hill_at_gps      = np.zeros(N, dtype=np.int32)
            turn_conf_at_gps = np.zeros(N, dtype=np.float32)
            hill_conf_at_gps = np.zeros(N, dtype=np.float32)
        else:
            #vsaki gps tocki priredi napoved najblizjega okna
            #nearest: (N,) - indeks okna za vsako gps tocko
            nearest          = _nearest_window_per_gps(win_center, M, N)
            turn_at_gps      = d.xgb_turn_preds[nearest].astype(np.int32)
            hill_at_gps      = d.xgb_hill_preds[nearest].astype(np.int32)
            #zaupanje napovedi za vsako gps tocko
            turn_conf_at_gps = d.xgb_turn_proba[nearest, turn_at_gps].astype(np.float32)
            hill_conf_at_gps = d.xgb_hill_proba[nearest, hill_at_gps].astype(np.float32)

        #filtriraj kratke dogodke ce je vklopljen rezim filtriranja
        if merged:
            turn_at_gps = _suppress_short_events(turn_at_gps, _MIN_EVENT_GPS_PTS)
            hill_at_gps = _suppress_short_events(hill_at_gps, _MIN_EVENT_GPS_PTS)
            #ponastavitve zaupanja na 0 za odseke ki so bili odstranjeni
            turn_conf_at_gps = np.where(
                turn_at_gps != 0, turn_conf_at_gps, 0.0).astype(np.float32)
            hill_conf_at_gps = np.where(
                hill_at_gps != 0, hill_conf_at_gps, 0.0).astype(np.float32)

        #ustvari folium karto - sredinisce je povprecje gps koordinat
        fmap = folium.Map(
            location=[float(lat.mean()), float(lon.mean())],
            zoom_start=15,
            tiles='OpenStreetMap',
        )

        #narisi pobarvano pot glede na izbrani nacin prikaza
        if mode == 'turns':
            #samo zavoji: siva=naravnost, modra=levo, oranzna=desno
            _draw_colored_route(
                fmap, lat, lon,
                turn_at_gps, turn_conf_at_gps,
                _TURN_COLOR, _TURN_LABEL,
                _TURN_WBASE, _TURN_WEVENT,
                opacity_base=0.45, opacity_event=0.92,
                task_name='turn',
            )

        elif mode == 'hills':
            #samo klanci: siva=ravno, zelena=gor, vijolicna=dol
            _draw_colored_route(
                fmap, lat, lon,
                hill_at_gps, hill_conf_at_gps,
                _HILL_COLOR, _HILL_LABEL,
                _HILL_WBASE, _HILL_WEVENT,
                opacity_base=0.45, opacity_event=0.92,
                task_name='hill',
            )

        else:
            #kombinirano: klanci kot ozadnje (debela linija) + zavoji spredaj (tanka)
            _draw_colored_route(
                fmap, lat, lon,
                hill_at_gps, hill_conf_at_gps,
                _HILL_COLOR, _HILL_LABEL,
                weight_base=4, weight_event=10,
                opacity_base=0.25, opacity_event=0.40,
                task_name='hill',
                add_popup=False,
            )
            _draw_colored_route(
                fmap, lat, lon,
                turn_at_gps, turn_conf_at_gps,
                _TURN_COLOR, _TURN_LABEL,
                weight_base=3, weight_event=5,
                opacity_base=0.45, opacity_event=0.92,
                task_name='turn',
                add_popup=True,
            )

        #opcijsko: en krog za vsako napovedno okno na mestu gps centra
        if self._show_markers:
            _draw_segment_dots(
                fmap, lat, lon,
                d.xgb_turn_preds, d.xgb_turn_proba,
                d.xgb_hill_preds, d.xgb_hill_proba,
                win_center, mode,
            )

        #oznaki zacetka in konca poti
        folium.Marker(
            [float(lat[0]),  float(lon[0])],
            tooltip='Start',
            icon=folium.Icon(color='green', icon='play',  prefix='fa'),
        ).add_to(fmap)
        folium.Marker(
            [float(lat[-1]), float(lon[-1])],
            tooltip='End',
            icon=folium.Icon(color='red',   icon='stop',  prefix='fa'),
        ).add_to(fmap)

        #pomocna funkcija za barvni vzorec v legendi
        def _swatch(color):
            return (
                f"<span style='display:inline-block;width:26px;height:5px;"
                f"background:{color};vertical-align:middle;"
                f"border-radius:2px;margin-right:5px'></span>"
            )

        #statistika za prikaz v legendi
        n_turns = int((turn_at_gps != 0).sum())
        n_hills = int((hill_at_gps != 0).sum())

        #sestavi vsebino legende glede na nacin prikaza
        if mode == 'turns':
            rows  = (f"{_swatch('#9e9e9e')}Straight<br>"
                     f"{_swatch('#1565c0')}Left Turn<br>"
                     f"{_swatch('#e65100')}Right Turn")
            title = "AutoDNA — Turns"
            stat  = f"{n_turns} / {N} GPS pts with turn prediction"
        elif mode == 'hills':
            rows  = (f"{_swatch('#9e9e9e')}Flat<br>"
                     f"{_swatch('#2e7d32')}Uphill<br>"
                     f"{_swatch('#6a1b9a')}Downhill")
            title = "AutoDNA — Hills"
            stat  = f"{n_hills} / {N} GPS pts with hill prediction"
        else:
            rows  = (
                f"<b style='font-size:10px;color:#555'>Background = Hill</b><br>"
                f"{_swatch('#9e9e9e')}Flat&nbsp;"
                f"{_swatch('#2e7d32')}Uphill&nbsp;"
                f"{_swatch('#6a1b9a')}Downhill<br>"
                f"<b style='font-size:10px;color:#555'>Foreground = Turn</b><br>"
                f"{_swatch('#9e9e9e')}Straight&nbsp;"
                f"{_swatch('#1565c0')}Left&nbsp;"
                f"{_swatch('#e65100')}Right"
            )
            title = "AutoDNA — Combined"
            stat  = f"Turns: {n_turns} · Hills: {n_hills} GPS pts"

        filter_label = "short-event filter ON" if merged else "no short-event filter"
        #vstavi legendu kot fiksni html element v spodnjem levem kotu karte
        legend = f"""
        <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                    background:white;padding:10px 14px;border-radius:6px;
                    border:1px solid #ccc;font-family:sans-serif;font-size:12px;
                    box-shadow:0 2px 6px rgba(0,0,0,.2)">
          <b style="font-size:13px">{title}</b><br>
          {rows}<br>
          <hr style="margin:6px 0">
          <span style="font-size:10px;color:#888">
            {N} GPS pts &bull; {M} windows<br>
            {filter_label}<br>
            {stat}<br>
            click event for details
          </span>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend))

        #zbrise staro zacasno datoteko in shrani novo - nato jo nalozi v webview
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
