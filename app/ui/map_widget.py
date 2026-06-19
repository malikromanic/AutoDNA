#============================================================================
#map widget - gps prikaz poti pobarvane z gps-only detekcijo
#
#pristop: sprememba gps smeri zazna zavoj (l/d), sprememba gps visine zazna
#klanec (gor/dol). xgboost napovedi so nalozene a ne zaprejo detekcije -
#dostopne so na drivedata za prihodnje regresijsko delo.
#
#nacini vizualizacije (neodvisni):
#  zavoji / klanci / kombinirano - katera gps detekcija se obarva
#  filtriraj kratke odseke       - iznici odseke krajse od min_run praga
#============================================================================

import math
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
_TURN_WBASE = 3  #debelina crte za odsek brez zavoja
_TURN_WEVENT = 6  #debelina crte za zaznani zavoj

_HILL_COLOR = {0: '#9e9e9e', 1: '#2e7d32', 2: '#6a1b9a'}
_HILL_LABEL = {0: 'Flat', 1: 'Uphill', 2: 'Downhill'}
_HILL_WBASE = 3
_HILL_WEVENT = 6

#minimalne dolzine zaporedja - krajse se steje za sum
_MIN_TURN_WIN   = 2    #~2 okni (~2 s)
_MIN_HILL_WIN   = 5    #~5 gps tock - krajse je sum gps visine
_MERGE_TURN_WIN = 4    #~4 okna pri vklopljenem filtriranju kratkih odsekov
_MERGE_HILL_WIN = 6    #~6 gps tock z vklopljenim filtrom

#gps detekcija zavojev - primarni vir za lokacijo zavoja in smer l/d
_GPS_GATE_CONTEXT   = 8     #gps tocke na vsaki strani za merjenje spremembe smeri
_GPS_GATE_THRESHOLD = 22.0  #stopinje - dvignjeno s 13 za zmanjsanje laznih zavojev na krivulji
_GPS_GATE_MIN_SPEED = 5.0   #km/h - pod tem pragom je gps smer sum, ignoriramo

#zapolnjevanje vrzeli - odseki iste vrste loceni z max toliko nicami se zdruzijo
_MERGE_GAP_GPS = 5

#gps detekcija klancev z nadmorsko visino - primarni vir za smer gor/dol
_GPS_ALT_CONTEXT   = 15    #gps tocke na vsaki strani za merjenje spremembe visine
_GPS_ALT_THRESHOLD = 2.5   #metri spremembe visine za potrditev klanca

#minimalna razdalja odseka v metrih - krajsi odseki so sum gps visine
_HILL_MIN_DIST_M   = 60.0
#maksimalen realisten naklon ceste v procentih - nad tem je gps napaka
_HILL_MAX_SLOPE_PCT = 15.0


def _route_segments(arr: np.ndarray):
    """Return list of (i0, i1, pred) for consecutive equal-value runs."""
    #razstavi polje na odseke z enako vrednostjo - uporablja se za barvanje poti
    segs, i, N = [], 0, len(arr)
    while i < N:
        p = int(arr[i])
        j = i
        while j < N and int(arr[j]) == p:
            j += 1
        segs.append((i, j, p))
        i = j
    return segs


def _filter_preds(preds: np.ndarray, min_run: int, max_run: int = 0) -> np.ndarray:
    """Zero out non-zero runs shorter than min_run or longer than max_run (0 = no cap)."""
    out = preds.copy()
    for i0, i1, p in _route_segments(preds):
        if p != 0 and ((i1 - i0) < min_run or (max_run > 0 and (i1 - i0) > max_run)):
            out[i0:i1] = 0
    return out.astype(np.int32)


def _merge_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Fill zero gaps of <= max_gap points between consecutive same-value non-zero runs.

    Example: [1,1,0,0,1,1] with max_gap=3 → [1,1,1,1,1,1]
             [1,1,0,2,0,1] — different values on each side → gaps are NOT filled

    Iterates until stable so chained merges are handled:
    [1,0,1,0,1] with max_gap=1 → [1,1,1,1,1] in two passes.
    """
    out = arr.copy()
    while True:
        changed = False
        segs = _route_segments(out)
        for k in range(len(segs) - 2):
            _, i1_a, pred_a = segs[k]
            i0_z, i1_z, pred_z = segs[k + 1]
            i0_b, _, pred_b = segs[k + 2]
            if pred_z == 0 and pred_a != 0 and pred_a == pred_b and (i1_z - i0_z) <= max_gap:
                out[i0_z:i1_z] = pred_a
                changed = True
        if not changed:
            break
    return out.astype(np.int32)


def _dilate_preds(arr: np.ndarray, pad: int) -> np.ndarray:
    """Extend each non-zero run by pad GPS points on both sides without changing direction."""
    out = arr.copy()
    N = len(arr)
    for i0, i1, pred in _route_segments(arr):
        if pred == 0:
            continue
        out[max(0, i0 - pad) : min(N, i1 + pad)] = pred
    return out.astype(np.int32)


def _filter_unreliable_hills(arr: np.ndarray,
                             lat: np.ndarray, lon: np.ndarray,
                             alt: np.ndarray,
                             min_dist_m: float = _HILL_MIN_DIST_M,
                             max_slope_pct: float = _HILL_MAX_SLOPE_PCT) -> np.ndarray:
    """Zero out hill segments that are too short or have unrealistic slopes."""
    out = arr.copy()
    for i0, i1, pred in _route_segments(arr):
        if pred == 0:
            continue
        phi1 = math.radians(float(lat[i0]))
        phi2 = math.radians(float(lat[min(i1, len(lat)) - 1]))
        dphi = phi2 - phi1
        dlam = math.radians(float(lon[min(i1, len(lon)) - 1]) - float(lon[i0]))
        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
        dist_m = 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        if dist_m < min_dist_m:
            out[i0:i1] = 0
            continue
        alt_change = abs(float(alt[min(i1, len(alt)) - 1]) - float(alt[i0]))
        slope_pct = alt_change / dist_m * 100.0
        if slope_pct > max_slope_pct:
            out[i0:i1] = 0
    return out.astype(np.int32)


def _gps_detect_turns(gps_heading: np.ndarray, gps_speed: np.ndarray):
    """
    GPS-only turn detection using heading change.
    Returns ((N,) int32 labels, (N,) float32 heading_delta_deg).
    Labels: 0=straight, 1=left, 2=right.
    heading_delta_deg: absolute heading-change magnitude at each GPS point (degrees).
    """
    N      = len(gps_heading)
    out    = np.zeros(N, dtype=np.int32)
    deltas = np.zeros(N, dtype=np.float32)

    h_rad = np.radians(gps_heading.astype(np.float64))
    k = np.ones(7) / 7
    h_s = np.degrees(np.arctan2(
        np.convolve(np.sin(h_rad), k, mode='same'),
        np.convolve(np.cos(h_rad), k, mode='same'),
    )) % 360

    spd    = gps_speed.astype(np.float64)
    moving = np.ones(N, dtype=bool) if spd.max() < 1.0 else spd >= _GPS_GATE_MIN_SPEED

    for i in range(N):
        if not moving[i]:
            continue
        i0 = max(0, i - _GPS_GATE_CONTEXT)
        i1 = min(N - 1, i + _GPS_GATE_CONTEXT)
        if i1 <= i0:
            continue
        delta      = (float(h_s[i1]) - float(h_s[i0]) + 180) % 360 - 180
        deltas[i]  = abs(delta)
        if abs(delta) >= _GPS_GATE_THRESHOLD:
            out[i] = 2 if delta > 0 else 1   #2=desno, 1=levo

    return out, deltas


def _gps_detect_hills(gps_alt: np.ndarray, gps_speed: np.ndarray):
    """
    GPS-only hill detection using altitude change.
    Returns ((N,) int32 labels, (N,) float32 alt_delta_m).
    Labels: 0=flat, 1=uphill, 2=downhill.
    alt_delta_m: signed altitude change at each GPS point (positive = uphill).
    """
    N      = len(gps_alt)
    out    = np.zeros(N, dtype=np.int32)
    deltas = np.zeros(N, dtype=np.float32)

    if float(np.abs(gps_alt).max()) < 1.0:
        return out, deltas

    k     = np.ones(11) / 11
    alt_s = np.convolve(gps_alt.astype(np.float64), k, mode='same')

    spd    = gps_speed.astype(np.float64)
    moving = np.ones(N, dtype=bool) if spd.max() < 1.0 else spd >= _GPS_GATE_MIN_SPEED

    for i in range(N):
        if not moving[i]:
            continue
        i0 = max(0, i - _GPS_ALT_CONTEXT)
        i1 = min(N - 1, i + _GPS_ALT_CONTEXT)
        if i1 <= i0:
            continue
        delta      = float(alt_s[i1]) - float(alt_s[i0])
        deltas[i]  = delta
        if delta >= _GPS_ALT_THRESHOLD:
            out[i] = 1
        elif delta <= -_GPS_ALT_THRESHOLD:
            out[i] = 2

    return out, deltas


def _build_segments(gps_ts, gps_lat, gps_lon, gps_speed, gps_heading, gps_altitude,
                    turn_at_gps, turn_deltas,
                    hill_at_gps, hill_deltas):
    """
    Extract all detected GPS event segments as a list of dicts.
    Stored on MapWidget.segments — ready for regression by teammates.

    Each segment includes:
      - GPS indices, coordinates, timing, speed stats
      - Turns: turn_angle_deg, turn_severity (0-1), max_heading_change_deg
      - Hills: alt_gain_m (signed), slope_pct, direction
    """
    _type = {
        ('turn', 1): 'left_turn',  ('turn', 2): 'right_turn',
        ('hill', 1): 'uphill',     ('hill', 2): 'downhill',
    }
    N        = len(gps_lat)
    segments = []

    for task, preds, deltas in (('turn', turn_at_gps, turn_deltas),
                                 ('hill', hill_at_gps, hill_deltas)):
        for i0, i1, pred in _route_segments(preds):
            if pred == 0:
                continue
            i1c = min(i1, N) - 1
            n   = i1 - i0
            dur = float(gps_ts[i1c] - gps_ts[i0]) if n > 1 else 0.0
            spd = gps_speed[i0:i1]

            seg: dict = {
                'type':           _type[(task, pred)],
                'gps_start':      int(i0),
                'gps_end':        int(i1c),
                'n_gps_points':   int(n),
                'duration_s':     round(dur, 2),
                'lat_start':      float(gps_lat[i0]),
                'lon_start':      float(gps_lon[i0]),
                'lat_end':        float(gps_lat[i1c]),
                'lon_end':        float(gps_lon[i1c]),
                'coords':         list(zip(gps_lat[i0:i1].tolist(),
                                           gps_lon[i0:i1].tolist())),
                'speed_mean_kmh': round(float(spd.mean()), 2) if n > 0 else 0.0,
                'speed_min_kmh':  round(float(spd.min()),  2) if n > 0 else 0.0,
                'speed_max_kmh':  round(float(spd.max()),  2) if n > 0 else 0.0,
            }

            #razdalja odseka - haversine vsota vzdolz vseh gps tock
            seg_dist_m = 0.0
            for k in range(i0, i1c):
                phi1k = math.radians(float(gps_lat[k]))
                phi2k = math.radians(float(gps_lat[k + 1]))
                dphik = phi2k - phi1k
                dlamk = math.radians(float(gps_lon[k + 1]) - float(gps_lon[k]))
                ak    = (math.sin(dphik / 2) ** 2 +
                         math.cos(phi1k) * math.cos(phi2k) * math.sin(dlamk / 2) ** 2)
                seg_dist_m += 6_371_000 * 2 * math.atan2(math.sqrt(ak), math.sqrt(1 - ak))
            seg['segment_distance_m'] = round(seg_dist_m, 1)

            if task == 'turn':
                h_start   = float(gps_heading[i0])
                h_end     = float(gps_heading[i1c])
                angle     = (h_end - h_start + 180) % 360 - 180   #podpisan kot v stopinjah
                max_delta = float(deltas[i0:i1].max()) if n > 0 else 0.0

                #kumulativni kot - vsota |dsmerni_kot| po korakih, s-krivulje se ne anulirajo
                cum_hdg = 0.0
                for k in range(i0, i1c):
                    dh = (float(gps_heading[k + 1]) - float(gps_heading[k]) + 180) % 360 - 180
                    cum_hdg += abs(dh)

                seg['turn_angle_deg']             = round(abs(angle), 1)
                seg['turn_direction']             = 'left' if pred == 1 else 'right'
                seg['turn_severity']              = round(min(abs(angle) / 180.0, 1.0), 3)
                seg['max_heading_change_deg']     = round(max_delta, 1)
                seg['cumulative_heading_deg']     = round(cum_hdg, 1)
                #ostrina na meter - locuje oster zavoj na 20m od blagega na 300m
                seg['heading_change_rate_deg_per_m'] = (
                    round(cum_hdg / seg_dist_m, 4) if seg_dist_m > 1.0 else 0.0
                )
            else:
                alt_start  = float(gps_altitude[i0])
                alt_end    = float(gps_altitude[i1c])
                alt_change = alt_end - alt_start   #pozitivno = pridobljena visina

                #najstrmejsa tocka v odseku - bolj diagnosticno od povprecnega naklona
                max_slope = 0.0
                for k in range(i0, i1c):
                    phi1k = math.radians(float(gps_lat[k]))
                    phi2k = math.radians(float(gps_lat[k + 1]))
                    dphik = phi2k - phi1k
                    dlamk = math.radians(float(gps_lon[k + 1]) - float(gps_lon[k]))
                    ak    = (math.sin(dphik / 2) ** 2 +
                             math.cos(phi1k) * math.cos(phi2k) * math.sin(dlamk / 2) ** 2)
                    dm    = 6_371_000 * 2 * math.atan2(math.sqrt(ak), math.sqrt(1 - ak))
                    if dm > 0.5:   #ignoriraj gps sum pod enim metrom
                        pt_slope = abs(float(gps_altitude[k + 1]) - float(gps_altitude[k])) / dm * 100
                        if pt_slope > max_slope:
                            max_slope = pt_slope

                phi1 = math.radians(float(gps_lat[i0]))
                phi2 = math.radians(float(gps_lat[i1c]))
                dphi = phi2 - phi1
                dlam = math.radians(float(gps_lon[i1c]) - float(gps_lon[i0]))
                a         = (math.sin(dphi / 2) ** 2 +
                             math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
                dist_m    = 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                slope_pct = (alt_change / dist_m * 100.0) if dist_m > 1.0 else 0.0
                seg['alt_gain_m']    = round(alt_change, 1)
                seg['slope_pct']     = round(slope_pct, 2)
                seg['max_slope_pct'] = round(max_slope, 2)
                seg['direction']     = 'uphill' if pred == 1 else 'downhill'

            segments.append(seg)
    return segments



# ── Oznake dogodkov ──────────────────────────────────────────────────────────

#neuporabljeno po migraciji na gps-only detekcijo
#prej risalo piko na gps center vsakega xgboost okna
#odstranjeno ker detekcija poteka na ravni gps tock; _draw_colored_route
#ze postavi circlemarker na zacetek vsakega odseka na ravni gps tock
#ohranjeno za morebitno prihodnje regresijsko delo (pike resnosti po oknih)
#
# def _draw_segment_dots(fmap, lat, lon,
#                        turn_preds: np.ndarray, turn_proba: np.ndarray,
#                        hill_preds: np.ndarray, hill_proba: np.ndarray,
#                        win_center_gps: np.ndarray, mode: str):
#     """Colored dot at the GPS center of each event window (none=0 skipped)."""
#     N = len(lat)
#     turn_color = {1: '#1565c0', 2: '#e65100'}
#     turn_label = {1: 'Left Turn', 2: 'Right Turn'}
#     hill_color = {1: '#2e7d32', 2: '#6a1b9a'}
#     hill_label = {1: 'Uphill',   2: 'Downhill'}
#
#     for i in range(len(turn_preds)):
#         gps_idx = int(min(win_center_gps[i], N - 1))
#         loc = [float(lat[gps_idx]), float(lon[gps_idx])]
#
#         if mode in ('turns', 'combined'):
#             tp = int(turn_preds[i])
#             if tp != 0:
#                 conf = float(turn_proba[i, tp])
#                 folium.CircleMarker(
#                     location=loc, radius=5,
#                     color='white', weight=1,
#                     fill=True, fillColor=turn_color[tp], fillOpacity=0.95,
#                     tooltip=f"{turn_label[tp]}  win#{i}  {conf:.0%}",
#                     popup=folium.Popup(
#                         f"<b style='color:{turn_color[tp]}'>{turn_label[tp]}</b>"
#                         f"<br>win #{i} · {conf:.1%}", max_width=180),
#                 ).add_to(fmap)
#
#         if mode in ('hills', 'combined'):
#             hp = int(hill_preds[i])
#             if hp != 0:
#                 conf = float(hill_proba[i, hp])
#                 folium.CircleMarker(
#                     location=loc, radius=5,
#                     color='white', weight=1,
#                     fill=True, fillColor=hill_color[hp], fillOpacity=0.95,
#                     tooltip=f"{hill_label[hp]}  win#{i}  {conf:.0%}",
#                     popup=folium.Popup(
#                         f"<b style='color:{hill_color[hp]}'>{hill_color[hp]}</b>"
#                         f"<br>win #{i} · {conf:.1%}", max_width=180),
#                 ).add_to(fmap)


# ── Risanje poti ─────────────────────────────────────────────────────────────

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
                f"({n_pts} GPS pts · {avg_c:.0%} GPS strength)"
            )
            kw['popup'] = folium.Popup(
                f"<div style='font-family:sans-serif;font-size:12px'>"
                f"<b style='color:{color};font-size:14px'>{label_map[pred]}</b><br>"
                f"<hr style='margin:4px 0'>"
                f"GPS pts&nbsp; {i0}–{i1-1} ({n_pts} pts)<br>"
                f"GPS strength&nbsp; <b>{avg_c:.1%}</b><br>"
                f"Source&nbsp; GPS {task_name}"
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


# ── Glavni gradnik ────────────────────────────────────────────────────────────

class MapWidget(QWidget):
    """Folium map with Turns/Hills/Combined × Per Window/Merged selectors."""

    def __init__(self):
        super().__init__()
        #podatki voznje - nastavljeni z set_drive_data
        self.drive_data = None
        #pot do zacasne html datoteke folium karte
        self._tmp_path: str | None = None
        #trenutni nacin prikaza: turns, hills ali combined
        self._mode   = 'turns'
        #ali filtriramo kratke odseke (manj kot 4 gps tocke)
        self._merged = False
        #neuporabljeno po migraciji na gps-only detekcijo
        #se je uporabljalo za prikaz/skrivanje per-okno xgboost pik (_draw_segment_dots)
        #ohranjeno kot opomnik da je podpora za per-okno oznacevalce obnovljiva
        # self._show_markers = True
        #zaznani odseki dogodkov - polnjeni po vsakem izrisu, pripravljeni za regresijo
        self.segments: list[dict] = []

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

        #neuporabljeno po migraciji na gps-only detekcijo
        #ui elementi so vklapljali/izklapljali _draw_segment_dots() per-okno pike
        #potrditveno polje ni vplivalo na barvanje poti na ravni gps - to se vedno izrise
        #obnovi ko bo podpora za per-okno pike resnosti/regresije znova uvedena
        #
        # sep = QLabel(" | ")
        # sep.setStyleSheet("color:#ccc; font-size:12px;")
        # bar2.addWidget(sep)
        #
        # self._markers_cb = QCheckBox("Show all segments")
        # self._markers_cb.setChecked(True)
        # self._markers_cb.setStyleSheet("font-size: 11px; color: #555;")
        # self._markers_cb.stateChanged.connect(self._on_markers_changed)
        # bar2.addWidget(self._markers_cb)

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

    # ── Javne metode ─────────────────────────────────────────────────────────

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

    #neuporabljeno po migraciji na gps-only detekcijo
    #upravljavec potrditvenega polja za per-okno xgboost pike
    #ohranjeno za morebitno prihodnje regresijsko delo (povezi z _markers_cb ko se pike vrnejo)
    # def _on_markers_changed(self, state):
    #     self._show_markers = bool(state)
    #     if self.drive_data:
    #         self._render_map()

    def _show_placeholder(self):
        #prikazi html stran z navodilom ko ni nalozene voznje
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
        #preveri da so podatki nalozeni in da ima gps sled vsaj 2 tocki
        if d is None or len(d.gps_lat) < 2:
            self._show_placeholder()
            return

        lat    = d.gps_lat
        lon    = d.gps_lon
        N      = len(lat)
        mode   = self._mode
        merged = self._merged

        #neuporabljeno po migraciji na gps-only detekcijo
        #m se je uporabljal za pogojitev _draw_segment_dots() in se je prikazoval v legendi
        # M = len(d.xgb_turn_preds)

        t_min = _MERGE_TURN_WIN if merged else _MIN_TURN_WIN
        h_min = _MERGE_HILL_WIN if merged else _MIN_HILL_WIN

        #gps-only detekcija zavojev - sprememba smeri zazna l/d
        #gps je edini vir resnice, brez xgboost potrditve
        gps_turns, turn_deltas = _gps_detect_turns(d.gps_heading, d.gps_speed)
        turn_at_gps            = _filter_preds(gps_turns, t_min).astype(np.int32)
        turn_at_gps            = _merge_gaps(turn_at_gps, _MERGE_GAP_GPS)
        turn_at_gps            = _dilate_preds(turn_at_gps, 5)
        #magnituda spremembe smeri normalizirana na [0,1] (90 stopinj = 100%) za popup
        turn_conf_at_gps       = np.minimum(turn_deltas / 90.0, 1.0).astype(np.float32)

        #gps-only detekcija klancev - sprememba visine zazna gor/dol
        gps_hills, hill_deltas = _gps_detect_hills(d.gps_altitude, d.gps_speed)
        if gps_hills.any():
            hill_at_gps      = _filter_preds(gps_hills, h_min).astype(np.int32)
            hill_at_gps      = _merge_gaps(hill_at_gps, _MERGE_GAP_GPS)
            hill_at_gps      = _filter_unreliable_hills(hill_at_gps, lat, lon, d.gps_altitude)
            #magnituda spremembe visine normalizirana na [0,1] (10 m = 100%) za popup
            hill_conf_at_gps = np.minimum(np.abs(hill_deltas) / 10.0, 1.0).astype(np.float32)
        else:
            #gps visina ni na voljo v csv-ju - klanci niso zaznani
            hill_at_gps      = np.zeros(N, dtype=np.int32)
            hill_conf_at_gps = np.zeros(N, dtype=np.float32)

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
            {N} GPS pts &bull; GPS-only detection<br>
            {filter_label}<br>
            {stat}<br>
            click event for details
          </span>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend))

        #sestavi odseke za regresijo in jih shrani na self.segments
        self.segments = _build_segments(
            d.gps_timestamps, d.gps_lat, d.gps_lon, d.gps_speed,
            d.gps_heading, d.gps_altitude,
            turn_at_gps, turn_deltas,
            hill_at_gps, hill_deltas,
        )

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
