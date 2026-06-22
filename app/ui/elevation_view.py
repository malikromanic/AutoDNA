# ============================================================================
# Elevation View — DEM elevation profile, road grade, and event distribution
# ============================================================================

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np

from app.gps_analysis import (
    TURN_ENTER_DEG, TURN_WINDOW_M, HILL_GRADE_THRESHOLD, HILL_GRADE_WINDOW_M,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _make_info_card(title: str, items: list) -> QFrame:
    frame = QFrame()
    frame.setStyleSheet("""
        QFrame {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(5)

    t = QLabel(title)
    t.setStyleSheet("font-weight: bold; font-size: 12px; color: #333; border: none;")
    layout.addWidget(t)

    for key, val in items:
        row = QLabel(f"<span style='color:#666'>{key}:</span>&nbsp;<span style='color:#000'>{val}</span>")
        row.setStyleSheet("font-size: 11px; border: none;")
        row.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(row)

    return frame


def _make_metrics_card(title: str, metrics: dict) -> QFrame:
    frame = QFrame()
    frame.setStyleSheet("""
        QFrame {
            background-color: #f0f8ff;
            border: 1px solid #b0c4de;
            border-radius: 8px;
        }
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(6)

    t = QLabel(title)
    t.setStyleSheet("font-weight: bold; font-size: 12px; color: #333; border: none;")
    layout.addWidget(t)

    frame.value_labels = {}
    for key, val in metrics.items():
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        k = QLabel(f"{key}:")
        k.setStyleSheet("font-size: 11px; color: #555; border: none;")
        v = QLabel(val)
        v.setStyleSheet("font-size: 11px; font-weight: bold; color: #0066cc; border: none;")
        row_layout.addWidget(k)
        row_layout.addStretch()
        row_layout.addWidget(v)
        layout.addLayout(row_layout)
        frame.value_labels[key] = v

    return frame


def _distance_per_class(preds, cum_m, n_classes=3):
    """Metres travelled in each class (attribute each step to its start point)."""
    preds = np.asarray(preds)
    cum_m = np.asarray(cum_m)
    out = np.zeros(n_classes)
    for i in range(len(preds) - 1):
        out[int(preds[i])] += cum_m[i + 1] - cum_m[i]
    return out


class ElevationView(QWidget):
    """Elevation profile, road grade, and turn/hill distribution."""

    def __init__(self):
        super().__init__()
        self.drive_data = None
        self._create_ui()

    def _create_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        title = QLabel("Elevation & Grade")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        outer.addWidget(title)

        subtitle = QLabel("DEM-based road grade and GPS event distribution")
        subtitle.setStyleSheet("color: #888888; font-size: 12px;")
        outer.addWidget(subtitle)

        # ── Top row: method info + live metrics ──────────────────────────────
        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        method_card = _make_info_card("Detection Method", [
            ("Turns",     "GPS heading change"),
            ("Turn rule", f"≥ {TURN_ENTER_DEG:.0f}° over {TURN_WINDOW_M:.0f} m"),
            ("Hills",     "DEM ground-elevation grade"),
            ("Hill rule", f"|grade| ≥ {HILL_GRADE_THRESHOLD:.0f}% over {HILL_GRADE_WINDOW_M:.0f} m"),
            ("Altitude",  "DEM lookup (device GPS altitude ignored)"),
        ])
        info_row.addWidget(method_card)

        self._metrics_card = _make_metrics_card("Live Statistics", {
            "GPS points":     "—",
            "Distance":       "—",
            "Turn segments":  "—",
            "Hill segments":  "—",
            "Elevation gain": "—",
            "Max grade":      "—",
            "Avg speed":      "—",
            "Source":         "—",
        })
        info_row.addWidget(self._metrics_card)

        outer.addLayout(info_row)

        # ── Elevation + grade chart ──────────────────────────────────────────
        prof_title = QLabel("Elevation & Grade Over Distance")
        prof_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50; padding-bottom: 4px;")
        outer.addWidget(prof_title)

        if HAS_MATPLOTLIB:
            self._prof_figure = Figure(figsize=(10, 3.4), facecolor='#ffffff')
            self._prof_canvas = FigureCanvas(self._prof_figure)
            self._prof_canvas.setMinimumHeight(230)
            outer.addWidget(self._prof_canvas)
        else:
            outer.addWidget(QLabel("Install matplotlib to see charts."))

        # ── Distribution chart ───────────────────────────────────────────────
        dist_title = QLabel("Distance by Class")
        dist_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50; padding-bottom: 4px;")
        outer.addWidget(dist_title)

        if HAS_MATPLOTLIB:
            self._dist_figure = Figure(figsize=(10, 2.8), facecolor='#ffffff')
            self._dist_canvas = FigureCanvas(self._dist_figure)
            self._dist_canvas.setMinimumHeight(190)
            outer.addWidget(self._dist_canvas)
        else:
            outer.addWidget(QLabel("Install matplotlib to see charts."))

        outer.addStretch()

    # ── Data update ──────────────────────────────────────────────────────────
    def set_drive_data(self, drive_data):
        self.drive_data = drive_data
        self._update_metrics()
        self._update_profile_chart()
        self._update_distribution_chart()

    def _update_metrics(self):
        from app.ui.dashboard_view import _count_segments
        d = self.drive_data
        lv = self._metrics_card.value_labels
        avg_speed = d.drive_distance_km / max(d.drive_duration_sec / 3600, 1e-3)
        lv["GPS points"].setText(str(d.n_points))
        lv["Distance"].setText(f"{d.drive_distance_km:.2f} km")
        lv["Turn segments"].setText(str(_count_segments(d.turn_preds)))
        lv["Hill segments"].setText(str(_count_segments(d.hill_preds)))
        lv["Elevation gain"].setText(f"+{d.elevation_gain_m:.0f} / -{d.elevation_loss_m:.0f} m")
        lv["Max grade"].setText(f"{np.abs(d.grade_pct).max():.1f}%")
        lv["Avg speed"].setText(f"{avg_speed:.0f} km/h")
        lv["Source"].setText(d.elevation_source)

    def _update_profile_chart(self):
        if not HAS_MATPLOTLIB or not self.drive_data:
            return
        d = self.drive_data
        x = d.cum_distance_m / 1000.0  # km

        self._prof_figure.clear()
        ax1 = self._prof_figure.add_subplot(211)
        ax1.plot(x, d.elevation_sm, color='#2c3e50', linewidth=1.6)
        ax1.fill_between(x, d.elevation_sm, d.elevation_sm.min(),
                         where=(d.hill_preds == 1), color='#2e7d32', alpha=0.25, label='Uphill')
        ax1.fill_between(x, d.elevation_sm, d.elevation_sm.min(),
                         where=(d.hill_preds == 2), color='#6a1b9a', alpha=0.25, label='Downhill')
        ax1.set_ylabel('Elevation (m)', fontsize=9)
        ax1.set_title('DEM Elevation Profile', fontsize=10, pad=6)
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.25)
        ax1.tick_params(labelsize=8)

        ax2 = self._prof_figure.add_subplot(212, sharex=ax1)
        ax2.plot(x, d.grade_pct, color='#e67e22', linewidth=1.2)
        ax2.axhline(HILL_GRADE_THRESHOLD, color='#2e7d32', linestyle='--', linewidth=1, alpha=0.6)
        ax2.axhline(-HILL_GRADE_THRESHOLD, color='#6a1b9a', linestyle='--', linewidth=1, alpha=0.6)
        ax2.axhline(0, color='#999', linewidth=0.8)
        ax2.set_xlabel('Distance (km)', fontsize=9)
        ax2.set_ylabel('Grade (%)', fontsize=9)
        ax2.grid(True, alpha=0.25)
        ax2.tick_params(labelsize=8)

        self._prof_figure.tight_layout(pad=1.0)
        self._prof_canvas.draw()

    def _update_distribution_chart(self):
        if not HAS_MATPLOTLIB or not self.drive_data:
            return
        d = self.drive_data
        turn_km = _distance_per_class(d.turn_preds, d.cum_distance_m) / 1000.0
        hill_km = _distance_per_class(d.hill_preds, d.cum_distance_m) / 1000.0

        self._dist_figure.clear()

        ax1 = self._dist_figure.add_subplot(121)
        bars = ax1.bar(["Straight", "Left", "Right"], turn_km,
                       color=['#9e9e9e', '#1565c0', '#e65100'], alpha=0.9)
        ax1.bar_label(bars, fmt='%.2f', fontsize=8)
        ax1.set_title('Turns (km)', fontsize=10)
        ax1.set_ylabel('Distance (km)', fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.grid(True, alpha=0.25, axis='y')

        ax2 = self._dist_figure.add_subplot(122)
        bars2 = ax2.bar(["Flat", "Uphill", "Downhill"], hill_km,
                        color=['#9e9e9e', '#2e7d32', '#6a1b9a'], alpha=0.9)
        ax2.bar_label(bars2, fmt='%.2f', fontsize=8)
        ax2.set_title('Hills (km)', fontsize=10)
        ax2.set_ylabel('Distance (km)', fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.grid(True, alpha=0.25, axis='y')

        self._dist_figure.tight_layout(pad=1.0)
        self._dist_canvas.draw()
