# ============================================================================
# Dashboard View — overview metrics, map, and GPS analysis summary
# ============================================================================

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np

from AutoDNA.app.ui.map_widget import MapWidget


# ── Segment helpers (a "segment" is a run of the same non-zero class) ────────
def _count_segments(preds) -> int:
    preds = np.asarray(preds)
    n, prev = 0, 0
    for v in preds:
        if v != 0 and v != prev:
            n += 1
        prev = int(v)
    return n


def _class_segments(preds, value):
    """Yield (start, end) index ranges where preds == value (consecutive)."""
    preds = np.asarray(preds)
    i, N = 0, len(preds)
    while i < N:
        if preds[i] == value:
            start = i
            while i < N and preds[i] == value:
                i += 1
            yield start, i - 1
        else:
            i += 1


def _longest_segment_m(preds, cum_distance_m, value) -> float:
    best = 0.0
    for s, e in _class_segments(preds, value):
        best = max(best, float(cum_distance_m[e] - cum_distance_m[s]))
    return best


class MetricCard(QFrame):
    """Single metric display card."""

    def __init__(self, title: str, value: str, unit: str = "", color: str = "#0066cc"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-top: 3px solid {color};
                border-radius: 8px;
            }}
        """)
        self.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(title_label)

        self.value_label = QLabel(value)
        value_font = QFont()
        value_font.setPointSize(22)
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        self.value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self.value_label)

        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet("color: #999999; font-size: 10px;")
            layout.addWidget(unit_label)

        layout.addStretch()

    def set_value(self, value: str, color: str = None):
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color}; font-size: 22pt; font-weight: bold;")


class DashboardView(QWidget):
    """Dashboard — overview metrics, route map, and GPS analysis summary."""

    def __init__(self):
        super().__init__()
        self.drive_data = None
        self._create_ui()

    def _create_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        scroll.setWidget(container)

        # --- Header ---
        title = QLabel("Dashboard")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Drive overview and GPS analysis summary")
        subtitle.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(subtitle)

        # --- Metric cards row ---
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        self.safety_card = MetricCard("Overall Safety Score", "—", "/ 100", "#27ae60")
        self.distance_card = MetricCard("Distance", "—", "km", "#0066cc")
        self.duration_card = MetricCard("Duration", "—", "minutes", "#8e44ad")
        self.events_card = MetricCard("Detected Events", "—", "turns + hills", "#e67e22")
        self.savings_card = MetricCard("Potential Savings", "—", "liters vs your records", "#16a085")

        for card in [self.safety_card, self.distance_card, self.duration_card,
                     self.events_card, self.savings_card]:
            cards_layout.addWidget(card)

        layout.addLayout(cards_layout)

        # --- Map + summary row ---
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        map_frame = QFrame()
        map_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        map_frame_layout = QVBoxLayout(map_frame)
        map_frame_layout.setContentsMargins(0, 0, 0, 0)

        map_title = QLabel("  Route Overview")
        map_title.setStyleSheet("font-weight: bold; font-size: 13px; padding: 10px 14px 6px 14px; color: #333;")
        map_frame_layout.addWidget(map_title)

        self.map_widget = MapWidget()
        self.map_widget.setMinimumHeight(350)
        map_frame_layout.addWidget(self.map_widget)

        content_row.addWidget(map_frame, 3)

        self.ai_summary_frame = self._build_summary()
        content_row.addWidget(self.ai_summary_frame, 2)

        layout.addLayout(content_row)

        self.scores_frame = self._build_scores_frame()
        layout.addWidget(self.scores_frame)

        layout.addStretch()

    def _build_summary(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("Drive Summary")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self.ai_insights_label = QLabel("Load a drive to see the analysis.")
        self.ai_insights_label.setWordWrap(True)
        self.ai_insights_label.setStyleSheet("color: #555555; font-size: 11px; line-height: 1.5;")
        layout.addWidget(self.ai_insights_label)

        # ── Turn Detections ──────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(sep)

        turns_title = QLabel("Turns (GPS heading)")
        turns_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #1565c0;")
        layout.addWidget(turns_title)

        self.turn_grid = QGridLayout()
        self.turn_grid.setSpacing(4)
        layout.addLayout(self.turn_grid)

        self._turn_rows: dict[str, QLabel] = {}
        for key in ["Left Turns", "Right Turns", "Longest Turn"]:
            k = QLabel(f"{key}:")
            k.setStyleSheet("font-size: 11px; color: #555;")
            v = QLabel("—")
            v.setStyleSheet("font-size: 11px; font-weight: bold; color: #1565c0;")
            r = len(self._turn_rows)
            self.turn_grid.addWidget(k, r, 0)
            self.turn_grid.addWidget(v, r, 1)
            self._turn_rows[key] = v

        # ── Hill Detections ──────────────────────────────────────────────────
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(sep2)

        hills_title = QLabel("Hills (DEM grade)")
        hills_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #2e7d32;")
        layout.addWidget(hills_title)

        self.hill_grid = QGridLayout()
        self.hill_grid.setSpacing(4)
        layout.addLayout(self.hill_grid)

        self._hill_rows: dict[str, QLabel] = {}
        for key in ["Uphill Segments", "Downhill Segments", "Elevation Gain", "Longest Hill"]:
            k = QLabel(f"{key}:")
            k.setStyleSheet("font-size: 11px; color: #555;")
            v = QLabel("—")
            v.setStyleSheet("font-size: 11px; font-weight: bold; color: #2e7d32;")
            r = len(self._hill_rows)
            self.hill_grid.addWidget(k, r, 0)
            self.hill_grid.addWidget(v, r, 1)
            self._hill_rows[key] = v

        # ── Sources ──────────────────────────────────────────────────────────
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(sep3)

        sources_title = QLabel("Sources")
        sources_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #333;")
        layout.addWidget(sources_title)

        self._source_rows: dict[str, QLabel] = {}
        for source in ["Turns", "Hills"]:
            row = QHBoxLayout()
            m_label = QLabel(source)
            m_label.setStyleSheet("font-size: 11px; color: #555;")
            s_label = QLabel("—")
            s_label.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            row.addWidget(m_label)
            row.addStretch()
            row.addWidget(s_label)
            layout.addLayout(row)
            self._source_rows[source] = s_label

        layout.addStretch()
        return frame

    def _build_scores_frame(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(24)

        title = QLabel("Category Scores:")
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #333;")
        layout.addWidget(title)

        self.cat_labels = {}
        for cat in ["Acceleration", "Cornering", "Braking", "Steering"]:
            cat_layout = QVBoxLayout()
            cat_layout.setSpacing(2)
            name = QLabel(cat)
            name.setStyleSheet("font-size: 10px; color: #666; text-align: center;")
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            score = QLabel("—")
            score.setStyleSheet("font-size: 16px; font-weight: bold; color: #0066cc;")
            score.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cat_layout.addWidget(name)
            cat_layout.addWidget(score)
            layout.addLayout(cat_layout)
            self.cat_labels[cat] = score

        layout.addStretch()
        return frame

    # ── Data update ──────────────────────────────────────────────────────────
    def set_drive_data(self, drive_data):
        self.drive_data = drive_data
        self._update_metric_cards()
        self.map_widget.set_drive_data(drive_data)
        self._update_insights()
        self._update_category_scores()

    def _update_metric_cards(self):
        d = self.drive_data
        self.distance_card.set_value(f"{d.drive_distance_km:.1f}")

        duration_min = d.drive_duration_sec / 60
        self.duration_card.set_value(f"{duration_min:.1f}")

        events = self._count_events()
        self.events_card.set_value(str(events))

        event_density = events / max(duration_min, 1)
        safety = max(55, min(100, int(100 - event_density * 6)))
        color = "#27ae60" if safety >= 85 else "#e67e22" if safety >= 70 else "#e74c3c"
        self.safety_card.set_value(str(safety), color)

        # potential fuel saved vs the per-vehicle records (lower = more efficient)
        savings = float(getattr(d, "total_savings_l", 0.0))
        s_color = "#27ae60" if savings <= 0.05 else "#e67e22" if savings <= 0.3 else "#e74c3c"
        self.savings_card.set_value(f"{savings:.2f}", s_color)

    def _update_insights(self):
        d = self.drive_data
        tp, hp = d.turn_preds, d.hill_preds

        left_seg  = _count_segments(np.where(tp == 1, 1, 0))
        right_seg = _count_segments(np.where(tp == 2, 2, 0))
        up_seg    = _count_segments(np.where(hp == 1, 1, 0))
        down_seg  = _count_segments(np.where(hp == 2, 2, 0))

        insights_html = (
            f"Analyzed <b>{d.n_points}</b> GPS points over "
            f"<b>{d.drive_distance_km:.1f} km</b>.<br>"
            f"Turns from heading change &bull; hills from DEM grade.<br>"
            f"Max grade: <b>{np.abs(d.grade_pct).max():.1f}%</b>"
        )
        self.ai_insights_label.setText(insights_html)
        self.ai_insights_label.setTextFormat(Qt.TextFormat.RichText)

        longest_turn = _longest_segment_m(np.where(tp != 0, 1, 0), d.cum_distance_m, 1)
        longest_hill = _longest_segment_m(np.where(hp != 0, 1, 0), d.cum_distance_m, 1)

        self._turn_rows["Left Turns"].setText(f"{left_seg}")
        self._turn_rows["Right Turns"].setText(f"{right_seg}")
        self._turn_rows["Longest Turn"].setText(
            f"{longest_turn:.0f} m" if longest_turn > 0 else "none"
        )

        self._hill_rows["Uphill Segments"].setText(f"{up_seg}")
        self._hill_rows["Downhill Segments"].setText(f"{down_seg}")
        self._hill_rows["Elevation Gain"].setText(
            f"+{d.elevation_gain_m:.0f} / -{d.elevation_loss_m:.0f} m"
        )
        self._hill_rows["Longest Hill"].setText(
            f"{longest_hill:.0f} m" if longest_hill > 0 else "none"
        )

        self._source_rows["Turns"].setText("GPS heading")
        self._source_rows["Hills"].setText(d.elevation_source)

    def _update_category_scores(self):
        d = self.drive_data
        # Speed-derived longitudinal acceleration (m/s²)
        ts = d.gps_timestamps
        v = d.gps_speed / 3.6  # m/s
        if len(v) >= 2 and np.ptp(ts) > 0:
            a = np.gradient(v, ts)
        else:
            a = np.zeros_like(v)
        accel = float(a[a > 0].mean()) if np.any(a > 0) else 0.0
        brake = float(-a[a < 0].mean()) if np.any(a < 0) else 0.0
        corner = float(np.abs(d.turn_rate).mean())

        scores = {
            "Acceleration": max(70, min(99, int(99 - accel * 18))),
            "Cornering":    max(70, min(99, int(99 - corner * 1.2))),
            "Braking":      max(70, min(99, int(99 - brake * 18))),
            "Steering":     max(70, min(99, int(99 - corner * 1.0))),
        }
        for cat, score in scores.items():
            color = "#27ae60" if score >= 90 else "#e67e22" if score >= 75 else "#e74c3c"
            self.cat_labels[cat].setText(str(score))
            self.cat_labels[cat].setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {color};"
            )

    def _count_events(self) -> int:
        d = self.drive_data
        return _count_segments(d.turn_preds) + _count_segments(d.hill_preds)
