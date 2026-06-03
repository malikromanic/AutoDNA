# ============================================================================
# Dashboard View - Overview with metrics, map, and AI insights
# ============================================================================

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from app.ui.map_widget import MapWidget


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
            current = self.value_label.styleSheet()
            self.value_label.setStyleSheet(f"color: {color}; font-size: 22pt; font-weight: bold;")


class DashboardView(QWidget):
    """Dashboard - overview metrics, route map, and AI insights summary."""

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

        subtitle = QLabel("Drive overview and AI analysis summary")
        subtitle.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(subtitle)

        # --- Metric cards row ---
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        self.safety_card = MetricCard("Overall Safety Score", "—", "/ 100", "#27ae60")
        self.distance_card = MetricCard("Distance", "—", "km", "#0066cc")
        self.duration_card = MetricCard("Duration", "—", "minutes", "#8e44ad")
        self.events_card = MetricCard("AI Events", "—", "detected", "#e67e22")

        for card in [self.safety_card, self.distance_card, self.duration_card, self.events_card]:
            cards_layout.addWidget(card)

        layout.addLayout(cards_layout)

        # --- Map + AI summary row ---
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        # Map (left 60%)
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

        # AI Summary (right 40%)
        self.ai_summary_frame = self._build_ai_summary()
        content_row.addWidget(self.ai_summary_frame, 2)

        layout.addLayout(content_row)

        # --- Category scores bar ---
        self.scores_frame = self._build_scores_frame()
        layout.addWidget(self.scores_frame)

        layout.addStretch()

    def _build_ai_summary(self) -> QFrame:
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

        self.ai_insights_label = QLabel("Load a drive to see AI analysis.")
        self.ai_insights_label.setWordWrap(True)
        self.ai_insights_label.setStyleSheet("color: #555555; font-size: 11px; line-height: 1.5;")
        layout.addWidget(self.ai_insights_label)

        # ── Turn Detections ───────────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e0e0e0;"); layout.addWidget(sep)

        turns_title = QLabel("Turn Detections")
        turns_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #1565c0;")
        layout.addWidget(turns_title)

        self.turn_grid = QGridLayout()
        self.turn_grid.setSpacing(4)
        layout.addLayout(self.turn_grid)

        # Placeholders filled by set_drive_data
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

        # ── Hill Detections ───────────────────────────────────────────────────
        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #e0e0e0;"); layout.addWidget(sep2)

        hills_title = QLabel("Hill Detections")
        hills_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #2e7d32;")
        layout.addWidget(hills_title)

        self.hill_grid = QGridLayout()
        self.hill_grid.setSpacing(4)
        layout.addLayout(self.hill_grid)

        self._hill_rows: dict[str, QLabel] = {}
        for key in ["Uphill Segments", "Downhill Segments", "Longest Hill"]:
            k = QLabel(f"{key}:")
            k.setStyleSheet("font-size: 11px; color: #555;")
            v = QLabel("—")
            v.setStyleSheet("font-size: 11px; font-weight: bold; color: #2e7d32;")
            r = len(self._hill_rows)
            self.hill_grid.addWidget(k, r, 0)
            self.hill_grid.addWidget(v, r, 1)
            self._hill_rows[key] = v

        # ── Models ────────────────────────────────────────────────────────────
        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet("color: #e0e0e0;"); layout.addWidget(sep3)

        models_title = QLabel("Models")
        models_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #333;")
        layout.addWidget(models_title)

        for model, status in [("XGBoost Turn", "Active"), ("XGBoost Hill", "Active")]:
            row = QHBoxLayout()
            m_label = QLabel(model)
            m_label.setStyleSheet("font-size: 11px; color: #555;")
            s_label = QLabel(f"✓ {status}")
            s_label.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            row.addWidget(m_label)
            row.addStretch()
            row.addWidget(s_label)
            layout.addLayout(row)

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

    def set_drive_data(self, drive_data):
        self.drive_data = drive_data
        self._update_metric_cards()
        self.map_widget.set_drive_data(drive_data)
        self._update_ai_insights()
        self._update_category_scores()

    def _update_metric_cards(self):
        d = self.drive_data

        self.distance_card.set_value(f"{d.drive_distance_km:.1f}")

        duration_min = d.drive_duration_sec / 60
        self.duration_card.set_value(f"{duration_min:.1f}")

        events = self._count_events()
        self.events_card.set_value(str(events))

        # Safety score: start at 100, subtract for event density
        event_density = events / max(duration_min, 1)
        safety = max(55, min(100, int(100 - event_density * 6)))
        color = "#27ae60" if safety >= 85 else "#e67e22" if safety >= 70 else "#e74c3c"
        self.safety_card.set_value(str(safety), color)

    def _update_ai_insights(self):
        d = self.drive_data
        tp = d.xgb_turn_preds   # (M,) 0=none 1=left 2=right
        hp = d.xgb_hill_preds   # (M,) 0=none 1=up   2=down
        M  = len(tp)

        left_count  = int((tp == 1).sum())
        right_count = int((tp == 2).sum())
        up_count    = int((hp == 1).sum())
        down_count  = int((hp == 2).sum())
        n_turns     = left_count + right_count
        n_hills     = up_count + down_count

        avg_turn_conf = float(d.xgb_turn_proba.max(axis=1).mean())
        avg_hill_conf = float(d.xgb_hill_proba.max(axis=1).mean())

        window_dur_s = d.window_size / d.target_fs
        insights_html = (
            f"Analyzed <b>{M}</b> windows "
            f"({window_dur_s:.1f}s each) with XGBoost.<br>"
            f"Turn conf avg: <b>{avg_turn_conf:.1%}</b> &nbsp;"
            f"Hill conf avg: <b>{avg_hill_conf:.1%}</b>"
        )
        self.ai_insights_label.setText(insights_html)
        self.ai_insights_label.setTextFormat(Qt.TextFormat.RichText)

        # ── Longest run helpers ───────────────────────────────────────────────
        def _longest_run_win(preds, val):
            """Length in windows of the longest consecutive run of preds==val."""
            best = cur = 0
            for p in preds:
                if p == val:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        def _longest_seg_win(preds, nonzero=True):
            """Longest consecutive run of any non-zero value."""
            best = cur = 0
            for p in preds:
                if (p != 0) if nonzero else (p == 0):
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        longest_turn_s = _longest_seg_win(tp) * window_dur_s
        longest_hill_s = _longest_seg_win(hp) * window_dur_s

        # ── Update turn summary rows ──────────────────────────────────────────
        self._turn_rows["Left Turns"].setText(
            f"{left_count} windows ({left_count * window_dur_s:.0f}s)"
        )
        self._turn_rows["Right Turns"].setText(
            f"{right_count} windows ({right_count * window_dur_s:.0f}s)"
        )
        self._turn_rows["Longest Turn"].setText(
            f"{longest_turn_s:.0f}s" if longest_turn_s > 0 else "none"
        )

        # ── Update hill summary rows ──────────────────────────────────────────
        self._hill_rows["Uphill Segments"].setText(
            f"{up_count} windows ({up_count * window_dur_s:.0f}s)"
        )
        self._hill_rows["Downhill Segments"].setText(
            f"{down_count} windows ({down_count * window_dur_s:.0f}s)"
        )
        self._hill_rows["Longest Hill"].setText(
            f"{longest_hill_s:.0f}s" if longest_hill_s > 0 else "none"
        )

    def _update_category_scores(self):
        d = self.drive_data
        duration_min = d.drive_duration_sec / 60
        turn_rate = self._count_events() / max(duration_min, 1)

        # Derive scores from IMU signal (cols 3-5=accel, cols 0-2=gyro)
        accel_mag = float(np.sqrt((d.imu_signal[:, 3:6] ** 2).sum(axis=1)).mean())
        gyro_mag  = float(np.sqrt((d.imu_signal[:, 0:3] ** 2).sum(axis=1)).mean())

        scores = {
            "Acceleration": max(70, min(99, int(100 - accel_mag * 3))),
            "Cornering":    max(70, min(99, int(100 - gyro_mag * 20 - turn_rate * 2))),
            "Braking":      max(70, min(99, int(94 - accel_mag * 1.5))),
            "Steering":     max(70, min(99, int(100 - gyro_mag * 15))),
        }
        for cat, score in scores.items():
            color = "#27ae60" if score >= 90 else "#e67e22" if score >= 75 else "#e74c3c"
            self.cat_labels[cat].setText(str(score))
            self.cat_labels[cat].setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {color};"
            )

    def _count_events(self) -> int:
        d = self.drive_data
        return int((d.xgb_turn_preds != 0).sum() + (d.xgb_hill_preds != 0).sum())
