# ============================================================================
# AI Analysis View — XGBoost prediction confidence and distribution
# ============================================================================

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
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
        row = QLabel(f"<span style='color:#666'>{key}:</span>&nbsp;<b>{val}</b>")
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


class AIAnalysisView(QWidget):
    """AI Analysis — XGBoost confidence charts and prediction distribution."""

    def __init__(self):
        super().__init__()
        self.drive_data = None
        self._create_ui()

    def _create_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        title = QLabel("AI Analysis")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        outer.addWidget(title)

        subtitle = QLabel("XGBoost prediction confidence and event distribution")
        subtitle.setStyleSheet("color: #888888; font-size: 12px;")
        outer.addWidget(subtitle)

        # ── Top row: model info + live metrics ───────────────────────────────
        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        model_card = _make_info_card("XGBoost Architecture", [
            ("Type",          "Gradient Boosting Trees"),
            ("Estimators",    "200"),
            ("Max Depth",     "6"),
            ("Learning Rate", "0.1"),
            ("Subsample",     "0.8"),
            ("Features",      "38 hand-crafted IMU features"),
            ("Window",        "100 samples / 2 s @ 50 Hz"),
            ("Stride",        "25 samples (75% overlap)"),
        ])
        info_row.addWidget(model_card)

        self._metrics_card = _make_metrics_card("Live Statistics", {
            "Windows":         "—",
            "Turn Events":     "—",
            "Hill Events":     "—",
            "Avg Turn Conf":   "—",
            "Avg Hill Conf":   "—",
            "Max Turn Conf":   "—",
            "Max Hill Conf":   "—",
        })
        info_row.addWidget(self._metrics_card)

        outer.addLayout(info_row)

        # ── Confidence chart ─────────────────────────────────────────────────
        conf_title = QLabel("Prediction Confidence Over Time")
        conf_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50; padding-bottom: 4px;")
        outer.addWidget(conf_title)

        if HAS_MATPLOTLIB:
            self._conf_figure = Figure(figsize=(10, 3), facecolor='#ffffff')
            self._conf_canvas = FigureCanvas(self._conf_figure)
            self._conf_canvas.setMinimumHeight(200)
            outer.addWidget(self._conf_canvas)
        else:
            outer.addWidget(QLabel("Install matplotlib to see charts."))

        # ── Distribution charts ───────────────────────────────────────────────
        dist_title = QLabel("Prediction Distribution")
        dist_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50; padding-bottom: 4px;")
        outer.addWidget(dist_title)

        if HAS_MATPLOTLIB:
            self._dist_figure = Figure(figsize=(10, 3), facecolor='#ffffff')
            self._dist_canvas = FigureCanvas(self._dist_figure)
            self._dist_canvas.setMinimumHeight(200)
            outer.addWidget(self._dist_canvas)
        else:
            outer.addWidget(QLabel("Install matplotlib to see charts."))

        outer.addStretch()

    # ── Data update ───────────────────────────────────────────────────────────

    def set_drive_data(self, drive_data):
        self.drive_data = drive_data
        self._update_metrics()
        self._update_confidence_chart()
        self._update_distribution_chart()

    def _update_metrics(self):
        d  = self.drive_data
        tp = d.xgb_turn_preds
        hp = d.xgb_hill_preds
        M  = len(tp)

        lv = self._metrics_card.value_labels
        lv["Windows"].setText(str(M))
        lv["Turn Events"].setText(str(int((tp != 0).sum())))
        lv["Hill Events"].setText(str(int((hp != 0).sum())))
        lv["Avg Turn Conf"].setText(f"{float(d.xgb_turn_proba.max(axis=1).mean()):.1%}")
        lv["Avg Hill Conf"].setText(f"{float(d.xgb_hill_proba.max(axis=1).mean()):.1%}")
        lv["Max Turn Conf"].setText(f"{float(d.xgb_turn_proba[:, 1:].max()):.1%}")
        lv["Max Hill Conf"].setText(f"{float(d.xgb_hill_proba[:, 1:].max()):.1%}")

    def _update_confidence_chart(self):
        if not HAS_MATPLOTLIB or not self.drive_data:
            return

        d  = self.drive_data
        tp = d.xgb_turn_preds
        hp = d.xgb_hill_preds
        M  = len(tp)
        t  = np.arange(M)

        turn_conf = d.xgb_turn_proba[t, tp]
        hill_conf = d.xgb_hill_proba[t, hp]

        self._conf_figure.clear()
        ax = self._conf_figure.add_subplot(111)
        ax.fill_between(t, turn_conf, alpha=0.20, color='#1565c0')
        ax.fill_between(t, hill_conf, alpha=0.20, color='#8e44ad')
        ax.plot(t, turn_conf, color='#1565c0', linewidth=1.5, label='Turn confidence')
        ax.plot(t, hill_conf, color='#8e44ad', linewidth=1.5, label='Hill confidence')
        ax.axhline(0.5, color='#e74c3c', linestyle='--', linewidth=1,
                   alpha=0.6, label='Threshold (0.5)')
        ax.set_xlabel('Window Index', fontsize=9)
        ax.set_ylabel('Confidence', fontsize=9)
        ax.set_title('XGBoost Prediction Confidence Per Window', fontsize=10, pad=6)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, M - 1)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        self._conf_figure.tight_layout(pad=1.0)
        self._conf_canvas.draw()

    def _update_distribution_chart(self):
        if not HAS_MATPLOTLIB or not self.drive_data:
            return

        d        = self.drive_data
        xgb_turn = d.xgb_turn_preds
        xgb_hill = d.xgb_hill_preds

        self._dist_figure.clear()

        ax1 = self._dist_figure.add_subplot(121)
        turn_vals, turn_counts = np.unique(xgb_turn, return_counts=True)
        label_map   = {0: "None", 1: "Left", 2: "Right"}
        colors_turn = ['#95a5a6', '#3498db', '#e74c3c']
        bars = ax1.bar(
            [label_map.get(int(v), str(v)) for v in turn_vals],
            turn_counts,
            color=[colors_turn[int(v)] for v in turn_vals],
            alpha=0.85,
        )
        ax1.bar_label(bars, fontsize=8)
        ax1.set_title('Turn Predictions', fontsize=10)
        ax1.set_ylabel('Count', fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.grid(True, alpha=0.25, axis='y')

        ax2 = self._dist_figure.add_subplot(122)
        hill_vals, hill_counts = np.unique(xgb_hill, return_counts=True)
        hill_label_map = {0: "None", 1: "Up", 2: "Down"}
        colors_hill    = ['#95a5a6', '#27ae60', '#8e44ad']
        bars2 = ax2.bar(
            [hill_label_map.get(int(v), str(v)) for v in hill_vals],
            hill_counts,
            color=[colors_hill[int(v)] for v in hill_vals],
            alpha=0.85,
        )
        ax2.bar_label(bars2, fontsize=8)
        ax2.set_title('Hill Predictions', fontsize=10)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.grid(True, alpha=0.25, axis='y')

        self._dist_figure.tight_layout(pad=1.0)
        self._dist_canvas.draw()
