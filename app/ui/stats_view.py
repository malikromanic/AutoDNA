# ============================================================================
# Stats View — fuel model feature importance and key takeaways
# ============================================================================

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

_PIE_COLORS = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
    '#59a14f', '#edc948', '#b07aa1', '#ff9da7',
    '#9c755f', '#bab0ac', '#d37295', '#a0cbe8',
]

_RED    = "#e74c3c"
_GREEN  = "#27ae60"
_YELLOW = "#f0a500"

_TAKEAWAY_DESCRIPTIONS = {
    'harsh_accel_rate':   ("Avoid sudden acceleration",  "Sharp acceleration wastes fuel."),
    'harsh_braking_rate': ("Avoid harsh braking",        "Sharp braking wastes fuel."),
    'rms_jerk':           ("Drive more smoothly",        "Jerk = frequent speed changes. Drive more smoothly to reduce."),
    'accel_x_std':        ("Maintain steady speed",      "Inconsistent throttle leads to higher overall consumption."),
    'avg_speed_kmh':      ("Average speed matters",      "Urban stop-go driving is less efficient than a steady cruise."),
    'pct_turning':        ("Route characteristics",      "Winding routes with frequent turns require more acceleration cycles."),
    'turns_per_km':       ("Turn frequency",             "More turns per km means more speed adjustments."),
    'pct_uphill':         ("Elevation increases load",   "Uphill driving significantly raises engine demand."),
    'pct_downhill':       ("Downhill helps efficiency",  "Descents reduce the energy required from the engine."),
    'distance_km':        ("Longer trips are efficient", "Short trips are costly per km due to cold-engine warm-up."),
    'duration_min':       ("Drive duration",             "Longer drives allow the engine to reach its optimal temperature."),
}


class _TakeawayCard(QFrame):
    """One of the three top-feature highlight boxes."""

    def __init__(self, rank: int, item: dict, importance_score: float,
                 confidence_info: dict | None = None):
        super().__init__()
        positive   = item['std_coef'] >= 0
        accent     = _RED if positive else _GREEN
        action     = "DECREASE" if positive else "INCREASE"
        confidence = confidence_info.get('confidence', 'high') if confidence_info else 'high'

        self.setStyleSheet(f"""
            QFrame {{
                background: #1e1e2e;
                border: 1px solid #333;
                border-top: 3px solid {accent};
                border-radius: 8px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        # rank + title + warning triangle
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        badge = QLabel(f"#{rank}")
        badge.setStyleSheet(
            f"color: {accent}; font-size: 10px; font-weight: bold;"
            " background: transparent; border: none;"
        )
        top_row.addWidget(badge)

        title = QLabel(item['label'])
        tf = QFont(); tf.setPointSize(11); tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet("color: #ffffff; border: none;")
        top_row.addWidget(title)
        top_row.addStretch()

        if confidence == 'low':
            tri = QLabel("⚠️")
            tri.setStyleSheet("font-size: 14px; border: none;")
            tri.setToolTip("Low confidence — direction may change as more drives are added.")
            top_row.addWidget(tri)

        layout.addLayout(top_row)

        # description
        key = item['key']
        _, body = _TAKEAWAY_DESCRIPTIONS.get(key, (item['label'], ""))
        desc = QLabel(body)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaaaaa; font-size: 10px; border: none;")
        layout.addWidget(desc)

        # low-confidence warning text
        if confidence == 'low':
            warn = QLabel("⚠️  Limited data — record more drives to confirm this finding.")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {_YELLOW}; font-size: 10px; font-style: italic; border: none;")
            layout.addWidget(warn)

        layout.addStretch()

        # bottom: action label + importance score
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        action_lbl = QLabel(
            f"<span style='color:#ffffff;'>TO SAVE FUEL: </span>"
            f"<span style='color:{accent};'>{action} {item['label'].upper()}</span>"
        )
        action_lbl.setTextFormat(Qt.TextFormat.RichText)
        action_lbl.setStyleSheet("font-size: 11px; font-weight: bold; border: none;")
        bottom_row.addWidget(action_lbl)
        bottom_row.addStretch()

        score_lbl = QLabel(f"importance: {importance_score:.2f}")
        score_lbl.setStyleSheet("color: #666; font-size: 10px; border: none;")
        bottom_row.addWidget(score_lbl)

        layout.addLayout(bottom_row)


class _FeatureRow(QWidget):
    """One row in the full importance list."""

    _BAR_MAX_W = 160

    def __init__(self, item: dict, max_abs_coef: float,
                 confidence_info: dict | None = None):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(44)

        std_coef   = item['std_coef']
        raw_coef   = item['raw_coef']
        positive   = std_coef >= 0
        accent     = _RED if positive else _GREEN
        sign       = "+" if positive else ""
        coef_str   = f"{sign}{raw_coef:.4f} L/100km / {item['unit']}"
        confidence = confidence_info.get('confidence', 'high') if confidence_info else 'high'

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(12)

        # coloured accent strip
        strip = QFrame()
        strip.setFixedSize(4, 32)
        strip.setStyleSheet(f"background: {accent}; border-radius: 2px; border: none;")
        row.addWidget(strip)

        # feature label
        lbl = QLabel(item['label'])
        lbl.setStyleSheet("font-size: 12px; color: #ffffff; border: none;")
        lbl.setMinimumWidth(170)
        row.addWidget(lbl)

        # coefficient value
        coef_lbl = QLabel(coef_str)
        coef_lbl.setStyleSheet(
            f"font-size: 11px; color: {accent}; font-weight: bold; border: none;"
        )
        coef_lbl.setMinimumWidth(220)
        row.addWidget(coef_lbl)

        # importance score
        score = abs(std_coef) / max_abs_coef if max_abs_coef > 0 else 0
        score_lbl = QLabel(f"importance: {score:.2f}")
        score_lbl.setStyleSheet("font-size: 10px; color: #666; border: none;")
        score_lbl.setMinimumWidth(110)
        row.addWidget(score_lbl)

        # confidence indicator
        if confidence == 'low':
            # yellow triangle — uncertain direction
            conf_lbl = QLabel("⚠️")
            conf_lbl.setStyleSheet(f"font-size: 13px; color: {_YELLOW}; border: none;")
            conf_lbl.setToolTip("Low confidence — direction uncertain. Add more diverse drives.")
            row.addWidget(conf_lbl)
        elif confidence == 'negligible':
            # red triangle — feature has no detectable effect
            conf_lbl = QLabel("🔴")
            conf_lbl.setStyleSheet("font-size: 11px; border: none;")
            conf_lbl.setToolTip("Negligible — this feature shows no detectable effect on fuel consumption.")
            row.addWidget(conf_lbl)

        row.addStretch()

        # magnitude bar
        bar_w = int(self._BAR_MAX_W * abs(std_coef) / max_abs_coef) if max_abs_coef > 0 else 0

        bar_bg = QFrame()
        bar_bg.setFixedSize(self._BAR_MAX_W, 10)
        bar_bg.setStyleSheet("background: #333; border-radius: 5px; border: none;")

        bar_fill = QFrame(bar_bg)
        bar_fill.setFixedSize(max(bar_w, 2), 10)
        bar_fill.setStyleSheet(f"background: {accent}; border-radius: 5px; border: none;")

        row.addWidget(bar_bg)


class _PieChart(FigureCanvasQTAgg):
    """Matplotlib pie chart embedded as a Qt widget."""

    def __init__(self, width=5, height=4):
        fig = Figure(figsize=(width, height), facecolor='#1a1a2e')
        super().__init__(fig)
        self._ax = fig.add_subplot(111)
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def update_chart(self, importance: list[dict]):
        self._ax.clear()
        self._ax.set_facecolor('#1a1a2e')

        labels  = [f['label'] for f in importance]
        sizes   = [abs(f['std_coef']) for f in importance]
        colors  = [_PIE_COLORS[i % len(_PIE_COLORS)] for i in range(len(importance))]
        explode = [0.05 if i == 0 else 0 for i in range(len(importance))]

        wedges, _, autotexts = self._ax.pie(
            sizes, labels=None, colors=colors, explode=explode,
            autopct='%1.1f%%', pctdistance=0.78, startangle=90,
            wedgeprops=dict(linewidth=0.5, edgecolor='#1a1a2e'),
        )
        for at in autotexts:
            at.set_color('#ffffff'); at.set_fontsize(8)

        self._ax.legend(wedges, labels, loc='center left',
                        bbox_to_anchor=(1.0, 0.5), fontsize=8,
                        framealpha=0, labelcolor='#cccccc')
        self._ax.set_title('Relative Feature Importance', color='#cccccc', fontsize=11, pad=10)
        self.figure.tight_layout()
        self.draw()


class StatsView(QWidget):
    """Feature importance view with key takeaways and full ranked list."""

    def __init__(self):
        super().__init__()
        self._result = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        self._container = QWidget()
        self._root = QVBoxLayout(self._container)
        self._root.setContentsMargins(20, 20, 20, 20)
        self._root.setSpacing(16)
        scroll.setWidget(self._container)

        title = QLabel("Fuel Efficiency Statistics")
        tf = QFont(); tf.setPointSize(20); tf.setBold(True)
        title.setFont(tf)
        self._root.addWidget(title)

        self._subtitle = QLabel("Load drives to build the model.")
        self._subtitle.setStyleSheet("color: #888; font-size: 12px;")
        self._root.addWidget(self._subtitle)

        # ── Potential fuel savings (per-segment records) ─────────────────────
        sav_frame = QFrame()
        sav_frame.setStyleSheet(
            "QFrame { background: #11261d; border: 1px solid #1f5d3f;"
            " border-left: 4px solid #27ae60; border-radius: 8px; }"
        )
        sav_layout = QVBoxLayout(sav_frame)
        sav_layout.setContentsMargins(14, 12, 14, 12)
        sav_layout.setSpacing(6)

        self._savings_title = QLabel("Potential Savings")
        _stf = QFont()
        _stf.setPointSize(14)
        _stf.setBold(True)
        self._savings_title.setFont(_stf)
        self._savings_title.setStyleSheet("color: #2ecc71; border: none;")
        sav_layout.addWidget(self._savings_title)

        self._savings_sub = QLabel("Load a drive to compare it against your records.")
        self._savings_sub.setWordWrap(True)
        self._savings_sub.setStyleSheet("color: #aaa; font-size: 11px; border: none;")
        self._savings_sub.setTextFormat(Qt.TextFormat.RichText)
        sav_layout.addWidget(self._savings_sub)

        self._savings_box = QVBoxLayout()
        self._savings_box.setSpacing(2)
        sav_layout.addLayout(self._savings_box)

        self._root.addWidget(sav_frame)

        # key takeaways
        sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine)
        sep0.setStyleSheet("color: #333;"); self._root.addWidget(sep0)

        kt_title = QLabel("Key Takeaways")
        ktf = QFont(); ktf.setPointSize(13); ktf.setBold(True)
        kt_title.setFont(ktf); self._root.addWidget(kt_title)

        kt_sub = QLabel(
            "Top 3 factors affecting your fuel consumption. "
            "⚠️ = low confidence, more drives needed."
        )
        kt_sub.setStyleSheet("color: #888; font-size: 11px;")
        self._root.addWidget(kt_sub)

        self._takeaways_row = QHBoxLayout()
        self._takeaways_row.setSpacing(12)
        self._root.addLayout(self._takeaways_row)

        # feature importance
        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #333;"); self._root.addWidget(sep1)

        imp_title = QLabel("Feature Importance")
        itf = QFont(); itf.setPointSize(13); itf.setBold(True)
        imp_title.setFont(itf); self._root.addWidget(imp_title)

        imp_sub = QLabel(
            "Red = increases consumption · Green = decreases consumption\n"
            "⚠️ yellow = low confidence · 🔴 red = negligible effect"
        )
        imp_sub.setStyleSheet("color: #888; font-size: 11px;")
        self._root.addWidget(imp_sub)

        self._pie = _PieChart()
        self._root.addWidget(self._pie)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._root.addWidget(self._rows_container)

        self._placeholder = QLabel("No model result yet.\nLoad a drive to generate statistics.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #555; font-size: 13px; padding: 40px 0;")
        self._rows_layout.addWidget(self._placeholder)

        self._root.addStretch()

    # ── public API ─────────────────────────────────────────────────────────

    def set_savings(self, d):
        """Show per-segment fuel-record savings for the loaded drive."""
        total     = float(getattr(d, "total_savings_l", 0.0))
        breakdown = list(getattr(d, "savings_breakdown", []) or [])
        worse     = sorted(
            (b for b in breakdown if b.get("perf", 1) >= 2),
            key=lambda b: b.get("savings_l", 0.0), reverse=True,
        )

        self._savings_title.setText(f"Potential Savings:  {total:.2f} L")
        self._savings_sub.setText(
            f"vs your records &middot; {len(worse)} of {len(breakdown)} segments "
            f"above their record. A record is the lowest fuel rate seen for that "
            f"turn/hill bucket."
        )

        while self._savings_box.count():
            it = self._savings_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        if not worse:
            ok = QLabel("Every segment matched or beat your records — nice driving!")
            ok.setStyleSheet("color: #2ecc71; font-size: 11px; border: none;")
            self._savings_box.addWidget(ok)
            return

        for b in worse[:8]:
            lbl = QLabel(
                f"<span style='color:#e74c3c;font-weight:bold'>+{b['savings_l']:.3f} L</span>"
                f" &nbsp; <span style='color:#ddd'>{b['bucket']}</span>"
                f" &nbsp; <span style='color:#888'>"
                f"{b['rate_l_s'] * 1000:.2f} vs {b['record_l_s'] * 1000:.2f} mL/s"
                f" &middot; {b['duration_s']:.0f}s</span>"
            )
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("font-size: 11px; border: none;")
            self._savings_box.addWidget(lbl)

    def set_result(self, result: dict):
        """Update view with result dict from fuel_model.fit_and_explain() or load_stats_cache()."""
        self._result = result

        if not result or not result.get('success'):
            msg = result.get('message', 'Not enough data.') if result else 'No data.'
            self._subtitle.setText(msg)
            return

        n            = result.get('n_drives', 0)
        last_updated = result.get('last_updated', '')
        ts_str       = f"  ·  last updated {last_updated}" if last_updated else ""
        self._subtitle.setText(
            f"Based on {n} drive{'s' if n != 1 else ''}{ts_str}. "
            "Results improve as more drives are added."
        )

        importance  = result.get('global_importance', [])
        confidence  = result.get('confidence', {})   # dict: feature_key -> confidence_info

        self._rebuild_takeaways(importance, confidence)
        self._rebuild_pie(importance)
        self._rebuild_rows(importance, confidence)

    # ── internal ───────────────────────────────────────────────────────────

    def _rebuild_takeaways(self, importance: list[dict], confidence: dict):
        while self._takeaways_row.count():
            item = self._takeaways_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # skip negligible features — choose top 3 from what remains
        eligible = [f for f in importance
                    if confidence.get(f['key'], {}).get('confidence', 'high') != 'negligible']

        top3    = eligible[:3]
        max_abs = max(abs(f['std_coef']) for f in importance) if importance else 1.0

        for rank, feat in enumerate(top3, start=1):
            score      = abs(feat['std_coef']) / max_abs if max_abs > 0 else 0.0
            conf_info  = confidence.get(feat['key'])
            card = _TakeawayCard(rank, feat, score, conf_info)
            self._takeaways_row.addWidget(card)

        for _ in range(3 - len(top3)):
            self._takeaways_row.addStretch()

    def _rebuild_pie(self, importance: list[dict]):
        if importance:
            self._pie.update_chart(importance)
            self._pie.setVisible(True)
        else:
            self._pie.setVisible(False)

    def _rebuild_rows(self, importance: list[dict], confidence: dict):
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not importance:
            self._rows_layout.addWidget(self._placeholder)
            return

        max_abs = max(abs(f['std_coef']) for f in importance) or 1.0
        for feat in importance:
            conf_info = confidence.get(feat['key'])
            row = _FeatureRow(feat, max_abs, conf_info)
            self._rows_layout.addWidget(row)
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("color: #222; border: none;")
            self._rows_layout.addWidget(sep)