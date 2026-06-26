# ============================================================================
# Drives View — browse, reload and delete stored drives
# ============================================================================

from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QPushButton, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor

from app.feature_loader import load_store, save_store


class DriveCard(QFrame):
    """
    Clickable card representing one stored drive.
    Clicking the card body triggers reloading the drive in the dashboard.
    The ✕ button deletes the drive from the store.
    """

    clicked          = pyqtSignal(str)   # emits drive_path
    delete_requested = pyqtSignal(str)   # emits drive_name

    def __init__(self, record: dict):
        super().__init__()
        self.record     = record
        self.drive_path = record.get('drive_path', '')
        self.drive_name = record.get('drive_name', 'Unknown')
        self._build_ui()

    # ── build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet("""
            DriveCard {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-left: 4px solid #0066cc;
                border-radius: 8px;
            }
            DriveCard:hover {
                background: #f4f8ff;
                border-left-color: #0044aa;
            }
        """)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(88)

        row = QHBoxLayout(self)
        row.setContentsMargins(16, 12, 12, 12)
        row.setSpacing(20)

        # ── left: name + date ──────────────────────────────────────────────
        info = QVBoxLayout()
        info.setSpacing(3)

        name_lbl = QLabel(self.drive_name)
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_lbl.setFont(name_font)
        name_lbl.setStyleSheet("color: #1a1a2e;")
        info.addWidget(name_lbl)

        ts_lbl = QLabel(self._format_timestamp(self.record.get('timestamp', '')))
        ts_lbl.setStyleSheet("color: #888; font-size: 10px;")
        info.addWidget(ts_lbl)

        row.addLayout(info, 2)

        # ── middle: stats ──────────────────────────────────────────────────
        feats = self.record.get('features', {})
        stats = [
            ("Distance",  f"{feats.get('distance_km', 0):.1f} km",                   "#1565c0"),
            ("Duration",  self._format_duration(feats.get('duration_min')),           "#7b1fa2"),
            ("Fuel",      f"{self.record.get('fuel_l100km', 0):.1f} L/100km",         "#2e7d32"),
        ]

        for label_text, value_text, color in stats:
            col = QVBoxLayout()
            col.setSpacing(2)

            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #999; font-size: 10px; letter-spacing: 0.5px;")

            val = QLabel(value_text)
            val.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: bold;"
            )

            col.addWidget(lbl)
            col.addWidget(val)
            row.addLayout(col, 1)

        # ── right: delete button ───────────────────────────────────────────
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(28, 28)
        del_btn.setToolTip("Delete this drive")
        del_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #ddd;
                border-radius: 14px;
                color: #bbb;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #fef2f2;
                border-color: #e74c3c;
                color: #e74c3c;
            }
        """)
        del_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        # use a lambda with default arg capture to avoid late-binding issues
        del_btn.clicked.connect(lambda _, n=self.drive_name: self.delete_requested.emit(n))
        row.addWidget(del_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_timestamp(ts: str) -> str:
        if not ts:
            return '—'
        try:
            dt = datetime.fromisoformat(ts)
            return dt.strftime('%d. %m. %Y  %H:%M')
        except Exception:
            return ts

    @staticmethod
    def _format_duration(duration_min) -> str:
        if duration_min is None or duration_min <= 0:
            return '—'
        mins = int(round(duration_min))
        if mins >= 60:
            return f"{mins // 60}h {mins % 60}min"
        return f"{mins} min"

    # ── events ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drive_path:
            self.clicked.emit(self.drive_path)
        super().mousePressEvent(event)


class DrivesView(QWidget):
    """
    Tab showing all drives saved in fuel_features_store.json.
    Newest drives appear at the top.
    Clicking a card reloads that drive in the dashboard.
    """

    # connected to AutoDNAApplication._load_drive in autodna_app.py
    drive_selected = pyqtSignal(str)   # emits drive_path

    def __init__(self):
        super().__init__()
        self._build_ui()

    # ── build ──────────────────────────────────────────────────────────────

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
        self._root.setSpacing(8)
        scroll.setWidget(self._container)

        # header
        title = QLabel("Drives")
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        title.setFont(f)
        self._root.addWidget(title)

        subtitle = QLabel("All recorded and processed drives. Click a drive to view it.")
        subtitle.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 8px;")
        self._root.addWidget(subtitle)

        # cards container — rebuilt on every refresh()
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(10)
        self._root.addWidget(self._cards_widget)

        self._root.addStretch()

    # ── public API ─────────────────────────────────────────────────────────

    def refresh(self):
        """Reload from store and rebuild the card list. Call after any store change."""
        # clear existing cards
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        records = load_store()

        if not records:
            empty = QLabel("No drives recorded yet.\nLoad a drive folder to get started.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(
                "color: #bbb; font-size: 13px; padding: 48px 0;"
            )
            self._cards_layout.addWidget(empty)
            return

        # newest first
        for record in reversed(records):
            card = DriveCard(record)
            card.clicked.connect(self.drive_selected.emit)
            card.delete_requested.connect(self._on_delete)
            self._cards_layout.addWidget(card)

    # ── internal ───────────────────────────────────────────────────────────

    def _on_delete(self, drive_name: str):
        reply = QMessageBox.question(
            self,
            "Delete Drive",
            f"Remove <b>{drive_name}</b> from stored drives?<br>"
            "The original recording files are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            records = load_store()
            records = [r for r in records if r.get('drive_name') != drive_name]
            save_store(records)
            self.refresh()