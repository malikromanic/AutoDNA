# ============================================================================
# Sidebar Navigation Component
# ============================================================================

#uvoz qt gradnikov
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from pathlib import Path


#seznam strani aplikacije: (id strani, prikazano ime z ikono)
NAV_PAGES = [
    ("Dashboard", "📊  Dashboard"),
]


class Sidebar(QWidget):
    """Left sidebar with branding, drive loading, and navigation."""

    #signala ki ju poslje glavn oknu ob izbiri mape ali menjavi strani
    drive_selected = pyqtSignal(str)
    page_changed   = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(220)
        self._create_ui()
        self._setup_styles()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(4)

        #blagovna znamka
        brand = QLabel("AutoDNA")
        brand_font = QFont()
        brand_font.setPointSize(16)
        brand_font.setBold(True)
        brand.setFont(brand_font)
        brand.setStyleSheet("color: #0066cc; border: none;")
        layout.addWidget(brand)

        sub = QLabel("Driving Analysis Platform")
        sub.setStyleSheet("color: #999999; font-size: 9px; border: none;")
        layout.addWidget(sub)

        layout.addSpacing(12)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(sep)

        layout.addSpacing(8)

        #gumb za odpiranje mape z voznjо
        self.load_btn = QPushButton("📁  Open Drive")
        self.load_btn.setFixedHeight(36)
        self.load_btn.clicked.connect(self._on_load_drive)
        layout.addWidget(self.load_btn)

        #informacija o trenutno nalozeni voznji
        self.drive_info = QLabel("No drive loaded")
        self.drive_info.setWordWrap(True)
        self.drive_info.setStyleSheet(
            "color: #888888; font-size: 10px; border: none; padding: 2px 0;"
        )
        layout.addWidget(self.drive_info)

        layout.addSpacing(12)

        #navigacijski gumbi med stranmi
        nav_label = QLabel("NAVIGATION")
        nav_label.setStyleSheet(
            "color: #aaaaaa; font-size: 9px; font-weight: bold;"
            " letter-spacing: 1px; border: none;"
        )
        layout.addWidget(nav_label)

        layout.addSpacing(4)

        self.page_buttons: dict[str, QPushButton] = {}
        for page_id, page_label in NAV_PAGES:
            btn = QPushButton(page_label)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda checked, p=page_id: self._on_page_click(p))
            self.page_buttons[page_id] = btn
            layout.addWidget(btn)

        #ob zagonu prikazi dashboard
        self.page_buttons["Dashboard"].setChecked(True)

        layout.addStretch()

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(sep2)

        #statistika voznje na dnu stranske vrstice
        stats_label = QLabel("DRIVE STATISTICS")
        stats_label.setStyleSheet(
            "color: #aaaaaa; font-size: 9px; font-weight: bold;"
            " letter-spacing: 1px; border: none;"
        )
        layout.addWidget(stats_label)

        layout.addSpacing(4)

        self.stats_text = QLabel("—")
        self.stats_text.setStyleSheet(
            "font-size: 10px; color: #555555; line-height: 1.6; border: none;"
        )
        self.stats_text.setWordWrap(True)
        layout.addWidget(self.stats_text)

    def _setup_styles(self):
        #skupni stil stranske vrstice in gumbov
        self.setStyleSheet("""
            QWidget {
                background-color: #f4f5f7;
                border-right: 1px solid #d0d0d0;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                color: #333333;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #eaf1fb;
                border-color: #0066cc;
                color: #0066cc;
            }
            QPushButton:checked {
                background-color: #0066cc;
                border-color: #0055aa;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:pressed {
                background-color: #004499;
            }
        """)

    def _on_load_drive(self):
        #odpre dialog za izbiro mape z .BIN in .csv datotekama
        drive_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Drive Data Directory",
            str(Path.home())
        )
        if drive_dir:
            self.drive_selected.emit(drive_dir)
            self.drive_info.setText(f"📂 {Path(drive_dir).name}")

    def _on_page_click(self, page_id: str):
        #odznaci vse gumbe razen izbranega in poslje signal
        for pid, btn in self.page_buttons.items():
            if pid != page_id:
                btn.setChecked(False)
        self.page_buttons[page_id].setChecked(True)
        self.page_changed.emit(page_id)

    def set_drive_statistics(self, distance_km: float, duration_sec: float):
        #posodobi statistiko po uspesni nalozitvi voznje
        duration_min = duration_sec / 60             #pretvorba: sekunde → minute
        avg_speed    = distance_km / max(duration_sec / 3600, 0.001)  #km / (sec/3600) = km/h; 0.001 prepreci deljenje z 0
        self.stats_text.setText(
            f"Distance: {distance_km:.1f} km\n"
            f"Duration: {duration_min:.1f} min\n"
            f"Avg Speed: {avg_speed:.1f} km/h"
        )
