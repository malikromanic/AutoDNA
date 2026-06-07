# ============================================================================
# Dashboard View - Drive statistics and route map
# ============================================================================

#uvoz qt gradnikov za postavitev vmesnika
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np

from app.ui.map_widget import MapWidget

#minimalna hitrost v km/h da se tocka steje kot "v gibanju" pri izracunu povprecne hitrosti
_MOVING_KMH_THRESHOLD = 2.0


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
    """Dashboard - drive statistics and route map."""

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

        title = QLabel("Dashboard")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Drive statistics and route overview")
        subtitle.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(subtitle)

        #zgornja vrsta: tri metricne kartice (razdalja, trajanje, gorivo)
        cards_top = QHBoxLayout()
        cards_top.setSpacing(12)

        self.distance_card = MetricCard("Distance",         "—", "km",      "#0066cc")
        self.duration_card = MetricCard("Duration",         "—", "minutes", "#8e44ad")
        self.fuel_card     = MetricCard("Fuel Consumption", "—", "L/100km", "#27ae60")

        for card in [self.distance_card, self.duration_card, self.fuel_card]:
            cards_top.addWidget(card)

        layout.addLayout(cards_top)

        #spodnja vrsta: karta (levo) in statistika (desno)
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
        map_title.setStyleSheet(
            "font-weight: bold; font-size: 13px;"
            " padding: 10px 14px 6px 14px; color: #333;"
        )
        map_frame_layout.addWidget(map_title)

        self.map_widget = MapWidget()
        self.map_widget.setMinimumHeight(380)
        map_frame_layout.addWidget(self.map_widget)

        content_row.addWidget(map_frame, 3)

        #desna ploscica s statistiko voznje
        self.stats_frame = self._build_stats_panel()
        content_row.addWidget(self.stats_frame, 2)

        layout.addLayout(content_row)
        layout.addStretch()

    def _build_stats_panel(self) -> QFrame:
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
        layout.setSpacing(10)

        title = QLabel("Drive Summary")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self._summary_label = QLabel("Load a drive to see statistics.")
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("color: #555555; font-size: 11px;")
        layout.addWidget(self._summary_label)

        #razdelek: hitrost
        layout.addWidget(self._section_sep())
        speed_title = QLabel("Speed")
        speed_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #1565c0;")
        layout.addWidget(speed_title)

        self._speed_grid = QGridLayout()
        self._speed_grid.setSpacing(4)
        layout.addLayout(self._speed_grid)
        self._speed_rows: dict[str, QLabel] = {}
        for key in ["Max Speed", "Avg Speed (moving)", "Avg Speed (overall)"]:
            self._add_stat_row(self._speed_grid, self._speed_rows, key, "#1565c0")

        #razdelek: gorivo
        layout.addWidget(self._section_sep())
        fuel_title = QLabel("Fuel")
        fuel_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #2e7d32;")
        layout.addWidget(fuel_title)

        self._fuel_grid = QGridLayout()
        self._fuel_grid.setSpacing(4)
        layout.addLayout(self._fuel_grid)
        self._fuel_rows: dict[str, QLabel] = {}
        for key in ["Total Consumed", "Avg Consumption"]:
            self._add_stat_row(self._fuel_grid, self._fuel_rows, key, "#2e7d32")

        #razdelek: podatki o snemanju
        layout.addWidget(self._section_sep())
        rec_title = QLabel("Recording")
        rec_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #555;")
        layout.addWidget(rec_title)

        self._rec_grid = QGridLayout()
        self._rec_grid.setSpacing(4)
        layout.addLayout(self._rec_grid)
        self._rec_rows: dict[str, QLabel] = {}
        for key in ["GPS Points", "IMU Samples", "IMU Duration"]:
            self._add_stat_row(self._rec_grid, self._rec_rows, key, "#555")

        layout.addStretch()
        return frame

    def _section_sep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e0e0e0;")
        return sep

    def _add_stat_row(self, grid: QGridLayout, store: dict, key: str, color: str):
        r = len(store)
        k = QLabel(f"{key}:")
        k.setStyleSheet("font-size: 11px; color: #555;")
        v = QLabel("—")
        v.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {color};")
        grid.addWidget(k, r, 0)
        grid.addWidget(v, r, 1)
        store[key] = v

    def set_drive_data(self, drive_data):
        self.drive_data = drive_data
        self._update_metric_cards()
        self.map_widget.set_drive_data(drive_data)
        self._update_stats_panel()

    def _update_metric_cards(self):
        d = self.drive_data
        #razdalja [km], trajanje [min] = sec/60, gorivo [l/100km] - vrednosti ze v DriveData
        self.distance_card.set_value(f"{d.drive_distance_km:.1f}")
        self.duration_card.set_value(f"{d.drive_duration_sec / 60:.1f}")
        self.fuel_card.set_value(f"{d.avg_fuel_l100km:.1f}")

    def _update_stats_panel(self):
        d = self.drive_data

        #kratki povzetek na vrhu
        self._summary_label.setText(
            f"Drive: <b>{d.drive_name}</b><br>"
            f"{d.drive_distance_km:.2f} km &nbsp;|&nbsp; "
            f"{d.drive_duration_sec / 60:.1f} min"
        )
        self._summary_label.setTextFormat(Qt.TextFormat.RichText)

        #hitrost - iz gps_speed polja (km/h)
        spd = d.gps_speed
        moving = spd[spd > _MOVING_KMH_THRESHOLD]
        max_spd = float(spd.max()) if len(spd) else 0.0
        avg_moving = float(moving.mean()) if len(moving) else 0.0
        avg_overall = d.drive_distance_km / max(d.drive_duration_sec / 3600.0, 1e-6)  #km / (sec/3600) = km/h; 1e-6 prepreci deljenje z 0

        self._speed_rows["Max Speed"].setText(f"{max_spd:.0f} km/h")
        self._speed_rows["Avg Speed (moving)"].setText(f"{avg_moving:.0f} km/h")
        self._speed_rows["Avg Speed (overall)"].setText(f"{avg_overall:.0f} km/h")

        #gorivo
        self._fuel_rows["Total Consumed"].setText(f"{d.fuel_consumption_l:.2f} L")
        self._fuel_rows["Avg Consumption"].setText(f"{d.avg_fuel_l100km:.1f} L/100km")

        #snemanje
        self._rec_rows["GPS Points"].setText(str(len(d.gps_lat)))
        self._rec_rows["IMU Samples"].setText(str(len(d.imu_timestamps)))
        self._rec_rows["IMU Duration"].setText(f"{d.imu_timestamps[-1]:.0f} s")
