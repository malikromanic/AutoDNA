# ============================================================================
# AutoDNA - Main Application Window  (GPS-only: turns + DEM hills, no AI)
# ============================================================================

import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget,
    QStatusBar, QMessageBox,
)
from PyQt6.QtGui import QKeySequence, QShortcut, QAction

from app.data_loader import DriveDataLoader
from app.ui.sidebar import Sidebar
from app.ui.dashboard_view import DashboardView
from app.ui.elevation_view import ElevationView


class AutoDNAApplication(QMainWindow):
    """AutoDNA main window — sidebar + stacked views."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDNA — GPS Driving Analysis")
        self.setGeometry(100, 80, 1440, 880)
        self.setMinimumSize(1100, 700)
        self.drive_data = None

        self._create_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._apply_styles()

    # ── UI construction ───────────────────────────────────────────────────────
    def _create_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.drive_selected.connect(self._on_drive_selected)
        self.sidebar.page_changed.connect(self._on_page_changed)
        root.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        self.dashboard_view = DashboardView()
        self.elevation_view = ElevationView()

        for view in (self.dashboard_view, self.elevation_view):
            self.stack.addWidget(view)

        self._page_index = {
            "Dashboard": 0,
            "Elevation": 1,
        }
        self.stack.setCurrentIndex(0)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — select a drive folder containing a GPS CSV.")

    def _setup_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        open_act = QAction("Open Drive…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self.sidebar._on_load_drive)
        file_menu.addAction(open_act)

        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.setShortcut(QKeySequence("Ctrl+Q"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        view_menu = mb.addMenu("View")
        for page_id, label in self._page_labels():
            act = QAction(label, self)
            act.triggered.connect(lambda _, p=page_id: self._on_page_changed(p))
            view_menu.addAction(act)

        help_menu = mb.addMenu("Help")
        about_act = QAction("About AutoDNA", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _setup_shortcuts(self):
        pairs = {"Ctrl+1": "Dashboard", "Ctrl+2": "Elevation"}
        for key, page in pairs.items():
            QShortcut(QKeySequence(key), self).activated.connect(
                lambda p=page: self._on_page_changed(p)
            )

    def _page_labels(self):
        return [("Dashboard", "Dashboard"), ("Elevation", "Elevation")]

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #f0f2f5; }
            QMenuBar {
                background: #2c3e50; color: #ecf0f1;
                padding: 2px 4px; font-size: 12px;
            }
            QMenuBar::item:selected { background: #34495e; border-radius: 4px; }
            QMenu { background: #fff; border: 1px solid #dee2e6; }
            QMenu::item:selected { background: #e8f0fb; color: #0066cc; }
            QStatusBar { background: #2c3e50; color: #bdc3c7; font-size: 11px; }
        """)

    # ── Events ────────────────────────────────────────────────────────────────
    def _on_drive_selected(self, drive_path: str):
        self._load_drive(drive_path)

    def _load_drive(self, drive_path: str):
        try:
            self.status.showMessage("Loading drive… (looking up DEM elevation)")
            loader = DriveDataLoader(Path(drive_path))
            self.drive_data = loader.load_drive(
                progress_cb=lambda msg: self.status.showMessage(msg)
            )

            for view in (self.dashboard_view, self.elevation_view):
                view.set_drive_data(self.drive_data)

            events = self._count_events()
            self.sidebar.set_drive_statistics(
                self.drive_data.drive_distance_km,
                self.drive_data.drive_duration_sec,
                events,
            )
            d = self.drive_data
            self.status.showMessage(
                f"Loaded: {d.drive_name}  |  "
                f"{d.drive_distance_km:.1f} km  |  "
                f"{d.drive_duration_sec / 60:.1f} min  |  "
                f"{events} events  |  "
                f"+{d.elevation_gain_m:.0f}/-{d.elevation_loss_m:.0f} m  |  "
                f"elevation: {d.elevation_source}"
            )
            self._on_page_changed("Dashboard")

        except Exception:
            tb = traceback.format_exc()
            print(tb)
            (Path(__file__).parent.parent / "autodna_crash.log").write_text(tb)
            self.status.showMessage("Load error — see autodna_crash.log")
            QMessageBox.critical(
                self, "Drive Load Error",
                f"{tb[-600:]}\n\nFull trace in autodna_crash.log"
            )

    def _on_page_changed(self, page_name: str):
        idx = self._page_index.get(page_name)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
            for pid, btn in self.sidebar.page_buttons.items():
                btn.setChecked(pid == page_name)

    def _count_events(self) -> int:
        if not self.drive_data:
            return 0
        from app.ui.dashboard_view import _count_segments
        return (_count_segments(self.drive_data.turn_preds) +
                _count_segments(self.drive_data.hill_preds))

    def _show_about(self):
        QMessageBox.about(
            self, "About AutoDNA",
            "<b>AutoDNA</b> — GPS Driving Analysis<br><br>"
            "Turns: GPS heading change<br>"
            "Hills: DEM road grade (device GPS altitude ignored)<br>"
            "Elevation: EU-DEM 25 m (Europe) / Copernicus DEM, online<br>"
            "Built with PyQt6 + Folium<br><br>"
            "© 2026 AutoDNA Project"
        )
