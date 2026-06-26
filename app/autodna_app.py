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

from AutoDNA.app.data_loader import DriveDataLoader
from AutoDNA.app.feature_loader import process_drive_fuel_features, load_store
from AutoDNA.app.fuel_model import fit_and_explain, save_stats_cache, load_stats_cache

from AutoDNA.app.ui.sidebar import Sidebar
from AutoDNA.app.ui.dashboard_view import DashboardView
from AutoDNA.app.ui.elevation_view import ElevationView
from AutoDNA.app.ui.drives_view import DrivesView
from AutoDNA.app.ui.stats_view import StatsView

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
        
        cached = load_stats_cache()
        if cached:
            self.stats_view.set_result(cached)
    
        self.status.showMessage("Ready — select a drive folder containing a GPS CSV.")

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
        self.stats_view = StatsView()
        self.drives_view = DrivesView()
        self.drives_view.drive_selected.connect(self._load_drive)
        
        self.stack.addWidget(self.dashboard_view)
        self.stack.addWidget(self.elevation_view)
        self.stack.addWidget(self.drives_view)
        self.stack.addWidget(self.stats_view)


        self._page_index = {
            "Dashboard": 0,
            "Elevation": 1,
            "Drives":    2,
            "Stats":    3,
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
        pairs = {"Ctrl+1": "Dashboard", "Ctrl+2": "Elevation", "Ctrl+3": "Drives", "Ctrl+4": "Stats"}
        for key, page in pairs.items():
            QShortcut(QKeySequence(key), self).activated.connect(
                lambda p=page: self._on_page_changed(p)
            )

    def _page_labels(self):
        return [
                ("Dashboard", "Dashboard"), 
                ("Elevation", "Elevation"),
                ("Drives",    "Drives"),
                ("Stats",     "Stats"),
                ]


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


    """def _load_drive(self, drive_path: str):
        try:
            self.status.showMessage("Loading drive… (looking up DEM elevation)")
            loader = DriveDataLoader(Path(drive_path))
            self.drive_data = loader.load_drive(
                progress_cb=lambda msg: self.status.showMessage(msg)
            )

            for view in (self.dashboard_view, self.elevation_view):
                view.set_drive_data(self.drive_data)

            events = self._count_events()
        self._load_drive_n(drive_path)"""
    
    
    def _load_drive(self, drive_path: str):
        try:
            self.status.showMessage("Loading drive… (looking up DEM elevation)")
            loader = DriveDataLoader(Path(drive_path))
            #self.drive_data = loader.load_drive()
            self.drive_data = loader.load_drive(
                progress_cb=lambda msg: self.status.showMessage(msg)
            )
            #self.dashboard_view.set_drive_data(self.drive_data)
            
            for view in (self.dashboard_view, self.elevation_view):
                view.set_drive_data(self.drive_data)
                
            
            events = self._count_events()
            d = self.drive_data

            self.sidebar.set_drive_statistics(
                self.drive_data.drive_distance_km,
                self.drive_data.drive_duration_sec,
                events,
            )
    
            self.status.showMessage(
                f"Loaded: {d.drive_name}  |  "
                f"{d.drive_distance_km:.1f} km  |  "
                f"{d.drive_duration_sec / 60:.1f} min  |  "
                f"{events} events  |  "
                f"+{d.elevation_gain_m:.0f}/-{d.elevation_loss_m:.0f} m  |  "
                f"elevation: {d.elevation_source}"
            )
            self._on_page_changed("Dashboard")
            
            try:
                features = process_drive_fuel_features(Path(drive_path), self.drive_data)
                records  = load_store()
                result = fit_and_explain(records, features)
                self.stats_view.set_result(result)
                save_stats_cache(result)
            except FileNotFoundError as e:
                print(f"[AutoDNA] Fuel features skipped: {e}")
            except Exception:
                print(f"[AutoDNA] Fuel feature extraction failed:\n{traceback.format_exc()}")
                
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
            # refresh drives list every time the tab is opened
            if page_name == "Drives":
                self.drives_view.refresh()

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
