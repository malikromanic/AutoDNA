# ============================================================================
# AutoDNA - Main Application Window
# ============================================================================

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QStatusBar, QMessageBox
)
from PyQt6.QtGui import QKeySequence, QShortcut, QAction
from pathlib import Path

from app.data_loader import DriveDataLoader
from app.ui.sidebar import Sidebar
from app.ui.dashboard_view import DashboardView
from app.ui.route_replay import RouteReplayWidget
from app.ui.ai_analysis_view import AIAnalysisView
from app.ui.recommendations_view import RecommendationsView
from app.ui.technical_view import TechnicalView


class AutoDNAApplication(QMainWindow):
    """Main application window for AutoDNA."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDNA - AI-Powered Driving Analysis")
        self.setGeometry(100, 80, 1440, 880)
        self.setMinimumSize(1100, 700)

        self.drive_data = None

        self._create_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._apply_global_styles()

    # ----------------------------------------------------------------- UI init
    def _create_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.drive_selected.connect(self._on_drive_selected)
        self.sidebar.page_changed.connect(self._on_page_changed)
        root_layout.addWidget(self.sidebar)

        # Main content stacked widget
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack)

        # ---- Views (order matches NAV_PAGES in sidebar) ----
        self.dashboard_view = DashboardView()
        self.replay_view = RouteReplayWidget()
        self.analysis_view = AIAnalysisView()
        self.recommendations_view = RecommendationsView()
        self.technical_view = TechnicalView()

        self.stack.addWidget(self.dashboard_view)       # index 0
        self.stack.addWidget(self.replay_view)          # index 1
        self.stack.addWidget(self.analysis_view)        # index 2
        self.stack.addWidget(self.recommendations_view) # index 3
        self.stack.addWidget(self.technical_view)       # index 4

        # Page index map matches NAV_PAGES order
        self._page_index = {
            "Dashboard":       0,
            "Route Replay":    1,
            "AI Analysis":     2,
            "Recommendations": 3,
            "Technical":       4,
        }

        # Show dashboard by default
        self.stack.setCurrentIndex(0)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — open a drive directory to begin.")

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Drive…", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.sidebar._on_load_drive)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        for page_id, label in [
            ("Dashboard",       "Dashboard"),
            ("Route Replay",    "Route Replay"),
            ("AI Analysis",     "AI Analysis"),
            ("Recommendations", "Recommendations"),
            ("Technical",       "Technical"),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, p=page_id: self._on_page_changed(p))
            view_menu.addAction(act)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About AutoDNA", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        shortcuts = {
            "Ctrl+1": "Dashboard",
            "Ctrl+2": "Route Replay",
            "Ctrl+3": "AI Analysis",
            "Ctrl+4": "Recommendations",
            "Ctrl+5": "Technical",
        }
        for key, page in shortcuts.items():
            QShortcut(QKeySequence(key), self).activated.connect(
                lambda p=page: self._on_page_changed(p)
            )

    def _apply_global_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QMenuBar {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 2px 4px;
                font-size: 12px;
            }
            QMenuBar::item:selected {
                background-color: #34495e;
                border-radius: 4px;
            }
            QMenu {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
            }
            QMenu::item:selected {
                background-color: #e8f0fb;
                color: #0066cc;
            }
            QStatusBar {
                background-color: #2c3e50;
                color: #bdc3c7;
                font-size: 11px;
            }
        """)

    # --------------------------------------------------------------- Events
    def _on_drive_selected(self, drive_path: str):
        try:
            self.status.showMessage("Loading drive data…")
            loader = DriveDataLoader(Path(drive_path))
            self.drive_data = loader.load_drive()

            # Push data to all views
            self.dashboard_view.set_drive_data(self.drive_data)
            self.replay_view.set_drive_data(self.drive_data)
            self.analysis_view.set_drive_data(self.drive_data)
            self.recommendations_view.set_drive_data(self.drive_data)
            self.technical_view.set_drive_data(self.drive_data)

            # Update sidebar stats
            events = self._count_events()
            self.sidebar.set_drive_statistics(
                self.drive_data.drive_distance_km,
                self.drive_data.drive_duration_sec,
                events
            )

            self.status.showMessage(
                f"Loaded: {self.drive_data.drive_name}  |  "
                f"{self.drive_data.drive_distance_km:.1f} km  |  "
                f"{self.drive_data.drive_duration_sec / 60:.1f} min  |  "
                f"{events} events detected"
            )

            # Navigate to dashboard
            self._on_page_changed("Dashboard")

        except Exception as e:
            import traceback
            full_tb = traceback.format_exc()
            print(full_tb)  # visible in console
            # Write crash log next to project root
            log_path = Path(__file__).parent.parent / "autodna_crash.log"
            log_path.write_text(full_tb)
            self.status.showMessage(f"Error: {e}  (see autodna_crash.log)")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load Error", f"{e}\n\nSee autodna_crash.log for details.")

    def _on_page_changed(self, page_name: str):
        idx = self._page_index.get(page_name)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
            # Update button checked state directly — do NOT call _on_page_click
            # because that emits page_changed again, causing infinite recursion.
            for pid, btn in self.sidebar.page_buttons.items():
                btn.setChecked(pid == page_name)

    def _count_events(self) -> int:
        if not self.drive_data:
            return 0
        bilstm = self.drive_data.bilstm_preds
        events = 0
        for i in range(1, len(bilstm)):
            if bilstm[i, 0] > 0.5 and bilstm[i - 1, 0] <= 0.5:
                events += 1
            if bilstm[i, 3] > 0.5 and bilstm[i - 1, 3] <= 0.5:
                events += 1
        return events

    def _show_about(self):
        QMessageBox.about(
            self,
            "About AutoDNA",
            "<b>AutoDNA</b> — AI-Powered Driving Analysis<br><br>"
            "Models: BiLSTM + XGBoost ensemble<br>"
            "Built with PyQt6 + Folium<br><br>"
            "© 2026 AutoDNA Project"
        )
