# ============================================================================
# AutoDNA - Main Application Window
# ============================================================================

import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget,
    QStatusBar, QMessageBox, QProgressDialog,
)
from PyQt6.QtGui import QKeySequence, QShortcut, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from app.ai_pipeline import models_exist, train_models
from app.data_loader import DriveDataLoader
from app.ui.sidebar import Sidebar
from app.ui.dashboard_view import DashboardView
from app.ui.ai_analysis_view import AIAnalysisView


# ── Background training thread ────────────────────────────────────────────────
class TrainThread(QThread):
    progress = pyqtSignal(str, int)    # (message, percent)
    finished = pyqtSignal(str)         # success message
    error    = pyqtSignal(str)         # error message

    def run(self):
        try:
            n, *_ = train_models(progress_cb=self.progress.emit)
            self.finished.emit(f"Models trained on {n} windows. Ready to load drives.")
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main window ───────────────────────────────────────────────────────────────
class AutoDNAApplication(QMainWindow):
    """AutoDNA main window — sidebar + stacked views."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDNA — AI-Powered Driving Analysis")
        self.setGeometry(100, 80, 1440, 880)
        self.setMinimumSize(1100, 700)
        self.drive_data = None

        self._create_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._apply_styles()

        # Check for trained models on startup
        if not models_exist():
            self.status.showMessage(
                "Models not trained. Use File → Train Models (or open a drive)."
            )

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
        self.analysis_view  = AIAnalysisView()

        for view in (self.dashboard_view, self.analysis_view):
            self.stack.addWidget(view)

        self._page_index = {
            "Dashboard":   0,
            "AI Analysis": 1,
        }
        self.stack.setCurrentIndex(0)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — select a drive folder containing BIN + GPS CSV.")

    def _setup_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        open_act = QAction("Open Drive…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self.sidebar._on_load_drive)
        file_menu.addAction(open_act)

        train_act = QAction("Train Models…", self)
        train_act.setShortcut(QKeySequence("Ctrl+T"))
        train_act.triggered.connect(self._on_train_models)
        file_menu.addAction(train_act)

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
        pairs = {"Ctrl+1": "Dashboard", "Ctrl+2": "AI Analysis"}
        for key, page in pairs.items():
            QShortcut(QKeySequence(key), self).activated.connect(
                lambda p=page: self._on_page_changed(p)
            )

    def _page_labels(self):
        return [("Dashboard", "Dashboard"), ("AI Analysis", "AI Analysis")]

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
        # Require models before loading
        if not models_exist():
            reply = QMessageBox.question(
                self, "Models Not Trained",
                "XGBoost models are not trained yet.\n\n"
                "Train them now? (takes ~1–2 minutes on first run)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_train_models(callback=lambda: self._load_drive(drive_path))
            return

        self._load_drive(drive_path)

    def _load_drive(self, drive_path: str):
        try:
            self.status.showMessage("Loading drive…")
            loader = DriveDataLoader(Path(drive_path))
            self.drive_data = loader.load_drive()

            for view in (self.dashboard_view, self.analysis_view):
                view.set_drive_data(self.drive_data)

            events = self._count_events()
            self.sidebar.set_drive_statistics(
                self.drive_data.drive_distance_km,
                self.drive_data.drive_duration_sec,
                events,
            )
            self.status.showMessage(
                f"Loaded: {self.drive_data.drive_name}  |  "
                f"{self.drive_data.drive_distance_km:.1f} km  |  "
                f"{self.drive_data.drive_duration_sec / 60:.1f} min  |  "
                f"{events} events  |  "
                f"{self.drive_data.window_size}-sample windows @ {self.drive_data.target_fs} Hz"
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

    def _on_train_models(self, callback=None):
        """Run model training in a background thread with a progress dialog."""
        dlg = QProgressDialog(
            "Training XGBoost models on all available drives…",
            "Cancel", 0, 100, self
        )
        dlg.setWindowTitle("AutoDNA — Training Models")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()

        self._train_thread = TrainThread()

        def on_progress(msg, pct):
            dlg.setLabelText(msg)
            dlg.setValue(pct)

        def on_done(msg):
            dlg.close()
            self.status.showMessage(msg)
            QMessageBox.information(self, "Training Complete", msg)
            if callback:
                callback()

        def on_err(msg):
            dlg.close()
            self.status.showMessage(f"Training error: {msg}")
            QMessageBox.critical(self, "Training Error", msg)

        self._train_thread.progress.connect(on_progress)
        self._train_thread.finished.connect(on_done)
        self._train_thread.error.connect(on_err)
        self._train_thread.start()

    def _on_page_changed(self, page_name: str):
        idx = self._page_index.get(page_name)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
            for pid, btn in self.sidebar.page_buttons.items():
                btn.setChecked(pid == page_name)

    def _count_events(self) -> int:
        if not self.drive_data:
            return 0
        return int((self.drive_data.xgb_turn_preds != 0).sum() +
                   (self.drive_data.xgb_hill_preds  != 0).sum())

    def _show_about(self):
        QMessageBox.about(
            self, "About AutoDNA",
            "<b>AutoDNA</b> — AI-Powered Driving Analysis<br><br>"
            "Models: XGBoost (turn + hill)<br>"
            "IMU: STM32 BIN → 50 Hz → 100-sample windows → 38 features<br>"
            "GPS: OBD2 CSV → deduplicated route<br>"
            "Built with PyQt6 + Folium<br><br>"
            "© 2026 AutoDNA Project"
        )
