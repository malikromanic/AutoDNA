# ============================================================================
# AutoDNA - Main Application Window
# ============================================================================

import traceback
from pathlib import Path

#uvoz qt gradnikov za glavno okno
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget,
    QStatusBar, QMessageBox, QProgressDialog,
)
from PyQt6.QtGui import QKeySequence, QShortcut, QAction

from app.feature_loader import process_drive_fuel_features, load_store
from app.fuel_model import fit_and_explain, save_stats_cache, load_stats_cache, compute_confidence
from app.data_loader import DriveDataLoader
from app.ui.sidebar import Sidebar
from app.ui.dashboard_view import DashboardView
from app.ui.drives_view import DrivesView
from app.ui.stats_view import StatsView


# ── Main window ───────────────────────────────────────────────────────────────
class AutoDNAApplication(QMainWindow):
    """AutoDNA main window — sidebar + stacked views."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDNA — AI-Powered Driving Analysis")
        self.setGeometry(100, 80, 1440, 880)
        self.setMinimumSize(1100, 700)
        #trenutno nalozeni podatki voznje - none dokler uporabnik ne odpre mape
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
        #osnovna postavitev: stranska vrstica levo, sklop pogledov desno
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.drive_selected.connect(self._on_drive_selected)
        self.sidebar.page_changed.connect(self._on_page_changed)
        root.addWidget(self.sidebar)

        #qstackedwidget preklapi med pogledi brez ponovnega ustvarjanja gradnikov
        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        #ustvari dashboard in ga dodaj v sklad kot edini pogled
        self.dashboard_view = DashboardView()
        self.stack.addWidget(self.dashboard_view)

        self.drives_view = DrivesView()
        self.drives_view.drive_selected.connect(self._load_drive_n)  # reuse existing _load_drive
        self.stack.addWidget(self.drives_view)
        
        self.stats_view = StatsView()
        self.stack.addWidget(self.stats_view)

        self._page_index = {
            "Dashboard": 0,
            "Drives":    1,
            "Stats":    2,
        }
        
        self.stack.setCurrentIndex(0)

        #statusna vrstica na dnu okna za kratka sporocila
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — select a drive folder containing a GPS CSV.")

    def _setup_menu(self):
        #menijska vrstica: file (odpri, ucenje, izhod) in view (menjava strani)
        mb = self.menuBar()

        #meni datoteka: odpri voznju, treniraj modele, izhod
        file_menu = mb.addMenu("File")
        open_act = QAction("Open Drive…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self.sidebar._on_load_drive)
        file_menu.addAction(open_act)

        """train_act = QAction("Train Models…", self)
        train_act.setShortcut(QKeySequence("Ctrl+T"))
        train_act.triggered.connect(self._on_train_models)
        file_menu.addAction(train_act)"""

        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.setShortcut(QKeySequence("Ctrl+Q"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        #meni pogled: bliznajknica za vsako stran (ctrl+1 = dashboard)
        view_menu = mb.addMenu("View")
        for page_id, label in self._page_labels():
            act = QAction(label, self)
            act.triggered.connect(lambda _, p=page_id: self._on_page_changed(p))
            view_menu.addAction(act)

        #meni pomoc: informacije o aplikaciji
        help_menu = mb.addMenu("Help")
        about_act = QAction("About AutoDNA", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(
            lambda: self._on_page_changed("Drives")
        )

    def _page_labels(self):
        return [
            ("Dashboard", "Dashboard"),
            ("Drives",    "Drives"),
            ("Stats",     "Stats"),
        ]

    def _apply_styles(self):
        #globalni qss stil: temna menijska vrstica in statusna vrstica
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
        self._load_drive_n(drive_path)
    
    def _load_drive_n(self, drive_path: str):
        try:
            self.status.showMessage("Loading drive…")
            loader = DriveDataLoader(Path(drive_path))
            self.drive_data = loader.load_drive()
    
            self.dashboard_view.set_drive_data(self.drive_data)
    
            self.sidebar.set_drive_statistics(
                self.drive_data.drive_distance_km,
                self.drive_data.drive_duration_sec,
            )
    
            # ── fuel-regression features from STM .npz, non-fatal if missing ──
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
                # drive still loads/displays normally even if this fails
    
            self.status.showMessage(
                f"Loaded: {self.drive_data.drive_name}  |  "
                f"{self.drive_data.drive_distance_km:.1f} km  |  "
                f"{self.drive_data.drive_duration_sec / 60:.1f} min"
            )
            self._on_page_changed("Dashboard")
    
        except FileNotFoundError as exc:
            self._handle_load_error(
                str(exc),
                "No valid drive recording was found.",
                "The selected folder does not contain a supported drive recording.\n\n"
                "Please make sure the selected folder contains:\n"
                "  • a valid GPS/OBD CSV file\n"
                "  • a supported AutoDNA recording\n\n"
                "Then try again.",
            )
        except ValueError as exc:
            self._handle_load_error(
                str(exc),
                "The drive data could not be read.",
                "The selected file exists but contains invalid or incomplete data.\n\n"
                "Possible causes:\n"
                "  • the CSV file is empty or corrupted\n"
                "  • required columns (latitude, longitude, seconds) are missing\n"
                "  • the GPS data has no valid coordinates\n\n"
                "Try selecting a different drive folder.",
            )
        except Exception:
            self._handle_load_error(
                traceback.format_exc(),
                "An unexpected error occurred.",
                "Something went wrong while loading the drive.\n\n"
                "The error details have been saved to autodna_crash.log.\n"
                "You can try selecting a different drive folder.",
            )

    def _handle_load_error(self, detail_text: str, title_line: str, body: str):
        #shrani traceback za debug, prikazi uporabniku prijazno sporocilo
        tb = detail_text if '\n' in detail_text else traceback.format_exc()
        print(tb)
        log_path = Path(__file__).parent.parent / "autodna_crash.log"
        log_path.write_text(tb)
        self.status.showMessage("Load error — see autodna_crash.log")

        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setWindowTitle("Drive Load Error")
        dlg.setText(f"<b>{title_line}</b>")
        dlg.setInformativeText(body)
        #ce uporabnik zahteva podrobnosti prikazi celoten traceback
        dlg.setDetailedText(tb)
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
        dlg.exec()

    def _on_page_changed(self, page_name: str):
        #preklopi aktivni pogled in oznaci ustrezen gumb v stranski vrstici
        idx = self._page_index.get(page_name)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
            for pid, btn in self.sidebar.page_buttons.items():
                btn.setChecked(pid == page_name)
            # refresh drives list every time the tab is opened
            if page_name == "Drives":
                self.drives_view.refresh()

    def _show_about(self):
        #dialog o aplikaciji z osnovnimi tehnicnimi podatki
        QMessageBox.about(
            self, "About AutoDNA",
            "<b>AutoDNA</b> — AI-Powered Driving Analysis<br><br>"
            "Detection: GPS heading change (turns) · GPS altitude change (hills)<br>"
            "GPS: OBD2 CSV → deduplicated route<br>"
            "Built with PyQt6 + Folium<br><br>"
            "© 2026 AutoDNA Project"
        )
