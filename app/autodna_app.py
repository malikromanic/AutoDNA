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
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from app.ai_pipeline import models_exist, train_models
from app.data_loader import DriveDataLoader
from app.ui.sidebar import Sidebar
from app.ui.dashboard_view import DashboardView


#ucenje xgboost modelov v loceni niti da ne zamrzne gui
class TrainThread(QThread):
    progress = pyqtSignal(str, int)  #sporocilo in odstotek napredka
    finished = pyqtSignal(str)       #sporocilo ob uspesnem zakljucku
    error    = pyqtSignal(str)       #sporocilo ob napaki

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
        #trenutno nalozeni podatki voznje - none dokler uporabnik ne odpre mape
        self.drive_data = None

        self._create_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._apply_styles()

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

        #slovar: ime_strani → indeks v skladu (zdaj samo dashboard)
        self._page_index = {
            "Dashboard": 0,
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

        train_act = QAction("Train Models…", self)
        train_act.setShortcut(QKeySequence("Ctrl+T"))
        train_act.triggered.connect(self._on_train_models)
        file_menu.addAction(train_act)

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
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(
            lambda: self._on_page_changed("Dashboard")
        )

    def _page_labels(self):
        #seznam (id_strani, ime_strani) za gradnjo menija view
        return [("Dashboard", "Dashboard")]

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
        self._load_drive(drive_path)

    def _load_drive(self, drive_path: str):
        #nalozi voznju in posodobi vse poglede z novimi podatki
        try:
            self.status.showMessage("Loading drive…")
            #ustvari naloznik in pozeni gps pipeline
            loader = DriveDataLoader(Path(drive_path))
            self.drive_data = loader.load_drive()

            #posodobi dashboard z novimi podatki voznje
            self.dashboard_view.set_drive_data(self.drive_data)

            #posodobi statistiko v stranski vrstici
            self.sidebar.set_drive_statistics(
                self.drive_data.drive_distance_km,
                self.drive_data.drive_duration_sec,
            )
            #prikazi povzetek v statusni vrstici
            self.status.showMessage(
                f"Loaded: {self.drive_data.drive_name}  |  "
                f"{self.drive_data.drive_distance_km:.1f} km  |  "
                f"{self.drive_data.drive_duration_sec / 60:.1f} min"
            )
            #preklopi na dashboard takoj po nalozitvi
            self._on_page_changed("Dashboard")

        except Exception:
            #zajemi celoten traceback in ga shrani v log datoteko za odpravljanje napak
            tb = traceback.format_exc()
            print(tb)
            (Path(__file__).parent.parent / "autodna_crash.log").write_text(tb)
            self.status.showMessage("Load error — see autodna_crash.log")
            QMessageBox.critical(
                self, "Drive Load Error",
                f"{tb[-600:]}\n\nFull trace in autodna_crash.log"
            )

    def _on_train_models(self, callback=None):
        #pozeni ucenje xgboost modelov v ozadje niti z napredkovnim dialogom
        #callback je opcijsen - klice se po uspesnem ucenju (npr. nalozi voznju takoj zatem)

        #modalni dialog blokira interakcijo z aplikacijo med ucenjem
        dlg = QProgressDialog(
            "Training XGBoost models on all available drives…",
            "Cancel", 0, 100, self
        )
        dlg.setWindowTitle("AutoDNA — Training Models")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()

        #ustvari delavsko nit - ucenje blokira gui ce tece v glavni niti
        self._train_thread = TrainThread()

        def on_progress(msg, pct):
            #posodobi dialog z vmesnim sporocilom in odstotkom napredka
            dlg.setLabelText(msg)
            dlg.setValue(pct)

        def on_done(msg):
            #ucenje uspesno - zapri dialog, prikazi sporocilo in opcijsko pozeni callback
            dlg.close()
            self.status.showMessage(msg)
            QMessageBox.information(self, "Training Complete", msg)
            if callback:
                callback()

        def on_err(msg):
            #napaka med ucenjem - zapri dialog in prikazi sporocilo o napaki
            dlg.close()
            self.status.showMessage(f"Training error: {msg}")
            QMessageBox.critical(self, "Training Error", msg)

        #prikljuci signale niti na ustrezne upravljavce
        self._train_thread.progress.connect(on_progress)
        self._train_thread.finished.connect(on_done)
        self._train_thread.error.connect(on_err)
        self._train_thread.start()

    def _on_page_changed(self, page_name: str):
        #preklopi aktivni pogled in oznaci ustrezen gumb v stranski vrstici
        idx = self._page_index.get(page_name)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
            for pid, btn in self.sidebar.page_buttons.items():
                btn.setChecked(pid == page_name)

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
