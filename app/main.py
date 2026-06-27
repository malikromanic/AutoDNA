# ============================================================================
# AutoDNA - AI-Powered Driving Analysis Platform
# Dashboard Application (PyQt6)
# ============================================================================

import sys
import traceback
import os
from pathlib import Path

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"

#sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# QWebEngineView MUST be imported before QApplication is instantiated
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
from PyQt6.QtWidgets import QApplication
from AutoDNA.app.autodna_app import AutoDNAApplication


def main():
    """Launch the AutoDNA GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = AutoDNAApplication()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        #shrani traceback v log datoteko za debug
        #tukaj sem dodal ADNA-54
        log_path = Path(__file__).resolve().parent.parent / "autodna_crash.log"
        tb = traceback.format_exc()
        with open(log_path, "w") as f:
            f.write(tb)

        #poskusi prikazati uporabniku prijazno sporocilo v gui oknu
        #tukaj sem dodal ADNA-54
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(sys.argv)
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Icon.Critical)
            dlg.setWindowTitle("AutoDNA — Startup Error")
            dlg.setText("<b>The application failed to start.</b>")
            dlg.setInformativeText(
                "AutoDNA could not launch due to an initialization error.\n\n"
                "Possible causes:\n"
                "  • missing Python packages (run: pip install -r requirements.txt)\n"
                "  • incompatible Python version\n"
                "  • corrupted installation\n\n"
                "Error details have been saved to autodna_crash.log."
            )
            dlg.setDetailedText(tb)
            dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
            dlg.exec()
        except Exception:
            #gui ni na voljo - izpisi v terminal
            print(f"\nAutoDNA failed to start: {e}")
            print(f"Full traceback saved to: {log_path}")
            traceback.print_exc()
            input("Press Enter to close...")
