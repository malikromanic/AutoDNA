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
        log_path = Path(__file__).resolve().parent.parent / "autodna_crash.log"
        with open(log_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"CRASH: {e}")
        traceback.print_exc()
        input("Press Enter to close...")
