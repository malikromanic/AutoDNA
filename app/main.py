# ============================================================================
# AutoDNA - AI-Powered Driving Analysis Platform
# Dashboard Application (PyQt6)
# ============================================================================

import sys
import os
import traceback
from pathlib import Path

# Add repo root to path so `app.*` and `stm32.*` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# QWebEngineView MUST be imported before QApplication is instantiated
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
from PyQt6.QtWidgets import QApplication
from app.autodna_app import AutoDNAApplication


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
