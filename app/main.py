# ============================================================================
# AutoDNA - AI-Powered Driving Analysis Platform
# Dashboard Application (PyQt6)
# ============================================================================

import sys
import traceback
import os
from pathlib import Path

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --disable-gpu --disable-software-rasterizer --disable-gpu-compositing"

if getattr(sys, 'frozen', False):
    exe_dir = Path(sys.executable).parent
    os.chdir(exe_dir)
    internal = exe_dir / '_internal'
    
    os.environ["QTWEBENGINEPROCESS_PATH"] = str(
        internal / "PyQt6" / "Qt6" / "bin" / "QtWebEngineProcess.exe"
    )
    # add _internal to PATH so Qt finds the DLLs there
    os.environ["PATH"] = str(internal) + os.pathsep + os.environ.get("PATH", "")
    
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QLibraryInfo
from AutoDNA.app.autodna_app import AutoDNAApplication
from AutoDNA.app.get_path import get_data_dir


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    print(f"Qt prefix:      {QLibraryInfo.path(QLibraryInfo.LibraryPath.PrefixPath)}")
    print(f"Qt data:        {QLibraryInfo.path(QLibraryInfo.LibraryPath.DataPath)}")
    print(f"Qt plugins:     {QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)}")
    print(f"Qt translations:{QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)}")

    window = AutoDNAApplication()
    window.show()
    sys.exit(app.exec())
    
"""import sys
import traceback
import os
from pathlib import Path

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --disable-gpu --disable-software-rasterizer --disable-gpu-compositing"

if getattr(sys, 'frozen', False):
    exe_dir = Path(sys.executable).parent
    os.chdir(exe_dir)
    
    # tell Qt where its resources are
    internal = exe_dir / '_internal'
    os.environ["QTWEBENGINEPROCESS_PATH"] = str(
        internal / "PyQt6" / "Qt6" / "bin" / "QtWebEngineProcess.exe"
    )
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
        internal / "PyQt6" / "Qt6" / "plugins" / "platforms"
    )
    
    # critical — tell WebEngine where its resources are
    from PyQt6.QtCore import QCoreApplication
    QCoreApplication.addLibraryPath(str(internal / "PyQt6" / "Qt6" / "plugins"))

   
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
from PyQt6.QtWidgets import QApplication
from AutoDNA.app.autodna_app import AutoDNAApplication
from AutoDNA.app.get_path import get_data_dir


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    from PyQt6.QtCore import QLibraryInfo 
    print(f"Qt prefix:      {QLibraryInfo.path(QLibraryInfo.LibraryPath.PrefixPath)}")
    print(f"Qt data:        {QLibraryInfo.path(QLibraryInfo.LibraryPath.DataPath)}")
    print(f"Qt plugins:     {QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)}")
    print(f"Qt translations:{QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)}")

    window = AutoDNAApplication()
    window.show()

    sys.exit(app.exec())"""


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log_path = get_data_dir() / "autodna_crash.log"
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
