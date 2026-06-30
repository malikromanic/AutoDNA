from PyInstaller.utils.hooks import collect_all
from PyQt6.QtCore import QLibraryInfo
from pathlib import Path
import shutil

QT_DATA         = QLibraryInfo.path(QLibraryInfo.LibraryPath.DataPath)
QT_TRANSLATIONS = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)

def find_webengine_process():
    candidates = [
        Path(QT_DATA).parent.parent / 'lib' / 'qt6' / 'QtWebEngineProcess.exe',
        Path(QT_DATA).parent / 'lib' / 'qt6' / 'QtWebEngineProcess.exe',
        Path(QT_DATA).parent / 'bin' / 'QtWebEngineProcess.exe',
        Path(QT_DATA) / 'bin' / 'QtWebEngineProcess.exe',
    ]
    for c in candidates:
        print(f"[spec] checking: {c}")
        if c.exists():
            return str(c)
    found = shutil.which('QtWebEngineProcess')
    return found or ''

WEBENGINE_PROCESS = find_webengine_process()

print(f"[spec] Qt data:          {QT_DATA}")
print(f"[spec] Qt translations:  {QT_TRANSLATIONS}")
print(f"[spec] WebEngineProcess: {WEBENGINE_PROCESS}")

datas = [
    ('.', 'AutoDNA'),
]

qt_resources = Path(QT_DATA) / 'resources'
if qt_resources.exists():
    datas.append((str(qt_resources), 'PyQt6/Qt6/resources'))
    print(f"[spec] Qt resources found: {qt_resources}")
else:
    print(f"[spec] WARNING: Qt resources not found at {qt_resources}")

qt_locales = Path(QT_TRANSLATIONS) / 'qtwebengine_locales'
if qt_locales.exists():
    datas.append((str(qt_locales), 'PyQt6/Qt6/translations/qtwebengine_locales'))
    print(f"[spec] Qt locales found: {qt_locales}")
else:
    print(f"[spec] WARNING: Qt locales not found at {qt_locales}")

binaries = []
if WEBENGINE_PROCESS:
    binaries.append((WEBENGINE_PROCESS, 'PyQt6/Qt6/bin'))
    print(f"[spec] WebEngineProcess added: {WEBENGINE_PROCESS}")
else:
    print("[spec] WARNING: QtWebEngineProcess.exe not found")

hiddenimports = [
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebChannel',
    'PyQt6.QtPrintSupport',
    'PyQt6.QtWidgets',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'sklearn.linear_model._ridge',
    'sklearn.preprocessing._data',
    'sklearn.model_selection._split',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors._partition_nodes',
    'sklearn.tree._utils',
    'scipy._lib.messagestream',
    'matplotlib.backends.backend_qtagg',
]

tmp_ret = collect_all('folium')
datas         += tmp_ret[0]
binaries      += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    ['app/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoDNA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AutoDNA',
)