# microexpression.spec
# ---------------------
# Archivo de configuración de PyInstaller para MicroExpression Analyzer.
#
# Uso:
#   pyinstaller microexpression.spec
#
# El resultado queda en:
#   dist/MicroExpressionAnalyzer/
#     MicroExpressionAnalyzer.exe   ← doble clic para lanzar
#     _internal/                    ← DLLs y paquetes (no borrar)
#     models/                       ← copia aquí tus .pth
#     data/                         ← sesiones (se crea automáticamente)

from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── Recolectar binarios/datos de paquetes complejos ───────────────────────────

datas, binaries, hiddenimports = [], [], []

for pkg in ["mediapipe", "cv2", "matplotlib"]:
    d, b, h = collect_all(pkg)
    datas       += d
    binaries    += b
    hiddenimports += h

# PyTorch: solo datos (los hooks de pyinstaller-hooks-contrib manejan los binarios)
datas += collect_data_files("torch", includes=["**/*.py"])

# ── Imports ocultos adicionales ───────────────────────────────────────────────

hiddenimports += [
    # PyQt6 backends
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",
    # Matplotlib backend Qt
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    # Torch
    "torch",
    "torchvision",
    "torchvision.models",
    "torchvision.transforms",
    # ML / data
    "numpy",
    "pandas",
    "PIL",
    "PIL.Image",
    # App interna
    "app",
    "app.main",
    "app.ui.main_window",
    "app.ui.analysis_screen",
    "app.ui.results_screen",
    "app.ui.history_screen",
    "app.ui.settings_screen",
    "app.ui.theme",
    "app.inference.inference_engine",
    "app.inference.video_pipeline",
    "app.inference.explainability",
    "app.storage.session_manager",
    "app.analytics.stats_engine",
]

# ── Análisis del código fuente ────────────────────────────────────────────────

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Excluir lo que no se usa para reducir tamaño
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,         # onedir: binarios van separados
    name="MicroExpressionAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                     # UPX puede romper DLLs de torch — desactivado
    console=False,                 # Sin ventana de consola (cambiar a True para depurar)
    disable_windowed_traceback=False,
    # icon="assets/icon.ico",      # Descomenta si tienes un .ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MicroExpressionAnalyzer",
)
