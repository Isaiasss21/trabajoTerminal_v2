"""
main.py
-------
Punto de entrada principal de la aplicación MicroExpression Analyzer.

Uso:
    # Desde el directorio microexpression-system/:
    python -m app.main

    # O directamente:
    python app/main.py [--model path/to/model.pth] [--device cpu|cuda|auto]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QFont
from PyQt6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


# ── Tema oscuro global ────────────────────────────────────────────────────────

def _apply_dark_palette(app: QApplication) -> None:
    """Aplica paleta oscura a toda la QApplication."""
    palette = QPalette()

    palette.setColor(QPalette.ColorRole.Window,          QColor("#121212"))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor("#E0E0E0"))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#252525"))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor("#E0E0E0"))
    palette.setColor(QPalette.ColorRole.Text,            QColor("#E0E0E0"))
    palette.setColor(QPalette.ColorRole.Button,          QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#E0E0E0"))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor("#FF5252"))
    palette.setColor(QPalette.ColorRole.Link,            QColor("#2979FF"))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#2979FF"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))

    # Colores deshabilitados
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,  QColor("#757575"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,        QColor("#757575"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,  QColor("#757575"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight,   QColor("#424242"))

    app.setPalette(palette)


# ── Argumentos CLI ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MicroExpression Analyzer — análisis de microexpresiones en tiempo real"
    )
    p.add_argument(
        "--model", "-m",
        metavar="PATH",
        default=None,
        help="Ruta al checkpoint .pth del FlowClassifier. "
             "Si se omite, puedes seleccionarlo desde la pantalla de Ajustes.",
    )
    p.add_argument(
        "--device", "-d",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo de inferencia (por defecto: auto).",
    )
    p.add_argument(
        "--session-dir",
        metavar="DIR",
        default=None,
        help="Directorio raíz para almacenar sesiones (por defecto: data/sessions/).",
    )
    return p.parse_args()


# ── Entrada principal ─────────────────────────────────────────────────────────

def main() -> int:  # noqa: D103
    args = _parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("MicroExpression Analyzer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Trabajo Terminal")

    # Asegurar que el directorio de trabajo sea microexpression-system/
    # para que las rutas relativas (models/, data/) funcionen correctamente.
    # Cuando la app está empaquetada como EXE (PyInstaller), sys.frozen es True
    # y sys.executable apunta al .exe — usamos su carpeta como raíz.
    import os
    if getattr(sys, "frozen", False):
        workspace_root = Path(sys.executable).resolve().parent
    else:
        workspace_root = Path(__file__).resolve().parent.parent
    os.chdir(workspace_root)

    # Estilo base + paleta oscura
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Ventana principal
    window = MainWindow(
        model_path  = args.model,
        session_dir = args.session_dir,
    )
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
