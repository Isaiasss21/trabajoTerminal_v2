"""
main_window.py
--------------
Ventana principal de la aplicación de análisis de microexpresiones.

Estructura:
  ┌──────────┬────────────────────────────────────┐
  │ Sidebar  │  QStackedWidget (pantallas)        │
  │ 200px    │                                    │
  │  Análisis│   AnalysisScreen                  │
  │  Results │   ResultsScreen                   │
  │  Historial│  HistoryScreen                   │
  │  Ajustes │   SettingsScreen                  │
  └──────────┴────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame, QSizePolicy,
)

from app.ui.analysis_screen import AnalysisScreen
from app.ui.results_screen import ResultsScreen
from app.ui.history_screen import HistoryScreen
from app.ui.settings_screen import SettingsScreen
from app.storage.session_manager import SessionManager
from app.inference.inference_engine import InferenceEngine


# ── Colores del tema oscuro ───────────────────────────────────────────────────

_BG        = "#121212"
_SURFACE   = "#1E1E1E"
_SIDEBAR   = "#1A1A1A"
_ACCENT    = "#2979FF"
_ACCENT_H  = "#5499FF"
_TEXT      = "#E0E0E0"
_SUBTEXT   = "#9E9E9E"
_DIVIDER   = "#2A2A2A"

_SIDEBAR_BTN_STYLE = """
QPushButton {{
    background: transparent;
    color: {text};
    border: none;
    border-left: 3px solid transparent;
    padding: 12px 16px;
    text-align: left;
    font-size: 14px;
    border-radius: 0px;
}}
QPushButton:hover {{
    background: rgba(255,255,255,0.05);
    border-left: 3px solid {accent_h};
    color: {text};
}}
QPushButton[active="true"] {{
    background: rgba(41,121,255,0.15);
    border-left: 3px solid {accent};
    color: {accent};
    font-weight: bold;
}}
""".format(text=_TEXT, accent=_ACCENT, accent_h=_ACCENT_H)


class _SidebarButton(QPushButton):
    """Botón del sidebar con estado activo/inactivo."""

    def __init__(self, text: str, icon_char: str = "", parent=None) -> None:
        super().__init__(f"  {icon_char}  {text}" if icon_char else text, parent)
        self.setStyleSheet(_SIDEBAR_BTN_STYLE)
        self.setCheckable(False)
        self.setProperty("active", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación.

    Args:
        model_path: Ruta al checkpoint .pth del modelo FlowClassifier.
                    Si no se pasa, el usuario puede seleccionarla desde Ajustes.
        session_dir: Directorio raíz donde se guardan las sesiones.
    """

    def __init__(
        self,
        model_path:  Optional[str | Path] = None,
        session_dir: Optional[str | Path] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Análisis de Microexpresiones")
        self.setMinimumSize(1100, 680)
        self.resize(1280, 760)

        # ── Servicios compartidos ─────────────────────────────────────────
        _sdir = Path(session_dir) if session_dir else Path("data/sessions")
        self._session_manager = SessionManager(sessions_dir=_sdir)
        self._engine          = InferenceEngine(model_path or "", device="auto")

        # ── Layout raíz ───────────────────────────────────────────────────
        root = QWidget()
        root.setObjectName("rootWidget")
        root.setStyleSheet(f"#rootWidget {{ background: {_BG}; }}")
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = self._build_sidebar()
        layout.addWidget(sidebar)

        # ── Separador vertical ────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {_DIVIDER};")
        sep.setFixedWidth(1)
        layout.addWidget(sep)

        # ── Stack de pantallas ────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {_BG};")
        layout.addWidget(self._stack, stretch=1)

        self._build_screens()
        self._navigate(0)  # pantalla inicial: Análisis

    # ── Construcción del sidebar ──────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet(f"background: {_SIDEBAR};")

        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Título / logo
        title = QLabel("MicroExp\nAnalyzer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: {_ACCENT};
            font-size: 16px;
            font-weight: bold;
            padding: 24px 8px 20px 8px;
        """)
        vbox.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {_DIVIDER};")
        vbox.addWidget(sep)

        vbox.addSpacing(8)

        # Botones de navegación
        self._nav_buttons: list[_SidebarButton] = []
        nav_items = [
            ("Análisis",   "▶"),
            ("Resultados", "📊"),
            ("Historial",  "🗂"),
            ("Ajustes",    "⚙"),
        ]
        for idx, (label, icon) in enumerate(nav_items):
            btn = _SidebarButton(label, icon)
            btn.clicked.connect(lambda checked, i=idx: self._navigate(i))
            self._nav_buttons.append(btn)
            vbox.addWidget(btn)

        vbox.addStretch()

        # Versión
        ver = QLabel("v1.0.0")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet(f"color: {_SUBTEXT}; font-size: 11px; padding: 12px;")
        vbox.addWidget(ver)

        return sidebar

    # ── Construcción de pantallas ─────────────────────────────────────────

    def _build_screens(self) -> None:
        self._analysis_screen  = AnalysisScreen(self._engine, self._session_manager)
        self._results_screen   = ResultsScreen(self._session_manager)
        self._history_screen   = HistoryScreen(self._session_manager)
        self._settings_screen  = SettingsScreen(self._engine, self._session_manager)

        for screen in (
            self._analysis_screen,
            self._results_screen,
            self._history_screen,
            self._settings_screen,
        ):
            self._stack.addWidget(screen)

        # Pasar a resultados al terminar una sesión
        self._analysis_screen.session_finished.connect(self._on_session_finished)
        # Ver sesión desde historial → navegar a resultados
        self._history_screen.view_session_requested.connect(self._on_view_session_from_history)
        # Actualizar motor desde ajustes
        self._settings_screen.model_changed.connect(self._on_model_changed)
        # Propagar selección de cámara desde ajustes
        self._settings_screen._camera_combo.currentIndexChanged.connect(self._on_camera_changed)

    # ── Navegación ────────────────────────────────────────────────────────

    def _navigate(self, index: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.set_active(i == index)
        self._stack.setCurrentIndex(index)

    # ── Slots de eventos cross-screen ─────────────────────────────────────

    def _on_session_finished(self, session_id: str) -> None:
        """Al terminar análisis, cargar resultados y navegar a la pantalla de resultados."""
        self._results_screen.load_session(session_id)
        self._navigate(1)  # pantalla Resultados
        self._history_screen.refresh()

    def _on_view_session_from_history(self, session_id: str) -> None:
        """Navega a la pantalla de resultados mostrando la sesión seleccionada."""
        self._results_screen.load_session(session_id)
        self._navigate(1)  # pantalla Resultados

    def _on_camera_changed(self) -> None:
        idx = self._settings_screen.selected_camera_index
        self._analysis_screen.set_camera_index(idx)

    def _on_model_changed(self, model_path: str) -> None:
        """Recarga el motor con la nueva ruta de modelo."""
        self._engine = InferenceEngine(model_path, device="auto")
        self._analysis_screen.set_engine(self._engine)

    # ── Cierre limpio ─────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._analysis_screen.stop_capture()
        super().closeEvent(event)
