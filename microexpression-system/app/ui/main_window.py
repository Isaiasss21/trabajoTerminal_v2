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
    QStackedWidget, QPushButton, QLabel, QFrame, QSizePolicy, QStyle,
    QMessageBox,
)

from app.ui.analysis_screen import AnalysisScreen
from app.ui.results_screen import ResultsScreen
from app.ui.history_screen import HistoryScreen
from app.ui.settings_screen import SettingsScreen
from app.ui.theme import Theme, ThemeManager, DARK
from app.storage.session_manager import SessionManager
from app.inference.inference_engine import InferenceEngine


def _mk_sidebar_btn_style(theme: Theme) -> str:
    return f"""
QPushButton {{
    background: transparent;
    color: {theme.text};
    border: none;
    border-left: 3px solid transparent;
    padding: 12px 16px;
    text-align: left;
    font-size: 14px;
    border-radius: 0px;
}}
QPushButton:hover {{
    background: {theme.sidebar_hover_bg};
    border-left: 3px solid {theme.accent_h};
    color: {theme.text};
}}
QPushButton[active="true"] {{
    background: {theme.accent_alpha};
    border-left: 3px solid {theme.accent};
    color: {theme.accent};
    font-weight: bold;
}}
"""


class _SidebarButton(QPushButton):
    """Botón del sidebar con estado activo/inactivo."""

    def __init__(self, text: str, style: str = "", icon: Optional[QIcon] = None, parent=None) -> None:
        super().__init__(f" {text}", parent)
        self.setStyleSheet(style)
        self.setCheckable(False)
        self.setProperty("active", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(16, 16))

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
        t = ThemeManager.current()

        self._root_widget = QWidget()
        self._root_widget.setObjectName("rootWidget")
        self._root_widget.setStyleSheet(f"#rootWidget {{ background: {t.bg}; }}")
        self.setCentralWidget(self._root_widget)

        layout = QHBoxLayout(self._root_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────
        self._sidebar_widget = self._build_sidebar(t)
        layout.addWidget(self._sidebar_widget)

        # ── Separador vertical ────────────────────────────────────────────
        self._sep_vline = QFrame()
        self._sep_vline.setFrameShape(QFrame.Shape.VLine)
        self._sep_vline.setStyleSheet(f"color: {t.divider};")
        self._sep_vline.setFixedWidth(1)
        layout.addWidget(self._sep_vline)

        # ── Stack de pantallas ────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {t.bg};")
        layout.addWidget(self._stack, stretch=1)

        self._build_screens()
        self._navigate(0)  # pantalla inicial: Análisis

        # ── Registrar cambios de tema ─────────────────────────────────────
        ThemeManager.on_change(self.apply_theme)

    # ── Construcción del sidebar ──────────────────────────────────────────

    def _build_sidebar(self, theme: Theme) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet(f"background: {theme.sidebar};")

        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Título / logo
        self._lbl_logo = QLabel("ZILU\nMicroExp\nAnalyzer")
        self._lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_logo.setStyleSheet(f"""
            color: {theme.accent};
            font-size: 16px;
            font-weight: bold;
            padding: 24px 8px 20px 8px;
        """)
        vbox.addWidget(self._lbl_logo)

        self._sep_sidebar = QFrame()
        self._sep_sidebar.setFrameShape(QFrame.Shape.HLine)
        self._sep_sidebar.setStyleSheet(f"color: {theme.divider};")
        vbox.addWidget(self._sep_sidebar)

        vbox.addSpacing(8)

        # Botones de navegación
        btn_style = _mk_sidebar_btn_style(theme)
        self._nav_buttons: list[_SidebarButton] = []
        nav_items = [
            ("Análisis",   ),
            ("Resultados", ),
            ("Historial",  ),
            ("Ajustes",    ),
        ]
        for idx, (label,) in enumerate(nav_items):
            btn = _SidebarButton(label, style=btn_style)
            btn.clicked.connect(lambda checked, i=idx: self._navigate(i))
            self._nav_buttons.append(btn)
            vbox.addWidget(btn)

        vbox.addStretch()

        # Versión
        self._lbl_version = QLabel("v1.0.0")
        self._lbl_version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_version.setStyleSheet(f"color: {theme.subtext}; font-size: 11px; padding: 12px;")
        vbox.addWidget(self._lbl_version)

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

    # ── Navegación ────────────────────────────────────────────────────────

    def _navigate(self, index: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.set_active(i == index)
        self._stack.setCurrentIndex(index)

    # ── Tema ──────────────────────────────────────────────────────────────

    def apply_theme(self, theme: Theme) -> None:
        """Aplica el tema a la ventana principal y todas las pantallas."""
        # Root
        self._root_widget.setStyleSheet(f"#rootWidget {{ background: {theme.bg}; }}")

        # Sidebar
        self._sidebar_widget.setStyleSheet(f"background: {theme.sidebar};")
        self._lbl_logo.setStyleSheet(f"""
            color: {theme.accent};
            font-size: 16px;
            font-weight: bold;
            padding: 24px 8px 20px 8px;
        """)
        self._sep_sidebar.setStyleSheet(f"color: {theme.divider};")
        self._lbl_version.setStyleSheet(f"color: {theme.subtext}; font-size: 11px; padding: 12px;")

        # Botones de navegación
        btn_style = _mk_sidebar_btn_style(theme)
        for btn in self._nav_buttons:
            btn.setStyleSheet(btn_style)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Separador vertical
        self._sep_vline.setStyleSheet(f"color: {theme.divider};")

        # Stack bg
        self._stack.setStyleSheet(f"background: {theme.bg};")

        # Pantallas
        self._analysis_screen.apply_theme(theme)
        self._results_screen.apply_theme(theme)
        self._history_screen.apply_theme(theme)
        self._settings_screen.apply_theme(theme)

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

    def _on_model_changed(self, model_path: str) -> None:
        """Recarga el motor con la nueva ruta de modelo."""
        try:
            new_engine = InferenceEngine(model_path, device="auto")
            if not new_engine.is_ready:
                raise RuntimeError("El modelo no se cargó correctamente (arquitectura no reconocida).")
            self._engine = new_engine
            self._analysis_screen.set_engine(self._engine)
            self._settings_screen.set_engine(self._engine)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error al cargar modelo",
                f"No se pudo cargar el modelo:\n{Path(model_path).name}\n\n{exc}",
            )
            self._settings_screen.set_engine(self._engine)

    # ── Cierre limpio ─────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._analysis_screen.stop_capture()
        super().closeEvent(event)
