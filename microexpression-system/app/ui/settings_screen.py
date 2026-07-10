"""
settings_screen.py
------------------
Pantalla de configuración de la aplicación.

Secciones:
  1. Modelo — selector de archivo .pth, estado de carga
  2. Apariencia — toggle claro / oscuro
  3. Acerca de — versión, créditos

Señales emitidas al exterior:
  model_changed(str) — ruta del nuevo modelo cuando el usuario lo carga
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QLineEdit,
    QGroupBox, QSizePolicy, QScrollArea,
    QMessageBox, QStyle,
)

from app.inference.inference_engine import InferenceEngine
from app.storage.session_manager import SessionManager
from app.ui.theme import Theme, ThemeManager

# ── Colores semánticos fijos ──────────────────────────────────────────────────

_GREEN = "#4CAF50"
_AMBER = "#FFC107"
_RED   = "#F44336"


def _group_style(theme: Theme) -> str:
    return (
        f"QGroupBox {{ color:{theme.text}; font-size:14px; font-weight:bold;"
        f" border:1px solid {theme.divider}; border-radius:8px; margin-top:14px;"
        f" padding:14px 12px; background:{theme.surface}; }}"
        f" QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; }}"
    )


def _input_style(theme: Theme) -> str:
    return (
        f"QLineEdit {{ background:{theme.input_bg}; color:{theme.text}; border:1px solid {theme.divider};"
        f" border-radius:5px; padding:6px 10px; font-size:13px; }}"
    )


def _btn_accent_style(theme: Theme) -> str:
    return (
        f"QPushButton {{ background:{theme.accent}; color:#fff; border:none; border-radius:6px; "
        f"padding:8px 18px; font-size:13px; }} "
        f"QPushButton:hover {{ background:{theme.accent_h}; }}"
    )


def _btn_neutral_style(theme: Theme) -> str:
    return (
        f"QPushButton {{ background:{theme.btn_neutral_bg}; color:{theme.text}; border:none; "
        f"border-radius:6px; padding:8px 18px; font-size:13px; }} "
        f"QPushButton:hover {{ background:{theme.btn_neutral_hover}; }}"
    )


class SettingsScreen(QWidget):
    """Pantalla de ajustes de la aplicación."""

    model_changed = pyqtSignal(str)   # ruta del nuevo modelo .pth

    def __init__(
        self,
        engine:          InferenceEngine,
        session_manager: SessionManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine          = engine
        self._session_manager = session_manager
        self._selected_model_path: Optional[str] = None
        self._theme: Theme = ThemeManager.current()

        self._build_ui()
        self._refresh_model_status()

    # ── API pública ───────────────────────────────────────────────────────

    def set_engine(self, engine: InferenceEngine) -> None:
        self._engine = engine
        self._refresh_model_status()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        t = self._theme
        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(0)

        # Título
        self._lbl_title = QLabel("Ajustes")
        self._lbl_title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t.text};")
        outer.addWidget(self._lbl_title)

        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.HLine)
        self._sep.setStyleSheet(f"color: {t.divider};")
        outer.addSpacing(10)
        outer.addWidget(self._sep)
        outer.addSpacing(16)

        # Área scrollable
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        outer.addWidget(self._scroll)

        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet(f"background: {t.bg};")
        self._scroll.setWidget(self._scroll_content)

        vbox = QVBoxLayout(self._scroll_content)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(20)

        self._grp_model      = self._build_model_group()
        self._grp_appearance = self._build_appearance_group()
        self._grp_about      = self._build_about_group()

        vbox.addWidget(self._grp_model)
        vbox.addWidget(self._grp_appearance)
        vbox.addWidget(self._grp_about)
        vbox.addStretch()

    # ── Sección: Modelo ───────────────────────────────────────────────────

    def _build_model_group(self) -> QGroupBox:
        t = self._theme
        grp = QGroupBox("Modelo de inferencia")
        grp.setStyleSheet(_group_style(t))
        vbox = QVBoxLayout(grp)
        vbox.setSpacing(10)

        self._model_path_edit = QLineEdit()
        self._model_path_edit.setStyleSheet(_input_style(t))
        self._model_path_edit.setReadOnly(True)
        self._model_path_edit.setPlaceholderText("Ningún modelo cargado")
        if self._engine.is_ready:
            self._model_path_edit.setText(str(self._engine.model_path))
        vbox.addWidget(self._model_path_edit)

        btn_row = QHBoxLayout()
        self._btn_browse = QPushButton("Examinar…")
        self._btn_browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self._btn_browse.setIconSize(QSize(16, 16))
        self._btn_browse.setStyleSheet(_btn_neutral_style(t))
        self._btn_browse.setFixedHeight(34)
        self._btn_browse.clicked.connect(self._browse_model)
        btn_row.addWidget(self._btn_browse)

        self._btn_load_model = QPushButton("Cargar modelo")
        self._btn_load_model.setStyleSheet(_btn_accent_style(t))
        self._btn_load_model.setFixedHeight(34)
        self._btn_load_model.setEnabled(False)
        self._btn_load_model.clicked.connect(self._load_model)
        btn_row.addWidget(self._btn_load_model)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self._lbl_model_status = QLabel()
        self._lbl_model_status.setStyleSheet(f"font-size: 12px; color: {t.subtext};")
        vbox.addWidget(self._lbl_model_status)

        return grp

    # ── Sección: Apariencia ───────────────────────────────────────────────

    def _build_appearance_group(self) -> QGroupBox:
        t = self._theme
        grp = QGroupBox("Apariencia")
        grp.setStyleSheet(_group_style(t))
        vbox = QVBoxLayout(grp)
        vbox.setSpacing(12)

        desc = QLabel("Selecciona el tema de la interfaz:")
        desc.setStyleSheet(f"color: {t.subtext}; font-size: 12px;")
        vbox.addWidget(desc)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_dark = QPushButton("Oscuro")
        self._btn_dark.setFixedHeight(34)
        self._btn_dark.clicked.connect(self._set_dark)

        self._btn_light = QPushButton("Claro")
        self._btn_light.setFixedHeight(34)
        self._btn_light.clicked.connect(self._set_light)

        btn_row.addWidget(self._btn_dark)
        btn_row.addWidget(self._btn_light)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self._grp_appearance_desc = desc
        self._refresh_theme_buttons()
        return grp

    def _refresh_theme_buttons(self) -> None:
        t = self._theme
        is_dark = ThemeManager.is_dark()
        # El botón activo usa acento; el inactivo usa neutral
        active_style = _btn_accent_style(t)
        neutral_style = _btn_neutral_style(t)
        self._btn_dark.setStyleSheet(active_style if is_dark else neutral_style)
        self._btn_light.setStyleSheet(neutral_style if is_dark else active_style)

    # ── Sección: Acerca de ────────────────────────────────────────────────

    def _build_about_group(self) -> QGroupBox:
        t = self._theme
        grp = QGroupBox("Acerca de")
        grp.setStyleSheet(_group_style(t))
        vbox = QVBoxLayout(grp)

        self._about_text = QLabel(
            "MicroExpression Analyzer  v1.0.0\n\n"
            "Modelo: FlowClassifier (DenseNet / ResNet + Optical Flow)\n"
            "Framework: PyQt6 · PyTorch · MediaPipe · OpenCV\n\n"
            "Trabajo Terminal — Análisis de Microexpresiones Faciales"
        )
        self._about_text.setStyleSheet(f"color: {t.subtext}; font-size: 12px; line-height: 1.6;")
        self._about_text.setWordWrap(True)
        vbox.addWidget(self._about_text)

        return grp

    # ── Tema ──────────────────────────────────────────────────────────────

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        t = theme

        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")
        self._lbl_title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t.text};")
        self._sep.setStyleSheet(f"color: {t.divider};")
        self._scroll_content.setStyleSheet(f"background: {t.bg};")

        # Grupos
        gs = _group_style(t)
        self._grp_model.setStyleSheet(gs)
        self._grp_appearance.setStyleSheet(gs)
        self._grp_about.setStyleSheet(gs)

        # Modelo
        self._model_path_edit.setStyleSheet(_input_style(t))
        self._btn_browse.setStyleSheet(_btn_neutral_style(t))
        self._btn_load_model.setStyleSheet(_btn_accent_style(t))
        self._lbl_model_status.setStyleSheet(f"font-size: 12px; color: {t.subtext};")

        # Apariencia
        self._grp_appearance_desc.setStyleSheet(f"color: {t.subtext}; font-size: 12px;")
        self._refresh_theme_buttons()

        # Acerca de
        self._about_text.setStyleSheet(f"color: {t.subtext}; font-size: 12px; line-height: 1.6;")

        # Refrescar estado del modelo (puede cambiar colores)
        self._refresh_model_status()

    # ── Acciones ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def _set_dark(self) -> None:
        from app.ui.theme import DARK
        ThemeManager.set_theme(DARK)

    @pyqtSlot()
    def _set_light(self) -> None:
        from app.ui.theme import LIGHT
        ThemeManager.set_theme(LIGHT)

    @pyqtSlot()
    def _browse_model(self) -> None:
        _models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        start_dir = str(_models_dir) if _models_dir.is_dir() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar checkpoint (.pth)",
            start_dir,
            "Checkpoint PyTorch (*.pth *.pt)",
        )
        if path:
            self._selected_model_path = path
            self._model_path_edit.setText(path)
            self._btn_load_model.setEnabled(True)

    @pyqtSlot()
    def _load_model(self) -> None:
        if not self._selected_model_path:
            return
        p = Path(self._selected_model_path)
        if not p.exists():
            QMessageBox.critical(self, "Error", f"Archivo no encontrado:\n{p}")
            return
        self._lbl_model_status.setText("Cargando…")
        self._lbl_model_status.setStyleSheet(f"color: {_AMBER}; font-size: 12px;")
        self.model_changed.emit(self._selected_model_path)
        self._refresh_model_status()

    def _refresh_model_status(self) -> None:
        if self._engine.is_ready:
            n = len(self._engine.label_map)
            self._lbl_model_status.setText(f"✔ Modelo cargado — {n} clases")
            self._lbl_model_status.setStyleSheet(f"color: {_GREEN}; font-size: 12px;")
        else:
            self._lbl_model_status.setText("✘ Sin modelo cargado")
            self._lbl_model_status.setStyleSheet(f"color: {_RED}; font-size: 12px;")
