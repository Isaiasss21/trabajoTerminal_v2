"""
settings_screen.py
------------------
Pantalla de configuración de la aplicación.

Secciones:
  1. Modelo — selector de archivo .pth, estado de carga
  2. Cámara — selector de índice de cámara disponible
  3. Consentimiento — checkbox GDPR (RB07: consentimiento informado)
  4. Acerca de — versión, créditos

Señales emitidas al exterior:
  model_changed(str) — ruta del nuevo modelo cuando el usuario lo carga
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QFileDialog, QCheckBox, QLineEdit,
    QGroupBox, QFormLayout, QSizePolicy, QScrollArea,
    QMessageBox,
)

from app.inference.inference_engine import InferenceEngine
from app.storage.session_manager import SessionManager
from app.inference.camera_pipeline import get_available_cameras


# ── Colores ────────────────────────────────────────────────────────────────────

_BG      = "#121212"
_SURFACE = "#1E1E1E"
_ACCENT  = "#2979FF"
_TEXT    = "#E0E0E0"
_SUBTEXT = "#9E9E9E"
_GREEN   = "#4CAF50"
_AMBER   = "#FFC107"
_RED     = "#F44336"

_BTN_ACCENT = (
    "QPushButton { background:#2979FF; color:#fff; border:none; border-radius:6px; "
    "padding:8px 18px; font-size:13px; } "
    "QPushButton:hover { background:#5499FF; }"
)
_BTN_NEUTRAL = (
    "QPushButton { background:#333; color:#E0E0E0; border:none; border-radius:6px; "
    "padding:8px 18px; font-size:13px; } "
    "QPushButton:hover { background:#444; }"
)
_INPUT_STYLE = (
    f"QLineEdit {{ background:#252525; color:{_TEXT}; border:1px solid #333;"
    f" border-radius:5px; padding:6px 10px; font-size:13px; }}"
)
_GROUP_STYLE = (
    f"QGroupBox {{ color:{_TEXT}; font-size:14px; font-weight:bold;"
    f" border:1px solid #2A2A2A; border-radius:8px; margin-top:14px; padding:14px 12px;"
    f" background:{_SURFACE}; }}"
    f" QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; }}"
)
_CHECKBOX_STYLE = (
    f"QCheckBox {{ color:{_TEXT}; font-size:13px; }}"
    f" QCheckBox::indicator {{ width:18px; height:18px; border:2px solid #555;"
    f" border-radius:4px; background:#252525; }}"
    f" QCheckBox::indicator:checked {{ background:{_ACCENT}; border-color:{_ACCENT}; }}"
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

        self._build_ui()
        self._refresh_model_status()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background: {_BG}; color: {_TEXT};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(0)

        # Título
        title = QLabel("Ajustes")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {_TEXT};")
        outer.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2A2A2A;")
        outer.addSpacing(10)
        outer.addWidget(sep)
        outer.addSpacing(16)

        # Área scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        outer.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet(f"background: {_BG};")
        scroll.setWidget(content)

        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(20)

        vbox.addWidget(self._build_model_group())
        vbox.addWidget(self._build_camera_group())
        vbox.addWidget(self._build_consent_group())
        vbox.addWidget(self._build_about_group())
        vbox.addStretch()

    # ── Sección: Modelo ───────────────────────────────────────────────────

    def _build_model_group(self) -> QGroupBox:
        grp = QGroupBox("Modelo de inferencia")
        grp.setStyleSheet(_GROUP_STYLE)
        vbox = QVBoxLayout(grp)
        vbox.setSpacing(10)

        # Ruta actual
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setStyleSheet(_INPUT_STYLE)
        self._model_path_edit.setReadOnly(True)
        self._model_path_edit.setPlaceholderText("Ningún modelo cargado")
        if self._engine.is_ready:
            self._model_path_edit.setText(str(self._engine.model_path))
        vbox.addWidget(self._model_path_edit)

        btn_row = QHBoxLayout()
        btn_browse = QPushButton("📂  Examinar…")
        btn_browse.setStyleSheet(_BTN_NEUTRAL)
        btn_browse.setFixedHeight(34)
        btn_browse.clicked.connect(self._browse_model)
        btn_row.addWidget(btn_browse)

        self._btn_load_model = QPushButton("Cargar modelo")
        self._btn_load_model.setStyleSheet(_BTN_ACCENT)
        self._btn_load_model.setFixedHeight(34)
        self._btn_load_model.setEnabled(False)
        self._btn_load_model.clicked.connect(self._load_model)
        btn_row.addWidget(self._btn_load_model)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        # Estado del modelo
        self._lbl_model_status = QLabel()
        self._lbl_model_status.setStyleSheet(f"font-size: 12px; color: {_SUBTEXT};")
        vbox.addWidget(self._lbl_model_status)

        return grp

    # ── Sección: Cámara ───────────────────────────────────────────────────

    def _build_camera_group(self) -> QGroupBox:
        grp = QGroupBox("Cámara")
        grp.setStyleSheet(_GROUP_STYLE)
        vbox = QVBoxLayout(grp)
        vbox.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(10)

        self._camera_combo = QComboBox()
        self._camera_combo.setStyleSheet(
            f"QComboBox {{ background:#252525; color:{_TEXT}; border:1px solid #333;"
            f" border-radius:5px; padding:5px 10px; font-size:13px; }}"
        )
        self._camera_combo.addItem("Cargando…")
        form.addRow(QLabel("Índice de cámara:"), self._camera_combo)
        vbox.addLayout(form)

        btn_detect = QPushButton("🔍  Detectar cámaras")
        btn_detect.setStyleSheet(_BTN_NEUTRAL)
        btn_detect.setFixedHeight(34)
        btn_detect.clicked.connect(self._detect_cameras)
        vbox.addWidget(btn_detect, alignment=Qt.AlignmentFlag.AlignLeft)

        return grp

    # ── Sección: Consentimiento ───────────────────────────────────────────

    def _build_consent_group(self) -> QGroupBox:
        grp = QGroupBox("Consentimiento informado (RB07)")
        grp.setStyleSheet(_GROUP_STYLE)
        vbox = QVBoxLayout(grp)
        vbox.setSpacing(10)

        info = QLabel(
            "Este sistema analiza expresiones faciales mediante cámara.\n"
            "Los datos de video no se transmiten externamente.\n"
            "Las sesiones se almacenan localmente y pueden eliminarse desde el Historial."
        )
        info.setStyleSheet(f"color: {_SUBTEXT}; font-size: 12px;")
        info.setWordWrap(True)
        vbox.addWidget(info)

        self._chk_consent = QCheckBox("Entiendo y doy mi consentimiento para el análisis facial")
        self._chk_consent.setStyleSheet(_CHECKBOX_STYLE)
        vbox.addWidget(self._chk_consent)

        return grp

    # ── Sección: Acerca de ────────────────────────────────────────────────

    def _build_about_group(self) -> QGroupBox:
        grp = QGroupBox("Acerca de")
        grp.setStyleSheet(_GROUP_STYLE)
        vbox = QVBoxLayout(grp)

        about_text = QLabel(
            "MicroExpression Analyzer  v1.0.0\n\n"
            "Modelo: FlowClassifier (DenseNet / ResNet + Optical Flow)\n"
            "Framework: PyQt6 · PyTorch · MediaPipe · OpenCV\n\n"
            "Trabajo Terminal — Análisis de Microexpresiones Faciales"
        )
        about_text.setStyleSheet(f"color: {_SUBTEXT}; font-size: 12px; line-height: 1.6;")
        about_text.setWordWrap(True)
        vbox.addWidget(about_text)

        return grp

    # ── Acciones ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def _browse_model(self) -> None:
        # Empezar en la carpeta models/ del proyecto si existe, si no en home
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

    @pyqtSlot()
    def _detect_cameras(self) -> None:
        self._camera_combo.clear()
        self._camera_combo.addItem("Buscando…")
        # Operación síncrona (breve); para producción usar QThread
        available = get_available_cameras(max_index=5)
        self._camera_combo.clear()
        if available:
            for idx in available:
                self._camera_combo.addItem(f"Cámara {idx}", userData=idx)
        else:
            self._camera_combo.addItem("No se encontraron cámaras")

    def _refresh_model_status(self) -> None:
        if self._engine.is_ready:
            n = len(self._engine.label_map)
            self._lbl_model_status.setText(f"✔ Modelo cargado — {n} clases")
            self._lbl_model_status.setStyleSheet(f"color: {_GREEN}; font-size: 12px;")
        else:
            self._lbl_model_status.setText("✘ Sin modelo cargado")
            self._lbl_model_status.setStyleSheet(f"color: {_RED}; font-size: 12px;")

    # ── Accesores ─────────────────────────────────────────────────────────

    @property
    def selected_camera_index(self) -> int:
        """Índice de cámara seleccionado actualmente."""
        data = self._camera_combo.currentData()
        return data if isinstance(data, int) else 0

    @property
    def consent_given(self) -> bool:
        return self._chk_consent.isChecked()
