"""
analysis_screen.py
------------------
Pantalla de análisis en tiempo real.

Layout:
  ┌───────────────────────────────────────────────────┐
  │  [▶ Iniciar]  [■ Detener]   ● Cara detectada     │  ← barra superior
  ├─────────────────────────┬─────────────────────────┤
  │                         │  Emoción:  Alegría      │
  │   Feed de cámara        │  Confianza: 87%         │
  │   (QLabel, JPEG)        │  ─────────────────────  │
  │                         │  Distribución de probs  │
  │                         │  (barras simples)       │
  └─────────────────────────┴─────────────────────────┘

Señales emitidas al exterior:
  session_finished(str) — session_id al cerrar sesión
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QProgressBar, QScrollArea,
)

from app.inference.inference_engine import InferenceEngine, InferenceResult, EMOTION_COLORS
from app.inference.camera_pipeline import CameraPipeline
from app.storage.session_manager import SessionManager, Prediction, Session


# ── Colores ────────────────────────────────────────────────────────────────────

_BG      = "#121212"
_SURFACE = "#1E1E1E"
_ACCENT  = "#2979FF"
_GREEN   = "#4CAF50"
_AMBER   = "#FFC107"
_RED     = "#F44336"
_TEXT    = "#E0E0E0"
_SUBTEXT = "#9E9E9E"

_BTN_START = f"QPushButton {{ background:{_GREEN}; color:#fff; border:none; border-radius:6px; padding:8px 20px; font-size:13px; }} QPushButton:hover {{ background:#66BB6A; }}"
_BTN_STOP  = f"QPushButton {{ background:{_RED}; color:#fff; border:none; border-radius:6px; padding:8px 20px; font-size:13px; }} QPushButton:hover {{ background:#EF5350; }}"
_BTN_DIS   = f"QPushButton {{ background:#424242; color:#757575; border:none; border-radius:6px; padding:8px 20px; font-size:13px; }}"


class AnalysisScreen(QWidget):
    """Pantalla de análisis en tiempo real con cámara."""

    session_finished = pyqtSignal(str)  # session_id

    def __init__(
        self,
        engine:          InferenceEngine,
        session_manager: SessionManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine          = engine
        self._session_manager = session_manager
        self._pipeline:       Optional[CameraPipeline] = None
        self._session:        Optional[Session]         = None
        self._last_result:    Optional[InferenceResult] = None
        self._frame_count:    int  = 0
        self._running:        bool = False
        self._inferring:      bool = False  # guard: evita inferencias en cascada

        self._build_ui()

    # ── API pública ───────────────────────────────────────────────────────

    def set_engine(self, engine: InferenceEngine) -> None:
        """Actualiza el motor de inferencia (llamado desde MainWindow al cambiar modelo)."""
        self._engine = engine

    def set_camera_index(self, index: int) -> None:
        """Actualiza el índice de cámara a usar en la siguiente sesión."""
        self._preferred_camera_index: int = index

    def stop_capture(self) -> None:
        """Detiene la captura limpiamente (llamado al cerrar la ventana)."""
        if self._pipeline and self._pipeline.isRunning():
            self._pipeline.stop()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background: {_BG}; color: {_TEXT};")
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Barra superior de controles ───────────────────────────────────
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(12)

        self._btn_start = QPushButton("▶  Iniciar")
        self._btn_start.setStyleSheet(_BTN_START)
        self._btn_start.setFixedHeight(38)
        self._btn_start.clicked.connect(self._start_session)
        ctrl_bar.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  Detener")
        self._btn_stop.setStyleSheet(_BTN_DIS)
        self._btn_stop.setFixedHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_session)
        ctrl_bar.addWidget(self._btn_stop)

        ctrl_bar.addStretch()

        self._lbl_face_status = QLabel("⬤  Sin cara")
        self._lbl_face_status.setStyleSheet(f"color: {_SUBTEXT}; font-size: 13px;")
        ctrl_bar.addWidget(self._lbl_face_status)

        self._lbl_pred_count = QLabel("Predicciones: 0")
        self._lbl_pred_count.setStyleSheet(f"color: {_SUBTEXT}; font-size: 12px;")
        ctrl_bar.addWidget(self._lbl_pred_count)

        root.addLayout(ctrl_bar)

        # ── Separador ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: #2A2A2A;")
        root.addWidget(sep)

        # ── Área principal: video + panel de emoción ──────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(16)

        # Feed de cámara
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet(f"background: #000; border-radius: 8px;")
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setMinimumSize(480, 360)
        self._video_label.setText("Sin señal de cámara")
        main_row.addWidget(self._video_label, stretch=3)

        # Panel lateral de resultados
        panel = self._build_result_panel()
        main_row.addWidget(panel, stretch=2)

        root.addLayout(main_row)

    def _build_result_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"background: {_SURFACE}; border-radius: 10px;")
        panel.setFixedWidth(280)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(8)

        # Etiqueta "Emoción detectada"
        lbl_title = QLabel("Emoción detectada")
        lbl_title.setStyleSheet(f"color: {_SUBTEXT}; font-size: 11px; font-weight: bold;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        vbox.addWidget(lbl_title)

        self._lbl_emotion = QLabel("–")
        self._lbl_emotion.setStyleSheet(f"color: {_TEXT}; font-size: 28px; font-weight: bold;")
        self._lbl_emotion.setAlignment(Qt.AlignmentFlag.AlignLeft)
        vbox.addWidget(self._lbl_emotion)

        self._lbl_confidence = QLabel("Confianza: –")
        self._lbl_confidence.setStyleSheet(f"color: {_SUBTEXT}; font-size: 13px;")
        vbox.addWidget(self._lbl_confidence)

        self._lbl_validity = QLabel("")
        self._lbl_validity.setStyleSheet(f"font-size: 12px;")
        vbox.addWidget(self._lbl_validity)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: #333;")
        vbox.addWidget(sep)

        # Barras de probabilidad por emoción
        lbl_dist = QLabel("Distribución")
        lbl_dist.setStyleSheet(f"color: {_SUBTEXT}; font-size: 11px; font-weight: bold;")
        vbox.addWidget(lbl_dist)

        self._prob_bars: dict[str, tuple[QLabel, QProgressBar]] = {}
        emotions_es = ["Alegría", "Asco", "Enojo", "Miedo", "Neutral", "Sorpresa", "Tristeza"]
        for emo in emotions_es:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(emo)
            lbl.setFixedWidth(70)
            lbl.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            color = EMOTION_COLORS.get(emo, _ACCENT)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: #333; border-radius: 5px; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}
            """)
            row.addWidget(lbl)
            row.addWidget(bar)
            vbox.addLayout(row)
            self._prob_bars[emo] = (lbl, bar)

        vbox.addStretch()
        return panel

    # ── Inicio / detención de sesión ──────────────────────────────────────

    def _start_session(self) -> None:
        if self._running:
            return
        self._running = True
        self._session = self._session_manager.new_session()

        from app.inference.camera_pipeline import get_available_cameras
        preferred = getattr(self, "_preferred_camera_index", None)
        if preferred is not None:
            cam_idx = preferred
        else:
            cameras = get_available_cameras()
            cam_idx = cameras[0] if cameras else 0
        self._pipeline = CameraPipeline(camera_index=cam_idx)
        self._pipeline.frame_ready.connect(self._on_frame)
        self._pipeline.sequence_ready.connect(self._on_sequence)
        self._pipeline.landmark_status.connect(self._on_landmark_status)
        self._pipeline.error.connect(self._on_pipeline_error)
        self._pipeline.start()

        self._btn_start.setEnabled(False)
        self._btn_start.setStyleSheet(_BTN_DIS)
        self._btn_stop.setEnabled(True)
        self._btn_stop.setStyleSheet(_BTN_STOP)

    def _stop_session(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

        session_id = ""
        if self._session:
            meta = self._session.close()
            session_id = meta.session_id
            self._session = None

        self._btn_start.setEnabled(True)
        self._btn_start.setStyleSheet(_BTN_START)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(_BTN_DIS)
        self._video_label.setText("Sin señal de cámara")
        self._lbl_face_status.setText("⬤  Sin cara")
        self._lbl_pred_count.setText("Predicciones: 0")
        self._reset_result_panel()

        if session_id:
            self.session_finished.emit(session_id)

    # ── Slots del pipeline ────────────────────────────────────────────────

    def _on_frame(self, jpeg_bytes: bytes) -> None:
        """Actualiza el QLabel con el frame JPEG recibido."""
        image = QImage.fromData(jpeg_bytes, "JPEG")
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def _on_sequence(self, flow_sequence: np.ndarray) -> None:
        """Recibe una secuencia de flujo, ejecuta inferencia y actualiza UI."""
        if not self._running or not self._engine.is_ready:
            return
        # Si ya hay una inferencia en curso, descartar esta secuencia para no
        # bloquear el event loop de Qt (cada pasada ResNet18 en CPU ~1-2 s).
        if self._inferring:
            return
        self._inferring = True
        try:
            result = self._engine.predict(flow_sequence)
        except Exception:
            return
        finally:
            self._inferring = False

        self._last_result = result
        self._update_result_panel(result)

        if self._session:
            pred = Prediction(
                timestamp          = datetime.now(timezone.utc).isoformat(),
                emotion            = result.emotion,
                raw_label          = result.raw_label,
                confidence         = result.confidence,
                is_valid           = result.is_valid,
                duration_ms        = 200,   # aproximado (SEQUENCE_LENGTH * ~13ms/frame)
                landmarks_detected = True,
            )
            self._session.add_prediction(pred)
            self._lbl_pred_count.setText(f"Predicciones: {self._session.prediction_count}")

    def _on_landmark_status(self, detected: bool) -> None:
        if detected:
            self._lbl_face_status.setText("⬤  Cara detectada")
            self._lbl_face_status.setStyleSheet(f"color: {_GREEN}; font-size: 13px;")
        else:
            self._lbl_face_status.setText("⬤  Sin cara")
            self._lbl_face_status.setStyleSheet(f"color: {_SUBTEXT}; font-size: 13px;")

    def _on_pipeline_error(self, msg: str) -> None:
        self._lbl_face_status.setText(f"⚠ {msg[:60]}")
        self._lbl_face_status.setStyleSheet(f"color: {_AMBER}; font-size: 12px;")

    # ── Actualización del panel de resultados ─────────────────────────────

    def _update_result_panel(self, result: InferenceResult) -> None:
        color = EMOTION_COLORS.get(result.emotion, _TEXT)
        self._lbl_emotion.setText(result.emotion)
        self._lbl_emotion.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")
        self._lbl_confidence.setText(f"Confianza: {result.confidence * 100:.1f}%")

        if result.is_valid:
            self._lbl_validity.setText("✔ Detección válida")
            self._lbl_validity.setStyleSheet(f"color: {_GREEN}; font-size: 12px;")
        elif result.is_uncertain:
            self._lbl_validity.setText("⚠ Detección incierta")
            self._lbl_validity.setStyleSheet(f"color: {_AMBER}; font-size: 12px;")
        else:
            self._lbl_validity.setText("✘ Baja confianza")
            self._lbl_validity.setStyleSheet(f"color: {_RED}; font-size: 12px;")

        for emo_es, (_, bar) in self._prob_bars.items():
            prob = result.probs.get(emo_es, 0.0)
            bar.setValue(int(prob * 100))

    def _reset_result_panel(self) -> None:
        self._lbl_emotion.setText("–")
        self._lbl_emotion.setStyleSheet(f"color: {_TEXT}; font-size: 28px; font-weight: bold;")
        self._lbl_confidence.setText("Confianza: –")
        self._lbl_validity.setText("")
        for _, (_, bar) in self._prob_bars.items():
            bar.setValue(0)
