"""
analysis_screen.py
------------------
Pantalla de análisis de video pre-grabado.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  [📂 Seleccionar video]  nombre_video.mp4   [▶ Analizar]    │  ← barra superior
  ├─────────────────────────┬────────────────────────────────────┤
  │                         │  Emoción:   Alegría                │
  │   Preview del frame     │  Confianza: 87%                    │
  │   en proceso            │  ──────────────────────────────    │
  │                         │  Distribución de probabilidades    │
  │                         │  (barras simples)                  │
  ├─────────────────────────┴────────────────────────────────────┤
  │  ████████████░░░░░░░  Frame 120 / 450   Secuencias: 8        │  ← progreso
  └──────────────────────────────────────────────────────────────┘

Señales emitidas al exterior:
  session_finished(str) — session_id al finalizar el análisis
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QProgressBar, QFileDialog, QCheckBox, QStyle,
)

from app.inference.inference_engine import (
    InferenceEngine, InferenceResult, EMOTION_COLORS, EMOTION_LABELS_ES,
    CONFIDENCE_VALID,
)
from app.inference.video_pipeline import VideoPipeline
from app.storage.session_manager import SessionManager, Prediction, Session
from app.ui.theme import Theme, ThemeManager


# ── Colores semánticos fijos (no cambian con el tema) ──────────────────────────

_GREEN = "#4CAF50"
_AMBER = "#FFC107"
_RED   = "#F44336"


class AnalysisScreen(QWidget):
    """Pantalla de análisis de video pre-grabado."""

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
        self._pipeline:       Optional[VideoPipeline]   = None
        self._session:        Optional[Session]          = None
        self._video_path:     Optional[Path]             = None
        self._running:        bool = False
        self._top_gradcam: list[tuple[float, str, bytes, bytes]] = []
        self._top_shap:    list[tuple[float, str, bytes, bytes]] = []
        self._theme: Theme = ThemeManager.current()

        self._build_ui()

    # ── API pública ───────────────────────────────────────────────────────

    def set_engine(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def stop_capture(self) -> None:
        if self._pipeline and self._pipeline.isRunning():
            self._pipeline.stop()

    # ── Estilos dinámicos (dependen del tema) ─────────────────────────────

    def _s_btn_neutral(self) -> str:
        t = self._theme
        return (f"QPushButton {{ background:{t.btn_neutral_bg}; color:{t.text}; border:none; "
                f"border-radius:6px; padding:8px 16px; font-size:13px; }} "
                f"QPushButton:hover {{ background:{t.btn_neutral_hover}; }}")

    def _s_btn_analyze(self) -> str:
        return (f"QPushButton {{ background:{_GREEN}; color:#fff; border:none; border-radius:6px; "
                f"padding:8px 20px; font-size:13px; }} QPushButton:hover {{ background:#66BB6A; }}")

    def _s_btn_stop(self) -> str:
        return (f"QPushButton {{ background:{_RED}; color:#fff; border:none; border-radius:6px; "
                f"padding:8px 20px; font-size:13px; }} QPushButton:hover {{ background:#EF5350; }}")

    def _s_btn_disabled(self) -> str:
        t = self._theme
        return (f"QPushButton {{ background:{t.btn_disabled_bg}; color:{t.btn_disabled_text}; "
                f"border:none; border-radius:6px; padding:8px 20px; font-size:13px; }}")

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        t = self._theme
        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Barra superior ─────────────────────────────────────────────────
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(10)

        self._btn_select = QPushButton("Seleccionar video")
        self._btn_select.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self._btn_select.setIconSize(QSize(16, 16))
        self._btn_select.setStyleSheet(self._s_btn_neutral())
        self._btn_select.setFixedHeight(38)
        self._btn_select.clicked.connect(self._select_video)
        ctrl_bar.addWidget(self._btn_select)

        self._lbl_video_path = QLabel("Ningún video seleccionado")
        self._lbl_video_path.setStyleSheet(f"color: {t.subtext}; font-size: 12px;")
        self._lbl_video_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ctrl_bar.addWidget(self._lbl_video_path, stretch=1)

        self._lbl_seq_count = QLabel("Secuencias: 0")
        self._lbl_seq_count.setStyleSheet(f"color: {t.subtext}; font-size: 12px;")
        ctrl_bar.addWidget(self._lbl_seq_count)

        self._btn_analyze = QPushButton("Analizar")
        self._btn_analyze.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._btn_analyze.setIconSize(QSize(16, 16))
        self._btn_analyze.setStyleSheet(self._s_btn_disabled())
        self._btn_analyze.setFixedHeight(38)
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self._start_analysis)
        ctrl_bar.addWidget(self._btn_analyze)

        self._btn_stop = QPushButton("Cancelar")
        self._btn_stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self._btn_stop.setIconSize(QSize(16, 16))
        self._btn_stop.setStyleSheet(self._s_btn_disabled())
        self._btn_stop.setFixedHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._cancel_analysis)
        ctrl_bar.addWidget(self._btn_stop)

        self._chk_gradcam = QCheckBox("Grad-CAM")
        self._chk_gradcam.setStyleSheet(
            f"color: {t.text}; font-size: 12px; padding-left: 4px;"
        )
        self._chk_gradcam.setToolTip(
            "Activa la explicabilidad Grad-CAM: muestra qué regiones del flujo óptico"
            " activaron más la predicción (agrega un paso extra por sección)."
        )
        ctrl_bar.addWidget(self._chk_gradcam)

        self._chk_shap = QCheckBox("SHAP")
        self._chk_shap.setStyleSheet(
            f"color: {t.text}; font-size: 12px; padding-left: 4px;"
        )
        self._chk_shap.setToolTip(
            "Activa las atribuciones SHAP (GradientSHAP / Integrated Gradients):\n"
            "muestra qué regiones del flujo óptico influyeron más en la predicción.\n"
            "Colormap PLASMA (distinto al JET de Grad-CAM).\n"
            "Agrega ≈ 1-3 seg extra por secuencia."
        )
        ctrl_bar.addWidget(self._chk_shap)

        root.addLayout(ctrl_bar)

        # ── Separador ─────────────────────────────────────────────────────
        self._sep_toolbar = QFrame()
        self._sep_toolbar.setFrameShape(QFrame.Shape.HLine)
        self._sep_toolbar.setStyleSheet(f"color: {t.divider};")
        root.addWidget(self._sep_toolbar)

        # ── Área principal ────────────────────────────────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # Columna 1: video original
        col1 = QVBoxLayout()
        col1.setSpacing(4)
        self._lbl_vid1 = QLabel("Video original")
        self._lbl_vid1.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        col1.addWidget(self._lbl_vid1)
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("background: #000; border-radius: 8px; color: #555; font-size: 13px;")
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setMinimumSize(300, 220)
        self._video_label.setText("Selecciona un video para comenzar")
        col1.addWidget(self._video_label)
        main_row.addLayout(col1, stretch=2)

        # Columna 2: video con landmarks y región de interés
        col2 = QVBoxLayout()
        col2.setSpacing(4)
        self._lbl_vid2 = QLabel("Landmarks / ROI")
        self._lbl_vid2.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        col2.addWidget(self._lbl_vid2)
        self._annotated_label = QLabel()
        self._annotated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._annotated_label.setStyleSheet("background: #000; border-radius: 8px; color: #555; font-size: 13px;")
        self._annotated_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._annotated_label.setMinimumSize(300, 220)
        self._annotated_label.setText("Sin cara detectada")
        col2.addWidget(self._annotated_label)
        main_row.addLayout(col2, stretch=2)

        # Panel lateral de última predicción
        panel = self._build_result_panel()
        main_row.addWidget(panel)

        root.addLayout(main_row)

        # ── Barra de progreso ─────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{ background: {t.bar_track}; border-radius: 7px; }}
            QProgressBar::chunk {{ background: {t.accent}; border-radius: 7px; }}
        """)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(f"color: {t.subtext}; font-size: 11px;")
        root.addWidget(self._lbl_status)

    def _build_result_panel(self) -> QWidget:
        t = self._theme
        self._result_panel = QFrame()
        self._result_panel.setStyleSheet(f"background: {t.surface}; border-radius: 10px;")
        self._result_panel.setFixedWidth(280)

        vbox = QVBoxLayout(self._result_panel)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(8)

        self._lbl_result_title = QLabel("Última predicción")
        self._lbl_result_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        vbox.addWidget(self._lbl_result_title)

        self._lbl_emotion = QLabel("–")
        self._lbl_emotion.setStyleSheet(f"color: {t.text}; font-size: 28px; font-weight: bold;")
        vbox.addWidget(self._lbl_emotion)

        self._lbl_confidence = QLabel("Confianza: –")
        self._lbl_confidence.setStyleSheet(f"color: {t.subtext}; font-size: 13px;")
        vbox.addWidget(self._lbl_confidence)

        self._lbl_validity = QLabel("")
        self._lbl_validity.setStyleSheet("font-size: 12px;")
        vbox.addWidget(self._lbl_validity)

        self._sep_panel = QFrame()
        self._sep_panel.setFrameShape(QFrame.Shape.HLine)
        self._sep_panel.setStyleSheet(f"color: {t.divider};")
        vbox.addWidget(self._sep_panel)

        self._lbl_dist_title = QLabel("Distribución")
        self._lbl_dist_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        vbox.addWidget(self._lbl_dist_title)

        bars_container = QWidget()
        bars_container.setStyleSheet("background: transparent;")
        self._bars_vbox = QVBoxLayout(bars_container)
        self._bars_vbox.setContentsMargins(0, 0, 0, 0)
        self._bars_vbox.setSpacing(4)
        vbox.addWidget(bars_container)

        self._prob_bars: dict[str, tuple[QLabel, QProgressBar]] = {}
        self._pct_labels: dict[str, QLabel] = {}
        emotions_es = ["Alegría", "Asco", "Neutral", "Sorpresa", "Tristeza"]
        for emo in emotions_es:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(emo)
            lbl.setFixedWidth(62)
            lbl.setStyleSheet(f"color: {t.text}; font-size: 11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            color = EMOTION_COLORS.get(emo, t.accent)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {t.bar_track}; border-radius: 5px; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}
            """)
            pct_lbl = QLabel("0%")
            pct_lbl.setFixedWidth(34)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct_lbl.setStyleSheet(f"color: {t.subtext}; font-size: 10px;")
            row.addWidget(lbl)
            row.addWidget(bar)
            row.addWidget(pct_lbl)
            self._bars_vbox.addLayout(row)
            self._prob_bars[emo] = (lbl, bar)
            self._pct_labels[emo] = pct_lbl

        vbox.addStretch()

        # ── Mini-panel Grad-CAM ──────────────────────────────────────────
        self._sep_gcam = QFrame()
        self._sep_gcam.setFrameShape(QFrame.Shape.HLine)
        self._sep_gcam.setStyleSheet(f"color: {t.divider};")
        vbox.addWidget(self._sep_gcam)

        self._lbl_gcam_title = QLabel("Grad-CAM (flujo óptico)")
        self._lbl_gcam_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        vbox.addWidget(self._lbl_gcam_title)

        self._gradcam_lbl = QLabel("Activa Grad-CAM para ver el mapa de calor")
        self._gradcam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gradcam_lbl.setStyleSheet(
            "background: #111; border-radius: 6px; color: #777; font-size: 10px;"
        )
        self._gradcam_lbl.setFixedHeight(160)
        self._gradcam_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        vbox.addWidget(self._gradcam_lbl)

        # ── Mini-panel SHAP ──────────────────────────────────────────────
        self._sep_shap = QFrame()
        self._sep_shap.setFrameShape(QFrame.Shape.HLine)
        self._sep_shap.setStyleSheet(f"color: {t.divider};")
        vbox.addWidget(self._sep_shap)

        self._lbl_shap_title = QLabel("SHAP (Integrated Gradients)")
        self._lbl_shap_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        vbox.addWidget(self._lbl_shap_title)

        self._shap_lbl = QLabel("Activa SHAP para ver las atribuciones")
        self._shap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._shap_lbl.setStyleSheet(
            "background: #111; border-radius: 6px; color: #777; font-size: 10px;"
        )
        self._shap_lbl.setFixedHeight(120)
        self._shap_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        vbox.addWidget(self._shap_lbl)

        return self._result_panel

    def _rebuild_bars(self, emotions_es: list[str]) -> None:
        """
        Reconstruye las filas de barras de probabilidad según las emociones del modelo.
        Solo toca el contenedor de barras (self._bars_vbox), nunca el Grad-CAM.
        """
        t = self._theme
        while self._bars_vbox.count() > 0:
            item = self._bars_vbox.takeAt(0)
            if item is None:
                continue
            sub = item.layout()
            if sub is not None:
                while sub.count() > 0:
                    sub_item = sub.takeAt(0)
                    if sub_item:
                        w = sub_item.widget()
                        if w:
                            w.setParent(None)
                            w.deleteLater()
            else:
                w = item.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()

        self._prob_bars.clear()
        self._pct_labels.clear()

        for emo in emotions_es:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(emo)
            lbl.setFixedWidth(62)
            lbl.setStyleSheet(f"color: {t.text}; font-size: 11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            color = EMOTION_COLORS.get(emo, t.accent)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {t.bar_track}; border-radius: 5px; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}
            """)
            pct_lbl = QLabel("0%")
            pct_lbl.setFixedWidth(34)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct_lbl.setStyleSheet(f"color: {t.subtext}; font-size: 10px;")
            row.addWidget(lbl)
            row.addWidget(bar)
            row.addWidget(pct_lbl)
            self._bars_vbox.addLayout(row)
            self._prob_bars[emo] = (lbl, bar)
            self._pct_labels[emo] = pct_lbl

    # ── Tema ──────────────────────────────────────────────────────────────

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        t = theme

        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")

        # Barra superior
        self._btn_select.setStyleSheet(self._s_btn_neutral())
        self._lbl_video_path.setStyleSheet(f"color: {t.subtext}; font-size: 12px;")
        self._lbl_seq_count.setStyleSheet(f"color: {t.subtext}; font-size: 12px;")
        self._chk_gradcam.setStyleSheet(f"color: {t.text}; font-size: 12px; padding-left: 4px;")
        self._chk_shap.setStyleSheet(f"color: {t.text}; font-size: 12px; padding-left: 4px;")
        self._sep_toolbar.setStyleSheet(f"color: {t.divider};")

        # Etiquetas de columna
        self._lbl_vid1.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        self._lbl_vid2.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")

        # Re-aplicar estilo a botones según su estado actual
        if self._running:
            self._btn_analyze.setStyleSheet(self._s_btn_disabled())
            self._btn_stop.setStyleSheet(self._s_btn_stop())
        else:
            if self._video_path:
                self._btn_analyze.setStyleSheet(self._s_btn_analyze())
            else:
                self._btn_analyze.setStyleSheet(self._s_btn_disabled())
            self._btn_stop.setStyleSheet(self._s_btn_disabled())

        # Panel de resultados
        self._result_panel.setStyleSheet(f"background: {t.surface}; border-radius: 10px;")
        self._lbl_result_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        self._lbl_confidence.setStyleSheet(f"color: {t.subtext}; font-size: 13px;")
        self._sep_panel.setStyleSheet(f"color: {t.divider};")
        self._lbl_dist_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        self._sep_gcam.setStyleSheet(f"color: {t.divider};")
        self._lbl_gcam_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")
        self._sep_shap.setStyleSheet(f"color: {t.divider};")
        self._lbl_shap_title.setStyleSheet(f"color: {t.subtext}; font-size: 11px; font-weight: bold;")

        # Barras de probabilidad
        for emo, (lbl, bar) in self._prob_bars.items():
            lbl.setStyleSheet(f"color: {t.text}; font-size: 11px;")
            color = EMOTION_COLORS.get(emo, t.accent)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {t.bar_track}; border-radius: 5px; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}
            """)
        for lbl in self._pct_labels.values():
            lbl.setStyleSheet(f"color: {t.subtext}; font-size: 10px;")

        # Progress bar
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{ background: {t.bar_track}; border-radius: 7px; }}
            QProgressBar::chunk {{ background: {t.accent}; border-radius: 7px; }}
        """)
        self._lbl_status.setStyleSheet(f"color: {t.subtext}; font-size: 11px;")

    # ── Selección de video ────────────────────────────────────────────────

    @pyqtSlot()
    def _select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar video",
            str(Path.home()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.MP4 *.AVI);;Todos los archivos (*)",
        )
        if path:
            self._video_path = Path(path)
            self._lbl_video_path.setText(self._video_path.name)
            self._btn_analyze.setEnabled(True)
            self._btn_analyze.setStyleSheet(self._s_btn_analyze())
            self._video_label.setText(f"Video: {self._video_path.name}\nPresiona Analizar para comenzar")
            self._annotated_label.clear()
            self._annotated_label.setText("Sin cara detectada")
            self._lbl_status.setText("")
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(False)

    # ── Análisis ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def _start_analysis(self) -> None:
        if self._running or not self._video_path or not self._engine.is_ready:
            if not self._engine.is_ready:
                self._lbl_status.setText("⚠ El modelo no está cargado. Ve a Ajustes para cargarlo.")
            return

        self._running = True
        self._session = self._session_manager.new_session()
        self._lbl_seq_count.setText("Secuencias: 0")

        active_emotions = [
            EMOTION_LABELS_ES[raw]
            for raw in sorted(self._engine.label_map, key=lambda k: self._engine.label_map[k])
            if raw in EMOTION_LABELS_ES
        ]
        self._rebuild_bars(active_emotions)

        self._pipeline = VideoPipeline(self._video_path, self._engine)
        self._pipeline.frame_ready.connect(self._on_frame)
        self._pipeline.annotated_frame_ready.connect(self._on_annotated_frame)
        self._pipeline.progress.connect(self._on_progress)
        self._pipeline.sequence_result.connect(self._on_sequence_result)
        self._pipeline.finished_processing.connect(self._on_finished)
        self._pipeline.error.connect(self._on_pipeline_error)
        self._pipeline.gradcam_ready.connect(self._on_gradcam_frame)
        self._pipeline.shap_ready.connect(self._on_shap_frame)
        if self._chk_gradcam.isChecked():
            self._pipeline.enable_gradcam(True)
        if self._chk_shap.isChecked():
            self._pipeline.enable_shap(True)
        self._pipeline.start()

        self._btn_select.setEnabled(False)
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.setStyleSheet(self._s_btn_disabled())
        self._btn_stop.setEnabled(True)
        self._btn_stop.setStyleSheet(self._s_btn_stop())
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._lbl_status.setText("Iniciando análisis…")

    @pyqtSlot()
    def _cancel_analysis(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._session:
            self._session.close()
            self._session = None
        self._top_gradcam.clear()
        self._top_shap.clear()
        self._reset_controls()
        self._lbl_status.setText("Análisis cancelado.")

    # ── Slots del pipeline ────────────────────────────────────────────────

    @pyqtSlot(bytes)
    def _on_frame(self, jpeg_bytes: bytes) -> None:
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

    @pyqtSlot(bytes)
    def _on_annotated_frame(self, jpeg_bytes: bytes) -> None:
        image = QImage.fromData(jpeg_bytes, "JPEG")
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self._annotated_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._annotated_label.setPixmap(scaled)

    @pyqtSlot(bytes, bytes, float, str)
    def _on_gradcam_frame(
        self, xai_jpeg: bytes, face_jpeg: bytes, confidence: float, emotion: str
    ) -> None:
        image = QImage.fromData(xai_jpeg, "JPEG")
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            self._gradcam_lbl.setPixmap(
                pixmap.scaled(
                    self._gradcam_lbl.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._collect_top_xai(self._top_gradcam, confidence, emotion, xai_jpeg, face_jpeg)

    @pyqtSlot(bytes, bytes, float, str)
    def _on_shap_frame(
        self, xai_jpeg: bytes, face_jpeg: bytes, confidence: float, emotion: str
    ) -> None:
        image = QImage.fromData(xai_jpeg, "JPEG")
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            self._shap_lbl.setPixmap(
                pixmap.scaled(
                    self._shap_lbl.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._collect_top_xai(self._top_shap, confidence, emotion, xai_jpeg, face_jpeg)

    @pyqtSlot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setValue(int(current * 100 / total))
            self._lbl_status.setText(f"Procesando frame {current} / {total}")

    @pyqtSlot(object)
    def _on_sequence_result(self, result: InferenceResult) -> None:
        self._update_result_panel(result)

        if self._session:
            pred = Prediction(
                timestamp          = datetime.now(timezone.utc).isoformat(),
                emotion            = result.emotion,
                raw_label          = result.raw_label,
                confidence         = result.confidence,
                is_valid           = result.is_valid,
                duration_ms        = 500,
                landmarks_detected = True,
                probs              = result.probs,
            )
            self._session.add_prediction(pred)
            self._lbl_seq_count.setText(f"Secuencias: {self._session.prediction_count}")

    @pyqtSlot(int)
    def _on_finished(self, seq_count: int) -> None:
        self._running = False
        self._progress_bar.setValue(100)
        self._lbl_status.setText(
            f"✔ Análisis completado — {seq_count} secuencias procesadas"
        )
        self._lbl_status.setStyleSheet(f"color: {_GREEN}; font-size: 11px;")

        session_id = ""
        if self._session:
            if self._top_gradcam or self._top_shap:
                self._session.save_xai_frames(self._top_gradcam, self._top_shap)
            meta = self._session.close()
            session_id = meta.session_id
            self._session = None

        self._top_gradcam.clear()
        self._top_shap.clear()
        self._reset_controls()

        if session_id and seq_count > 0:
            self.session_finished.emit(session_id)

    @pyqtSlot(str)
    def _on_pipeline_error(self, msg: str) -> None:
        self._lbl_status.setText(f"⚠ {msg[:80]}")
        self._lbl_status.setStyleSheet(f"color: {_AMBER}; font-size: 11px;")

    # ── Helpers de UI ─────────────────────────────────────────────────────

    def _reset_controls(self) -> None:
        self._btn_select.setEnabled(True)
        self._btn_analyze.setEnabled(self._video_path is not None)
        if self._video_path:
            self._btn_analyze.setStyleSheet(self._s_btn_analyze())
        else:
            self._btn_analyze.setStyleSheet(self._s_btn_disabled())
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(self._s_btn_disabled())

    @staticmethod
    def _collect_top_xai(
        frames: list,
        confidence: float,
        emotion: str,
        xai_jpeg: bytes,
        face_jpeg: bytes,
    ) -> None:
        if emotion == "Neutral" and confidence >= CONFIDENCE_VALID:
            return
        frames.append((confidence, emotion, xai_jpeg, face_jpeg))

    def _update_result_panel(self, result: InferenceResult) -> None:
        color = EMOTION_COLORS.get(result.emotion, self._theme.text)
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
            self._pct_labels[emo_es].setText(f"{prob * 100:.0f}%")
