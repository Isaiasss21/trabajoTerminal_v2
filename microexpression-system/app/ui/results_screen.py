"""
results_screen.py
-----------------
Pantalla de resultados de sesión:
  - Resumen textual de métricas
  - Gráfica de barras de distribución de emociones (matplotlib embebido)
  - Botón de exportar CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QScrollArea, QFileDialog, QMessageBox,
    QGridLayout, QStyle,
)

from app.storage.session_manager import SessionManager, SessionMeta
from app.analytics.stats_engine import StatsEngine, SessionStats
from app.ui.theme import Theme, ThemeManager

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ── Colores semánticos fijos ──────────────────────────────────────────────────

_GREEN = "#4CAF50"
_AMBER = "#FFC107"
_RED   = "#F44336"


class ResultsScreen(QWidget):
    """Pantalla que muestra los resultados de la sesión más reciente."""

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._session_manager = session_manager
        self._stats_engine    = StatsEngine()
        self._current_session_id: Optional[str] = None
        self._current_stats:      Optional[SessionStats] = None
        self._current_meta:       Optional[SessionMeta]  = None
        self._theme: Theme = ThemeManager.current()

        self._build_ui()
        self._show_empty()

    # ── API pública ───────────────────────────────────────────────────────

    def load_session(self, session_id: str) -> None:
        """Carga y muestra los resultados de la sesión indicada."""
        predictions = self._session_manager.load_predictions(session_id)
        meta_list = [m for m in self._session_manager.list_sessions() if m.session_id == session_id]
        self._current_session_id = session_id
        self._current_meta       = meta_list[0] if meta_list else None
        self._current_stats      = self._stats_engine.compute(predictions)
        self._refresh_ui()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        t = self._theme
        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # Título + botón exportar
        header = QHBoxLayout()
        self._lbl_title = QLabel("Resultados de sesión")
        self._lbl_title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t.text};")
        header.addWidget(self._lbl_title)
        header.addStretch()
        self._btn_export = QPushButton("Exportar CSV")
        self._btn_export.setStyleSheet(self._btn_export_style())
        self._btn_export.setFixedHeight(36)
        self._btn_export.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
        self._btn_export.setIconSize(QSize(16, 16))
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        header.addWidget(self._btn_export)
        root.addLayout(header)

        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.HLine)
        self._sep.setStyleSheet(f"color: {t.divider};")
        root.addWidget(self._sep)

        # Panel central con scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        root.addWidget(scroll)

        self._content = QWidget()
        self._content.setStyleSheet(f"background: {t.bg};")
        scroll.setWidget(self._content)

        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setSpacing(16)
        self._content_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder inicial
        self._placeholder = QLabel("No hay resultados.\nEjecuta una sesión de análisis primero.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {t.subtext}; font-size: 14px;")
        self._content_layout.addWidget(self._placeholder, alignment=Qt.AlignmentFlag.AlignCenter)
        self._content_layout.addStretch()

    def _btn_export_style(self) -> str:
        t = self._theme
        return (
            f"QPushButton {{ background:{t.accent}; color:#fff; border:none; border-radius:6px; "
            f"padding:8px 20px; font-size:13px; }} "
            f"QPushButton:hover {{ background:{t.accent_h}; }} "
            f"QPushButton:disabled {{ background:{t.btn_disabled_bg}; color:{t.btn_disabled_text}; }}"
        )

    def _show_empty(self) -> None:
        self._placeholder.show()
        self._btn_export.setEnabled(False)

    def _refresh_ui(self) -> None:
        if not self._current_stats:
            return

        # Limpiar contenido previo (todo excepto el stretch final)
        while self._content_layout.count() > 1:
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._placeholder:
                w.deleteLater()
        try:
            self._placeholder.hide()
        except RuntimeError:
            pass

        stats = self._current_stats
        t = self._theme

        metrics_card = self._build_metrics_card(stats)
        self._content_layout.insertWidget(0, metrics_card)

        if _MPL_AVAILABLE and stats.valid_predictions > 0:
            chart = self._build_evolution_chart(stats)
            self._content_layout.insertWidget(1, chart)

        xai_strip = self._build_xai_strip()
        if xai_strip is not None:
            self._content_layout.insertWidget(2, xai_strip)

        self._btn_export.setEnabled(True)
        if self._current_session_id:
            self._lbl_title.setText(f"Resultados — {self._current_session_id}")

    def _build_metrics_card(self, stats: SessionStats) -> QWidget:
        t = self._theme
        card = QFrame()
        card.setStyleSheet(f"background: {t.surface}; border-radius: 10px;")

        grid = QGridLayout(card)
        grid.setContentsMargins(20, 16, 20, 16)
        grid.setSpacing(12)

        metrics = [
            ("Predicciones totales",  str(stats.total_predictions),           t.text),
            ("Predicciones válidas",  str(stats.valid_predictions),            _GREEN),
            ("Descartadas",           str(stats.discarded),                    _AMBER if stats.discarded else t.subtext),
            ("Emoción dominante",     stats.emotion_stats.dominant_emotion,    t.accent),
            ("Confianza promedio",    f"{stats.mean_confidence:.1%}",          t.text),
            ("Calidad landmarks",     f"{stats.landmark_quality:.1%}",
             _GREEN if stats.quality_ok else _RED),
        ]

        for row_idx, (label, value, color) in enumerate(metrics):
            lbl_k = QLabel(label)
            lbl_k.setStyleSheet(f"color: {t.subtext}; font-size: 12px;")
            lbl_v = QLabel(value)
            lbl_v.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold;")
            col = (row_idx % 3) * 2
            row = row_idx // 3
            grid.addWidget(lbl_k, row * 2,     col)
            grid.addWidget(lbl_v, row * 2 + 1, col)

        return card

    def _build_evolution_chart(self, stats: SessionStats) -> QWidget:
        from collections import defaultdict
        from app.inference.inference_engine import EMOTION_COLORS

        t = self._theme

        predictions = self._session_manager.load_predictions(self._current_session_id or "")
        valid_preds = [p for p in predictions if p.is_valid and getattr(p, "probs", None)]

        if not valid_preds:
            rows = self._stats_engine.timeline_data(stats)["rows"]
            if not rows:
                return QWidget()
            emotion_data: dict[str, dict] = defaultdict(lambda: {"x": [], "y": []})
            for idx, row in enumerate(rows):
                emotion_data[row["emotion"]]["x"].append(idx + 1)
                emotion_data[row["emotion"]]["y"].append(row["confidence"])
            top3_emotions = sorted(
                emotion_data, key=lambda e: sum(emotion_data[e]["y"]), reverse=True
            )[:3]
        else:
            emotion_sum: dict[str, float] = defaultdict(float)
            for p in valid_preds:
                for emotion, prob in p.probs.items():
                    emotion_sum[emotion] += prob
            top3_emotions = sorted(emotion_sum, key=lambda e: emotion_sum[e], reverse=True)[:3]
            emotion_data = {}
            for emotion in top3_emotions:
                emotion_data[emotion] = {
                    "x": list(range(1, len(valid_preds) + 1)),
                    "y": [p.probs.get(emotion, 0.0) for p in valid_preds],
                }

        if not emotion_data:
            return QWidget()

        fig = Figure(figsize=(7, 3.5), facecolor=t.surface)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(t.surface)

        for emotion in top3_emotions:
            data = emotion_data[emotion]
            color = EMOTION_COLORS.get(emotion, "#78909C")
            ax.plot(
                data["x"], data["y"],
                "o-", color=color, label=emotion,
                linewidth=1.8, markersize=5, alpha=0.88,
            )

        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Predicción #", color=t.subtext, fontsize=9)
        ax.set_ylabel("Confianza", color=t.subtext, fontsize=9)
        ax.set_title("Evolución de emociones en la sesión", color=t.text, fontsize=12, pad=12)
        ax.tick_params(colors=t.subtext, labelsize=8)
        ax.spines[:].set_color(t.mpl_spine)
        for lbl in ax.get_xticklabels():
            lbl.set_color(t.subtext)
        for lbl in ax.get_yticklabels():
            lbl.set_color(t.subtext)
        ax.yaxis.label.set_color(t.subtext)
        ax.xaxis.label.set_color(t.subtext)
        ax.grid(True, color=t.mpl_grid, linestyle="--", linewidth=0.6, alpha=0.7)

        legend = ax.legend(
            fontsize=8, labelcolor=t.text,
            framealpha=0.25, facecolor=t.surface,
            edgecolor=t.divider,
        )

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f"background: {t.surface}; border-radius: 10px;")
        canvas.setFixedHeight(300)

        wrapper = QFrame()
        wrapper.setStyleSheet(f"background: {t.surface}; border-radius: 10px;")
        vbox = QVBoxLayout(wrapper)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.addWidget(canvas)
        return wrapper

    def _build_xai_strip(self) -> "QWidget | None":
        if not self._current_session_id:
            return None

        xai_data = self._session_manager.load_xai_frames(self._current_session_id)
        gradcam_frames = xai_data.get("gradcam", [])
        shap_frames    = xai_data.get("shap",    [])

        if not gradcam_frames and not shap_frames:
            return None

        from app.inference.inference_engine import EMOTION_COLORS
        t = self._theme

        outer = QFrame()
        outer.setStyleSheet(f"background: {t.surface}; border-radius: 10px;")
        vbox = QVBoxLayout(outer)
        vbox.setContentsMargins(16, 14, 16, 14)
        vbox.setSpacing(12)

        n_gcam = len(gradcam_frames)
        n_shap = len(shap_frames)
        count_txt = f"({max(n_gcam, n_shap)} momentos de cambio)"
        title = QLabel(f"Momentos de emoción no neutra — XAI  {count_txt}")
        title.setStyleSheet(f"color: {t.text}; font-size: 13px; font-weight: bold;")
        vbox.addWidget(title)

        IMG_W, IMG_H = 140, 140

        def _make_row(frames: list, row_title: str, tag_color: str) -> None:
            if not frames:
                return
            row_lbl = QLabel(row_title)
            row_lbl.setStyleSheet(
                f"color: {tag_color}; font-size: 11px; font-weight: bold; padding-bottom: 2px;"
            )
            vbox.addWidget(row_lbl)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedHeight(IMG_H * 2 + 70)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setStyleSheet(
                "QScrollArea { border: none; background: transparent; }"
                f"QScrollBar:horizontal {{ height: 6px; background: {t.scrollbar_bg}; }}"
                f"QScrollBar::handle:horizontal {{ background: {t.scrollbar_handle}; border-radius: 3px; }}"
            )

            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)

            for conf, emotion, xai_jpeg, face_jpeg in frames:
                card = QFrame()
                card.setStyleSheet(f"QFrame {{ background: {t.card_dark}; border-radius: 8px; }}")
                card.setFixedWidth(IMG_W + 8)
                card_vbox = QVBoxLayout(card)
                card_vbox.setContentsMargins(4, 4, 4, 6)
                card_vbox.setSpacing(3)

                emo_color = EMOTION_COLORS.get(emotion, t.subtext)

                info_lbl = QLabel(f"{emotion}  {conf * 100:.0f}%")
                info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                info_lbl.setStyleSheet(
                    f"color: {emo_color}; font-size: 10px; font-weight: bold;"
                )
                card_vbox.addWidget(info_lbl)

                def _img_label(data: bytes) -> QLabel:
                    lbl = QLabel()
                    lbl.setFixedSize(IMG_W, IMG_H)
                    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    lbl.setStyleSheet("background: #1A1A1A; border-radius: 4px;")
                    if data:
                        qimg = QImage.fromData(data, "JPEG")
                        if not qimg.isNull():
                            lbl.setPixmap(
                                QPixmap.fromImage(qimg).scaled(
                                    IMG_W, IMG_H,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation,
                                )
                            )
                    return lbl

                face_title = QLabel("ROI")
                face_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                face_title.setStyleSheet(f"color: {t.subtext}; font-size: 9px;")
                card_vbox.addWidget(face_title)
                card_vbox.addWidget(_img_label(face_jpeg))

                xai_title = QLabel("XAI")
                xai_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                xai_title.setStyleSheet(f"color: {t.subtext}; font-size: 9px;")
                card_vbox.addWidget(xai_title)
                card_vbox.addWidget(_img_label(xai_jpeg))

                row_layout.addWidget(card)

            row_layout.addStretch()
            scroll.setWidget(row_widget)
            vbox.addWidget(scroll)

        _make_row(gradcam_frames, "Grad-CAM  (JET — activaciones espaciales)", "#EF5350")
        _make_row(shap_frames,    "SHAP / Integrated Gradients  (PLASMA — atribuciones)", "#AB47BC")

        return outer

    # ── Tema ──────────────────────────────────────────────────────────────

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        t = theme

        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")
        self._lbl_title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t.text};")
        self._btn_export.setStyleSheet(self._btn_export_style())
        self._sep.setStyleSheet(f"color: {t.divider};")
        self._content.setStyleSheet(f"background: {t.bg};")
        self._placeholder.setStyleSheet(f"color: {t.subtext}; font-size: 14px;")

        # Reconstruir contenido con el nuevo tema si hay datos
        if self._current_stats:
            self._refresh_ui()

    # ── Exportación ───────────────────────────────────────────────────────

    @pyqtSlot()
    def _export_csv(self) -> None:
        if not self._current_session_id:
            return
        try:
            csv_path = self._session_manager.export_session_csv(self._current_session_id)
            dest, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar CSV",
                str(Path.home() / f"{self._current_session_id}.csv"),
                "Archivos CSV (*.csv)",
            )
            if dest:
                import shutil
                shutil.copy(str(csv_path), dest)
                QMessageBox.information(self, "Exportación exitosa",
                                        f"Sesión exportada a:\n{dest}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))
