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

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QScrollArea, QFileDialog, QMessageBox,
    QGridLayout,
)

from app.storage.session_manager import SessionManager, SessionMeta
from app.analytics.stats_engine import StatsEngine, SessionStats

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ── Colores ────────────────────────────────────────────────────────────────────

_BG      = "#121212"
_SURFACE = "#1E1E1E"
_ACCENT  = "#2979FF"
_TEXT    = "#E0E0E0"
_SUBTEXT = "#9E9E9E"
_GREEN   = "#4CAF50"
_AMBER   = "#FFC107"
_RED     = "#F44336"

_BTN_EXPORT = (
    "QPushButton { background:#2979FF; color:#fff; border:none; border-radius:6px; "
    "padding:8px 20px; font-size:13px; } "
    "QPushButton:hover { background:#5499FF; } "
    "QPushButton:disabled { background:#424242; color:#757575; }"
)


class ResultsScreen(QWidget):
    """Pantalla que muestra los resultados de la sesión más reciente."""

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._session_manager = session_manager
        self._stats_engine    = StatsEngine()
        self._current_session_id: Optional[str] = None
        self._current_stats:      Optional[SessionStats] = None
        self._current_meta:       Optional[SessionMeta]  = None

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
        self.setStyleSheet(f"background: {_BG}; color: {_TEXT};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # Título + botón exportar
        header = QHBoxLayout()
        self._lbl_title = QLabel("Resultados de sesión")
        self._lbl_title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {_TEXT};")
        header.addWidget(self._lbl_title)
        header.addStretch()
        self._btn_export = QPushButton("⬇  Exportar CSV")
        self._btn_export.setStyleSheet(_BTN_EXPORT)
        self._btn_export.setFixedHeight(36)
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        header.addWidget(self._btn_export)
        root.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2A2A2A;")
        root.addWidget(sep)

        # Panel central con scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        root.addWidget(scroll)

        self._content = QWidget()
        self._content.setStyleSheet(f"background: {_BG};")
        scroll.setWidget(self._content)

        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setSpacing(16)
        self._content_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder inicial
        self._placeholder = QLabel("No hay resultados.\nEjecuta una sesión de análisis primero.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {_SUBTEXT}; font-size: 14px;")
        self._content_layout.addWidget(self._placeholder, alignment=Qt.AlignmentFlag.AlignCenter)
        self._content_layout.addStretch()

    def _show_empty(self) -> None:
        self._placeholder.show()
        self._btn_export.setEnabled(False)

    def _refresh_ui(self) -> None:
        if not self._current_stats:
            return

        # Limpiar contenido previo
        self._placeholder.hide()
        while self._content_layout.count() > 1:
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        stats = self._current_stats

        # Métricas resumen
        metrics_card = self._build_metrics_card(stats)
        self._content_layout.insertWidget(0, metrics_card)

        # Gráfica de barras
        if _MPL_AVAILABLE and stats.valid_predictions > 0:
            chart = self._build_bar_chart(stats)
            self._content_layout.insertWidget(1, chart)

        self._btn_export.setEnabled(True)
        if self._current_session_id:
            self._lbl_title.setText(f"Resultados — {self._current_session_id}")

    def _build_metrics_card(self, stats: SessionStats) -> QWidget:
        card = QFrame()
        card.setStyleSheet(f"background: {_SURFACE}; border-radius: 10px;")

        grid = QGridLayout(card)
        grid.setContentsMargins(20, 16, 20, 16)
        grid.setSpacing(12)

        metrics = [
            ("Predicciones totales",  str(stats.total_predictions),           _TEXT),
            ("Predicciones válidas",  str(stats.valid_predictions),            _GREEN),
            ("Descartadas",           str(stats.discarded),                    _AMBER if stats.discarded else _SUBTEXT),
            ("Emoción dominante",     stats.emotion_stats.dominant_emotion,    _ACCENT),
            ("Confianza promedio",    f"{stats.mean_confidence:.1%}",          _TEXT),
            ("Calidad landmarks",     f"{stats.landmark_quality:.1%}",
             _GREEN if stats.quality_ok else _RED),
        ]

        for row_idx, (label, value, color) in enumerate(metrics):
            lbl_k = QLabel(label)
            lbl_k.setStyleSheet(f"color: {_SUBTEXT}; font-size: 12px;")
            lbl_v = QLabel(value)
            lbl_v.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold;")
            col = (row_idx % 3) * 2
            row = row_idx // 3
            grid.addWidget(lbl_k, row * 2,     col)
            grid.addWidget(lbl_v, row * 2 + 1, col)

        return card

    def _build_bar_chart(self, stats: SessionStats) -> QWidget:
        chart_data = self._stats_engine.bar_chart_data(stats)
        if not chart_data["labels"]:
            return QWidget()

        fig = Figure(figsize=(6, 3), facecolor=_SURFACE)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(_SURFACE)

        bars = ax.bar(
            chart_data["labels"],
            chart_data["values"],
            color=chart_data["colors"],
            width=0.6,
        )

        # Etiquetas de porcentaje sobre cada barra
        for bar_rect, pct in zip(bars, chart_data["percentages"]):
            height = bar_rect.get_height()
            if height > 0:
                ax.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2.0,
                    height + 0.05,
                    f"{pct:.1f}%",
                    ha="center", va="bottom",
                    color=_TEXT, fontsize=9,
                )

        ax.set_ylabel("Predicciones válidas", color=_SUBTEXT, fontsize=10)
        ax.set_title("Distribución de emociones", color=_TEXT, fontsize=12, pad=12)
        ax.tick_params(colors=_TEXT)
        ax.spines[:].set_color("#333")
        for label in ax.get_xticklabels():
            label.set_color(_TEXT)
            label.set_fontsize(9)
        for label in ax.get_yticklabels():
            label.set_color(_SUBTEXT)
        ax.yaxis.label.set_color(_SUBTEXT)

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f"background: {_SURFACE}; border-radius: 10px;")
        canvas.setFixedHeight(280)

        wrapper = QFrame()
        wrapper.setStyleSheet(f"background: {_SURFACE}; border-radius: 10px;")
        vbox = QVBoxLayout(wrapper)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.addWidget(canvas)
        return wrapper

    # ── Exportación ───────────────────────────────────────────────────────

    @pyqtSlot()
    def _export_csv(self) -> None:
        if not self._current_session_id:
            return
        try:
            csv_path = self._session_manager.export_session_csv(self._current_session_id)
            # Ofrecer guardar en ubicación elegida por el usuario
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
