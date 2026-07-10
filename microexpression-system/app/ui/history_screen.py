"""
history_screen.py
-----------------
Pantalla de historial de sesiones.

Muestra una tabla con todas las sesiones almacenadas:
  Fecha/Hora | Predicciones | Válidas | Emoción dominante | Confianza | Duración

Acciones disponibles:
  - Ver resultados de una sesión
  - Exportar CSV
  - Eliminar sesión
  - Refrescar lista
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QMessageBox, QAbstractItemView, QFileDialog, QStyle,
)

from app.storage.session_manager import SessionManager, SessionMeta
from app.ui.theme import Theme, ThemeManager


# ── Colores semánticos fijos ──────────────────────────────────────────────────

_GREEN = "#4CAF50"
_RED   = "#F44336"
_AMBER = "#FFC107"

_COLUMNS = ["Fecha / Hora", "Predicciones", "Válidas", "Emoción dominante",
            "Confianza media", "Duración (s)"]


def _table_style(theme: Theme) -> str:
    t = theme
    return f"""
QTableWidget {{
    background: {t.surface};
    color: {t.text};
    border: none;
    gridline-color: {t.divider};
    font-size: 13px;
    border-radius: 8px;
    selection-background-color: {t.accent_alpha};
}}
QTableWidget::item {{
    padding: 6px 10px;
}}
QHeaderView::section {{
    background: {t.table_header_bg};
    color: {t.subtext};
    border: none;
    border-bottom: 1px solid {t.divider};
    padding: 8px 10px;
    font-size: 12px;
    font-weight: bold;
}}
QScrollBar:vertical {{
    background: {t.scrollbar_bg};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {t.scrollbar_handle};
    border-radius: 4px;
}}
"""


def _btn_style(bg: str, fg: str, hover: str) -> str:
    return (
        f"QPushButton {{ background:{bg}; color:{fg}; border:none; border-radius:6px; "
        f"padding:7px 16px; font-size:12px; }} "
        f"QPushButton:hover {{ background:{hover}; }} "
        f"QPushButton:disabled {{ background:#424242; color:#757575; }}"
    )


class HistoryScreen(QWidget):
    """Pantalla de historial de sesiones pasadas."""

    view_session_requested = pyqtSignal(str)  # session_id

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._session_manager = session_manager
        self._metas: list[SessionMeta] = []
        self._theme: Theme = ThemeManager.current()

        self._build_ui()
        self.refresh()

    # ── API pública ───────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Recarga la lista de sesiones desde disco."""
        self._metas = self._session_manager.list_sessions()
        self._populate_table()
        self._update_btn_states()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        t = self._theme
        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # Título + botones globales
        header = QHBoxLayout()
        self._lbl_header = QLabel("Historial de sesiones")
        self._lbl_header.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t.text};")
        header.addWidget(self._lbl_header)
        header.addStretch()

        self._btn_refresh = QPushButton("Actualizar")
        self._btn_refresh.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self._btn_refresh.setIconSize(QSize(16, 16))
        self._btn_refresh.setStyleSheet(
            _btn_style(t.btn_neutral_bg, t.text, t.btn_neutral_hover)
        )
        self._btn_refresh.setFixedHeight(34)
        self._btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._btn_refresh)

        root.addLayout(header)

        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.HLine)
        self._sep.setStyleSheet(f"color: {t.divider};")
        root.addWidget(self._sep)

        # Tabla
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setStyleSheet(_table_style(t))
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.selectionModel().selectionChanged.connect(self._update_btn_states)
        root.addWidget(self._table)

        # Barra de acciones sobre selección
        action_bar = QHBoxLayout()
        action_bar.setSpacing(10)

        self._btn_view = QPushButton("Ver resultados")
        self._btn_view.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self._btn_view.setIconSize(QSize(16, 16))
        self._btn_view.setStyleSheet(_btn_style(t.accent, "#fff", t.accent_h))
        self._btn_view.setFixedHeight(34)
        self._btn_view.setEnabled(False)
        self._btn_view.clicked.connect(self._view_selected)
        action_bar.addWidget(self._btn_view)

        self._btn_export = QPushButton("Exportar CSV")
        self._btn_export.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
        self._btn_export.setIconSize(QSize(16, 16))
        self._btn_export.setStyleSheet(
            _btn_style(t.btn_neutral_bg, t.text, t.btn_neutral_hover)
        )
        self._btn_export.setFixedHeight(34)
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_selected)
        action_bar.addWidget(self._btn_export)

        action_bar.addStretch()

        self._btn_delete = QPushButton("Eliminar")
        self._btn_delete.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self._btn_delete.setIconSize(QSize(16, 16))
        self._btn_delete.setStyleSheet(_btn_style(_RED, "#fff", "#EF5350"))
        self._btn_delete.setFixedHeight(34)
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._delete_selected)
        action_bar.addWidget(self._btn_delete)

        root.addLayout(action_bar)

        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet(f"color: {t.subtext}; font-size: 11px;")
        root.addWidget(self._lbl_count)

    # ── Tema ──────────────────────────────────────────────────────────────

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        t = theme

        self.setStyleSheet(f"background: {t.bg}; color: {t.text};")
        self._lbl_header.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t.text};")
        self._sep.setStyleSheet(f"color: {t.divider};")
        self._lbl_count.setStyleSheet(f"color: {t.subtext}; font-size: 11px;")

        # Botones
        self._btn_refresh.setStyleSheet(_btn_style(t.btn_neutral_bg, t.text, t.btn_neutral_hover))
        self._btn_view.setStyleSheet(_btn_style(t.accent, "#fff", t.accent_h))
        self._btn_export.setStyleSheet(_btn_style(t.btn_neutral_bg, t.text, t.btn_neutral_hover))
        self._btn_delete.setStyleSheet(_btn_style(_RED, "#fff", "#EF5350"))

        # Tabla
        self._table.setStyleSheet(_table_style(t))

    # ── Tabla ─────────────────────────────────────────────────────────────

    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        for meta in self._metas:
            row = self._table.rowCount()
            self._table.insertRow(row)

            dt_str = meta.created_at[:19].replace("T", "  ") if meta.created_at else "–"

            cells = [
                dt_str,
                str(meta.total_predictions),
                str(meta.valid_predictions),
                meta.dominant_emotion or "–",
                f"{meta.mean_confidence:.1%}",
                f"{meta.duration_secs:.1f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                self._table.setItem(row, col, item)

        count = len(self._metas)
        self._lbl_count.setText(f"{count} sesión{'es' if count != 1 else ''} almacenada{'s' if count != 1 else ''}")

    # ── Estado de botones ─────────────────────────────────────────────────

    def _update_btn_states(self, *_) -> None:
        has = bool(self._selected_row() is not None)
        self._btn_view.setEnabled(has)
        self._btn_export.setEnabled(has)
        self._btn_delete.setEnabled(has)

    def _selected_row(self) -> Optional[int]:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _selected_session_id(self) -> Optional[str]:
        row = self._selected_row()
        if row is None or row >= len(self._metas):
            return None
        return self._metas[row].session_id

    # ── Acciones ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def _view_selected(self) -> None:
        sid = self._selected_session_id()
        if sid:
            self.view_session_requested.emit(sid)

    @pyqtSlot()
    def _export_selected(self) -> None:
        sid = self._selected_session_id()
        if not sid:
            return
        try:
            csv_path = self._session_manager.export_session_csv(sid)
            dest, _ = QFileDialog.getSaveFileName(
                self, "Guardar CSV",
                str(csv_path.parent / f"{sid}.csv"),
                "Archivos CSV (*.csv)",
            )
            if dest:
                import shutil
                shutil.copy(str(csv_path), dest)
                QMessageBox.information(self, "Exportación exitosa", f"Guardado en:\n{dest}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))

    @pyqtSlot()
    def _delete_selected(self) -> None:
        sid = self._selected_session_id()
        if not sid:
            return
        reply = QMessageBox.question(
            self,
            "Eliminar sesión",
            f"¿Eliminar la sesión '{sid}'?\nEsta acción no se puede deshacer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._session_manager.delete_session(sid)
            self.refresh()
