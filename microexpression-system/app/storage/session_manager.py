"""
session_manager.py
------------------
Gestión de sesiones de análisis de microexpresiones.

Responsabilidades:
  - Crear y persistir sesiones (RB06: máx. 50 sesiones)
  - Almacenar predicciones individuales por sesión
  - Exportar sesión a CSV (RB08)
  - Listar y eliminar sesiones históricas

Formato CSV (RB08):
  session_id, timestamp, emotion, confidence, intensity, duration_ms,
  landmarks_detected
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Constantes ─────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.parent.parent  # microexpression-system/
SESSIONS_DIR  = BASE_DIR / "data" / "sessions"
EXPORTS_DIR   = BASE_DIR / "data" / "exports"
MAX_SESSIONS  = 50   # RB06


# ── Estructuras de datos ────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """Una predicción individual dentro de una sesión."""
    timestamp: str            # ISO-8601 del momento de la detección
    emotion: str              # e.g. "Alegría"
    raw_label: str            # label del modelo, e.g. "felicidad"
    confidence: float         # 0.0 – 1.0
    is_valid: bool            # confidence >= 0.70 y duración válida (RB01, RB02)
    duration_ms: int          # duración estimada en ms (RB01: 100-500 ms)
    landmarks_detected: bool  # si MediaPipe detectó rostro en ese frame (RB03)
    intensity: float = 0.0    # placeholder; puede calcularse externamente

    def to_csv_row(self, session_id: str) -> dict:
        return {
            "session_id":         session_id,
            "timestamp":          self.timestamp,
            "emotion":            self.emotion,
            "confidence":         f"{self.confidence:.4f}",
            "intensity":          f"{self.intensity:.4f}",
            "duration_ms":        self.duration_ms,
            "landmarks_detected": int(self.landmarks_detected),
        }


@dataclass
class SessionMeta:
    """Metadatos de una sesión almacenada."""
    session_id: str
    created_at: str           # ISO-8601
    total_predictions: int
    valid_predictions: int
    dominant_emotion: str
    mean_confidence: float
    duration_secs: float


# ── Session (activa) ────────────────────────────────────────────────────────────

class Session:
    """
    Sesión de análisis en curso.

    Uso:
        session = SessionManager().new_session()
        session.add_prediction(pred)
        session.close()
    """

    def __init__(self, session_id: str, sessions_dir: Path):
        self.session_id   = session_id
        self._dir         = sessions_dir / session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._predictions: list[Prediction] = []
        self._started_at  = datetime.now()

    # ── API pública ─────────────────────────────────────────────────────────

    def add_prediction(self, prediction: Prediction) -> None:
        """Agrega una predicción a la sesión en memoria."""
        self._predictions.append(prediction)

    def close(self) -> SessionMeta:
        """
        Persiste la sesión en disco y devuelve sus metadatos.
        Escribe:
          - predictions.json  (raw)
          - meta.json         (resumen)
        """
        # escribir predicciones
        preds_path = self._dir / "predictions.json"
        preds_path.write_text(
            json.dumps([asdict(p) for p in self._predictions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # calcular metadatos
        duration_secs = (datetime.now() - self._started_at).total_seconds()
        valid = [p for p in self._predictions if p.is_valid]
        dominant = _dominant_emotion(valid or self._predictions)
        mean_conf = (
            sum(p.confidence for p in valid) / len(valid)
            if valid else 0.0
        )

        meta = SessionMeta(
            session_id        = self.session_id,
            created_at        = self._started_at.isoformat(),
            total_predictions = len(self._predictions),
            valid_predictions = len(valid),
            dominant_emotion  = dominant,
            mean_confidence   = round(mean_conf, 4),
            duration_secs     = round(duration_secs, 2),
        )

        meta_path = self._dir / "meta.json"
        meta_path.write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta

    @property
    def predictions(self) -> list[Prediction]:
        return list(self._predictions)

    @property
    def prediction_count(self) -> int:
        return len(self._predictions)


# ── SessionManager ──────────────────────────────────────────────────────────────

class SessionManager:
    """
    Punto de entrada principal para gestión de sesiones.

    Uso:
        manager = SessionManager()
        session = manager.new_session()
        ...
        meta = session.close()
        csv_path = manager.export_session_csv(session.session_id)
    """

    def __init__(
        self,
        sessions_dir: Path = SESSIONS_DIR,
        exports_dir:  Path = EXPORTS_DIR,
        max_sessions: int  = MAX_SESSIONS,
    ):
        self._sessions_dir = sessions_dir
        self._exports_dir  = exports_dir
        self._max_sessions = max_sessions
        sessions_dir.mkdir(parents=True, exist_ok=True)
        exports_dir.mkdir(parents=True, exist_ok=True)

    # ── Sesión nueva ────────────────────────────────────────────────────────

    def new_session(self) -> Session:
        """
        Crea una sesión nueva.
        Si ya hay MAX_SESSIONS sesiones, elimina la más antigua (RB06).
        """
        self._enforce_session_limit()
        session_id = _make_session_id()
        return Session(session_id, self._sessions_dir)

    # ── Consulta de historial ───────────────────────────────────────────────

    def list_sessions(self) -> list[SessionMeta]:
        """Devuelve los metadatos de todas las sesiones, ordenadas por fecha (más reciente primero)."""
        metas = []
        for d in self._sessions_dir.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                metas.append(SessionMeta(**data))
            except Exception:
                continue
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas

    def load_predictions(self, session_id: str) -> list[Prediction]:
        """Carga las predicciones de una sesión persisted."""
        pred_path = self._sessions_dir / session_id / "predictions.json"
        if not pred_path.exists():
            return []
        raw = json.loads(pred_path.read_text(encoding="utf-8"))
        return [Prediction(**r) for r in raw]

    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión del disco. Devuelve True si existía."""
        session_dir = self._sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            return True
        return False

    # ── Exportación CSV ─────────────────────────────────────────────────────

    def export_session_csv(self, session_id: str) -> Path:
        """
        Exporta la sesión a CSV (RB08).
        Devuelve la ruta del archivo generado.
        """
        predictions = self.load_predictions(session_id)
        if not predictions:
            raise ValueError(f"No se encontraron predicciones para la sesión '{session_id}'")

        csv_path = self._exports_dir / f"{session_id}.csv"
        fieldnames = [
            "session_id", "timestamp", "emotion", "confidence",
            "intensity", "duration_ms", "landmarks_detected",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pred in predictions:
                writer.writerow(pred.to_csv_row(session_id))

        return csv_path

    # ── Interno ─────────────────────────────────────────────────────────────

    def _enforce_session_limit(self) -> None:
        """Si hay >= MAX_SESSIONS sesiones, elimina la(s) más antigua(s)."""
        sessions = self.list_sessions()  # ya ordenadas: [más reciente, ..., más antigua]
        while len(sessions) >= self._max_sessions:
            oldest = sessions.pop()
            self.delete_session(oldest.session_id)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _make_session_id() -> str:
    """ID único basado en timestamp: session_20250101_153045."""
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _dominant_emotion(predictions: list[Prediction]) -> str:
    """Emoción más frecuente en la lista de predicciones."""
    if not predictions:
        return "Neutral"
    counts: dict[str, int] = {}
    for p in predictions:
        counts[p.emotion] = counts.get(p.emotion, 0) + 1
    return max(counts, key=lambda e: counts[e])
