"""
stats_engine.py
---------------
Motor de análisis estadístico para sesiones de microexpresiones.

Responsabilidades:
  - Computar distribución de emociones de una sesión
  - Calcular emoción dominante y confianza promedio
  - Aplicar filtros de calidad (RB01, RB02, RB03)
  - Proveer datos listos para gráficas (barras, pastel, timeline)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from app.inference.inference_engine import (
    CONFIDENCE_VALID,
    CONFIDENCE_UNCERTAIN,
)

# ── Constantes de reglas de negocio ────────────────────────────────────────────

DURATION_MIN_MS  = 100   # RB01: mínimo de duración de microexpresión
DURATION_MAX_MS  = 500   # RB01: máximo de duración de microexpresión
LANDMARK_QUALITY = 0.60  # RB03: fracción mínima de frames con landmarks


# ── Estructuras de datos de salida ──────────────────────────────────────────────

@dataclass
class EmotionStats:
    """Estadísticas de distribución de emociones."""
    counts:            dict[str, int]    # emoción → nº de ocurrencias
    percentages:       dict[str, float]  # emoción → % del total
    dominant_emotion:  str
    dominant_count:    int
    dominant_pct:      float


@dataclass
class SessionStats:
    """Resumen estadístico completo de una sesión."""
    total_predictions:   int
    valid_predictions:   int
    discarded:           int           # predicciones descartadas por filtros
    mean_confidence:     float
    std_confidence:      float
    landmark_quality:    float         # fracción [0,1] de frames con landmarks
    emotion_stats:       EmotionStats
    timeline:            list[dict]    # lista de {timestamp, emotion, confidence, is_valid}
    quality_ok:          bool          # True si landmark_quality >= 0.60 (RB03)


# ── Motor principal ─────────────────────────────────────────────────────────────

class StatsEngine:
    """
    Computa estadísticas de una lista de predicciones (Prediction dataclass).

    Uso:
        engine = StatsEngine()
        stats  = engine.compute(predictions)
        chart_data = engine.bar_chart_data(stats)
    """

    def compute(self, predictions: list) -> SessionStats:
        """
        Recibe lista de `Prediction` (de session_manager.py).
        No importa directamente para evitar dependencia circular; usa duck typing.
        """
        if not predictions:
            return self._empty_stats()

        # ── Filtros de calidad ───────────────────────────────────────────────

        valid = [
            p for p in predictions
            if p.is_valid
            and DURATION_MIN_MS <= p.duration_ms <= DURATION_MAX_MS
        ]
        discarded = len(predictions) - len(valid)

        # ── Confianza ────────────────────────────────────────────────────────

        confs = [p.confidence for p in valid] if valid else [0.0]
        mean_conf = sum(confs) / len(confs)
        variance  = sum((c - mean_conf) ** 2 for c in confs) / len(confs)
        std_conf  = variance ** 0.5

        # ── Calidad de landmarks (RB03) ──────────────────────────────────────

        lm_detected = sum(1 for p in predictions if p.landmarks_detected)
        lm_quality  = lm_detected / len(predictions) if predictions else 0.0
        quality_ok  = lm_quality >= LANDMARK_QUALITY

        # ── Distribución de emociones ────────────────────────────────────────

        emotion_counts = Counter(p.emotion for p in valid)
        emotion_stats  = _build_emotion_stats(emotion_counts)

        # ── Timeline ─────────────────────────────────────────────────────────

        timeline = [
            {
                "timestamp":  p.timestamp,
                "emotion":    p.emotion,
                "confidence": round(p.confidence, 4),
                "is_valid":   p.is_valid,
                "duration_ms": p.duration_ms,
            }
            for p in predictions
        ]

        return SessionStats(
            total_predictions = len(predictions),
            valid_predictions = len(valid),
            discarded         = discarded,
            mean_confidence   = round(mean_conf, 4),
            std_confidence    = round(std_conf, 4),
            landmark_quality  = round(lm_quality, 4),
            emotion_stats     = emotion_stats,
            timeline          = timeline,
            quality_ok        = quality_ok,
        )

    # ── Helpers para gráficas ────────────────────────────────────────────────

    def bar_chart_data(self, stats: SessionStats) -> dict:
        """
        Datos para gráfica de barras de distribución por emoción.
        Devuelve:
          {
            "labels":     ["Alegría", "Neutral", ...],
            "values":     [5, 3, ...],
            "percentages": [62.5, 37.5, ...],
            "colors":     ["#FFD700", "#90A4AE", ...],
          }
        """
        from app.inference.inference_engine import EMOTION_COLORS  # lazy import

        labels, values, percentages, colors = [], [], [], []
        for emotion, count in sorted(
            stats.emotion_stats.counts.items(),
            key=lambda kv: -kv[1],
        ):
            labels.append(emotion)
            values.append(count)
            percentages.append(round(stats.emotion_stats.percentages.get(emotion, 0.0), 1))
            colors.append(EMOTION_COLORS.get(emotion, "#78909C"))

        return {"labels": labels, "values": values, "percentages": percentages, "colors": colors}

    def timeline_data(self, stats: SessionStats) -> dict:
        """
        Datos de timeline listos para tabla o gráfica temporal.
        Sólo incluye predicciones válidas.
        """
        rows = [e for e in stats.timeline if e["is_valid"]]
        return {"rows": rows, "total": len(rows)}

    def summary_text(self, stats: SessionStats) -> str:
        """Texto de resumen legible para reportes."""
        em = stats.emotion_stats
        lines = [
            f"Predicciones totales: {stats.total_predictions}",
            f"Predicciones válidas: {stats.valid_predictions}",
            f"Descartadas por calidad: {stats.discarded}",
            f"Emoción dominante: {em.dominant_emotion} ({em.dominant_pct:.1f}%)",
            f"Confianza promedio: {stats.mean_confidence:.2%}",
            f"Calidad de landmarks: {stats.landmark_quality:.1%}"
            + (" ✓" if stats.quality_ok else " ✗ (baja calidad)"),
        ]
        return "\n".join(lines)

    # ── Privados ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_stats() -> SessionStats:
        empty_emotion = EmotionStats(
            counts={}, percentages={}, dominant_emotion="Neutral",
            dominant_count=0, dominant_pct=0.0,
        )
        return SessionStats(
            total_predictions=0, valid_predictions=0, discarded=0,
            mean_confidence=0.0, std_confidence=0.0, landmark_quality=0.0,
            emotion_stats=empty_emotion, timeline=[], quality_ok=False,
        )


# ── Funciones helper ────────────────────────────────────────────────────────────

def _build_emotion_stats(counts: Counter) -> EmotionStats:
    total = sum(counts.values()) or 1
    percentages = {e: (c / total) * 100 for e, c in counts.items()}

    if not counts:
        return EmotionStats(
            counts={}, percentages={}, dominant_emotion="Neutral",
            dominant_count=0, dominant_pct=0.0,
        )

    dominant = max(counts, key=lambda e: counts[e])
    return EmotionStats(
        counts=dict(counts),
        percentages={e: round(p, 2) for e, p in percentages.items()},
        dominant_emotion=dominant,
        dominant_count=counts[dominant],
        dominant_pct=round(percentages[dominant], 2),
    )
