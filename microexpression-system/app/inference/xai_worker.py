"""
xai_worker.py
-------------
Hilo separado para cálculo asíncrono de Grad-CAM y SHAP.

El pipeline de video llama a enqueue() sin bloquearse. Si la cola
(maxsize=2) está llena se descarta el job más antiguo para evitar
que el trabajo se acumule indefinidamente cuando el XAI es más lento
que la tasa de predicción.
"""
from __future__ import annotations

import queue
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from app.inference.inference_engine import InferenceEngine, InferenceResult


class XAIWorker(QThread):
    """
    Procesa Grad-CAM y SHAP en background para no bloquear el pipeline.

    Señales:
        gradcam_ready(bytes, float, str) — (jpeg, confidence, emotion)
        shap_ready(bytes, float, str)    — (jpeg, confidence, emotion)
    """

    gradcam_ready = pyqtSignal(bytes, float, str)
    shap_ready    = pyqtSignal(bytes, float, str)

    _STOP = object()

    def __init__(self, engine: "InferenceEngine", parent=None) -> None:
        super().__init__(parent)
        self._engine:          "InferenceEngine" = engine
        self._queue:           queue.Queue       = queue.Queue(maxsize=2)
        self._gradcam_enabled: bool              = False
        self._shap_enabled:    bool              = False

    def enable_gradcam(self, enabled: bool) -> None:
        self._gradcam_enabled = enabled

    def enable_shap(self, enabled: bool) -> None:
        self._shap_enabled = enabled

    def enqueue(self, flow_sequence: np.ndarray, result: "InferenceResult") -> None:
        """Encola sin bloquear. Si la cola está llena descarta el más antiguo."""
        try:
            self._queue.put_nowait((flow_sequence.copy(), result))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((flow_sequence.copy(), result))
            except queue.Full:
                pass

    def stop(self) -> None:
        self._queue.put(self._STOP)
        self.wait(5000)

    def run(self) -> None:  # noqa: C901
        while True:
            job = self._queue.get()
            if job is self._STOP:
                break

            flow_sequence, result = job

            if self._gradcam_enabled:
                try:
                    heatmap = self._engine.compute_gradcam(
                        flow_sequence, result=result, out_size=(224, 224)
                    )
                    if heatmap is not None:
                        ok, buf = cv2.imencode(
                            ".jpg", heatmap, [cv2.IMWRITE_JPEG_QUALITY, 90]
                        )
                        if ok:
                            self.gradcam_ready.emit(
                                bytes(buf), result.confidence, result.emotion
                            )
                except Exception:
                    pass

            if self._shap_enabled:
                try:
                    shap_map = self._engine.compute_shap(
                        flow_sequence, result=result, out_size=(224, 224)
                    )
                    if shap_map is not None:
                        ok, buf = cv2.imencode(
                            ".jpg", shap_map, [cv2.IMWRITE_JPEG_QUALITY, 90]
                        )
                        if ok:
                            self.shap_ready.emit(
                                bytes(buf), result.confidence, result.emotion
                            )
                except Exception:
                    pass
