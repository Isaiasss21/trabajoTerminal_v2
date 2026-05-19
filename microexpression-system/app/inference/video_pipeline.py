"""
video_pipeline.py
-----------------
Pipeline de procesamiento de video pre-grabado para análisis de microexpresiones.

Ejecuta en un QThread dedicado:
  1. Abre el archivo de video con OpenCV.
  2. Detecta landmarks faciales con MediaPipe por cada frame.
  3. Extrae ROI facial y calcula flujo óptico Farneback entre frames consecutivos.
  4. Acumula SEQUENCE_LENGTH frames de flujo → emite secuencia lista para inferencia.
  5. Ejecuta InferenceEngine.predict() en el hilo del pipeline (no bloquea el UI thread).
  6. Emite InferenceResult, progreso y estado de finalización.

Señales:
    frame_ready(bytes)         — frame JPEG codificado para previsualización (cada 5 frames).
    progress(int, int)         — (frame_actual, total_frames).
    sequence_result(object)    — InferenceResult de cada secuencia procesada.
    finished_processing(int)   — número total de secuencias inferidas al terminar.
    error(str)                 — mensaje de error no fatal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

from app.inference.camera_pipeline import (
    SEQUENCE_LENGTH,
    _DEFAULT_MODEL_PATH,
    _extract_face_and_landmarks,
    _create_roi_mask,
    _compute_flow,
)
from app.inference.inference_engine import InferenceEngine, InferenceResult


class VideoPipeline(QThread):
    """
    Hilo de procesamiento de video + inferencia de microexpresiones.

    Uso:
        pipeline = VideoPipeline(video_path, engine)
        pipeline.sequence_result.connect(on_result)
        pipeline.progress.connect(on_progress)
        pipeline.finished_processing.connect(on_done)
        pipeline.start()
        ...
        pipeline.stop()   # detener antes de tiempo si se necesita
    """

    frame_ready         = pyqtSignal(bytes)   # JPEG preview
    progress            = pyqtSignal(int, int) # (frame_actual, total_frames)
    sequence_result     = pyqtSignal(object)   # InferenceResult
    finished_processing = pyqtSignal(int)      # total secuencias procesadas
    error               = pyqtSignal(str)

    def __init__(
        self,
        video_path:      str | Path,
        engine:          InferenceEngine,
        model_path:      str | Path = _DEFAULT_MODEL_PATH,
        sequence_length: int = SEQUENCE_LENGTH,
        preview_every:   int = 5,   # emitir frame preview cada N frames
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video_path      = Path(video_path)
        self._engine          = engine
        self._model_path      = Path(model_path)
        self._sequence_length = sequence_length
        self._preview_every   = preview_every
        self._stop_flag       = False

    # ── API pública ───────────────────────────────────────────────────────

    def stop(self) -> None:
        """Solicita la detención del procesamiento y espera hasta 5 s."""
        self._stop_flag = True
        self.wait(5000)

    # ── QThread.run ───────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901
        self._stop_flag = False

        # ── 1. Inicializar MediaPipe ──────────────────────────────────────
        landmarker = self._init_landmarker()

        # ── 2. Abrir video ────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            self.error.emit(f"No se pudo abrir el video: {self._video_path.name}")
            if landmarker:
                landmarker.close()
            self.finished_processing.emit(0)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0  # stream o archivo sin metadatos de duración

        # ── 3. Procesar frames ────────────────────────────────────────────
        sequence:  list[np.ndarray] = []
        prev_gray: Optional[np.ndarray] = None
        prev_mask: Optional[np.ndarray] = None
        frame_idx  = 0
        seq_count  = 0

        try:
            while not self._stop_flag:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                self.progress.emit(frame_idx, total_frames)

                # ── Preview JPEG (cada N frames) ──────────────────────────
                if frame_idx % self._preview_every == 0:
                    ok_enc, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    if ok_enc:
                        self.frame_ready.emit(bytes(buf))

                # ── Detección de landmarks ────────────────────────────────
                if landmarker is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = landmarker.detect(mp_image)

                if not detection.face_landmarks:
                    # Sin cara → resetear secuencia acumulada
                    prev_gray = None
                    prev_mask = None
                    sequence.clear()
                    continue

                landmarks = detection.face_landmarks[0]
                roi_gray, mapped, _ = _extract_face_and_landmarks(frame, landmarks)

                if roi_gray is None or mapped is None:
                    prev_gray = None
                    prev_mask = None
                    sequence.clear()
                    continue

                mask = _create_roi_mask(mapped)

                if prev_gray is not None:
                    flow_frame = _compute_flow(prev_gray, roi_gray, mask)
                    sequence.append(flow_frame)

                    # ── Secuencia completa → inferir ──────────────────────
                    if len(sequence) >= self._sequence_length:
                        seq_array = np.stack(sequence, axis=0)  # (N,64,64,3)
                        sequence.clear()
                        try:
                            result = self._engine.predict(seq_array)
                            self.sequence_result.emit(result)
                            seq_count += 1
                        except Exception as exc:
                            self.error.emit(f"Error en inferencia: {exc}")

                prev_gray = roi_gray
                prev_mask = mask  # noqa: F841

        finally:
            cap.release()
            if landmarker is not None:
                landmarker.close()

        self.finished_processing.emit(seq_count)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _init_landmarker(self):
        """Carga el detector de landmarks de MediaPipe. Retorna None si no disponible."""
        if not _MEDIAPIPE_AVAILABLE:
            self.error.emit(
                "MediaPipe no está instalado. Instálalo con: pip install mediapipe"
            )
            return None
        if not self._model_path.exists():
            self.error.emit(
                f"Modelo MediaPipe no encontrado: {self._model_path}\n"
                "Descárgalo desde: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )
            return None
        base_options = mp_python.BaseOptions(model_asset_path=str(self._model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)
