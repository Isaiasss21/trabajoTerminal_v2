"""
camera_pipeline.py
------------------
Pipeline de captura de cámara en tiempo real para PyQt6.

Ejecuta en un QThread dedicado:
  1. Captura frames desde OpenCV.
  2. Detecta landmarks faciales con MediaPipe.
  3. Calcula optical flow Farneback enmascarado.
  4. Emite `frame_ready` (JPEG) cada frame para la previsualización.
  5. Emite `sequence_ready` (np.ndarray N×64×64×3) al acumular SEQUENCE_LENGTH frames.
  6. Emite `error` (str) ante cualquier fallo no fatal.

Uso:
    pipeline = CameraPipeline(camera_index=0)
    pipeline.frame_ready.connect(on_frame)
    pipeline.sequence_ready.connect(on_sequence)
    pipeline.start()
    ...
    pipeline.stop()
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


# ── Constantes de pipeline ────────────────────────────────────────────────────

FACE_SIZE       = (64, 64)
SEQUENCE_LENGTH = 15
SEQUENCE_STEP   = 5   # ventana deslizante: predice cada N frames nuevos
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)

# CLAHE reutilizable: mejora contraste local en imágenes IR y con poca luz
# sin degradar imágenes bien iluminadas (clipLimit moderado)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Tabla LUT de gamma correction (gamma=0.55): aclara medios tonos oscuros del IR
_GAMMA_LUT = np.array(
    [int((i / 255.0) ** 0.55 * 255 + 0.5) for i in range(256)], dtype=np.uint8
)


def _apply_ir_filter(frame: np.ndarray) -> np.ndarray:
    """
    Convierte un frame IR (NIR, night-vision) a pseudo-RGB optimizado para MediaPipe.

    Pipeline:
      1. Escala de grises
      2. Gamma correction (gamma=0.55) → aclara medios tonos oscuros del NIR
      3. CLAHE fuerte → realza texturas faciales (cejas, párpados, labios)
      4. Gaussian blur 3×3 → reduce el ruido salt-and-pepper del sensor IR
      5. Pseudo-RGB (3 canales iguales) para que MediaPipe lo acepte

    Returns:
        np.ndarray (H, W, 3) uint8 contiguo, formato SRGB.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = _GAMMA_LUT[gray]               # gamma correction rápida por LUT
    gray = _CLAHE.apply(gray)             # realce adaptativo de contraste
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # suavizado de ruido
    return np.ascontiguousarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

REGIONS: dict[str, tuple[list[int], float]] = {
    "left_eyebrow":  ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  1.0),
    "right_eyebrow": ([300, 293, 334, 296, 336, 285, 295, 282, 283, 276], 1.0),
    "mouth":         ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                       409, 270, 269, 267, 0, 37, 39, 40], 0.9),
    "left_eye":      ([33, 160, 158, 133, 153, 144], 0.25),
    "right_eye":     ([362, 385, 387, 263, 373, 380], 0.25),
    "nose":          ([8, 193, 114, 129, 219, 2, 439, 358, 343, 417], 0.85),
}

# Ruta al modelo de MediaPipe relativa a la raíz del workspace del sistema
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent  # microexpression-system/
    / "models" / "face_landmarker.task"
)


# ── Helpers de procesamiento ──────────────────────────────────────────────────

def _extract_face_and_landmarks(
    image: np.ndarray, face_landmarks
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[tuple]]:
    """Extrae ROI gris 64×64 y re-mapea landmarks al espacio de la ROI."""
    h, w = image.shape[:2]
    px = np.array([[int(pt.x * w), int(pt.y * h)] for pt in face_landmarks])
    x_min, y_min = np.min(px, axis=0)
    x_max, y_max = np.max(px, axis=0)
    mx = int((x_max - x_min) * 0.1)
    my = int((y_max - y_min) * 0.1)
    x1 = max(0, x_min - mx);  y1 = max(0, y_min - my)
    x2 = min(w, x_max + mx);  y2 = min(h, y_max + my)
    if x2 <= x1 or y2 <= y1:
        return None, None, None
    roi_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_gray = _CLAHE.apply(roi_gray)   # mejora contraste en IR / poca luz
    roi_resized = cv2.resize(roi_gray, FACE_SIZE)
    sx = FACE_SIZE[0] / (x2 - x1)
    sy = FACE_SIZE[1] / (y2 - y1)
    mapped = [[(lx - x1) * sx, (ly - y1) * sy] for lx, ly in px]
    return roi_resized, np.array(mapped), (x1, y1, x2, y2)


def _create_roi_mask(mapped_landmarks: np.ndarray) -> np.ndarray:
    """Construye máscara de pesos 64×64 basada en regiones faciales."""
    mask = np.zeros(FACE_SIZE, dtype=np.float32)
    for _name, (indices, weight) in REGIONS.items():
        pts = np.array([mapped_landmarks[i] for i in indices if i < len(mapped_landmarks)],
                       dtype=np.int32)
        if len(pts) < 3:
            continue
        layer = np.zeros(FACE_SIZE, dtype=np.float32)
        cv2.fillConvexPoly(layer, cv2.convexHull(pts), 1.0)
        mask = np.maximum(mask, layer * weight)
    return cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5)


def _compute_flow(
    prev_gray: np.ndarray, curr_gray: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Calcula flujo Farneback enmascarado. Devuelve (64,64,3) float32 [dx,dy,mag]."""
    flow   = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)
    flow_f = flow * mask[..., np.newaxis]
    mag, _ = cv2.cartToPolar(flow_f[..., 0], flow_f[..., 1])
    dx  = cv2.normalize(flow_f[..., 0], None, -1.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dy  = cv2.normalize(flow_f[..., 1], None, -1.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mag = cv2.normalize(mag,            None,  0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.stack([dx, dy, mag], axis=-1)  # (64,64,3)


def get_available_cameras(max_index: int = 5) -> list[int]:
    """Detecta cámaras disponibles probando índices consecutivos."""
    available: list[int] = []
    for idx in range(max_index + 1):
        # Probar primero DirectShow; si falla, intentar backend por defecto
        opened = False
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                cap.release()
                if ok:
                    available.append(idx)
                    opened = True
                    break
            else:
                cap.release()
        if opened:
            continue   # ya registrada, no duplicar
    return available


# ── CameraPipeline ────────────────────────────────────────────────────────────

class CameraPipeline(QThread):
    """
    Hilo de captura + pipeline de optical flow facial.

    Señales:
        frame_ready   (bytes)         — frame JPEG codificado para previsualización.
        sequence_ready (np.ndarray)   — secuencia (N,64,64,3) float32 lista para inferencia.
        landmark_status (bool)        — True si se detectó cara en el frame actual.
        error          (str)          — mensaje de error no fatal.
    """

    frame_ready    = pyqtSignal(bytes)
    sequence_ready = pyqtSignal(object)   # np.ndarray
    landmark_status = pyqtSignal(bool)
    error          = pyqtSignal(str)

    def __init__(
        self,
        camera_index:    int  = 0,
        model_path:      str | Path = _DEFAULT_MODEL_PATH,
        sequence_length: int  = SEQUENCE_LENGTH,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera_index    = camera_index
        self._model_path      = Path(model_path)
        self._sequence_length = sequence_length
        self._running         = False

    # ── API pública ───────────────────────────────────────────────────────

    def stop(self) -> None:
        """Detiene el hilo de captura de forma ordenada."""
        self._running = False
        self.wait(3000)  # esperar hasta 3 s

    # ── QThread.run ───────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901
        self._running = True

        # Inicializar MediaPipe
        landmarker = self._init_landmarker()

        # Inicializar cámara — intentar DirectShow primero, luego backend por defecto
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._camera_index, cv2.CAP_ANY)
        if not cap.isOpened():
            self.error.emit(f"No se pudo abrir la cámara {self._camera_index}.")
            return

        sequence: list[np.ndarray] = []  # frames de flujo acumulados
        prev_gray:    Optional[np.ndarray] = None
        prev_mask:    Optional[np.ndarray] = None

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error.emit("Error leyendo frame de la cámara.")
                    break

                # ── Emitir frame JPEG para previsualización ───────────────
                ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok_enc:
                    self.frame_ready.emit(bytes(buf))

                # ── Detección de landmarks ────────────────────────────────
                face_detected = False
                if landmarker is not None:
                    # Preprocesamiento IR: convertir a gris, CLAHE fuerte, volver a pseudo-RGB.
                    # Mejora la detección de cara en cámaras de infrarrojo cercano (NIR).
                    gray_ir = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_ir = _CLAHE.apply(gray_ir)
                    rgb = np.ascontiguousarray(
                        cv2.cvtColor(gray_ir, cv2.COLOR_GRAY2RGB)
                    )
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result   = landmarker.detect(mp_image)

                    if result.face_landmarks:
                        face_detected = True
                        landmarks = result.face_landmarks[0]
                        roi_gray, mapped, _ = _extract_face_and_landmarks(frame, landmarks)

                        if roi_gray is not None and mapped is not None:
                            mask = _create_roi_mask(mapped)

                            if prev_gray is not None and prev_mask is not None:
                                flow_frame = _compute_flow(prev_gray, roi_gray, mask)
                                sequence.append(flow_frame)

                                if len(sequence) >= self._sequence_length:
                                    seq_array = np.stack(sequence, axis=0)  # (N,64,64,3)
                                    self.sequence_ready.emit(seq_array)
                                    sequence.clear()

                            prev_gray = roi_gray
                            prev_mask = mask
                        else:
                            prev_gray = None
                            prev_mask = None
                    else:
                        prev_gray = None
                        prev_mask = None
                        sequence.clear()

                self.landmark_status.emit(face_detected)

        finally:
            cap.release()
            if landmarker is not None:
                landmarker.close()

    # ── Helpers privados ──────────────────────────────────────────────────

    def _init_landmarker(self):
        """Carga el detector de landmarks de MediaPipe. Retorna None si no disponible."""
        if not _MEDIAPIPE_AVAILABLE:
            self.error.emit("MediaPipe no está instalado. La detección facial no estará disponible.")
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
            min_face_detection_confidence=0.3,   # más bajo para detectar caras en IR
            min_face_presence_confidence=0.3,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)
