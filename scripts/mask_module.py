"""
mask_module.py
--------------
Enmascarado basado en landmarks faciales para filtrar flujo óptico.

Responsabilidades:
  - Detectar 68 landmarks faciales con dlib
  - Construir una máscara espacial ponderada por fotograma
  - Aplicar la máscara al flujo óptico

Pesos por región facial (índices del modelo de 68 puntos de dlib):
  - Cejas    (17–26) → 1.0
  - Boca     (48–67) → 0.9
  - Ojos     (36–47) → 0.25  (bajo para suprimir ruido por parpadeo)
  - Resto            → 0.0
"""

from __future__ import annotations

import bz2
from pathlib import Path

import cv2
import dlib
import numpy as np
from urllib.request import urlopen


# Inicialización del modelo (descarga el predictor en la primera ejecución)


_PREDICTOR_URLS = [
    "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
]
_PREDICTOR_PATH = Path("assets/models/shape_predictor_68_face_landmarks.dat")


def _download_archive(dst: Path) -> None:
    """Descarga de forma segura el archivo comprimido del predictor de dlib."""
    last_error = None
    part_path = dst.with_suffix(dst.suffix + ".part")

    for url in _PREDICTOR_URLS:
        try:
            if part_path.exists():
                part_path.unlink()

            print(f"[mask_module] Descargando predictor desde {url} ...")
            with urlopen(url, timeout=120) as response, part_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            if part_path.stat().st_size < 10_000_000:
                raise RuntimeError("El archivo descargado es inesperadamente pequeño.")

            part_path.replace(dst)
            return
        except Exception as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink()

    raise RuntimeError(
        "No se pudo descargar el archivo del predictor de landmarks de dlib."
    ) from last_error


def _unpack_archive(src_path: Path, dst_path: Path) -> None:
    """Descomprime el archivo bz2 en el predictor final."""
    tmp_dat = dst_path.with_suffix(".dat.tmp")
    if tmp_dat.exists():
        tmp_dat.unlink()

    with bz2.BZ2File(str(src_path), "rb") as src, tmp_dat.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)

    if tmp_dat.stat().st_size < 10_000_000:
        raise RuntimeError("El predictor descomprimido es inesperadamente pequeño.")

    tmp_dat.replace(dst_path)


def _ensure_shape_predictor() -> None:
    """Descarga y descomprime el predictor de dlib si falta o está corrupto."""
    if _PREDICTOR_PATH.exists() and _PREDICTOR_PATH.stat().st_size > 10_000_000:
        return

    _PREDICTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    bz2_path = _PREDICTOR_PATH.with_suffix(".dat.bz2")

    if not bz2_path.exists() or bz2_path.stat().st_size < 10_000_000:
        _download_archive(bz2_path)

    try:
        print("[mask_module] Descomprimiendo predictor de landmarks …")
        _unpack_archive(bz2_path, _PREDICTOR_PATH)
    except EOFError:
        print("[mask_module] Archivo corrupto detectado. Reintentando descarga …")
        if bz2_path.exists():
            bz2_path.unlink()
        if _PREDICTOR_PATH.exists():
            _PREDICTOR_PATH.unlink()
        _download_archive(bz2_path)
        print("[mask_module] Descomprimiendo predictor de landmarks …")
        _unpack_archive(bz2_path, _PREDICTOR_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Detector y predictor a nivel de módulo (inicialización diferida)
# ──────────────────────────────────────────────────────────────────────────────

_detector: dlib.fhog_object_detector | None = None
_predictor: dlib.shape_predictor | None = None


def _get_models():
    """Devuelve detector y predictor, inicializándolos una sola vez."""
    global _detector, _predictor
    if _detector is None:
        _ensure_shape_predictor()
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(str(_PREDICTOR_PATH))
    return _detector, _predictor


# ──────────────────────────────────────────────────────────────────────────────
# Definición de regiones faciales (índices del modelo de 68 puntos de dlib)
# ──────────────────────────────────────────────────────────────────────────────

# Cada entrada tiene: (nombre, índices_de_puntos, peso)
# Los ojos usan un peso bajo para reducir artefactos por parpadeo.
FACIAL_REGIONS = [
    ("left_eyebrow",  range(17, 22), 1.0),
    ("right_eyebrow", range(22, 27), 1.0),
    ("mouth",         range(48, 68), 0.9),
    ("left_eye",      range(36, 42), 0.25),
    ("right_eye",     range(42, 48), 0.25),
]


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────

def detect_landmarks(frame: np.ndarray) -> np.ndarray | None:
    """
    Detecta 68 landmarks faciales en el fotograma de entrada.

    Parámetros
    ----------
    frame : np.ndarray
        Imagen uint8 en formato BGR o RGB.

    Retorna
    -------
    landmarks : np.ndarray de forma (68, 2) con coordenadas (x, y),
                o None si no se detecta un rostro.
    """
    detector, predictor = _get_models()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)           # mejora la detección con poca iluminación

    faces = detector(gray, 1)
    if not faces:
        return None

    # Elige el rostro más grande, normalmente el sujeto principal
    face = max(faces, key=lambda r: r.width() * r.height())
    shape = predictor(gray, face)

    landmarks = np.array(
        [(shape.part(i).x, shape.part(i).y) for i in range(68)],
        dtype=np.float32,
    )
    return landmarks


def create_mask(
    landmarks: np.ndarray,
    frame_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    """
    Construye una máscara espacial flotante de tamaño (H, W).

    Cada región facial se dibuja como un convex hull relleno con su peso.
    Si dos regiones se solapan, se conserva el mayor peso por píxel.

    Parámetros
    ----------
    landmarks   : arreglo (68, 2) con coordenadas (x, y).
    frame_shape : (H, W) o (H, W, C).

    Retorna
    -------
    mask : np.ndarray de forma (H, W), tipo float32 y valores en el rango [0, 1].
    """
    H, W = frame_shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)

    for _name, indices, weight in FACIAL_REGIONS:
        pts = landmarks[list(indices)].astype(np.int32)

        # El convex hull genera una región cerrada y limpia para cada zona facial
        hull = cv2.convexHull(pts)

        region_layer = np.zeros((H, W), dtype=np.float32)
        cv2.fillConvexPoly(region_layer, hull, 1.0)

        # Conserva el mayor peso por píxel cuando hay superposición entre regiones
        mask = np.maximum(mask, region_layer * weight)

    # Suaviza bordes para evitar transiciones demasiado bruscas
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=3)

    return mask


def apply_mask_to_flow(
    flow: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Multiplica el flujo óptico por la máscara espacial.

    Parámetros
    ----------
    flow : arreglo float32 de forma (H, W, 2) con componentes [dx, dy].
    mask : arreglo float32 de forma (H, W).

    Retorna
    -------
    filtered_flow : np.ndarray de forma (H, W, 2).
    """
    if flow.shape[:2] != mask.shape:
        raise ValueError(
            f"flow spatial dims {flow.shape[:2]} != mask shape {mask.shape}"
        )

    # Expande la máscara a (H, W, 1) para aplicarla a ambos canales del flujo
    return flow * mask[..., np.newaxis]
