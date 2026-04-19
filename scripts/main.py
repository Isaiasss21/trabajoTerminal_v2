"""
main.py
-------
Punto de entrada del pipeline de flujo óptico enmascarado con landmarks.

Responsabilidades:
  - Cargar una fuente de video (archivo o webcam)
  - Calcular flujo óptico denso (Farneback) entre fotogramas consecutivos
  - Llamar a mask_module para detectar landmarks, crear la máscara y filtrar el flujo
  - Visualizar: magnitud del flujo original | máscara | magnitud del flujo filtrado
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Permite ejecutar el archivo directamente desde la carpeta scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mask_module import apply_mask_to_flow, create_mask, detect_landmarks

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

# Cambia esto a la ruta de un video si no quieres usar la webcam
VIDEO_SOURCE: int | str = 0

# Parámetros del flujo óptico de Farneback
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# Resolución de trabajo (cada fotograma se redimensiona antes de procesarse)
PROC_SIZE = (640, 480)

# ──────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────────────────

def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Devuelve el flujo óptico denso de forma (H, W, 2) entre dos imágenes en gris."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, **FARNEBACK_PARAMS
    )


def flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    """Devuelve la magnitud normalizada del flujo como imagen para visualización."""
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return mag_u8


def mask_to_display(mask: np.ndarray) -> np.ndarray:
    """Convierte una máscara flotante en una imagen BGR coloreada para mostrarla."""
    mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)


def draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    """Dibuja los 68 landmarks faciales directamente sobre el fotograma."""
    for x, y in landmarks.astype(np.int32):
        cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)


def build_display(
    frame_bgr: np.ndarray,
    flow_raw: np.ndarray,
    mask: np.ndarray | None,
    flow_filtered: np.ndarray | None,
) -> np.ndarray:
    """
    Coloca cuatro paneles uno junto a otro:
      | Fotograma original | Flujo original | Máscara | Flujo filtrado |
    """
    H, W = frame_bgr.shape[:2]

    raw_mag  = cv2.cvtColor(flow_to_magnitude(flow_raw),  cv2.COLOR_GRAY2BGR)

    if mask is not None:
        mask_vis = mask_to_display(mask)
    else:
        mask_vis = np.zeros_like(frame_bgr)
        cv2.putText(mask_vis, "Sin rostro", (10, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    if flow_filtered is not None:
        filt_mag = cv2.cvtColor(flow_to_magnitude(flow_filtered), cv2.COLOR_GRAY2BGR)
    else:
        filt_mag = np.zeros_like(frame_bgr)

    # Agrega etiquetas a cada panel
    for img, label in [
        (frame_bgr, "Fotograma"),
        (raw_mag,   "Flujo (original)"),
        (mask_vis,  "Máscara"),
        (filt_mag,  "Flujo (filtrado)"),
    ]:
        cv2.putText(img, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)

    return np.hstack([frame_bgr, raw_mag, mask_vis, filt_mag])


# ──────────────────────────────────────────────────────────────────────────────
# Bucle principal
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE!r}")

    print("[main] Presiona 'q' para salir.")

    prev_gray: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            # Reinicia archivos de video; se detiene si la webcam se desconecta
            if isinstance(VIDEO_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # ── 1. Redimensionar a la resolución de trabajo ─────────────────────
        frame = cv2.resize(frame, PROC_SIZE)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 2. Flujo óptico (requiere dos fotogramas consecutivos) ──────────
        if prev_gray is None:
            prev_gray = curr_gray
            continue

        flow_raw = compute_optical_flow(prev_gray, curr_gray)

        # ── 3. Detección de landmarks → máscara → flujo filtrado ────────────
        landmarks = detect_landmarks(frame)

        mask: np.ndarray | None = None
        flow_filtered: np.ndarray | None = None

        frame_vis = frame.copy()
        if landmarks is not None:
            draw_landmarks(frame_vis, landmarks)
            mask = create_mask(landmarks, frame.shape)
            flow_filtered = apply_mask_to_flow(flow_raw, mask)

        # ── 4. Visualización ─────────────────────────────────────────────────
        display = build_display(frame_vis, flow_raw, mask, flow_filtered)
        cv2.imshow("Flujo óptico con máscara de landmarks", display)

        prev_gray = curr_gray

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
