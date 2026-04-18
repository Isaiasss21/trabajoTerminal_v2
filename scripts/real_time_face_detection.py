import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

# ─── Configuración del detector ───────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='models/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
prev_time = 0

# ─── Estado del flujo óptico ──────────────────────────────────────────────────
prev_gray_face = None          # Frame anterior (ROI en gris)
FACE_SIZE = (64, 64)           # Tamaño fijo al que normalizamos la ROI

# Parámetros de Farneback (ajustables)
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,     # Factor de escala entre niveles de la pirámide
    levels=3,          # Niveles de la pirámide
    winsize=15,        # Tamaño de ventana de promediado
    iterations=3,      # Iteraciones por nivel
    poly_n=5,          # Vecindad para aproximación polinomial
    poly_sigma=1.2,    # Sigma gaussiano para la aproximación
    flags=0
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_most_centered_detection(detections, image_shape):
    if not detections:
        return None
    img_h, img_w = image_shape[:2]
    image_center = np.array([img_w * 0.5, img_h * 0.5], dtype=np.float32)
    best_detection, best_distance = None, float("inf")
    for detection in detections:
        bbox = detection.bounding_box
        face_center = np.array(
            [bbox.origin_x + bbox.width * 0.5, bbox.origin_y + bbox.height * 0.5],
            dtype=np.float32,
        )
        dist = np.linalg.norm(face_center - image_center)
        if dist < best_distance:
            best_distance, best_detection = dist, detection
    return best_detection


def extract_face_roi(image, detection):
    """
    Recorta y redimensiona la ROI del rostro a FACE_SIZE.
    Retorna (roi_gray, bbox_coords) o (None, None) si el bbox está fuera de imagen.
    """
    bbox = detection.bounding_box
    h, w = image.shape[:2]

    # Añadir margen del 10% para no cortar el rostro
    margin_x = int(bbox.width * 0.1)
    margin_y = int(bbox.height * 0.1)

    x1 = max(0, bbox.origin_x - margin_x)
    y1 = max(0, bbox.origin_y - margin_y)
    x2 = min(w, bbox.origin_x + bbox.width + margin_x)
    y2 = min(h, bbox.origin_y + bbox.height + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = image[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, FACE_SIZE)
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    return roi_gray, (x1, y1, x2, y2)


def compute_optical_flow(prev_gray, curr_gray):
    """
    Calcula flujo óptico denso (Farneback) entre dos frames en gris.
    Retorna:
        flow       → shape (H, W, 2), componentes [dx, dy]
        magnitude  → shape (H, W)   , magnitud del desplazamiento
        angle      → shape (H, W)   , dirección en radianes
        hsv_visual → imagen BGR para visualización
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Visualización en HSV: tono=dirección, saturación=1, valor=magnitud
    hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2          # Hue: dirección (0-180)
    hsv[..., 1] = 255                                # Saturación máxima
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv_visual = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow, magnitude, angle, hsv_visual


def prepare_flow_for_model(flow, magnitude):
    """
    Prepara el flujo para la CNN:
        - Normaliza dx, dy al rango [-1, 1]
        - Canal 3 opcional: magnitud normalizada [0, 1]
        - Shape final: (H, W, 3)  →  listo para apilar en secuencias
    """
    dx = cv2.normalize(flow[..., 0], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dy = cv2.normalize(flow[..., 1], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mag = cv2.normalize(magnitude,   None,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Stack → (64, 64, 3): los 3 "canales" que verá la CNN en vez de RGB
    flow_tensor = np.stack([dx, dy, mag], axis=-1)
    return flow_tensor


def draw_detections(image, detection, bbox_coords):
    x1, y1, x2, y2 = bbox_coords
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for keypoint in detection.keypoints:
        kp = (int(keypoint.x * image.shape[1]), int(keypoint.y * image.shape[0]))
        cv2.circle(image, kp, 2, (255, 0, 0), 2)


# ─── Loop principal ───────────────────────────────────────────────────────────
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Detección de rostro
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)
    detection = get_most_centered_detection(detection_result.detections, image.shape)

    flow_tensor = None  # Lo que se pasará a la CNN

    if detection is not None:
        curr_gray_face, bbox_coords = extract_face_roi(image, detection)

        if curr_gray_face is not None:
            draw_detections(image, detection, bbox_coords)

            if prev_gray_face is not None:
                # ── Aquí ocurre el flujo óptico ──────────────────────────
                flow, magnitude, angle, flow_visual = compute_optical_flow(
                    prev_gray_face, curr_gray_face
                )
                flow_tensor = prepare_flow_for_model(flow, magnitude)
                # ─────────────────────────────────────────────────────────

                # Mostrar visualización del flujo en esquina
                flow_display = cv2.resize(flow_visual, (128, 128))
                image[10:138, 10:138] = flow_display
                cv2.putText(image, 'Optical Flow', (10, 150),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

            prev_gray_face = curr_gray_face
    else:
        # Sin rostro: resetear para no calcular flujo entre rostros distintos
        prev_gray_face = None

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow('Microexpression Pipeline', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()