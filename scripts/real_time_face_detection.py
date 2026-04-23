"real_time_face_detection.py"
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from collections import deque
import tkinter as tk

# ─── Configuración Global ────────────────────────────────────────────────────
FACE_SIZE = (64, 64)           # Tamaño del tensor de entrada a la CNN
SEQUENCE_LENGTH = 15           # Longitud del Sliding Window (aprox 0.5s a 30fps)

# Parámetros del flujo óptico de Farneback
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, 
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# ─── Configuración del Face Landmarker (Tasks API) ───────────────────────────
# Asegúrate de que la ruta coincida con donde guardaste el archivo descargado
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ─── Índices de MediaPipe para Unidades de Acción (AUs) ──────────────────────
REGIONS = {
    # Cejas (AU1, AU2, AU4) -> Peso 1.0 (Movimientos clave en microexpresiones)
    "left_eyebrow": ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46], 1.0),
    "right_eyebrow": ([300, 293, 334, 296, 336, 285, 295, 282, 283, 276], 1.0),
    
    # Boca (AU12, AU15, etc.) -> Peso 0.9
    "mouth": ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40], 0.9),
    
    # Ojos (Blinking / AU45) -> Peso 0.25 (Suprimir ruido por parpadeo natural)
    "left_eye": ([33, 160, 158, 133, 153, 144], 0.25),
    "right_eye": ([362, 385, 387, 263, 373, 380], 0.25),

    # Nariz (AU9 - Nose Wrinkler / Asco) -> Peso 0.85
    # Puntos: 8 (Puente superior), 2 (Punta), 129/219 (Aleta izq), 358/439 (Aleta der)
    "nose": ([8, 193, 114, 129, 219, 2, 439, 358, 343, 417], 0.85)
}

# ─── Funciones Core ──────────────────────────────────────────────────────────

def extract_face_and_landmarks(image, face_landmarks):
    """
    Extrae la ROI de 64x64 y mapea los landmarks de la imagen completa a la ROI.
    """
    h, w = image.shape[:2]
    
    # 1. Convertir coordenadas normalizadas a píxeles (Adaptado para Tasks API)
    px_landmarks = np.array([
        [int(pt.x * w), int(pt.y * h)] for pt in face_landmarks
    ])
    
    # 2. Calcular Bounding Box con margen del 10%
    x_min, y_min = np.min(px_landmarks, axis=0)
    x_max, y_max = np.max(px_landmarks, axis=0)
    
    margin_x = int((x_max - x_min) * 0.1)
    margin_y = int((y_max - y_min) * 0.1)
    
    x1, y1 = max(0, x_min - margin_x), max(0, y_min - margin_y)
    x2, y2 = min(w, x_max + margin_x), min(h, y_max + margin_y)
    
    # Prevención de crasheos si el rostro sale de cámara
    if x2 <= x1 or y2 <= y1:
        return None, None, None
        
    roi = image[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, FACE_SIZE)
    
    # 3. Trasladar y escalar los landmarks al nuevo espacio de 64x64
    scale_x = FACE_SIZE[0] / (x2 - x1)
    scale_y = FACE_SIZE[1] / (y2 - y1)
    
    mapped_landmarks = []
    for (lx, ly) in px_landmarks:
        nx = (lx - x1) * scale_x
        ny = (ly - y1) * scale_y
        mapped_landmarks.append([nx, ny])
        
    return roi_resized, np.array(mapped_landmarks), (x1, y1, x2, y2)

def create_roi_mask(mapped_landmarks):
    """
    Crea una máscara de pesos de 64x64 basada en los landmarks mapeados.
    """
    mask = np.zeros(FACE_SIZE, dtype=np.float32)
    
    for region_name, (indices, weight) in REGIONS.items():
        pts = np.array([mapped_landmarks[i] for i in indices], dtype=np.int32)
        hull = cv2.convexHull(pts)
        region_layer = np.zeros(FACE_SIZE, dtype=np.float32)
        cv2.fillConvexPoly(region_layer, hull, 1.0)
        mask = np.maximum(mask, region_layer * weight)
        
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5)
    return mask

def process_optical_flow(prev_gray, curr_gray, mask):
    """
    Calcula Farneback en 64x64, aplica la máscara, y empaqueta para la CNN.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)
    flow_filtered = flow * mask[..., np.newaxis]
    magnitude, angle = cv2.cartToPolar(flow_filtered[..., 0], flow_filtered[..., 1])
    
    dx = cv2.normalize(flow_filtered[..., 0], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dy = cv2.normalize(flow_filtered[..., 1], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mag = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    flow_tensor = np.stack([dx, dy, mag], axis=-1) 
    
    hsv = np.zeros((*FACE_SIZE, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_visual = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow_tensor, flow_visual


def get_screen_size():
    """Obtiene el tamaño actual de la pantalla para redimensionar la ventana."""
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

# ─── Bucle Principal ─────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
prev_gray_face = None
flow_buffer = deque(maxlen=SEQUENCE_LENGTH)
prev_time = 0
window_name = 'Microexpression Hybrid Pipeline'
screen_width, screen_height = get_screen_size()

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_width, screen_height)

print("\n[INFO] Pipeline iniciado. Presiona 'q' para salir.\n")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue
    
    # Preparar imagen para MediaPipe Tasks API
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Inferencia del Face Landmarker
    detection_result = detector.detect(mp_image)
    
    # Si detecta al menos un rostro
    if len(detection_result.face_landmarks) > 0:
        face_landmarks = detection_result.face_landmarks[0]
        
        curr_gray_face, mapped_lms, bbox = extract_face_and_landmarks(image, face_landmarks)
        
        if curr_gray_face is not None:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            if prev_gray_face is not None:
                mask = create_roi_mask(mapped_lms)
                flow_tensor, flow_visual = process_optical_flow(prev_gray_face, curr_gray_face, mask)
                
                # --- SISTEMA DE INFERENCIA SECUENCIAL ---
                flow_buffer.append(flow_tensor)
                
                if len(flow_buffer) == SEQUENCE_LENGTH:
                    sequence_volume = np.array(flow_buffer) 
                    input_tensor = np.expand_dims(sequence_volume, axis=0) # (1, 15, 64, 64, 3)
                    
                    cv2.putText(image, 'Buffer Lleno: Listo para CNN/LSTM', (20, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # --- VISUALIZACIONES EN PANTALLA ---
                flow_display = cv2.resize(flow_visual, (150, 150))
                image[10:160, 10:160] = flow_display
                
                mask_display = cv2.resize(cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (150, 150))
                image[10:160, 170:320] = mask_display
                
                cv2.putText(image, 'Flujo Filtrado', (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                cv2.putText(image, 'Mascara de Pesos', (170, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            prev_gray_face = curr_gray_face
    else:
        prev_gray_face = None
        flow_buffer.clear()

    # Cálculo de FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()