import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

# Opciones para el detector de rostros
base_options = python.BaseOptions(model_asset_path='models/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)

# Crear el detector de rostros
detector = vision.FaceDetector.create_from_options(options)

# Iniciar captura de video desde la webcam
cap = cv2.VideoCapture(0)

# Variables para el cálculo de FPS
prev_time = 0
curr_time = 0

def draw_landmarks(image, detection_result):
    """Dibuja los cuadros delimitadores y los puntos clave en la imagen."""
    for detection in detection_result.detections:
        # Dibujar cuadro delimitador
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

        # Dibujar puntos clave (ojos, nariz, boca)
        for keypoint in detection.keypoints:
            keypoint_px = (int(keypoint.x * image.shape[1]), int(keypoint.y * image.shape[0]))
            cv2.circle(image, keypoint_px, 2, (255, 0, 0), 2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convertir la imagen de BGR a RGB y luego a un objeto de imagen de MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detectar rostros en la imagen
    detection_result = detector.detect(mp_image)

    # Dibujar las detecciones en la imagen
    draw_landmarks(image, detection_result)

    # Calcular y mostrar los FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Mostrar la imagen
    cv2.imshow('MediaPipe Face Detection', image)

    # Salir con la tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
