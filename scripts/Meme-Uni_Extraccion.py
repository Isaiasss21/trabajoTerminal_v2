# ============================================================
# extraccion_datasetVideosTT.py
# Extracción de flujo óptico — dataset datasetVideosTT
# Estructura: p{NNN}/camara/{emocion}/frames{camara}/frame_XXXX.jpg
# Salida:     <OUTPUT_DIR>/<persona>/<camara>/<emocion>.npy
#             <OUTPUT_DIR>/extraction_index.csv
# ============================================================

# %% [0] EXTRAER ZIP (solo la primera vez)
from pathlib import Path
'''
'''
import zipfile

BASE_DIR = Path.cwd()          # /Microexpresiones/   ← raíz del Jupyter
ZIP_PATH = BASE_DIR / "datasetVideosTT.zip"
EXTRACT_DIR = BASE_DIR / "datasetVideosTT"

if not EXTRACT_DIR.exists():
    print(f"Extrayendo {ZIP_PATH}  →  {BASE_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(BASE_DIR)
    print("Extracción completa ✓")
else:
    print(f"Ya extraído: {EXTRACT_DIR}  ✓")


# %% [1] CONFIGURACIÓN
from pathlib import Path   # ya importado, no hace daño repetirlo

DATASET_ROOT = EXTRACT_DIR / "03_grabaciones"
OUTPUT_DIR   = BASE_DIR / "extracted_flow"
MODEL_PATH   = BASE_DIR / "models" / "face_landmarker.task"

FACE_SIZE = (64, 64)
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5, "levels": 3, "winsize": 15,
    "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
}
REGIONS = {
    "left_eyebrow":  ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46],                   1.0),
    "right_eyebrow": ([300, 293, 334, 296, 336, 285, 295, 282, 283, 276],            1.0),
    "mouth":         ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                       291, 409, 270, 269, 267, 0, 37, 39, 40],                      0.9),
    "left_eye":      ([33, 160, 158, 133, 153, 144],                                 0.25),
    "right_eye":     ([362, 385, 387, 263, 373, 380],                                0.25),
    "nose":          ([8, 193, 114, 129, 219, 2, 439, 358, 343, 417],                0.85),
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CAMERAS = {
    "camaraInfrarroja": "framesInfrarroja",
    "camaraWeb":        "framesWeb",
}

print(f"Dataset : {DATASET_ROOT}")
print(f"Salida  : {OUTPUT_DIR}")
print(f"Modelo  : {MODEL_PATH}  (existe: {MODEL_PATH.exists()})")


# %% [2] EXPLORAR ESTRUCTURA
import re
from collections import defaultdict

def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

def scan_dataset(root: Path):
    sequences = []
    emotion_counts = defaultdict(int)
    cam_counts = defaultdict(int)
    personas = sorted([d for d in root.iterdir() if d.is_dir() and re.match(r'^p\d{3}$', d.name)])

    for persona_dir in personas:
        for cam_name, frames_folder in CAMERAS.items():
            cam_dir = persona_dir / cam_name
            if not cam_dir.is_dir():
                continue
            for emotion_dir in sorted(cam_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                frames_dir = emotion_dir / frames_folder
                if not frames_dir.is_dir():
                    continue
                frames = sorted(
                    [f for f in frames_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS],
                    key=natural_key
                )
                if not frames:
                    continue
                sequences.append({
                    "persona": persona_dir.name,
                    "camara":  cam_name,
                    "emocion": emotion_dir.name,
                    "frames":  frames,
                    "n_frames": len(frames),
                })
                emotion_counts[emotion_dir.name] += 1
                cam_counts[cam_name] += 1

    return sequences, emotion_counts, cam_counts

sequences, emotion_counts, cam_counts = scan_dataset(DATASET_ROOT)

print(f"Total secuencias : {len(sequences)}")
print(f"Personas únicas  : {len(set(s['persona'] for s in sequences))}")
print("\nPor emoción:")
for em, c in sorted(emotion_counts.items()):
    print(f"  {em:<25} {c:>4}")
print("\nPor cámara:")
for cam, c in sorted(cam_counts.items()):
    print(f"  {cam:<30} {c:>4}")
if sequences:
    s = sequences[0]
    print(f"\nEjemplo: {s['persona']}/{s['camara']}/{s['emocion']} — {s['n_frames']} frames")


# %% [3] FUNCIONES DEL PIPELINE
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def init_landmarker(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

def extract_face_and_landmarks(image, face_landmarks):
    h, w = image.shape[:2]
    px = np.array([[int(pt.x * w), int(pt.y * h)] for pt in face_landmarks])
    x_min, y_min = np.min(px, axis=0)
    x_max, y_max = np.max(px, axis=0)
    mx = int((x_max - x_min) * 0.1)
    my = int((y_max - y_min) * 0.1)
    x1 = max(0, x_min - mx); y1 = max(0, y_min - my)
    x2 = min(w, x_max + mx); y2 = min(h, y_max + my)
    if x2 <= x1 or y2 <= y1:
        return None, None
    roi_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, FACE_SIZE)
    sx = FACE_SIZE[0] / (x2 - x1)
    sy = FACE_SIZE[1] / (y2 - y1)
    mapped = np.array([[(lx - x1) * sx, (ly - y1) * sy] for lx, ly in px])
    return roi_resized, mapped

def create_roi_mask(mapped_landmarks):
    mask = np.zeros(FACE_SIZE, dtype=np.float32)
    for _, (indices, weight) in REGIONS.items():
        pts = np.array([mapped_landmarks[i] for i in indices], dtype=np.int32)
        layer = np.zeros(FACE_SIZE, dtype=np.float32)
        cv2.fillConvexPoly(layer, cv2.convexHull(pts), 1.0)
        mask = np.maximum(mask, layer * weight)
    return cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5)

def compute_flow_tensor(prev_gray, curr_gray, mask):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)
    flow_f = flow * mask[..., np.newaxis]
    magnitude, _ = cv2.cartToPolar(flow_f[..., 0], flow_f[..., 1])
    dx  = cv2.normalize(flow_f[..., 0], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dy  = cv2.normalize(flow_f[..., 1], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mag = cv2.normalize(magnitude,      None,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.stack([dx, dy, mag], axis=-1)

def extract_sequence_from_frames(frame_paths, detector):
    tensors = []
    prev_gray = None
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)
        if not result.face_landmarks:
            prev_gray = None
            continue
        curr_gray, mapped_lms = extract_face_and_landmarks(frame, result.face_landmarks[0])
        if curr_gray is None:
            prev_gray = None
            continue
        if prev_gray is not None:
            mask = create_roi_mask(mapped_lms)
            tensors.append(compute_flow_tensor(prev_gray, curr_gray, mask))
        prev_gray = curr_gray
    if not tensors:
        return None
    return np.stack(tensors, axis=0).astype(np.float32)

print("Funciones del pipeline cargadas ✓")


# %% [4] EXTRACCIÓN EN LOTE
import csv
from datetime import datetime

OUTPUT_CSV = OUTPUT_DIR / "extraction_index.csv"
CSV_FIELDS = ["persona", "camara", "emocion", "archivo", "num_frames",
              "alto", "ancho", "canales", "timestamp"]

def append_csv_row(csv_path, row):
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

print("Cargando modelo MediaPipe...")
detector = init_landmarker(MODEL_PATH)
print("Modelo cargado ✓\n")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Comentar la siguiente línea si quieres REANUDAR en lugar de empezar desde cero
if OUTPUT_CSV.exists():
    OUTPUT_CSV.unlink()

total = len(sequences)
success = failed = skipped = 0

for idx, seq in enumerate(sequences, start=1):
    persona = seq["persona"]
    camara  = seq["camara"]
    emocion = seq["emocion"]
    frames  = seq["frames"]

    out_dir  = OUTPUT_DIR / persona / camara
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{emocion}.npy"

    if npy_path.exists():          # reanudación automática
        skipped += 1
        print(f"[{idx:>4}/{total}] SKIP  {persona}/{camara}/{emocion}")
        continue

    try:
        volume = extract_sequence_from_frames(frames, detector)
        if volume is None or volume.shape[0] == 0:
            raise ValueError("Sin frames válidos")
        np.save(str(npy_path), volume)
        n, h, w, c = volume.shape
        append_csv_row(OUTPUT_CSV, {
            "persona":    persona,
            "camara":     camara,
            "emocion":    emocion,
            "archivo":    str(npy_path.relative_to(OUTPUT_DIR)),
            "num_frames": n, "alto": h, "ancho": w, "canales": c,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        success += 1
        print(f"[{idx:>4}/{total}] ✓  {persona}/{camara}/{emocion}  ({n} frames)")
    except Exception as e:
        failed += 1
        print(f"[{idx:>4}/{total}] ✗  {persona}/{camara}/{emocion}  — {e}")

if hasattr(detector, "close"):
    detector.close()

print(f"\n{'='*50}")
print(f"  ✓ Exitosas : {success}")
print(f"  ✗ Fallidas : {failed}")
print(f"  ↷ Saltadas : {skipped}")
print(f"  CSV        : {OUTPUT_CSV}")


# %% [5] VERIFICAR RESULTADO
import pandas as pd

df = pd.read_csv(OUTPUT_CSV)
print(f"Filas: {len(df)}")
print("\nPor emoción:\n", df["emocion"].value_counts().to_string())
print("\nPor cámara:\n",  df["camara"].value_counts().to_string())
print(f"\nFrames — min: {df['num_frames'].min()}  max: {df['num_frames'].max()}  media: {df['num_frames'].mean():.1f}")
df.head(10)