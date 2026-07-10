"""
extract_sequences.py
--------------------
Extracción OAO + EVM — 3 flujos por secuencia.
Shape de salida: (3, 64, 64, 3)
  [0] onset→apex    (aparición)
  [1] apex→offset   (relajación)
  [2] onset→offset  (movimiento total)

Datasets soportados:
  - CASME II  (xlsx) — excluye repression y others
  - SMIC-HS-E (xlsx) — solo surprise
  - MEME/Propio (xlsx hoja Resumen) 

Cada emoción se guarda en su propia carpeta.
La fusión se controla en train_mex.py con --merge_classes.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# ── Parámetros del pipeline ───────────────────────────────────────────────────
FACE_SIZE = (64, 64)
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5, "levels": 3, "winsize": 15,
    "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
}
REGIONS = {
    "left_eyebrow":  ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46],                   1.0),
    "right_eyebrow": ([300, 293, 334, 296, 336, 285, 295, 282, 283, 276],           1.0),
    "mouth":         ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                       291, 409, 270, 269, 267, 0, 37, 39, 40],                      0.9),
    "left_eye":      ([33, 160, 158, 133, 153, 144],                                0.25),
    "right_eye":     ([362, 385, 387, 263, 373, 380],                               0.25),
    "nose":          ([8, 193, 114, 129, 219, 2, 439, 358, 343, 417],               0.85),
}

SUPPORTED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_CSV_NAME       = "extraction_index.csv"
_MODEL_PATH           = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
LANDMARK_FALLBACK_WINDOW = 5

# ── Exclusiones CASME II ──────────────────────────────────────────────────────
EXCLUDE_CASME_EXCEL    = {"repression", "others"}
EXCLUDE_CASME_CARPETAS = {"represion", "repression", "otros", "others", "other"}

# ── Mapeo canónico — SIN fusionar miedo/enojo ────────────────────────────────
# La fusión se hace en train_mex.py con --merge_classes
EMOTION_CANONICAL: dict[str, str] = {
    # CASME II (inglés)
    "happiness":  "felicidad",
    "disgust":    "asco",
    "surprise":   "sorpresa",
    "sadness":    "tristeza",
    "fear":       "miedo",
    # MEME (español) — cada emoción en su carpeta propia
    "asco":       "asco",
    "felicidad":  "felicidad",
    "sorpresa":   "sorpresa",
    "tristeza":   "tristeza",
    "miedo":      "miedo",
    "enojo":      "enojo",
    "neutral":    "neutral",
}

# ── Parámetros EVM ────────────────────────────────────────────────────────────
EVM_ALPHA     = 25.0
EVM_LOW_FREQ  = 0.5
EVM_HIGH_FREQ = 4.0
EVM_LEVELS    = 4
EVM_CLIP_SECS = 3.0

# ── Paleta GUI ────────────────────────────────────────────────────────────────
_C_BG      = "#1a1a2e"
_C_SURFACE = "#16213e"
_C_BORDER  = "#0f3460"
_C_GREEN   = "#22c55e"
_C_GREEN_H = "#16a34a"
_C_RED     = "#ef4444"
_C_RED_H   = "#dc2626"
_C_AMBER   = "#f59e0b"
_C_TEXT    = "#e2e8f0"
_C_SUBTEXT = "#64748b"
_C_DISBLD  = "#374151"


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE TABLAS
# ══════════════════════════════════════════════════════════════════════════════

def load_coding_table(excel_path: Path) -> tuple[pd.DataFrame, str]:
    def read_file(path: Path, **kwargs) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(str(path), **kwargs)
        return pd.read_excel(str(path), **kwargs)

    try:
        df_test = read_file(excel_path, nrows=5)
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo: {e}")

    # ── SMIC-E ────────────────────────────────────────────────────────────────
    if "OnsetF" in df_test.columns and "FirstF" in df_test.columns:
        try:
            df_smic = pd.read_excel(str(excel_path), sheet_name="HS")
        except Exception:
            df_smic = read_file(excel_path)
        df_smic = df_smic[df_smic["Emotion"] == "surprise"].copy()
        std = pd.DataFrame()
        std["Subject"]     = df_smic["Subject"].apply(lambda x: f"s{int(x):02d}")
        std["Sequence"]    = df_smic["Filename"].astype(str).str.strip()
        std["Emotion"]     = "sorpresa"
        std["OnsetFrame"]  = (df_smic["OnsetF"]  - df_smic["FirstF"] + 1).astype(int)
        std["OffsetFrame"] = (df_smic["OffsetF"] - df_smic["FirstF"] + 1).astype(int)
        std["ApexFrame"]   = ((std["OnsetFrame"] + std["OffsetFrame"]) // 2).astype(int)
        std["TotalFrames"] = df_smic["TotalMF1"].fillna(
            df_smic["OffsetF"] - df_smic["FirstF"] + 1).astype(int)
        mask = (std["OnsetFrame"] > 0) & (std["OffsetFrame"] > std["OnsetFrame"])
        return std[mask].reset_index(drop=True), "SMIC_E"

    # ── MEME / PROPIO ─────────────────────────────────────────────────────────
    try:
        df_p = read_file(excel_path, sheet_name="Resumen", skiprows=2)
        df_p.columns = [str(c).replace("\n", " ").strip() for c in df_p.columns]
        if "Participante" in df_p.columns and "Emoción" in df_p.columns:
            # Excluir cámara infrarroja
            if "Cámara" in df_p.columns:
                df_p = df_p[
                    ~df_p["Cámara"].astype(str).str.lower().str.contains(
                        "infrarroja|infrared", na=False)
                ].copy()
            std = pd.DataFrame()
            std["Subject"]     = df_p["Participante"].astype(str).str.strip()
            std["Sequence"]    = df_p["Cámara"].astype(str).str.strip()
            # Mapear emoción a canónico en minúsculas
            std["Emotion"]     = df_p["Emoción"].astype(str).str.strip().apply(
                lambda e: EMOTION_CANONICAL.get(e.strip().lower(), e.strip().lower()))
            std["OnsetFrame"]  = pd.to_numeric(df_p["Onset (frame)"],  errors="coerce").fillna(-1).astype(int)
            std["ApexFrame"]   = pd.to_numeric(df_p["Apex (frame)"],   errors="coerce").fillna(-1).astype(int)
            std["OffsetFrame"] = pd.to_numeric(df_p["Offset (frame)"], errors="coerce").fillna(-1).astype(int)
            std["TotalFrames"] = pd.to_numeric(df_p["Total Frames"],   errors="coerce").fillna(-1).astype(int)
            mask = (std["OnsetFrame"] > 0) & (std["OffsetFrame"] > 0) & (std["ApexFrame"] > 0)
            return std[mask].reset_index(drop=True), "PROPIO"
    except Exception:
        pass

    # ── CASME II ──────────────────────────────────────────────────────────────
    df_c = read_file(excel_path)
    if "Subject" in df_c.columns and "Filename" in df_c.columns:
        df_c = df_c[~df_c["Estimated Emotion"].isin(EXCLUDE_CASME_EXCEL)].copy()
        std = pd.DataFrame()
        std["Subject"]     = df_c["Subject"].apply(lambda x: f"Usuario_{int(x):02d}")
        std["Sequence"]    = df_c["Filename"].astype(str).str.strip()
        std["Emotion"]     = df_c["Estimated Emotion"].astype(str).str.strip().apply(
            lambda e: EMOTION_CANONICAL.get(e.lower(), e.lower()))
        std["OnsetFrame"]  = pd.to_numeric(df_c["OnsetFrame"],  errors="coerce").fillna(-1).astype(int)
        std["ApexFrame"]   = pd.to_numeric(df_c["ApexFrame"],   errors="coerce").fillna(-1).astype(int)
        std["OffsetFrame"] = pd.to_numeric(df_c["OffsetFrame"], errors="coerce").fillna(-1).astype(int)
        std["TotalFrames"] = std["OffsetFrame"]
        mask = (
            (std["OnsetFrame"]  > 0) &
            (std["OffsetFrame"] > std["OnsetFrame"]) &
            (std["ApexFrame"]   >= std["OnsetFrame"]) &
            (std["ApexFrame"]   <= std["OffsetFrame"])
        )
        return std[mask].reset_index(drop=True), "CASME2"

    raise ValueError("Formato no reconocido. Debe ser SMIC-E, CASME II o MEME/Propio.")


# ══════════════════════════════════════════════════════════════════════════════
# VISIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _init_landmarker(model_path: Path):
    if not model_path.exists():
        return None
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _extract_face_and_landmarks(image: np.ndarray, face_landmarks) -> tuple:
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
    roi_gray    = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, FACE_SIZE)
    sx = FACE_SIZE[0] / (x2 - x1)
    sy = FACE_SIZE[1] / (y2 - y1)
    mapped = [[(lx - x1) * sx, (ly - y1) * sy] for lx, ly in px]
    return roi_resized, np.array(mapped), (x1, y1, x2, y2)


def _create_roi_mask(mapped_landmarks: np.ndarray) -> np.ndarray:
    mask = np.zeros(FACE_SIZE, dtype=np.float32)
    for _name, (indices, weight) in REGIONS.items():
        pts   = np.array([mapped_landmarks[i] for i in indices], dtype=np.int32)
        layer = np.zeros(FACE_SIZE, dtype=np.float32)
        cv2.fillConvexPoly(layer, cv2.convexHull(pts), 1.0)
        mask = np.maximum(mask, layer * weight)
    return cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5)


def _compute_flow_tensor(prev_gray: np.ndarray, curr_gray: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
    flow   = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)
    flow_f = flow * mask[..., np.newaxis]
    magnitude, _ = cv2.cartToPolar(flow_f[..., 0], flow_f[..., 1])
    dx  = cv2.normalize(flow_f[..., 0], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dy  = cv2.normalize(flow_f[..., 1], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mag = cv2.normalize(magnitude,       None,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.stack([dx, dy, mag], axis=-1)


def _get_frame_paths(sequence_dir: Path) -> list[Path]:
    return sorted(
        p for p in sequence_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _frame_number(path: Path) -> int:
    digits = "".join(c for c in path.stem if c.isdigit())
    return int(digits) if digits else 0


def _load_gray_frame(path: Path, detector) -> tuple[np.ndarray | None, np.ndarray | None]:
    frame = cv2.imread(str(path))
    if frame is None:
        return None, None
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result    = detector.detect(mp_image)
    if not result.face_landmarks:
        return None, None
    roi_gray, mapped_lms, _ = _extract_face_and_landmarks(frame, result.face_landmarks[0])
    return roi_gray, mapped_lms


def _load_gray_frame_with_fallback(
    target_number: int, frame_map: dict[int, Path], detector,
    window: int = LANDMARK_FALLBACK_WINDOW,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if target_number in frame_map:
        gray, lms = _load_gray_frame(frame_map[target_number], detector)
        if gray is not None and lms is not None:
            return gray, lms, target_number
    candidates = sorted(
        [k for k in frame_map if abs(k - target_number) <= window and k != target_number],
        key=lambda k: abs(k - target_number),
    )
    for candidate in candidates:
        gray, lms = _load_gray_frame(frame_map[candidate], detector)
        if gray is not None and lms is not None:
            return gray, lms, candidate
    return None, None, target_number


# ══════════════════════════════════════════════════════════════════════════════
# EVM
# ══════════════════════════════════════════════════════════════════════════════

def _apply_evm_to_sequence(
    all_frames_sorted: list[Path], detector, total_frames_excel: int,
    alpha: float = EVM_ALPHA, low_freq: float = EVM_LOW_FREQ,
    high_freq: float = EVM_HIGH_FREQ, levels: int = EVM_LEVELS,
    clip_secs: float = EVM_CLIP_SECS,
) -> dict[int, np.ndarray]:
    fps = max(1.0, total_frames_excel / clip_secs)
    gray_frames: list[np.ndarray | None] = []
    for path in all_frames_sorted:
        gray, _ = _load_gray_frame(path, detector)
        gray_frames.append(gray.astype(np.float32) if gray is not None else None)
    valid_idx = [i for i, g in enumerate(gray_frames) if g is not None]
    if len(valid_idx) < 8:
        raise ValueError(f"Solo {len(valid_idx)} frames con cara; mínimo 8 para EVM.")
    for i, g in enumerate(gray_frames):
        if g is None:
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            gray_frames[i] = gray_frames[nearest].copy()
    n = len(gray_frames)

    def gaussian_pyramid(img, lvls):
        pyr = [img]
        for _ in range(lvls):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    pyramids = [gaussian_pyramid(f, levels) for f in gray_frames]

    def bandpass(signal, low, high, fs):
        nyq   = fs / 2.0
        low_n = np.clip(low  / nyq, 0.001, 0.999)
        hig_n = np.clip(high / nyq, 0.001, 0.999)
        if low_n >= hig_n:
            hig_n = min(low_n + 0.05, 0.999)
        b, a = butter(2, [low_n, hig_n], btype="band")
        return filtfilt(b, a, signal, axis=0)

    filtered_pyrs = []
    for lvl in range(levels + 1):
        stack    = np.stack([pyramids[i][lvl] for i in range(n)], axis=0)
        filtered = bandpass(stack, low_freq, high_freq, fps)
        filtered_pyrs.append(filtered)

    amplified: dict[int, np.ndarray] = {}
    for i in range(n):
        recon = filtered_pyrs[levels][i] * alpha
        for lvl in range(levels - 1, -1, -1):
            th, tw = filtered_pyrs[lvl][i].shape[:2]
            recon  = cv2.pyrUp(recon, dstsize=(tw, th))
            recon  = recon + filtered_pyrs[lvl][i] * alpha
        amplified[i + 1] = np.clip(gray_frames[i] + recon, 0, 255).astype(np.uint8)
    return amplified


def _get_amp_frame(number: int, amp_dict: dict[int, np.ndarray]) -> np.ndarray | None:
    if number in amp_dict:
        return amp_dict[number]
    if amp_dict:
        closest = min(amp_dict, key=lambda k: abs(k - number))
        return amp_dict[closest]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def extract_sequence_oao(
    sequence_dir: Path, onset: int, apex: int, offset: int,
    total_frames_excel: int, detector, use_evm: bool = True,
) -> np.ndarray:
    """Retorna (3, 64, 64, 3): [0]=onset→apex  [1]=apex→offset  [2]=onset→offset"""
    all_frames = _get_frame_paths(sequence_dir)
    if not all_frames:
        raise ValueError("No se encontraron imágenes en la secuencia.")

    all_frames_sorted = sorted(all_frames, key=_frame_number)
    # frame_map 1-based: frame 1 = primer archivo ordenado
    frame_map: dict[int, Path] = {i + 1: p for i, p in enumerate(all_frames_sorted)}
    n_frames = len(frame_map)

    onset_c  = max(1, min(onset,  n_frames))
    apex_c   = max(1, min(apex,   n_frames))
    offset_c = max(1, min(offset, n_frames))

    amp_dict: dict[int, np.ndarray] = {}
    if use_evm:
        amp_dict = _apply_evm_to_sequence(
            all_frames_sorted, detector, total_frames_excel=total_frames_excel)

    # Landmarks con fallback
    onset_gray_lm,  onset_lms,  onset_used  = _load_gray_frame_with_fallback(onset_c,  frame_map, detector)
    apex_gray_lm,   apex_lms,   apex_used   = _load_gray_frame_with_fallback(apex_c,   frame_map, detector)
    offset_gray_lm, offset_lms, offset_used = _load_gray_frame_with_fallback(offset_c, frame_map, detector)

    # Grays para el flujo: EVM si está activo, original si no
    if use_evm:
        _ao = _get_amp_frame(onset_used,  amp_dict)
        _aa = _get_amp_frame(apex_used,   amp_dict)
        _af = _get_amp_frame(offset_used, amp_dict)
        onset_gray  = _ao if _ao  is not None else onset_gray_lm
        apex_gray   = _aa if _aa  is not None else apex_gray_lm
        offset_gray = _af if _af  is not None else offset_gray_lm
    else:
        onset_gray  = onset_gray_lm
        apex_gray   = apex_gray_lm
        offset_gray = offset_gray_lm

    missing_gray = [n for n, g in [("onset", onset_gray), ("apex", apex_gray), ("offset", offset_gray)] if g is None]
    if missing_gray:
        raise ValueError(f"No se pudo obtener gray en: {', '.join(missing_gray)}")

    missing_lms = [n for n, l in [("onset", onset_lms), ("apex", apex_lms), ("offset", offset_lms)] if l is None]
    if missing_lms:
        raise ValueError(f"No se detectó cara en: {', '.join(missing_lms)} ni en ±{LANDMARK_FALLBACK_WINDOW} frames vecinos.")

    mask_oa = _create_roi_mask(apex_lms)
    mask_ao = _create_roi_mask(offset_lms)
    mask_oo = _create_roi_mask(offset_lms)

    flow_onset_apex   = _compute_flow_tensor(onset_gray,  apex_gray,   mask_oa)
    flow_apex_offset  = _compute_flow_tensor(apex_gray,   offset_gray, mask_ao)
    flow_onset_offset = _compute_flow_tensor(onset_gray,  offset_gray, mask_oo)

    return np.stack([flow_onset_apex, flow_apex_offset, flow_onset_offset],
                    axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ESCANEO DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _collect_sequences(
    dataset_root: Path, coding_df: pd.DataFrame, dataset_type: str,
) -> list[tuple[Path, str, str, str, int, int, int, int]]:
    """
    Retorna lista de (seq_dir, subject, emotion_canonical, seq_name,
                      onset, apex, offset, total_frames)
    """
    sequences = []

    if dataset_type == "SMIC_E":
        for subject_dir in sorted(dataset_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for seq_dir in sorted(subject_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                mask = (coding_df["Subject"] == subject_dir.name) & \
                       (coding_df["Sequence"] == seq_dir.name)
                row = coding_df[mask]
                if row.empty:
                    continue
                r = row.iloc[0]
                sequences.append((
                    seq_dir, subject_dir.name, "sorpresa", seq_dir.name,
                    int(r["OnsetFrame"]), int(r["ApexFrame"]),
                    int(r["OffsetFrame"]), int(r["TotalFrames"])
                ))

    elif dataset_type == "CASME2":
        for subject_dir in sorted(dataset_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for emotion_dir in sorted(subject_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                if emotion_dir.name.lower() in EXCLUDE_CASME_CARPETAS:
                    continue
                for seq_dir in sorted(emotion_dir.iterdir()):
                    if not seq_dir.is_dir():
                        continue
                    mask = (coding_df["Subject"]  == subject_dir.name) & \
                           (coding_df["Sequence"] == seq_dir.name)
                    row = coding_df[mask]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    emotion_canonical = r["Emotion"]
                    if emotion_canonical in EXCLUDE_CASME_EXCEL:
                        continue
                    sequences.append((
                        seq_dir, subject_dir.name, emotion_canonical, seq_dir.name,
                        int(r["OnsetFrame"]), int(r["ApexFrame"]),
                        int(r["OffsetFrame"]), int(r["TotalFrames"])
                    ))

    elif dataset_type == "PROPIO":
        for subject_dir in sorted(dataset_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for cam_dir in sorted(subject_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                # Solo cámara web — saltar infrarroja
                if "infrarroja" in cam_dir.name.lower() or "infrared" in cam_dir.name.lower():
                    continue
                for emotion_dir in sorted(cam_dir.iterdir()):
                    if not emotion_dir.is_dir():
                        continue
                    # Mapear nombre de carpeta a canónico
                    emotion_canonical = EMOTION_CANONICAL.get(
                        emotion_dir.name.strip().lower(),
                        emotion_dir.name.strip().lower()
                    )
                    # Buscar en el DataFrame: subject + camara + emocion canónica
                    # Si no existe registro → saltar (la carpeta existe pero no fue anotada)
                    mask = (
                        (coding_df["Subject"]  == subject_dir.name) &
                        (coding_df["Sequence"] == cam_dir.name) &
                        (coding_df["Emotion"]  == emotion_canonical)
                    )
                    row = coding_df[mask]
                    if row.empty:
                        continue  # carpeta sin registro en el Excel → saltar
                    r = row.iloc[0]
                    onset  = int(r["OnsetFrame"])
                    apex   = int(r["ApexFrame"])
                    offset = int(r["OffsetFrame"])
                    total  = int(r["TotalFrames"])
                    seq_name = f"{cam_dir.name}_{emotion_dir.name}"
                    sequences.append((
                        emotion_dir, subject_dir.name, emotion_canonical, seq_name,
                        onset, apex, offset, total
                    ))

    return sequences


def _append_csv_row(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════

class ExtractionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Extracción OAO+EVM — (3, 64, 64, 3)")
        self.root.geometry("860x780")
        self.root.configure(bg=_C_BG)
        self.root.resizable(True, True)
        self.root.minsize(720, 600)
        self._stop_flag    = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Green.Horizontal.TProgressbar",
            troughcolor=_C_SURFACE, background=_C_GREEN,
            bordercolor=_C_BORDER, lightcolor=_C_GREEN, darkcolor=_C_GREEN_H)

        container = tk.Frame(self.root, bg=_C_BG, padx=20, pady=16)
        container.pack(fill="both", expand=True)
        tk.Label(container, text="Extracción OAO + EVM  —  (3, 64, 64, 3)",
                 bg=_C_BG, fg=_C_TEXT, font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0,12))

        self._dataset_var = tk.StringVar()
        self._output_var  = tk.StringVar()
        self._model_var   = tk.StringVar(value=str(_MODEL_PATH))
        self._excel_var   = tk.StringVar()

        paths_frame = tk.Frame(container, bg=_C_SURFACE, padx=12, pady=10)
        paths_frame.pack(fill="x", pady=(0, 8))
        for label_text, var, cmd in [
            ("Dataset raíz:",          self._dataset_var, self._browse_dataset),
            ("Directorio de salida:",  self._output_var,  self._browse_output),
            ("Modelo MediaPipe:",      self._model_var,   self._browse_model),
            ("Tabla de codificación:", self._excel_var,   self._browse_excel),
        ]:
            r = tk.Frame(paths_frame, bg=_C_SURFACE)
            r.pack(fill="x", pady=4)
            tk.Label(r, text=label_text, bg=_C_SURFACE, fg=_C_SUBTEXT,
                     font=("Segoe UI", 9), width=24, anchor="w").pack(side="left")
            tk.Entry(r, textvariable=var, bg="#0f172a", fg=_C_TEXT,
                     insertbackground=_C_TEXT, relief="flat",
                     font=("Segoe UI", 9), width=46).pack(side="left", padx=(4, 6))
            tk.Button(r, text="...", command=cmd, bg=_C_BORDER, fg=_C_TEXT,
                      relief="flat", padx=8, pady=2,
                      font=("Segoe UI", 9), cursor="hand2").pack(side="left")

        evm_frame = tk.Frame(container, bg=_C_SURFACE, padx=12, pady=10)
        evm_frame.pack(fill="x", pady=(0, 8))
        tk.Label(evm_frame, text="Opciones EVM", bg=_C_SURFACE, fg=_C_TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))
        self._use_evm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(evm_frame, text="Activar Eulerian Video Magnification",
                       variable=self._use_evm_var, bg=_C_SURFACE, fg=_C_TEXT,
                       selectcolor=_C_BORDER, activebackground=_C_SURFACE,
                       activeforeground=_C_TEXT, font=("Segoe UI", 9)).pack(anchor="w")
        pr = tk.Frame(evm_frame, bg=_C_SURFACE)
        pr.pack(fill="x", pady=(6, 0))
        self._alpha_var    = tk.StringVar(value=str(EVM_ALPHA))
        self._low_var      = tk.StringVar(value=str(EVM_LOW_FREQ))
        self._high_var     = tk.StringVar(value=str(EVM_HIGH_FREQ))
        self._clip_sec_var = tk.StringVar(value=str(EVM_CLIP_SECS))
        for lbl, var, w in [("Alpha:", self._alpha_var, 6),
                             ("Frec. baja Hz:", self._low_var, 6),
                             ("Frec. alta Hz:", self._high_var, 6),
                             ("Duración clip s:", self._clip_sec_var, 6)]:
            tk.Label(pr, text=lbl, bg=_C_SURFACE, fg=_C_SUBTEXT,
                     font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
            tk.Entry(pr, textvariable=var, bg="#0f172a", fg=_C_TEXT,
                     insertbackground=_C_TEXT, relief="flat",
                     font=("Segoe UI", 9), width=w).pack(side="left", padx=(0, 12))

        tk.Label(container,
                 text="ℹ  Salida: (3,64,64,3)  [0]=onset→apex  [1]=apex→offset  [2]=onset→offset\n"
                      "   Fallback ±5 frames si MediaPipe no detecta cara\n"
                      "   CASME II: excluye repression+others  |  SMIC: solo surprise\n"
                      "   MEME: cada emoción en su carpeta propia \n"
                      "   La fusión se controla en train_mex.py con --merge_classes",
                 justify="left", bg=_C_BG, fg=_C_SUBTEXT,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 8))

        btn_frame = tk.Frame(container, bg=_C_BG)
        btn_frame.pack(fill="x", pady=(0, 10))
        self.start_btn = tk.Button(btn_frame, text="▶  Iniciar Extracción",
            bg=_C_GREEN, fg="white", activebackground=_C_GREEN_H, activeforeground="white",
            font=("Segoe UI", 10, "bold"), relief="flat", padx=18, pady=8,
            cursor="hand2", command=self._start_extraction)
        self.start_btn.pack(side="left", padx=(0, 10))
        self.stop_btn = tk.Button(btn_frame, text="■  Detener",
            bg=_C_DISBLD, fg=_C_SUBTEXT, activebackground=_C_RED_H, activeforeground="white",
            font=("Segoe UI", 10, "bold"), relief="flat", padx=18, pady=8,
            cursor="hand2", state="disabled", command=self._stop_extraction)
        self.stop_btn.pack(side="left")

        prog_frame = tk.Frame(container, bg=_C_SURFACE, padx=12, pady=10)
        prog_frame.pack(fill="x", pady=(0, 8))
        self._progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(prog_frame, variable=self._progress_var, maximum=100,
                        mode="determinate",
                        style="Green.Horizontal.TProgressbar").pack(fill="x", pady=(0, 8))
        sr = tk.Frame(prog_frame, bg=_C_SURFACE)
        sr.pack(fill="x")
        self._pct_label = tk.Label(sr, text="0%", bg=_C_SURFACE, fg=_C_TEXT,
                                   font=("Segoe UI", 10, "bold"))
        self._pct_label.pack(side="left")
        self._stats_label = tk.Label(sr, text="Total: 0  |  Procesadas: 0  |  ✓ 0  |  ✗ 0",
                                     bg=_C_SURFACE, fg=_C_SUBTEXT, font=("Segoe UI", 9))
        self._stats_label.pack(side="right")

        log_outer = tk.Frame(container, bg=_C_BORDER, padx=1, pady=1)
        log_outer.pack(fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(
            log_outer, bg="#0a0f1e", fg=_C_TEXT, insertbackground=_C_TEXT,
            relief="flat", font=("Consolas", 9), state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True)
        for tag, color in [("ok", _C_GREEN), ("fail", _C_RED),
                           ("info", _C_SUBTEXT), ("warn", _C_AMBER), ("done", _C_TEXT)]:
            self.log_text.tag_config(tag, foreground=color)

    def _browse_dataset(self):
        p = filedialog.askdirectory(title="Carpeta raíz del dataset")
        if p: self._dataset_var.set(p)

    def _browse_output(self):
        p = filedialog.askdirectory(title="Directorio de salida")
        if p: self._output_var.set(p)

    def _browse_model(self):
        p = filedialog.askopenfilename(title="Modelo face_landmarker.task",
            filetypes=[("MediaPipe Task", "*.task"), ("Todos", "*.*")])
        if p: self._model_var.set(p)

    def _browse_excel(self):
        p = filedialog.askopenfilename(title="Archivo de codificación",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("Todos", "*.*")])
        if p: self._excel_var.set(p)

    def _log(self, msg: str, tag: str = "info"):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_progress(self, total, processed, success, failed):
        pct = (processed / total * 100) if total > 0 else 0.0
        self._progress_var.set(pct)
        self._pct_label.configure(text=f"{pct:.1f}%")
        self._stats_label.configure(
            text=f"Total: {total}  |  Procesadas: {processed}  |  ✓ {success}  |  ✗ {failed}")

    def _start_extraction(self):
        dataset_root = Path(self._dataset_var.get().strip())
        output_dir   = Path(self._output_var.get().strip())
        model_path   = Path(self._model_var.get().strip())
        excel_path   = Path(self._excel_var.get().strip())

        if not self._dataset_var.get().strip() or not dataset_root.is_dir():
            self._log("✗ El directorio del dataset no existe.", "fail"); return
        if not self._output_var.get().strip():
            self._log("✗ Especifica un directorio de salida.", "fail"); return
        if not model_path.exists():
            self._log(f"✗ Modelo no encontrado: {model_path}", "fail"); return
        if not excel_path.exists():
            self._log(f"✗ Archivo no encontrado: {excel_path}", "fail"); return
        try:
            alpha    = float(self._alpha_var.get())
            low_freq = float(self._low_var.get())
            high_freq= float(self._high_var.get())
            clip_sec = float(self._clip_sec_var.get())
        except ValueError:
            self._log("✗ Los parámetros EVM deben ser números.", "fail"); return

        self._stop_flag.clear()
        self._progress_var.set(0.0)
        self._pct_label.configure(text="0%")
        self._stats_label.configure(text="Total: 0  |  Procesadas: 0  |  ✓ 0  |  ✗ 0")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.start_btn.configure(state="disabled", bg=_C_DISBLD, fg=_C_SUBTEXT)
        self.stop_btn.configure(state="normal", bg=_C_RED, fg="white")

        self._worker_thread = threading.Thread(
            target=self._extraction_worker,
            args=(dataset_root, output_dir, model_path, excel_path,
                  self._use_evm_var.get(), alpha, low_freq, high_freq, clip_sec),
            daemon=True)
        self._worker_thread.start()

    def _stop_extraction(self):
        self._stop_flag.set()
        self.stop_btn.configure(state="disabled", bg=_C_DISBLD, fg=_C_SUBTEXT)
        self.root.after(0, self._log, "⏹  Deteniendo...", "warn")

    def _extraction_worker(self, dataset_root, output_dir, model_path, excel_path,
                            use_evm, alpha, low_freq, high_freq, clip_sec):
        global EVM_ALPHA, EVM_LOW_FREQ, EVM_HIGH_FREQ, EVM_CLIP_SECS
        EVM_ALPHA, EVM_LOW_FREQ, EVM_HIGH_FREQ, EVM_CLIP_SECS = alpha, low_freq, high_freq, clip_sec

        self.root.after(0, self._log, f"Cargando tabla: {excel_path}", "info")
        try:
            coding_df, dataset_type = load_coding_table(excel_path)
        except Exception as exc:
            self.root.after(0, self._log, f"✗ Error: {exc}", "fail")
            self.root.after(0, self._finish_worker); return

        evm_txt = f"EVM={'ON' if use_evm else 'OFF'}  α={alpha}  [{low_freq}–{high_freq} Hz]"
        self.root.after(0, self._log,
            f"Tipo: {dataset_type}  |  {evm_txt}  |  Registros: {len(coding_df)}", "ok")

        sequences = _collect_sequences(dataset_root, coding_df, dataset_type)
        total = len(sequences)
        if total == 0:
            self.root.after(0, self._log, "✗ No se encontraron secuencias.", "fail")
            self.root.after(0, self._finish_worker); return

        self.root.after(0, self._log,
            f"Secuencias encontradas: {total} — cargando modelo...", "info")
        detector = _init_landmarker(model_path)
        if detector is None:
            self.root.after(0, self._log, "✗ No se pudo cargar el modelo.", "fail")
            self.root.after(0, self._finish_worker); return

        self.root.after(0, self._log, f"Modelo OK. Iniciando extracción...\n", "info")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / OUTPUT_CSV_NAME
        success = failed = 0

        for idx, (seq_dir, subject, emotion, seq_name, onset, apex, offset, total_f) \
                in enumerate(sequences, 1):
            if self._stop_flag.is_set():
                self.root.after(0, self._log, f"\n⏹  Detenido en {idx-1}/{total}.", "warn")
                break
            try:
                volume = extract_sequence_oao(
                    seq_dir, onset, apex, offset, total_f, detector, use_evm=use_evm)

                out_dir  = output_dir / subject / emotion
                out_dir.mkdir(parents=True, exist_ok=True)
                npy_path = out_dir / f"{seq_name}.npy"
                np.save(str(npy_path), volume)

                n_f, h, w, c = volume.shape
                _append_csv_row(csv_path, {
                    "sujeto":       subject,
                    "emocion":      emotion,
                    "secuencia":    seq_name,
                    "archivo":      str(npy_path.relative_to(output_dir)),
                    "onset_frame":  onset,
                    "apex_frame":   apex,
                    "offset_frame": offset,
                    "total_frames": total_f,
                    "evm":          use_evm,
                    "num_flujos":   n_f,
                    "alto":         h,
                    "ancho":        w,
                    "canales":      c,
                    "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                success += 1
                self.root.after(0, self._log,
                    f"[{idx:>4}/{total}] ✓  {subject}/{emotion}/{seq_name}  "
                    f"(O:{onset} A:{apex} F:{offset})", "ok")

            except Exception as exc:
                failed += 1
                self.root.after(0, self._log,
                    f"[{idx:>4}/{total}] ✗  {subject}/{emotion}/{seq_name} — {exc}", "fail")

            self.root.after(0, self._update_progress, total, idx, success, failed)

        if hasattr(detector, "close"):
            detector.close()

        self.root.after(0, self._log,
            f"\n{'─'*50}\nFinalizado.  ✓{success}  ✗{failed}  total:{total}\n"
            f"CSV: {output_dir / OUTPUT_CSV_NAME}", "done")
        self.root.after(0, self._finish_worker)

    def _finish_worker(self):
        self.start_btn.configure(state="normal", bg=_C_GREEN, fg="white")
        self.stop_btn.configure(state="disabled", bg=_C_DISBLD, fg=_C_SUBTEXT)


def main() -> None:
    root = tk.Tk()
    ExtractionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()