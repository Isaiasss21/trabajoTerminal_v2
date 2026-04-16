from __future__ import annotations

import bz2
import hashlib
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import dlib
import numpy as np
import requests
from ultralytics import YOLO


@dataclass
class SpotSegment:
    start: int
    end: int
    apex_idx: int
    motion_score: float


@dataclass
class SequenceExtractionBundle:
    faces: List[np.ndarray]
    landmarks: List[np.ndarray]
    segment: Optional[SpotSegment]
    bboxes: List[Tuple[int, int, int, int]]

# -------------------------
# Utilidades de descarga
# -------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dst: Path, min_bytes: int = 1_000_000, timeout: int = 60) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code != 200:
            return False
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return dst.exists() and dst.stat().st_size >= min_bytes
    except Exception:
        return False

def ensure_yolo_face_model(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return
    urls = [
        "https://github.com/SannketNikam/Face-Detection/raw/main/yolov8n-face.pt",
        "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face.pt",
    ]
    ok = False
    for u in urls:
        if download_file(u, dst, min_bytes=1_000_000):
            ok = True
            break
    if not ok:
        raise FileNotFoundError(
            f"No se pudo obtener {dst.name}. Descárgalo manualmente y colócalo en: {dst}"
        )

def ensure_dlib_shape_predictor(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return
    bz2_path = dst.with_suffix(dst.suffix + ".bz2")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    if not bz2_path.exists():
        if not download_file(url, bz2_path, min_bytes=10_000_000):
            raise FileNotFoundError(
                "No se pudo descargar shape_predictor_68_face_landmarks.dat.bz2. "
                "Descárgalo manualmente desde dlib.net y colócalo en assets/."
            )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with bz2.BZ2File(str(bz2_path), "rb") as fin, dst.open("wb") as fout:
        fout.write(fin.read())
    if not dst.exists():
        raise FileNotFoundError(f"No se pudo crear {dst}")

# -------------------------
# Preprocesador
# -------------------------
@dataclass
class PreprocesadorFacial:
    img_size: Tuple[int,int] = (640, 640)
    assets_dir: Path = Path("assets")
    device: str = "cpu"

    def __post_init__(self):
        self.assets_dir = Path("assets/models")
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_path = self.assets_dir / "yolov8n-face.pt"
        self.shape_path = self.assets_dir / "shape_predictor_68_face_landmarks.dat"

        ensure_yolo_face_model(self.yolo_path)
        ensure_dlib_shape_predictor(self.shape_path)

        # CORRECCIÓN: Inicializamos YOLO en el dispositivo solicitado
        self.detector_yolo = YOLO(str(self.yolo_path))
        try:
            self.detector_yolo.to(self.device)
        except Exception:
            self.detector_yolo.to('cpu')
        
        self.predictor_landmarks = dlib.shape_predictor(str(self.shape_path))
        
        self.frame_anterior = None
        self.bbox_anterior = None

    def _compute_optical_flow_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(magnitude.mean())

    def _load_resized_frames(self, frame_paths: List[str]) -> List[Optional[np.ndarray]]:
        frames: List[Optional[np.ndarray]] = []
        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is None:
                frames.append(None)
                continue
            frame_rgb = self._convertir_a_rgb(frame)
            if frame_rgb is None:
                frames.append(None)
                continue
            frames.append(cv2.resize(frame_rgb, self.img_size))
        return frames

    @staticmethod
    def _segment_from_scores(scores: List[float], start_idx: int, end_idx: int) -> SpotSegment:
        segment_scores = scores[start_idx:end_idx + 1]
        apex_rel = int(np.argmax(segment_scores))
        apex_idx = start_idx + apex_rel
        return SpotSegment(
            start=start_idx,
            end=end_idx,
            apex_idx=apex_idx,
            motion_score=float(np.max(segment_scores)),
        )

    def _build_spot_segments_from_scores(
        self,
        scores: List[float],
        threshold: float,
        min_frames: int,
        max_frames: int,
    ) -> List[SpotSegment]:
        segments: List[SpotSegment] = []
        in_segment = False
        start_idx = 0

        for i in range(1, len(scores)):
            if scores[i] > threshold:
                if not in_segment:
                    in_segment = True
                    start_idx = i - 1
                continue

            if in_segment:
                end_idx = i
                duration = end_idx - start_idx
                if min_frames <= duration <= max_frames:
                    segments.append(self._segment_from_scores(scores, start_idx, end_idx))
                in_segment = False

        if in_segment:
            end_idx = len(scores) - 1
            duration = end_idx - start_idx
            if min_frames <= duration <= max_frames:
                segments.append(self._segment_from_scores(scores, start_idx, end_idx))

        return segments

    def _extract_aligned_faces_and_landmarks(
        self,
        selected_paths: List[str],
        save_aligned_size: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int, int, int]]]:
        faces: List[np.ndarray] = []
        landmarks: List[np.ndarray] = []
        bboxes: List[Tuple[int, int, int, int]] = []

        for fp in selected_paths:
            frame = cv2.imread(fp)
            if frame is None:
                continue

            frame_rgb = self._convertir_a_rgb(frame)
            if frame_rgb is None:
                continue

            frame_resized = cv2.resize(frame_rgb, self.img_size)
            extraccion = self._extraer_landmarks_y_bbox(frame_resized)
            if extraccion is None:
                continue

            landmarks_norm, bbox = extraccion
            landmarks.append(landmarks_norm.astype(np.float32))
            bboxes.append(bbox)

            landmarks_abs = self._reconstruir_landmarks_absolutos(landmarks_norm, bbox)
            aligned = self._alinear_rostro_afin(frame_resized, landmarks_abs, out_size=save_aligned_size)
            if aligned is None:
                x1, y1, x2, y2 = bbox
                crop = frame_resized[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                aligned = cv2.resize(crop, (save_aligned_size, save_aligned_size))
            faces.append(aligned)

        return faces, landmarks, bboxes

    def spot_segments(
        self,
        frame_paths: List[str],
        threshold: float = 0.015,
        min_frames: int = 8,
        max_frames: int = 32,
    ) -> List[SpotSegment]:
        if len(frame_paths) < 2:
            return []

        frames = self._load_resized_frames(frame_paths)
        valid_idx = [i for i, frame in enumerate(frames) if frame is not None]
        if len(valid_idx) < 2:
            return []

        scores = [0.0] * len(frames)
        for i in range(1, len(frames)):
            if frames[i - 1] is None or frames[i] is None:
                continue
            scores[i] = self._compute_optical_flow_score(frames[i - 1], frames[i])

        return self._build_spot_segments_from_scores(scores, threshold, min_frames, max_frames)

    def extract_sequence_bundle(
        self,
        frame_paths: List[str],
        window_size: int = 5,
        spot_threshold: float = 0.015,
        min_frames: int = 8,
        max_frames: int = 32,
        save_aligned_size: int = 224,
        fallback_to_fixed_window: bool = True,
    ) -> Optional[SequenceExtractionBundle]:
        if not frame_paths:
            return None

        segments = self.spot_segments(
            frame_paths,
            threshold=spot_threshold,
            min_frames=min_frames,
            max_frames=max_frames,
        )

        if segments:
            segment = max(segments, key=lambda seg: seg.motion_score)
            start_idx, end_idx = segment.start, segment.end
        elif fallback_to_fixed_window:
            segment = None
            apex_idx = len(frame_paths) // 2
            half_window = max(1, window_size // 2)
            start_idx = max(0, apex_idx - half_window)
            end_idx = min(len(frame_paths), start_idx + window_size)
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)
        else:
            return None

        faces, landmarks, bboxes = self._extract_aligned_faces_and_landmarks(
            frame_paths[start_idx:end_idx],
            save_aligned_size,
        )
        if len(faces) < 2 or len(landmarks) < 2:
            return None

        return SequenceExtractionBundle(
            faces=faces,
            landmarks=landmarks,
            segment=segment,
            bboxes=bboxes,
        )

    def _convertir_a_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 3:
            return frame
        else:
            try:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            except Exception:
                return None
            
    def _filtrar_puntos_criticos(self, landmarks_1d: np.ndarray) -> np.ndarray:
        coords = landmarks_1d.reshape(68, 2)
        ancla_nariz = coords[30]
        criticos = coords[17:]
        criticos_centrados = criticos - ancla_nariz
        return criticos_centrados.flatten()

    def procesar_frame(self, frame: np.ndarray, _frame_path: str = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if frame is None:
            return None, None
        
        frame_rgb = self._convertir_a_rgb(frame)
        if frame_rgb is None:
            return None, None
            
        frame_resized = cv2.resize(frame_rgb, self.img_size)
        extraccion = self._extraer_landmarks_y_bbox(frame_resized)
        if extraccion is None:
            return None, None
            
        landmarks_norm, bbox_actual = extraccion
        
        flow_features = None
        if self.frame_anterior is not None and self.bbox_anterior is not None:
            flow_features = self._calcular_flow_features_roi(self.frame_anterior, frame_resized, bbox_actual)
        
        self.frame_anterior = frame_resized.copy()
        self.bbox_anterior = bbox_actual
        
        return landmarks_norm, flow_features
    
    def _extraer_landmarks_y_bbox(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        try:
            frame_rgb = self._convertir_a_rgb(frame)
            if frame_rgb is None:
                return None
                
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            results = self.detector_yolo(frame_rgb, verbose=False)
            if not results or len(results) == 0:
                return None

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return None

            best = None
            best_conf = -1.0
            for b in boxes:
                conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                xyxy = b.xyxy[0].cpu().numpy()
                if conf > best_conf:
                    best_conf = conf
                    best = xyxy

            if best is None:
                return None

            x1, y1, x2, y2 = best.astype(float)
            H, W = gray_eq.shape[:2]
            x1 = max(0.0, min(x1, W-1))
            x2 = max(0.0, min(x2, W-1))
            y1 = max(0.0, min(y1, H-1))
            y2 = max(0.0, min(y2, H-1))
            
            if x2 <= x1 or y2 <= y1:
                return None

            rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
            shape = self.predictor_landmarks(gray_eq, rect)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

            try:
                landmarks_norm = self._normalizar_landmarks_en_bbox(landmarks, (x1, y1, x2, y2))
            except Exception:
                return None

            bbox_entero = (int(x1), int(y1), int(x2), int(y2))
            return landmarks_norm.reshape(-1).astype(np.float32), bbox_entero
            
        except Exception:
            return None
            
    def _extraer_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        res = self._extraer_landmarks_y_bbox(frame)
        return res[0] if res else None

    def _calcular_flow_features_roi(self, frame1: np.ndarray, frame2: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        frame1_rgb = self._convertir_a_rgb(frame1)
        frame2_rgb = self._convertir_a_rgb(frame2)
        
        if frame1_rgb is None or frame2_rgb is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
        gray1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_rgb, cv2.COLOR_BGR2GRAY)

        x1, y1, x2, y2 = bbox
        
        roi1 = gray1[y1:y2, x1:x2]
        roi2 = gray2[y1:y2, x1:x2]

        if roi1.size == 0 or roi2.size == 0 or roi1.shape != roi2.shape:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            roi1, roi2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_mean = float(np.mean(mag))
        mag_std  = float(np.std(mag))
        ang_mean = float(np.mean(ang))
        ang_std  = float(np.std(ang))
        
        return np.array([mag_mean, mag_std, ang_mean, ang_std], dtype=np.float32)

    def _calcular_flow_features(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        H, W = frame1.shape[:2]
        return self._calcular_flow_features_roi(frame1, frame2, (0, 0, W, H))

    @staticmethod
    def _normalizar_landmarks_en_bbox(landmarks: np.ndarray, bbox: Tuple[float,float,float,float]) -> np.ndarray:
        landmarks = np.asarray(landmarks, dtype=np.float32)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            raise ValueError(f"BBox inválido: {bbox}")
        xs_norm = (landmarks[:, 0] - x1) / w
        ys_norm = (landmarks[:, 1] - y1) / h
        return np.stack([xs_norm, ys_norm], axis=1)

    @staticmethod
    def _reconstruir_landmarks_absolutos(
        landmarks_norm: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        pts = landmarks_norm.reshape(68, 2).copy()
        x1, y1, x2, y2 = bbox
        w = float(max(1, x2 - x1))
        h = float(max(1, y2 - y1))
        pts[:, 0] = pts[:, 0] * w + float(x1)
        pts[:, 1] = pts[:, 1] * h + float(y1)
        return pts.astype(np.float32)

    def _alinear_rostro_afin(
        self,
        frame: np.ndarray,
        landmarks_abs: np.ndarray,
        out_size: int = 112,
    ) -> Optional[np.ndarray]:
        if landmarks_abs is None or landmarks_abs.shape != (68, 2):
            return None

        left_eye = landmarks_abs[36:42].mean(axis=0)
        right_eye = landmarks_abs[42:48].mean(axis=0)
        nose_tip = landmarks_abs[30]
        mouth_l = landmarks_abs[48]
        mouth_r = landmarks_abs[54]

        src = np.array([left_eye, right_eye, nose_tip, mouth_l, mouth_r], dtype=np.float32)
        dst_112 = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        dst = dst_112 * (float(out_size) / 112.0)

        try:
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            if M is None:
                return None
            aligned = cv2.warpAffine(
                frame,
                M,
                (out_size, out_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(128, 128, 128),
            )
            return aligned
        except Exception:
            return None

    @staticmethod
    def _extraer_hog_pyfeat_like(
        face_alineada: np.ndarray,
        orientations: int = 8,
        pixels_per_cell: int = 8,
        cells_per_block: int = 2,
    ) -> Optional[np.ndarray]:
        if face_alineada is None:
            return None

        if len(face_alineada.shape) == 3:
            gray = cv2.cvtColor(face_alineada, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_alineada

        h, w = gray.shape[:2]
        block = pixels_per_cell * cells_per_block
        if h < block or w < block:
            return None

        win_w = (w // pixels_per_cell) * pixels_per_cell
        win_h = (h // pixels_per_cell) * pixels_per_cell
        gray = gray[:win_h, :win_w]

        try:
            hog = cv2.HOGDescriptor(
                _winSize=(win_w, win_h),
                _blockSize=(block, block),
                _blockStride=(pixels_per_cell, pixels_per_cell),
                _cellSize=(pixels_per_cell, pixels_per_cell),
                _nbins=orientations,
            )
            vec = hog.compute(gray)
            if vec is None:
                return None
            return vec.reshape(-1).astype(np.float32)
        except Exception:
            return None

    def procesar_secuencia_frames(
        self,
        frames_paths: List[str],
        usar_flow: bool = True,
        salto_frames: int = 3,
        feature_mode: str = "legacy208",
        usar_alineacion_afin: bool = False,
        baseline_frames: int = 0,
        face_size_alineada: int = 112,
    ) -> Optional[np.ndarray]:
        if len(frames_paths) < 2: return None

        mode = str(feature_mode).lower().strip()
        if mode not in {"legacy208", "hog", "hybrid"}:
            raise ValueError("feature_mode debe ser: legacy208 | hog | hybrid")

        frames_filtrados = frames_paths[::salto_frames]
        
        if len(frames_filtrados) < 2: 
            return None

        feats: List[np.ndarray] = []
        self.frame_anterior = None
        self.bbox_anterior = None
        landmarks_referencia = None
        landmarks_anteriores = None

        for frame_path in frames_filtrados:
            vec_base, landmarks_referencia, landmarks_anteriores, frame_resized, bbox_actual = self._sequence_step(
                frame_path,
                mode,
                landmarks_referencia,
                landmarks_anteriores,
                usar_alineacion_afin,
                face_size_alineada,
                usar_flow,
            )
            if vec_base is None or frame_resized is None or bbox_actual is None:
                continue

            feats.append(vec_base)
            self.frame_anterior = frame_resized.copy()
            self.bbox_anterior = bbox_actual

        if usar_flow:
            if len(feats) < 1: return None
        else:
            if len(feats) < 3: return None

        seq = np.stack(feats, axis=0).astype(np.float32)

        if baseline_frames and baseline_frames > 0 and seq.shape[0] > 0:
            k = min(int(baseline_frames), int(seq.shape[0]))
            baseline = seq[:k].mean(axis=0, keepdims=True)
            seq = seq - baseline

        return seq

    def _build_sequence_base_vector(
        self,
        mode: str,
        frame_resized: np.ndarray,
        landmarks_norm: np.ndarray,
        bbox_actual: Tuple[int, int, int, int],
        delta: np.ndarray,
        velocidad: np.ndarray,
        usar_alineacion_afin: bool,
        face_size_alineada: int,
    ) -> Optional[np.ndarray]:
        vec_legacy = np.concatenate([delta, velocidad])

        if mode == "legacy208":
            return vec_legacy

        landmarks_abs = self._reconstruir_landmarks_absolutos(landmarks_norm, bbox_actual)
        if usar_alineacion_afin:
            face_hog = self._alinear_rostro_afin(frame_resized, landmarks_abs, out_size=int(face_size_alineada))
        else:
            x1, y1, x2, y2 = bbox_actual
            face_hog = frame_resized[y1:y2, x1:x2]
            if face_hog is not None and face_hog.size != 0:
                face_hog = cv2.resize(face_hog, (int(face_size_alineada), int(face_size_alineada)))
            else:
                face_hog = None

        hog_vec = self._extraer_hog_pyfeat_like(face_hog)
        if mode == "hog":
            return hog_vec
        if hog_vec is None:
            return None
        return np.concatenate([vec_legacy, hog_vec], axis=0)

    def _sequence_step(
        self,
        frame_path: str,
        mode: str,
        landmarks_referencia: Optional[np.ndarray],
        landmarks_anteriores: Optional[np.ndarray],
        usar_alineacion_afin: bool,
        face_size_alineada: int,
        usar_flow: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        frame = cv2.imread(frame_path)
        if frame is None:
            return None, landmarks_referencia, landmarks_anteriores, None, None

        frame_rgb = self._convertir_a_rgb(frame)
        if frame_rgb is None:
            return None, landmarks_referencia, landmarks_anteriores, None, None

        frame_resized = cv2.resize(frame_rgb, self.img_size)
        extraccion = self._extraer_landmarks_y_bbox(frame_resized)
        if extraccion is None:
            return None, landmarks_referencia, landmarks_anteriores, None, None

        landmarks_norm, bbox_actual = extraccion
        puntos_criticos = self._filtrar_puntos_criticos(landmarks_norm)

        if landmarks_referencia is None:
            landmarks_referencia = puntos_criticos.copy()

        delta = (puntos_criticos - landmarks_referencia) * 10.0
        velocidad = np.zeros_like(puntos_criticos) if landmarks_anteriores is None else (puntos_criticos - landmarks_anteriores) * 5.0

        vec_base = self._build_sequence_base_vector(
            mode,
            frame_resized,
            landmarks_norm,
            bbox_actual,
            delta,
            velocidad,
            usar_alineacion_afin,
            face_size_alineada,
        )
        if vec_base is None:
            return None, landmarks_referencia, puntos_criticos, frame_resized.copy(), bbox_actual

        if usar_flow and self.frame_anterior is not None and self.bbox_anterior is not None:
            flow_features = self._calcular_flow_features_roi(self.frame_anterior, frame_resized, bbox_actual)
            vec_base = np.concatenate([vec_base, flow_features], axis=0)

        return vec_base, landmarks_referencia, puntos_criticos, frame_resized.copy(), bbox_actual

    def procesar_imagen(self, image_path: str) -> Optional[Dict]:
        p = Path(image_path)
        if not p.exists():
            return None
        frame = cv2.imread(str(p))
        if frame is None:
            return None
        frame_rgb = self._convertir_a_rgb(frame)
        if frame_rgb is None:
            return None
            
        landmarks_norm = self._extraer_landmarks(frame_rgb)
        if landmarks_norm is None:
            return None
            
        return {"feature_vector": landmarks_norm}

    def calcular_optical_flow_roi(self, img_path_1: str, img_path_2: str) -> Optional[np.ndarray]:
        p1 = Path(img_path_1); p2 = Path(img_path_2)
        if not p1.exists() or not p2.exists():
            return None
        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))
        if img1 is None or img2 is None:
            return None
        img1_rgb = self._convertir_a_rgb(img1)
        img2_rgb = self._convertir_a_rgb(img2)
        if img1_rgb is None or img2_rgb is None:
            return None
        img1_resized = cv2.resize(img1_rgb, self.img_size)
        img2_resized = cv2.resize(img2_rgb, self.img_size)
        
        H, W = img1_resized.shape[:2]
        return self._calcular_flow_features_roi(img1_resized, img2_resized, (0, 0, W, H))