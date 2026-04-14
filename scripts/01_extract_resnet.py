import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.preprocessing import PreprocesadorFacial

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extraer características espaciales con ResNet-18 + Apex Spotting")
    p.add_argument("--sequences_csv", required=True, help="Ruta al sequences.csv")
    p.add_argument("--out_dir", default="pipeline_out_resnet", help="Carpeta de salida")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--aligned_face_size", type=int, default=224, help="Tamaño esperado por ResNet")
    p.add_argument("--window_size", type=int, default=5, help="Frames a conservar alrededor del Apex")
    return p.parse_args()

def extract_faces_with_apex_spotting(
    pre: PreprocesadorFacial,
    frame_paths: list[str],
    out_size: int,
    window_size: int
) -> list[np.ndarray]:
    rostros_grises = []
    
    for fp in frame_paths:
        try:
            frame = cv2.imread(fp)
            if frame is None:
                rostros_grises.append(None)
                continue
            
            frame_rgb = pre._convertir_a_rgb(frame)
            results = pre.detector_yolo(frame_rgb, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                rostros_grises.append(None)
                continue
                
            best_conf = -1.0
            best_box = None
            for b in results[0].boxes:
                conf = float(b.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = b.xyxy[0].cpu().numpy().astype(int)
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                x1, y1 = max(0, x1), max(0, y1)
                roi = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                if roi.size > 0:
                    roi_std = cv2.resize(roi, (64, 64))
                    rostros_grises.append(roi_std)
                else:
                    rostros_grises.append(None)
            else:
                rostros_grises.append(None)
        except Exception:
            rostros_grises.append(None)

    idx_baseline = 0
    while idx_baseline < len(rostros_grises) and rostros_grises[idx_baseline] is None:
        idx_baseline += 1
        
    if idx_baseline >= len(rostros_grises): return []

    baseline_roi = rostros_grises[idx_baseline]
    diferencias = [0.0 if roi is None else np.sum(cv2.absdiff(baseline_roi, roi)) for roi in rostros_grises]

    apex_idx = int(np.argmax(diferencias))
    half_window = window_size // 2
    start_idx = max(0, apex_idx - half_window)
    end_idx = min(len(frame_paths), apex_idx + half_window + 1)
    
    if end_idx - start_idx < window_size:
        if start_idx == 0: end_idx = min(len(frame_paths), start_idx + window_size)
        elif end_idx == len(frame_paths): start_idx = max(0, end_idx - window_size)

    frames_ventana = frame_paths[start_idx:end_idx]
    faces: list[np.ndarray] = []
    
    for fp in frames_ventana:
        try:
            frame = cv2.imread(fp)
            if frame is None: continue
            
            frame_rgb = pre._convertir_a_rgb(frame)
            frame_resized = cv2.resize(frame_rgb, pre.img_size)
            extr = pre._extraer_landmarks_y_bbox(frame_resized)
            
            if extr is not None:
                landmarks_norm, bbox = extr
                lm_abs = pre._reconstruir_landmarks_absolutos(landmarks_norm, bbox)
                aligned = pre._alinear_rostro_afin(frame_resized, lm_abs, out_size=out_size)
                if aligned is not None:
                    faces.append(aligned)
                    continue

            results = pre.detector_yolo(frame_resized, verbose=False)
            if results and len(results[0].boxes) > 0:
                b = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), b[2], b[3]
                crop = frame_resized[y1:y2, x1:x2]
                if crop.size > 0:
                    faces.append(cv2.resize(crop, (out_size, out_size)))
        except Exception:
            continue

    return faces

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    print("[INFO] Cargando modelo pre-entrenado: ResNet-18")
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((args.aligned_face_size, args.aligned_face_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pre = PreprocesadorFacial(device="cpu" if device.type == "cpu" else "cuda")
    
    out_dir = Path(args.out_dir)
    feat_dir = out_dir / "features_resnet"
    feat_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(args.sequences_csv)
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader: rows.append(r)

    out_csv = out_dir / "features_index_resnet.csv"
    ok_count, fail_count = 0, 0

    print(f"[INFO] Extrayendo características de {len(rows)} secuencias...")
    
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["persona", "camara", "emocion", "feature_path", "num_frames"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(tqdm(rows, desc="Extrayendo ResNet")):
            persona = row.get("persona", f"p{idx:03d}")
            camara = row.get("camara", "cam")
            emocion = row.get("emocion", "unk")
            paths_str = row.get("paths_joined", "")
            if not paths_str:
                fail_count += 1
                continue

            frame_paths = paths_str.split("|")
            faces = extract_faces_with_apex_spotting(pre, frame_paths, args.aligned_face_size, args.window_size)
            
            if len(faces) < 3: 
                fail_count += 1
                continue

            batch_tensors = torch.stack([transform(face) for face in faces]).to(device)

            with torch.no_grad():
                embeddings = resnet(batch_tensors)
            
            emb_np = embeddings.cpu().numpy().astype(np.float32)
            safe_emocion = emocion.replace(" ", "_").replace("/", "_")
            out_name = f"{persona}_{camara}_{safe_emocion}_{idx:06d}.npy"
            out_file = feat_dir / out_name
            np.save(out_file, emb_np)

            writer.writerow({
                "persona": persona, "camara": camara, "emocion": emocion,
                "feature_path": str(out_file), "num_frames": len(faces)
            })
            ok_count += 1

    print(f"\n[HECHO] ok={ok_count} fail={fail_count}")
    print(f"[INFO] Dimensión por frame: 512D")

if __name__ == "__main__":
    main()