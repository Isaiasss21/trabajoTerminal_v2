import argparse
import csv
import json
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
    p.add_argument("--spot_threshold", type=float, default=0.015, help="Umbral de movimiento para spotting")
    p.add_argument("--spot_min_frames", type=int, default=8, help="Longitud mínima de un segmento relevante")
    p.add_argument("--spot_max_frames", type=int, default=32, help="Longitud máxima de un segmento relevante")
    p.add_argument("--save_clips", action="store_true", help="Guardar el clip alineado por secuencia")
    p.add_argument("--save_landmarks", action="store_true", help="Guardar landmarks normalizados por secuencia")
    return p.parse_args()


def process_sequence(
    pre: PreprocesadorFacial,
    resnet: nn.Module,
    transform: T.Compose,
    row: dict,
    idx: int,
    args: argparse.Namespace,
    device: torch.device,
    feat_dir: Path,
    clip_dir: Path,
    lm_dir: Path,
    meta_dir: Path,
) -> tuple[bool, dict | None]:
    persona = row.get("persona", f"p{idx:03d}")
    camara = row.get("camara", "cam")
    emocion = row.get("emocion", "unk")
    paths_str = row.get("paths_joined", "")
    if not paths_str:
        return False, None

    frame_paths = paths_str.split("|")
    bundle = pre.extract_sequence_bundle(
        frame_paths,
        window_size=args.window_size,
        spot_threshold=args.spot_threshold,
        min_frames=args.spot_min_frames,
        max_frames=args.spot_max_frames,
        save_aligned_size=args.aligned_face_size,
        fallback_to_fixed_window=True,
    )
    if bundle is None or len(bundle.faces) < 2:
        return False, None

    payload = build_sequence_payload(
        bundle=bundle,
        persona=persona,
        camara=camara,
        emocion=emocion,
        idx=idx,
        args=args,
        device=device,
        resnet=resnet,
        transform=transform,
        feat_dir=feat_dir,
        clip_dir=clip_dir,
        lm_dir=lm_dir,
        meta_dir=meta_dir,
    )
    return True, payload


def build_sequence_payload(
    bundle,
    persona: str,
    camara: str,
    emocion: str,
    idx: int,
    args: argparse.Namespace,
    device: torch.device,
    resnet: nn.Module,
    transform: T.Compose,
    feat_dir: Path,
    clip_dir: Path,
    lm_dir: Path,
    meta_dir: Path,
) -> dict:
    faces = bundle.faces
    batch_tensors = torch.stack([transform(face) for face in faces]).to(device)
    with torch.no_grad():
        embeddings = resnet(batch_tensors)

    emb_np = embeddings.cpu().numpy().astype(np.float32)
    safe_emocion = emocion.replace(" ", "_").replace("/", "_")
    out_name = f"{persona}_{camara}_{safe_emocion}_{idx:06d}.npy"
    out_file = feat_dir / out_name
    np.save(out_file, emb_np)

    clip_path = None
    landmarks_path = None
    if args.save_clips:
        clip_path = clip_dir / f"{persona}_{camara}_{safe_emocion}_{idx:06d}.npy"
        np.save(clip_path, np.stack(faces, axis=0).astype(np.uint8))
    if args.save_landmarks:
        landmarks_path = lm_dir / f"{persona}_{camara}_{safe_emocion}_{idx:06d}.npy"
        np.save(landmarks_path, np.stack(bundle.landmarks, axis=0).astype(np.float32))

    segment = bundle.segment
    meta_payload = {
        "persona": persona,
        "camara": camara,
        "emocion": emocion,
        "num_frames": len(faces),
        "apex_idx": None if segment is None else segment.apex_idx,
        "frame_start": None if segment is None else segment.start,
        "frame_end": None if segment is None else segment.end,
        "motion_score": None if segment is None else segment.motion_score,
        "clip_path": None if clip_path is None else str(clip_path),
        "landmarks_path": None if landmarks_path is None else str(landmarks_path),
    }
    (meta_dir / f"{persona}_{camara}_{safe_emocion}_{idx:06d}.json").write_text(
        json.dumps(meta_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "persona": persona,
        "camara": camara,
        "emocion": emocion,
        "feature_path": str(out_file),
        "num_frames": len(faces),
        "apex_idx": None if segment is None else segment.apex_idx,
        "frame_start": None if segment is None else segment.start,
        "frame_end": None if segment is None else segment.end,
        "clip_path": None if clip_path is None else str(clip_path),
        "landmarks_path": None if landmarks_path is None else str(landmarks_path),
        "motion_score": None if segment is None else segment.motion_score,
    }


def write_sequence_row(writer: csv.DictWriter, payload: dict | None) -> None:
    if payload is not None:
        writer.writerow(payload)

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
    clip_dir = out_dir / "aligned_clips"
    lm_dir = out_dir / "landmarks"
    meta_dir = out_dir / "metadata"
    feat_dir.mkdir(parents=True, exist_ok=True)
    if args.save_clips:
        clip_dir.mkdir(parents=True, exist_ok=True)
    if args.save_landmarks:
        lm_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(args.sequences_csv)
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader: rows.append(r)

    out_csv = out_dir / "features_index_resnet.csv"
    ok_count, fail_count = 0, 0

    print(f"[INFO] Extrayendo características de {len(rows)} secuencias...")
    
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["persona", "camara", "emocion", "feature_path", "num_frames", "apex_idx", "frame_start", "frame_end", "clip_path", "landmarks_path", "motion_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(tqdm(rows, desc="Extrayendo ResNet")):
            ok, payload = process_sequence(
                pre=pre,
                resnet=resnet,
                transform=transform,
                row=row,
                idx=idx,
                args=args,
                device=device,
                feat_dir=feat_dir,
                clip_dir=clip_dir,
                lm_dir=lm_dir,
                meta_dir=meta_dir,
            )
            if not ok or payload is None:
                fail_count += 1
                continue

            write_sequence_row(writer, payload)
            ok_count += 1

    print(f"\n[HECHO] ok={ok_count} fail={fail_count}")
    print("[INFO] Dimensión por frame: 512D")

if __name__ == "__main__":
    main()