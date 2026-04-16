import argparse
from pathlib import Path
import sys

import pandas as pd

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.dataio import load_feature_index
from src.train import train_lstm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning with modular LOSO engine")
    parser.add_argument("--features_index", required=True)
    parser.add_argument("--pretrained_weights", required=True)
    parser.add_argument("--out_dir", default="outputs/modelo_webcam_robusto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--loss_type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--validation_mode", choices=["split", "loso"], default="loso")
    parser.add_argument("--input_mode", choices=["auto", "feature", "multimodal"], default="auto")
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--test_subject", default=None)
    return parser.parse_args()


def normalize_emotion(value: str) -> str:
    emotion = str(value).strip().lower()
    if emotion in ["felicidad", "positive", "happiness"]:
        return "Positivo"
    if emotion in ["asco", "tristeza", "miedo", "enojo", "negativo", "negative", "disgust", "sadness", "fear", "anger"]:
        return "Negativo"
    if emotion in ["sorpresa", "surprise"]:
        return "Sorpresa"
    if emotion in ["neutral", "non_micro"]:
        return "Neutral"
    return "Desconocida"


def main() -> None:
    args = parse_args()
    df = load_feature_index(args.features_index)
    if "emocion" in df.columns:
        df["label"] = df["emocion"].apply(normalize_emotion)
        df = df[df["label"] != "Desconocida"].copy()
    if "persona" in df.columns and "subject_id" not in df.columns:
        df["subject_id"] = df["persona"].astype(str)

    if args.input_mode == "auto":
        has_multimodal = "clip_path" in df.columns and "landmarks_path" in df.columns and df["clip_path"].notna().any() and df["landmarks_path"].notna().any()
        input_mode = "multimodal" if has_multimodal else "feature"
    else:
        input_mode = args.input_mode

    train_lstm(
        df,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        validation_mode=args.validation_mode,
        input_mode=input_mode,
        subject_col=args.subject_col,
        label_col=args.label_col,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        test_subject=args.test_subject,
        pretrained_weights=args.pretrained_weights,
    )


if __name__ == "__main__":
    main()