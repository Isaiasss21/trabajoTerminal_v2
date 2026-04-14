import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.dataio import load_feature_index
from src.model import EmotionLSTM
from src.train import NpySeqDataset, collate_pad, compute_mean_std, default_person_split, split_by_persona

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Base BiLSTM on ResNet embeddings")
    p.add_argument("--features_index", required=True, help="CSV maestro")
    p.add_argument("--out_dir", default="outputs/exp_base", help="Output folder")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()

def agrupar_emociones(emocion_original: str) -> str:
    emocion = str(emocion_original).strip().lower()
    if emocion in ['felicidad', 'positive', 'happiness']: return 'Positivo'
    elif emocion in ['asco', 'tristeza', 'miedo', 'enojo', 'negativo', 'negative', 'disgust', 'sadness', 'fear', 'anger']: return 'Negativo'
    elif emocion in ['sorpresa', 'surprise']: return 'Sorpresa'
    elif emocion in ['neutral', 'non_micro']: return 'Neutral'
    else: return 'Desconocida'

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    args = parse_args()
    set_seed(42)
    device = torch.device("cpu") # Forzado por incompatibilidad de RTX 5050
    print(f"[INFO] Device: {device}")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_index(args.features_index)
    df["emocion"] = df["emocion"].apply(agrupar_emociones)
    df = df[df["emocion"] != "Desconocida"].copy()

    le = LabelEncoder()
    _ = le.fit_transform(df["emocion"].astype(str).tolist())
    classes = le.classes_.tolist()

    personas = sorted(df["persona"].unique().tolist())
    train_p, val_p, test_p = default_person_split(personas, n_train=max(1, int(len(personas) * 0.67)), n_val=max(1, int(len(personas) * 0.20)))
    df_tr, df_va, df_te = split_by_persona(df, train_p, val_p, test_p)

    y_tr = le.transform(df_tr["emocion"].astype(str).tolist())
    y_va = le.transform(df_va["emocion"].astype(str).tolist())
    
    mean, std = compute_mean_std(df_tr["feature_path"].tolist())
    np.save(out_dir / "train_mean.npy", mean)
    np.save(out_dir / "train_std.npy", std)

    train_ds = NpySeqDataset(df_tr["feature_path"].tolist(), y_tr, mean=mean, std=std)
    val_ds = NpySeqDataset(df_va["feature_path"].tolist(), y_va, mean=mean, std=std)

    counts = np.bincount(y_tr).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    sample_weights = (len(y_tr) / counts).astype(np.float64)[y_tr]
    sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, collate_fn=collate_pad)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pad)

    input_dim = train_ds[0][0].shape[1]
    
    model = EmotionLSTM(input_dim=input_dim, hidden_dim=128, num_layers=1, num_classes=len(classes), dropout=0.5, bidirectional=False).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = -1.0
    best_path = ckpt_dir / "best_model.pth"

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb, lengths in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model.forward_packed(xb, lengths)
            loss_fn(logits, yb).backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb, lengths in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model.forward_packed(xb, lengths)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

        va_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        scheduler.step(va_f1)

        print(f"[EP {ep:03d}] val_f1_macro={va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({"model_state": model.state_dict(), "label_encoder": classes}, best_path)

if __name__ == "__main__":
    main()