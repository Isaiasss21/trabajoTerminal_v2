import argparse
import json
from pathlib import Path
import sys
import os

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.train import NpySeqDataset, compute_mean_std, collate_pad, split_by_persona, default_person_split, set_seed
from src.model import EmotionLSTM

def agrupar_emociones(emocion_original: str) -> str:
    emocion = str(emocion_original).strip().lower()
    if emocion in ['felicidad', 'positive', 'happiness']: return 'Positivo'
    elif emocion in ['asco', 'tristeza', 'miedo', 'enojo', 'negativo', 'negative', 'disgust', 'sadness', 'fear', 'anger']: return 'Negativo'
    elif emocion in ['sorpresa', 'surprise']: return 'Sorpresa'
    elif emocion in ['neutral', 'non_micro']: return 'Neutral'
    else: return 'Desconocida'

def main():
    parser = argparse.ArgumentParser(description="Transfer Learning: Fine-Tuning robusto")
    parser.add_argument("--features_index", required=True)
    parser.add_argument("--pretrained_weights", required=True)
    parser.add_argument("--out_dir", default="outputs/modelo_webcam_robusto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001) 
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    set_seed(42)
    device = torch.device("cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    df = pd.read_csv(args.features_index)
    df["emocion"] = df["emocion"].apply(agrupar_emociones)
    df = df[df["emocion"] != "Desconocida"].copy()

    le = LabelEncoder()
    _ = le.fit_transform(df["emocion"].astype(str).tolist())
    num_classes_nuevas = len(le.classes_)

    personas = sorted(df["persona"].unique().tolist())
    train_p, val_p, test_p = default_person_split(personas, n_train=max(1, int(len(personas)*0.70)), n_val=max(1, int(len(personas)*0.15)))
    df_tr, df_va, df_te = split_by_persona(df, train_p, val_p, test_p)

    y_tr = le.transform(df_tr["emocion"].astype(str).tolist())
    y_va = le.transform(df_va["emocion"].astype(str).tolist())
    y_te = le.transform(df_te["emocion"].astype(str).tolist())

    mean, std = compute_mean_std(df_tr["feature_path"].tolist())
    np.save(out_dir / "train_mean.npy", mean)
    np.save(out_dir / "train_std.npy", std)

    train_ds = NpySeqDataset(df_tr["feature_path"].tolist(), y_tr, mean=mean, std=std)
    val_ds   = NpySeqDataset(df_va["feature_path"].tolist(), y_va, mean=mean, std=std)
    test_ds  = NpySeqDataset(df_te["feature_path"].tolist(), y_te, mean=mean, std=std)

    counts = np.bincount(y_tr, minlength=num_classes_nuevas).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    sample_weights = (len(y_tr) / counts)[y_tr]
    sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, collate_fn=collate_pad, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pad, num_workers=0)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pad, num_workers=0)

    input_dim = train_ds[0][0].shape[1]
    
    # DROPOUT ALTO para evitar overfitting
    model = EmotionLSTM(input_dim=input_dim, hidden_dim=128, num_layers=1, num_classes=4, dropout=0.65, bidirectional=False).to(device)
    
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)

    for param in model.parameters(): param.requires_grad = True
    in_features = model.fc2.in_features
    model.fc2 = nn.Linear(in_features, num_classes_nuevas).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loss_fn = nn.CrossEntropyLoss()
    val_loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_path = ckpt_dir / "best_finetuned.pt"
    
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb, lengths in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model.forward_packed(xb, lengths)
            loss = train_loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for xb, yb, lengths in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model.forward_packed(xb, lengths)
                loss = val_loss_fn(logits, yb)
                va_losses.append(loss.item())
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

        tr_loss = np.mean(tr_losses)
        va_loss = np.mean(va_losses)
        va_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        scheduler.step(va_f1)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "label_encoder": le.classes_.tolist()}, best_path)
        else:
            epochs_no_improve += 1

        print(f"[EP {ep:03d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_f1={va_f1:.4f} best_f1={best_val_f1:.4f}")

        if epochs_no_improve >= 10:
            print("\n[EARLY STOP] Deteniendo entrenamiento.")
            break

    print("\n[INFO] Evaluando Test...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb, lengths in test_dl:
            xb = xb.to(device)
            logits = model.forward_packed(xb, lengths)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_true.extend(yb.numpy())
            
    print(f"\n[TEST FINAL] Accuracy: {accuracy_score(y_true, y_pred):.4f} | F1-Macro: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_.tolist(), digits=4))

if __name__ == "__main__":
    main()