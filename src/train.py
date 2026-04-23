"train.py"
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from .model import EmotionLSTM


class NpySeqDataset(Dataset):
    def __init__(self, feature_paths: List[str], labels: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.feature_paths = feature_paths
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        x = np.load(self.feature_paths[idx]).astype(np.float32)
        x = x.reshape(x.shape[0], -1) 
        y = int(self.labels[idx])
        if self.mean is not None and self.std is not None:
            # normalize per feature dimension
            x = (x - self.mean[None, :]) / (self.std[None, :] + 1e-8)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mean_std(feature_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean and std across all frames in the given .npy files."""
    total = 0
    s = None
    ss = None
    for p in feature_paths:
        a = np.load(p).astype(np.float64)
        n = a.shape[0]
        flat = a.reshape(-1, a.shape[1])
        if s is None:
            s = flat.sum(axis=0)
            ss = (flat ** 2).sum(axis=0)
        else:
            s += flat.sum(axis=0)
            ss += (flat ** 2).sum(axis=0)
        total += n
    if total == 0:
        raise ValueError("No frames found when computing mean/std")
    mean = (s / total).astype(np.float32)
    var = (ss / total) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


def collate_pad(batch):
    """Collate that pads variable-length sequences in the batch.

    Returns:
        xs_padded: Tensor [B, T_max, D]
        ys: Tensor [B]
        lengths: Tensor [B] (original lengths)
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([int(x.shape[0]) for x in xs], dtype=torch.long)
    # pad_sequence expects a list of tensors [T, D]
    xs_padded = pad_sequence(xs, batch_first=True)  # pads with 0.0
    ys = torch.stack(ys)
    return xs_padded, ys, lengths

def split_by_persona(df: pd.DataFrame, train_personas: List[str], val_personas: List[str], test_personas: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["persona"].isin(train_personas)].reset_index(drop=True)
    va = df[df["persona"].isin(val_personas)].reset_index(drop=True)
    te = df[df["persona"].isin(test_personas)].reset_index(drop=True)
    return tr, va, te

def default_person_split(personas: List[str], n_train: int = 10, n_val: int = 3) -> Tuple[List[str], List[str], List[str]]:
    personas = list(personas)
    random.seed(42) 
    random.shuffle(personas) 
    train = personas[:n_train]
    val = personas[n_train:n_train+n_val]
    test = personas[n_train+n_val:]
    
    return train, val, test


class WarmupScheduler:
    """Learning rate scheduler with warmup phase."""
    def __init__(self, optimizer, base_lr: float, warmup_steps: int = 500):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            # linear warmup
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            # decay after warmup: 1 / sqrt(steps_after_warmup)
            steps_after = self.step_num - self.warmup_steps
            lr = self.base_lr * (1.0 / (1.0 + 0.1 * steps_after ** 0.5))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train_lstm(
    df_index: pd.DataFrame,
    out_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.0,
    seed: int = 42,
    bidirectional: bool = False,
    arch: str = "lstm",
    transformer_kwargs: dict | None = None,
    balanced_sampler: bool = False,
    device: str | None = None,
    normalize: bool = True,
    patience: int = 5,
    clip: float = 1.0,
    use_scheduler: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    set_seed(seed)

    # Label encoder
    le = LabelEncoder()
    y = le.fit_transform(df_index["emocion"].astype(str).tolist())

    # Split por persona
    personas = sorted(df_index["persona"].unique().tolist())
    train_p, val_p, test_p = default_person_split(personas, n_train=max(1, int(len(personas)*0.67)), n_val=max(1, int(len(personas)*0.2)))
    df_tr, df_va, df_te = split_by_persona(df_index, train_p, val_p, test_p)

    y_tr = le.transform(df_tr["emocion"].astype(str).tolist())
    y_va = le.transform(df_va["emocion"].astype(str).tolist())
    y_te = le.transform(df_te["emocion"].astype(str).tolist())

    # Optional normalization: compute mean/std from training set and pass to dataset
    mean = std = None
    if normalize:
        print("[INFO] Computing mean/std from training features...")
        mean, std = compute_mean_std(df_tr["feature_path"].tolist())

    train_ds = NpySeqDataset(df_tr["feature_path"].tolist(), y_tr, mean=mean, std=std)
    val_ds   = NpySeqDataset(df_va["feature_path"].tolist(), y_va, mean=mean, std=std)
    test_ds  = NpySeqDataset(df_te["feature_path"].tolist(), y_te, mean=mean, std=std)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_pad)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_pad)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_pad)

    # Optionally use WeightedRandomSampler to balance classes per-batch
    # compute class counts (used by sampler and class weights)
    counts = np.bincount(y_tr).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)

    if balanced_sampler:
        try:
            from torch.utils.data import WeightedRandomSampler
            # compute per-sample weights from y_tr and counts
            sample_weights = (len(y_tr) / counts).astype(np.float64)[y_tr]
            sample_weights = torch.tensor(sample_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False, collate_fn=collate_pad)
            print(f"[INFO] Using WeightedRandomSampler for training (balanced_sampler=True)")
        except Exception as e:
            print(f"[WARN] Could not enable WeightedRandomSampler: {e}")

    # determine device: allow override via `device` parameter
    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # allow strings 'cpu' or 'cuda'
        try:
            device = torch.device(device)
        except Exception:
            # fallback to auto
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    # Infer dims from first sample
    x0, _ = train_ds[0]
    seq_len, input_dim = x0.shape
    num_classes = len(le.classes_)

    # Create model
    arch_l = arch.lower() if isinstance(arch, str) else 'lstm'
    saved_transformer_kwargs = None
    
    if arch_l == 'lstm':
        model = EmotionLSTM(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            num_classes=num_classes, 
            dropout=dropout, 
            bidirectional=bidirectional
        ).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # class weights to mitigate imbalance (inverse freq)
    # counts already computed above
    if counts.shape[0] < num_classes:
        # pad if necessary
        counts = np.pad(counts, (0, num_classes - counts.shape[0]), constant_values=1.0)
    class_weights = (len(y_tr) / counts).astype(np.float32)
    class_weights = class_weights / np.mean(class_weights)  # normalize around 1
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    # Use warmup for Transformer, standard scheduler for LSTM
    warmup_scheduler = None
    plateau_scheduler = None
    if arch_l == 'transformer':
        # Warmup for first few batches
        warmup_steps = max(100, len(train_dl) // 2)  # warmup for ~0.5 epoch
        warmup_scheduler = WarmupScheduler(opt, base_lr=lr, warmup_steps=warmup_steps)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3) if use_scheduler else None
    else:
        # Standard scheduler for LSTM
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2) if use_scheduler else None

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = -1.0
    best_path = ckpt_dir / "best.pt"
    epochs_no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        tr_losses = []
        for xb, yb, lengths in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            # lengths used for packing; keep on cpu for pack_padded_sequence
            opt.zero_grad()
            logits = model.forward_packed(xb, lengths)
            loss = loss_fn(logits, yb)
            loss.backward()
            # gradient clipping
            if clip and clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            
            # Warmup step for Transformer
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            
            tr_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        va_losses = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb, lengths in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model.forward_packed(xb, lengths)
                loss = loss_fn(logits, yb)
                va_losses.append(float(loss.detach().cpu().item()))
                pred = torch.argmax(logits, dim=1)
                y_true.extend(yb.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        va_acc  = float(accuracy_score(y_true, y_pred)) if y_true else 0.0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # save epoch checkpoint
        epoch_path = ckpt_dir / f"epoch_{ep:03d}.pt"
        torch.save({"model_state": model.state_dict(), "epoch": ep, "label_encoder": le.classes_.tolist()}, epoch_path)

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model_state": model.state_dict(), "label_encoder": le.classes_.tolist()}, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # scheduler step on validation loss
        if plateau_scheduler is not None:
            plateau_scheduler.step(va_loss)

        print(f"[EP {ep:03d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} best={best_val:.4f} eps_no_imp={epochs_no_improve}")

        # early stopping
        if epochs_no_improve >= patience and patience > 0:
            print(f"[EARLY STOP] No improvement in {patience} epochs. Stopping at epoch {ep}.")
            break

    # guardar split y metadata
    meta = {
        "device": str(device),
        "classes": le.classes_.tolist(),
        "personas": personas,
        "split": {"train": train_p, "val": val_p, "test": test_p},
        "best_checkpoint": str(best_path),
        "history": history,
        "balanced_sampler": bool(balanced_sampler),
    }
    if saved_transformer_kwargs is not None:
        meta["transformer_kwargs"] = saved_transformer_kwargs
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # evaluar test con best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb, lengths in test_dl:
            xb = xb.to(device)
            logits = model.forward_packed(xb, lengths)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.numpy().tolist())
    test_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    print(f"[TEST] acc={test_acc:.4f}")

    return meta

import torchvision.transforms as T

class NpyFrameDataset(Dataset):
    """
    Dataset para archivos .npy con forma (T, H, W, C) — salida de extract_sequences.py.
    Devuelve tensores (T, C, H, W) float32 normalizados [0,1] listos para ResNet.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, feature_paths: List[str], labels: np.ndarray, img_size: int = 64):
        self.feature_paths = feature_paths
        self.labels = labels
        # Normalización ImageNet estándar para ResNet18
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        self.img_size = img_size

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        # Cargar (T, H, W, C) float32 con valores en [−1,1] aprox (flujo óptico)
        x = np.load(self.feature_paths[idx]).astype(np.float32)  # (T, H, W, C)

        # Mover canales: (T, H, W, C) → (T, C, H, W)
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # (T, 3, H, W)

        # Escalar de [−1,1] a [0,1] para normalización ImageNet
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)

        # Normalización ImageNet por frame
        x = torch.stack([self.normalize(frame) for frame in x])  # (T, 3, H, W)

        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def collate_pad_frames(batch):
    """
    Collate para NpyFrameDataset: rellena secuencias de longitud variable.

    Returns:
        xs_padded : (B, T_max, C, H, W) float32
        ys        : (B,) long
        lengths   : (B,) long
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    # Dimensiones del frame
    _, C, H, W = xs[0].shape
    T_max = int(lengths.max().item())
    B = len(xs)

    padded = torch.zeros(B, T_max, C, H, W, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x

    ys = torch.stack(ys)
    return padded, ys, lengths


def train_frame_lstm(
    df_index: pd.DataFrame,
    out_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    frame_embedding_dim: int = 256,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.5,
    seed: int = 42,
    bidirectional: bool = True,
    balanced_sampler: bool = False,
    device: str | None = None,
    patience: int = 7,
    clip: float = 1.0,
    use_scheduler: bool = True,
    baseline_subtract: bool = True,
    baseline_frames: int = 3,
    freeze_encoder_epochs: int = 0,
    affectnet_weights_path: str | None = None,
):
    """
    Entrena EmotionFrameLSTM (ResNet18 + BiLSTM + Atención Temporal).

    Parámetros clave vs train_lstm:
      - frame_embedding_dim : dimensión del embedding por frame (salida ResNet18 → proj)
      - freeze_encoder_epochs: épocas iniciales con ResNet18 congelado (transfer learning)
      - affectnet_weights_path: checkpoint AffectNet/FER opcional para el encoder
    """
    from src.model import EmotionFrameLSTM  # import local para evitar circularidad

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    set_seed(seed)

    # ── Label encoder ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df_index["emocion"].astype(str).tolist())

    # ── Split por persona ──────────────────────────────────────────────────────
    personas = sorted(df_index["persona"].unique().tolist())
    n_train  = max(1, int(len(personas) * 0.67))
    n_val    = max(1, int(len(personas) * 0.20))
    train_p, val_p, test_p = default_person_split(personas, n_train=n_train, n_val=n_val)
    df_tr, df_va, df_te = split_by_persona(df_index, train_p, val_p, test_p)

    print(f"[INFO] Split — train: {len(df_tr)} | val: {len(df_va)} | test: {len(df_te)}")
    print(f"[INFO] Personas train ({len(train_p)}): {train_p}")
    print(f"[INFO] Personas val   ({len(val_p)}):   {val_p}")
    print(f"[INFO] Personas test  ({len(test_p)}):  {test_p}")

    y_tr = le.transform(df_tr["emocion"].astype(str).tolist())
    y_va = le.transform(df_va["emocion"].astype(str).tolist())
    y_te = le.transform(df_te["emocion"].astype(str).tolist())

    # ── Datasets y DataLoaders ────────────────────────────────────────────────
    train_ds = NpyFrameDataset(df_tr["feature_path"].tolist(), y_tr)
    val_ds   = NpyFrameDataset(df_va["feature_path"].tolist(), y_va)
    test_ds  = NpyFrameDataset(df_te["feature_path"].tolist(), y_te)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=False, collate_fn=collate_pad_frames)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          drop_last=False, collate_fn=collate_pad_frames)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          drop_last=False, collate_fn=collate_pad_frames)

    # ── Balanceo de clases ────────────────────────────────────────────────────
    counts = np.bincount(y_tr).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)

    if balanced_sampler:
        try:
            from torch.utils.data import WeightedRandomSampler
            sample_weights = (len(y_tr) / counts).astype(np.float64)[y_tr]
            sampler = WeightedRandomSampler(
                torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights), replacement=True
            )
            train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  drop_last=False, collate_fn=collate_pad_frames)
            print("[INFO] Usando WeightedRandomSampler (balanced_sampler=True)")
        except Exception as e:
            print(f"[WARN] No se pudo activar WeightedRandomSampler: {e}")

    # ── Dispositivo ───────────────────────────────────────────────────────────
    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    num_classes = len(le.classes_)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = EmotionFrameLSTM(
        frame_embedding_dim  = frame_embedding_dim,
        hidden_dim           = hidden_dim,
        num_layers           = num_layers,
        num_classes          = num_classes,
        dropout              = dropout,
        bidirectional        = bidirectional,
        baseline_subtract    = baseline_subtract,
        baseline_frames      = baseline_frames,
        imagenet_pretrained  = True,
        affectnet_weights_path = affectnet_weights_path,
    ).to(device)

    # ── Congelar encoder (transfer learning warmup) ───────────────────────────
    def _set_encoder_grad(requires_grad: bool):
        for p in model.frame_encoder.parameters():
            p.requires_grad = requires_grad

    if freeze_encoder_epochs > 0:
        _set_encoder_grad(False)
        print(f"[INFO] ResNet18 congelado por {freeze_encoder_epochs} épocas.")

    # ── Optimizador y loss ────────────────────────────────────────────────────
    # Usar lr más bajo para el encoder pre-entrenado
    encoder_params = list(model.frame_encoder.parameters())
    other_params   = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]

    opt = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1},   # encoder: lr/10
        {"params": other_params,   "lr": lr},          # LSTM + head: lr completo
    ], weight_decay=weight_decay)

    # Pesos de clase para CrossEntropy
    num_cls = len(le.classes_)
    if counts.shape[0] < num_cls:
        counts = np.pad(counts, (0, num_cls - counts.shape[0]), constant_values=1.0)
    class_weights = (len(y_tr) / counts).astype(np.float32)
    class_weights = class_weights / np.mean(class_weights)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    plateau_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        if use_scheduler else None
    )

    # ── Loop de entrenamiento ─────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val       = -1.0
    best_path      = ckpt_dir / "best.pt"
    epochs_no_imp  = 0

    for ep in range(1, epochs + 1):

        # Descongelar encoder después de freeze_encoder_epochs
        if freeze_encoder_epochs > 0 and ep == freeze_encoder_epochs + 1:
            _set_encoder_grad(True)
            print(f"[EP {ep:03d}] ResNet18 descongelado — fine-tuning completo.")

        model.train()
        tr_losses = []
        for xb, yb, lengths in train_dl:
            xb = xb.to(device)    # (B, T, C, H, W)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model.forward_packed(xb, lengths)
            loss   = loss_fn(logits, yb)
            loss.backward()
            if clip and clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            tr_losses.append(float(loss.detach().cpu()))

        model.eval()
        va_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for xb, yb, lengths in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model.forward_packed(xb, lengths)
                va_losses.append(float(loss_fn(logits, yb).detach().cpu()))
                pred = torch.argmax(logits, dim=1)
                y_true.extend(yb.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        va_acc  = float(accuracy_score(y_true, y_pred)) if y_true else 0.0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Checkpoint por época
        torch.save({
            "model_state":    model.state_dict(),
            "epoch":          ep,
            "label_encoder":  le.classes_.tolist(),
        }, ckpt_dir / f"epoch_{ep:03d}.pt")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "model_state":   model.state_dict(),
                "label_encoder": le.classes_.tolist(),
            }, best_path)
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        if plateau_scheduler is not None:
            plateau_scheduler.step(va_loss)

        print(
            f"[EP {ep:03d}] train={tr_loss:.4f}  val={va_loss:.4f}  "
            f"val_acc={va_acc:.4f}  best={best_val:.4f}  no_imp={epochs_no_imp}"
        )

        if patience > 0 and epochs_no_imp >= patience:
            print(f"[EARLY STOP] Sin mejora en {patience} épocas. Parando en época {ep}.")
            break

    # ── Evaluación final en test ──────────────────────────────────────────────
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb, lengths in test_dl:
            xb = xb.to(device)
            logits = model.forward_packed(xb, lengths)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            y_true.extend(yb.numpy().tolist())

    test_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    print(f"[TEST] acc={test_acc:.4f}")

    # ── Guardar metadata ──────────────────────────────────────────────────────
    meta = {
        "model":       "EmotionFrameLSTM",
        "device":      str(device),
        "classes":     le.classes_.tolist(),
        "personas":    personas,
        "split":       {"train": train_p, "val": val_p, "test": test_p},
        "best_checkpoint": str(best_path),
        "test_acc":    test_acc,
        "history":     history,
        "balanced_sampler": bool(balanced_sampler),
        "hparams": {
            "frame_embedding_dim": frame_embedding_dim,
            "hidden_dim":          hidden_dim,
            "num_layers":          num_layers,
            "dropout":             dropout,
            "bidirectional":       bidirectional,
            "baseline_subtract":   baseline_subtract,
            "baseline_frames":     baseline_frames,
            "freeze_encoder_epochs": freeze_encoder_epochs,
        },
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return meta