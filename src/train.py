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
from .model_transformer import get_sequence_model


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

    # Create model via factory so we can switch architectures easily
    arch_l = arch.lower() if isinstance(arch, str) else 'lstm'
    saved_transformer_kwargs = None
    if arch_l == 'lstm':
        model = get_sequence_model('lstm', input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes, dropout=dropout, bidirectional=bidirectional).to(device)
    elif arch_l == 'transformer':
        # Prepare transformer kwargs: allow CLI to override d_model/nhead/dim_feedforward/num_layers
        tkwargs = {} if transformer_kwargs is None else dict(transformer_kwargs)
        # default mapping from --hidden to d_model if not provided
        if 'd_model' not in tkwargs or tkwargs.get('d_model') is None:
            tkwargs['d_model'] = hidden_dim
        # default num_layers mapping
        if 'num_layers' not in tkwargs or tkwargs.get('num_layers') is None:
            tkwargs['num_layers'] = num_layers
        # ensure num_classes passed
        tkwargs['num_classes'] = num_classes
        model = get_sequence_model('transformer', input_dim=input_dim, **tkwargs).to(device)
        saved_transformer_kwargs = tkwargs
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
