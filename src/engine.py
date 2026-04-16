from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

from .loso_manager import build_split_bundle, resolve_series
from .model import SwinLandmarkFusionModel
from .model_transformer import get_sequence_model
from .train_utils import (
    TrainConfig,
    build_datasets,
    build_loss_function,
    compute_class_weights,
    compute_mean_std,
    set_seed,
    WarmupScheduler,
)


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device_arg)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_lstm_model(config: TrainConfig, input_dim: int | None, num_classes: int) -> torch.nn.Module:
    return get_sequence_model(
        "lstm",
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=num_classes,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    )


def _build_transformer_model(config: TrainConfig, input_dim: int | None, num_classes: int) -> torch.nn.Module:
    tkwargs = {} if config.transformer_kwargs is None else dict(config.transformer_kwargs)
    tkwargs.setdefault("d_model", config.hidden_dim)
    tkwargs.setdefault("num_layers", config.num_layers)
    tkwargs["num_classes"] = num_classes
    return get_sequence_model("transformer", input_dim=input_dim, **tkwargs)


def _maybe_load_pretrained_weights(model: torch.nn.Module, pretrained_weights: str | None) -> torch.nn.Module:
    if not pretrained_weights:
        return model

    ckpt = torch.load(pretrained_weights, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    target = model.state_dict()
    matched = {}
    for key, value in state.items():
        if key in target and hasattr(value, "shape") and target[key].shape == value.shape:
            matched[key] = value
    model.load_state_dict(matched, strict=False)
    return model


def _build_model(config: TrainConfig, input_dim: int | None, num_classes: int) -> torch.nn.Module:
    device = _resolve_device(config.device)
    arch_l = config.arch.lower() if isinstance(config.arch, str) else "lstm"

    if config.input_mode == "multimodal":
        return SwinLandmarkFusionModel(num_classes=num_classes).to(device)

    if arch_l == "lstm":
        model = _build_lstm_model(config, input_dim, num_classes)
    elif arch_l == "transformer":
        model = _build_transformer_model(config, input_dim, num_classes)
    else:
        raise ValueError(f"Unknown arch: {config.arch}")

    model = model.to(device)
    return _maybe_load_pretrained_weights(model, config.pretrained_weights)


def _unpack_batch(batch, input_mode: str, device: torch.device):
    if input_mode == "multimodal":
        xb, lm, yb, lengths = batch
        return xb.to(device), lm.to(device), yb.to(device), lengths
    xb, yb, lengths = batch
    return xb.to(device), None, yb.to(device), lengths


def _encode_labels(df_index: pd.DataFrame, config: TrainConfig):
    label_series = resolve_series(df_index, config.label_col, "emocion")
    le = LabelEncoder()
    _ = le.fit_transform(label_series.tolist())
    return le


def _prepare_splits_and_targets(df_index: pd.DataFrame, config: TrainConfig, le: LabelEncoder):
    split = build_split_bundle(
        df_index,
        validation_mode=config.validation_mode,
        subject_col=config.subject_col,
        test_subject=config.test_subject,
        val_fraction=0.2,
    )
    y_tr = le.transform(resolve_series(split.train_df, config.label_col, "emocion").tolist())
    y_va = le.transform(resolve_series(split.val_df, config.label_col, "emocion").tolist())
    y_te = le.transform(resolve_series(split.test_df, config.label_col, "emocion").tolist())
    return split, y_tr, y_va, y_te


def _build_dataloaders(split, y_tr, y_va, y_te, config: TrainConfig, mean, std):
    train_ds, val_ds, test_ds, collate_fn = build_datasets(
        split.train_df,
        split.val_df,
        split.test_df,
        y_tr,
        y_va,
        y_te,
        input_mode=config.input_mode,
        mean=mean,
        std=std,
    )

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=0)
    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, collate_fn


def _build_training_setup(model, config: TrainConfig, train_dl, y_tr, device: torch.device):
    counts = np.bincount(y_tr).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    class_weights = compute_class_weights(y_tr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = build_loss_function(config.loss_type, class_weights, device, focal_gamma=config.focal_gamma)

    if config.arch.lower() == "transformer":
        warmup_steps = max(100, len(train_dl) // 2)
        warmup_scheduler = WarmupScheduler(optimizer, base_lr=config.lr, warmup_steps=warmup_steps)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3) if config.use_scheduler else None
    else:
        warmup_scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2) if config.use_scheduler else None

    return optimizer, loss_fn, warmup_scheduler, plateau_scheduler


@dataclass
class TrainingState:
    out_dir: Path
    ckpt_dir: Path
    device: torch.device
    le: LabelEncoder
    split: object
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    warmup_scheduler: WarmupScheduler | None
    plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None
    input_mode: str


def _prepare_training_state(df_index: pd.DataFrame, config: TrainConfig) -> TrainingState:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    device = _resolve_device(config.device)
    le = _encode_labels(df_index, config)
    split, y_tr, y_va, y_te = _prepare_splits_and_targets(df_index, config, le)

    mean = std = None
    if config.normalize and config.input_mode != "multimodal":
        print("[INFO] Computing mean/std from training features...")
        mean, std = compute_mean_std(split.train_df["feature_path"].tolist())

    train_ds, _, _, train_dl, val_dl, test_dl, collate_fn = _build_dataloaders(split, y_tr, y_va, y_te, config, mean, std)

    if config.balanced_sampler:
        try:
            counts = np.bincount(y_tr).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            sample_weights = (len(y_tr) / counts).astype(np.float64)[y_tr]
            sample_weights = torch.tensor(sample_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_dl = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, drop_last=False, collate_fn=collate_fn, num_workers=0)
            print("[INFO] Using WeightedRandomSampler for training (balanced_sampler=True)")
        except Exception as e:
            print(f"[WARN] Could not enable WeightedRandomSampler: {e}")

    if config.input_mode == "multimodal":
        x0, _, _ = train_ds[0]
        input_dim = None
        if x0.ndim != 4:
            raise ValueError(f"Entrada multimodal inválida: se esperaba [T, C, H, W], recibido {tuple(x0.shape)}")
        h, w = int(x0.shape[-2]), int(x0.shape[-1])
        if (h % 32) != 0 or (w % 32) != 0:
            print(f"[WARN][SWIN_SHAPE] Primer clip con shape ({h}, {w}) no múltiplo de 32.")
        else:
            print(f"[CHECK][SWIN_SHAPE] Primer clip OK para Swin: ({h}, {w}).")
    else:
        x0, _ = train_ds[0]
        _, input_dim = x0.shape

    num_classes = len(le.classes_)
    model = _build_model(config, input_dim=input_dim, num_classes=num_classes)
    optimizer, loss_fn, warmup_scheduler, plateau_scheduler = _build_training_setup(model, config, train_dl, y_tr, device)

    return TrainingState(
        out_dir=out_dir,
        ckpt_dir=ckpt_dir,
        device=device,
        le=le,
        split=split,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        warmup_scheduler=warmup_scheduler,
        plateau_scheduler=plateau_scheduler,
        input_mode=config.input_mode,
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device: torch.device, input_mode: str, clip: float, warmup_scheduler=None):
    model.train()
    tr_losses = []
    for batch_idx, batch in enumerate(loader):
        xb, lm, yb, lengths = _unpack_batch(batch, input_mode, device)
        _log_swin_batch_shape(xb, input_mode, batch_idx)
        optimizer.zero_grad()
        logits = _forward_logits(model, xb, lm, lengths, input_mode)
        loss = loss_fn(logits, yb)
        loss.backward()
        if clip and clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        tr_losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(tr_losses)) if tr_losses else float("nan")


def evaluate(model, loader, loss_fn, device: torch.device, input_mode: str):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            xb, lm, yb, lengths = _unpack_batch(batch, input_mode, device)
            logits = model(xb, lm, lengths) if input_mode == "multimodal" else model.forward_packed(xb, lengths)
            loss = loss_fn(logits, yb)
            losses.append(float(loss.detach().cpu().item()))
            pred = torch.argmax(logits, dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _log_epoch_diagnostics(ep: int, val_metrics: dict, config: TrainConfig, history: dict) -> None:
    acc = float(val_metrics["acc"])
    f1_macro = float(val_metrics["f1_macro"])
    gap = acc - f1_macro
    print(f"[CHECK][F1_VS_ACC] ep={ep:03d} acc={acc:.4f} f1={f1_macro:.4f} gap={gap:.4f}")

    if gap > 0.25:
        print("[WARN][F1_VS_ACC] Brecha alta Acc-F1: posible desbalance de clases.")

    if ep <= 2 and acc > 0.90:
        print("[WARN][ALIGNMENT_LEAK] Convergencia muy rápida; revisar bordes/alineación para posibles atajos por sujeto.")

    if config.loss_type.lower() == "focal" and ep >= 10 and len(history["val_f1"]) >= 10:
        f1_ep10 = float(history["val_f1"][9])
        delta_vs_ep10 = f1_macro - f1_ep10
        print(f"[CHECK][FOCAL_TREND] ep={ep:03d} f1_delta_vs_ep10={delta_vs_ep10:+.4f}")


def _log_swin_batch_shape(xb: torch.Tensor, input_mode: str, batch_idx: int) -> None:
    if input_mode != "multimodal" or batch_idx != 0:
        return

    h, w = int(xb.shape[-2]), int(xb.shape[-1])
    if (h % 32) == 0 and (w % 32) == 0:
        print(f"[CHECK][SWIN_BATCH] batch_shape={tuple(xb.shape)} (H,W)=({h},{w}) múltiplos de 32")
    else:
        print(f"[WARN][SWIN_BATCH] batch_shape={tuple(xb.shape)} (H,W)=({h},{w}) no múltiplos de 32")


def _forward_logits(model, xb: torch.Tensor, lm: torch.Tensor | None, lengths: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "multimodal":
        return model(xb, lm, lengths)
    return model.forward_packed(xb, lengths)


def run_training(df_index: pd.DataFrame, config: TrainConfig):
    set_seed(config.seed)
    state = _prepare_training_state(df_index, config)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val = -1.0
    best_path = state.ckpt_dir / "best.pt"
    epochs_no_improve = 0

    for ep in range(1, config.epochs + 1):
        tr_loss = train_one_epoch(state.model, state.train_dl, state.optimizer, state.loss_fn, state.device, state.input_mode, config.clip, warmup_scheduler=state.warmup_scheduler)
        val_metrics = evaluate(state.model, state.val_dl, state.loss_fn, state.device, state.input_mode)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1_macro"])

        epoch_path = state.ckpt_dir / f"epoch_{ep:03d}.pt"
        torch.save({"model_state": state.model.state_dict(), "epoch": ep, "label_encoder": state.le.classes_.tolist()}, epoch_path)

        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            torch.save({"model_state": state.model.state_dict(), "label_encoder": state.le.classes_.tolist()}, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if state.plateau_scheduler is not None:
            state.plateau_scheduler.step(val_metrics["loss"])

        print(
            f"[EP {ep:03d}] train_loss={tr_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1_macro']:.4f} "
            f"best={best_val:.4f} eps_no_imp={epochs_no_improve}"
        )
        _log_epoch_diagnostics(ep, val_metrics, config, history)

        if epochs_no_improve >= config.patience and config.patience > 0:
            print(f"[EARLY STOP] No improvement in {config.patience} epochs. Stopping at epoch {ep}.")
            break

    meta = {
        "device": str(state.device),
        "classes": state.le.classes_.tolist(),
        "personas": state.split.train_subjects + state.split.val_subjects + state.split.test_subjects,
        "split": {
            "train": state.split.train_subjects,
            "val": state.split.val_subjects,
            "test": state.split.test_subjects,
        },
        "best_checkpoint": str(best_path),
        "history": history,
        "balanced_sampler": bool(config.balanced_sampler),
        "validation_mode": config.validation_mode,
        "input_mode": config.input_mode,
    }
    if config.transformer_kwargs is not None:
        meta["transformer_kwargs"] = config.transformer_kwargs
    (state.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    ckpt = torch.load(best_path, map_location=state.device)
    state.model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(state.model, state.test_dl, state.loss_fn, state.device, state.input_mode)
    print(f"[TEST] acc={test_metrics['acc']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    return meta