
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import time

# Utilidad para guardar la matriz de confusión como imagen

def save_confusion_matrix(cm, class_names, out_path, title=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix_normalized(cm, class_names, out_path, title=None):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm / row_sums
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _analyze_confusions(cm: np.ndarray, class_names: list[str], top_k: int = 6) -> list[str]:
    lines: list[str] = []
    pairs: list[tuple[float, str]] = []
    for i in range(cm.shape[0]):
        row_total = float(cm[i, :].sum())
        if row_total <= 0:
            continue
        for j in range(cm.shape[1]):
            if i == j:
                continue
            rate = float(cm[i, j]) / row_total
            if cm[i, j] > 0:
                pairs.append(
                    (
                        rate,
                        f"Se observa una tasa de confusion del {rate * 100:.2f}% entre la clase '{class_names[i]}' y '{class_names[j]}' "
                        f"({int(cm[i, j])}/{int(row_total)} muestras de '{class_names[i]}').",
                    )
                )
    pairs.sort(key=lambda x: x[0], reverse=True)
    for _, txt in pairs[:top_k]:
        lines.append(txt)
    if not lines:
        lines.append("No se observaron confusiones fuera de la diagonal para este split.")
    return lines


def _plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_f1_present: list[float],
    best_epoch: int,
    early_stop_epoch: int | None,
    fold_tag: str,
) -> tuple[str, str]:
    n = len(train_losses)
    if n == 0:
        return "", ""
    # Plot up to the early-stop epoch if available, otherwise up to the best epoch
    if early_stop_epoch is not None:
        max_e = min(early_stop_epoch, n)
    else:
        max_e = min(best_epoch, n)
    epochs = np.arange(1, max_e + 1)

    loss_path = f"learning_curve_loss_{fold_tag}.png"
    metric_path = f"learning_curve_metric_{fold_tag}.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_losses[:max_e], label="Train Loss", color="#1f77b4")
    plt.plot(epochs, val_losses[:max_e], label="Val Loss", color="#d62728")
    plt.axvline(best_epoch, color="green", linestyle="--", linewidth=1.2, label=f"Best epoch={best_epoch}")
    if early_stop_epoch is not None and early_stop_epoch <= max_e:
        plt.axvline(early_stop_epoch, color="purple", linestyle=":", linewidth=1.4, label=f"Early stop={early_stop_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, val_f1_present[:max_e], label="macro_f1_present", color="#2ca02c")
    plt.axvline(best_epoch, color="green", linestyle="--", linewidth=1.2, label=f"Best epoch={best_epoch}")
    if early_stop_epoch is not None and early_stop_epoch <= max_e:
        plt.axvline(early_stop_epoch, color="purple", linestyle=":", linewidth=1.4, label=f"Early stop={early_stop_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("macro_f1_present")
    plt.title("Validation macro_f1_present")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_path)
    plt.close()

    return loss_path, metric_path


def _estimate_tta_latency(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    tta_passes: int = 30,
    repeats: int = 3,
) -> tuple[float, float]:
    # Returns mean and std latency (seconds) for one sample inference with TTA.
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return 0.0, 0.0
    x, y = batch[:2]
    x1 = x[:1]
    y1 = y[:1]
    ds = TensorDataset(x1, y1)
    one_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    times: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        evaluate_with_tta(
            model,
            one_loader,
            criterion,
            device,
            num_classes,
            tta_passes=max(1, int(tta_passes)),
        )
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _save_temporal_case_study(
    model: nn.Module,
    val_ds,
    label_names: list[str],
    device: torch.device,
    out_path: str,
) -> str:
    # Try to find a correctly predicted 'enojo' sample; fallback to first sample.
    target_idx = label_names.index("enojo") if "enojo" in label_names else None
    chosen = None
    max_scan = min(len(val_ds), 128)
    for i in range(max_scan):
        item = val_ds[i]
        frames = item[0]
        label = int(item[1])
        logits = model(frames.unsqueeze(0).to(device))
        pred = int(torch.argmax(logits, dim=1).item())
        if target_idx is not None and label == target_idx and pred == target_idx:
            chosen = (i, frames, label, pred)
            break
        if chosen is None and pred == label:
            chosen = (i, frames, label, pred)
    if chosen is None:
        item = val_ds[0]
        chosen = (0, item[0], int(item[1]), int(item[1]))

    i_sel, frames, label, pred = chosen
    T = frames.shape[0]
    pick = sorted({0, max(0, T // 4), max(0, T // 2), max(0, (3 * T) // 4), T - 1})

    plt.figure(figsize=(12, 2.8))
    for k, t in enumerate(pick, start=1):
        flow = frames[t].detach().cpu().numpy()  # (3,H,W)
        mag = np.clip((flow[2] + 1.0) / 2.0, 0.0, 1.0)
        ax = plt.subplot(1, len(pick), k)
        ax.imshow(mag, cmap="inferno")
        ax.set_title(f"t={t}")
        ax.axis("off")
    plt.suptitle(
        f"Temporal flow case | idx={i_sel} | true={label_names[label]} | pred={label_names[pred]}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return (
        f"Caso temporal guardado en '{out_path}' usando muestra idx={i_sel}, "
        f"true='{label_names[label]}' y pred='{label_names[pred]}'."
    )


def write_detailed_training_report(
    fold_tag: str,
    class_names: list[str],
    cm: np.ndarray,
    train_losses: list[float],
    val_losses: list[float],
    val_f1_present: list[float],
    epoch_times_sec: list[float],
    best_epoch: int,
    early_stop_epoch: int | None,
    args: argparse.Namespace,
    penalty_matrix: torch.Tensor,
    class_weights: torch.Tensor,
    loss_curve_path: str,
    metric_curve_path: str,
    cm_raw_path: str,
    cm_norm_path: str,
    temporal_case_path: str,
    temporal_case_note: str,
    infer_latency_mean: float,
    infer_latency_std: float,
) -> str:
    report_path = f"detailed_report_{fold_tag}.txt"
    conf_lines = _analyze_confusions(cm, class_names, top_k=8)
    mean_epoch_time = float(np.mean(epoch_times_sec)) if epoch_times_sec else 0.0
    std_epoch_time = float(np.std(epoch_times_sec)) if epoch_times_sec else 0.0
    total_train_time = float(np.sum(epoch_times_sec)) if epoch_times_sec else 0.0

    # Describe penalty and class-cap effect from runtime values
    feli_idx = class_names.index("felicidad") if "felicidad" in class_names else None
    enojo_idx = class_names.index("enojo") if "enojo" in class_names else None
    neutral_idx = class_names.index("neutral") if "neutral" in class_names else None
    penalty_note = "No aplica"
    if feli_idx is not None and neutral_idx is not None:
        p_fn = float(penalty_matrix[feli_idx, neutral_idx].item())
        p_en = float(penalty_matrix[enojo_idx, neutral_idx].item()) if enojo_idx is not None else 1.0
        penalty_note = (
            f"Penalizacion activa: felicidad->neutral={p_fn:.3f}, "
            f"enojo->neutral={p_en:.3f}."
        )
    cw_note = (
        f"class_weight_cap configurado en {float(args.class_weight_cap):.3f}; "
        f"pesos efectivos={class_weights.detach().cpu().numpy().round(3).tolist()}"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DETALLADO DE ENTRENAMIENTO\n")
        f.write(f"Fold: {fold_tag}\n\n")
        f.write(f"Tiempo total de entrenamiento (s): {total_train_time:.2f}\n\n")

        if train_losses and val_losses:
            f.write(
                f"Resumen historial: train_loss inicio={train_losses[0]:.4f}, fin={train_losses[-1]:.4f}; "
                f"val_loss inicio={val_losses[0]:.4f}, fin={val_losses[-1]:.4f}; "
                f"macro_f1_present max={max(val_f1_present):.4f}.\n\n"
            )

        f.write("A) Matriz de Confusion (normalizada y no normalizada)\n")
        f.write(f"- Imagen no normalizada: {cm_raw_path}\n")
        f.write(f"- Imagen normalizada: {cm_norm_path}\n")
        f.write("- Analisis de principales confusiones:\n")
        for ln in conf_lines:
            f.write(f"  * {ln}\n")
        f.write(f"- Vinculacion tecnica (penalty matrix): {penalty_note}\n")
        f.write(f"- Vinculacion tecnica (class_weight_cap): {cw_note}\n\n")

        f.write("B) Curvas de Aprendizaje (loss y metrica)\n")
        f.write(f"- Curva de perdida: {loss_curve_path}\n")
        f.write(f"- Curva de macro_f1_present: {metric_curve_path}\n")
        f.write(
            "- Observacion: se grafican las primeras 60 epocas (o menos si el entrenamiento se detuvo antes). "
            "La linea vertical marca best_epoch y, cuando aplica, la activacion del early stopping.\n"
        )
        f.write(
            f"- best_epoch={best_epoch}; early_stop_epoch={early_stop_epoch if early_stop_epoch is not None else 'no activado'}.\n\n"
        )

        f.write("C) Impacto del balanceo de datos (ablation analitico)\n")
        f.write(
            "- El pipeline combina undersampling del percentil objetivo y oversampling sintetico de minorias. "
            "Esto suele mejorar recall de clases raras, pero puede introducir patrones repetitivos si la fuerza de aumento es muy alta.\n"
        )
        f.write(
            f"- Config usada: undersample_target_percentile={float(getattr(args, 'undersample_target_percentile', 0.0)):.1f}, "
            f"oversample_target_percentile={float(getattr(args, 'oversample_target_percentile', 0.0)):.1f}, "
            f"minority_aug_strength={float(getattr(args, 'minority_aug_strength', 0.0)):.2f}, "
            f"oversample_max_multiplier={float(getattr(args, 'oversample_max_multiplier', 0.0)):.2f}.\n"
        )
        f.write(
            "- Interpretacion: si la curva de validacion se estanca mientras train loss sigue bajando, hay senal de sobreajuste inducido por sinteticos.\n\n"
        )

        f.write("D) Costo computacional y eficiencia\n")
        f.write(
            f"- Tiempo por epoca: media={mean_epoch_time:.2f}s, desviacion={std_epoch_time:.2f}s "
            f"(batch_size={args.batch_size}, num_frames={args.num_frames}).\n"
        )
        f.write(
            f"- Latencia de inferencia con TTA=30 (1 video): media={infer_latency_mean:.3f}s, std={infer_latency_std:.3f}s.\n"
        )
        f.write(
            "- Justificacion: TTA alto incrementa robustez pero multiplica costo temporal casi linealmente con el numero de pases.\n\n"
        )

        f.write("E) Comportamiento temporal de la LSTM\n")
        f.write(f"- Imagen de caso temporal (mapas de flujo): {temporal_case_path}\n")
        f.write(f"- Nota: {temporal_case_note}\n")
        f.write(
            "- Interpretacion: dual_phase_flow separa la dinamica onset->apex y apex->offset; la Bi-LSTM explota esa secuencia para reducir ambiguedad frente a frames aislados.\n"
        )

    return report_path

import torch.nn.functional as F

    # Matriz de penalización para confusiones específicas (Ekman-7 orden: asco, enojo, felicidad, miedo, neutral, sorpresa, tristeza)
    # Penaliza más la confusión entre felicidad (2), enojo (1) y neutral (4)
def get_confusion_penalty_matrix(
    num_classes: int,
    penalty: float = 2.0,
    label_map: dict[str, int] | None = None,
    custom_penalties: str | None = None,
) -> torch.Tensor:
    """Build a penalty matrix for the custom cross-entropy loss.

    If *custom_penalties* is provided (format: 'true_cls:pred_cls:value,...'),
    those entries are added on top of the default matrix using *label_map*
    to resolve class names to indices.  If *label_map* is None, custom entries
    are ignored with a warning.
    """
    penalty_matrix = torch.ones((num_classes, num_classes))

    def _set_penalty_if_valid(true_idx: int, pred_idx: int, value: float) -> None:
        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            penalty_matrix[true_idx, pred_idx] = value

    # Default penalties focus on confusion among felicidad/enojo/neutral.
    # Use class names when label_map is available (robust to filtered class subsets).
    if label_map:
        _aliases = {
            "feli": "felicidad", "hap": "felicidad", "happiness": "felicidad",
            "ang": "enojo", "anger": "enojo",
            "neu": "neutral", "others": "neutral",
        }

        def _resolve_default(name: str) -> int | None:
            key = _aliases.get(name.strip().lower(), name.strip().lower())
            return label_map.get(key)

        feli = _resolve_default("felicidad")
        enojo = _resolve_default("enojo")
        neutral = _resolve_default("neutral")

        if feli is not None and enojo is not None:
            _set_penalty_if_valid(feli, enojo, penalty)
            _set_penalty_if_valid(enojo, feli, penalty)
        if feli is not None and neutral is not None:
            _set_penalty_if_valid(feli, neutral, penalty)
            _set_penalty_if_valid(neutral, feli, penalty)
        if enojo is not None and neutral is not None:
            _set_penalty_if_valid(enojo, neutral, penalty)
            _set_penalty_if_valid(neutral, enojo, penalty)
    else:
        # Backward-compatible fallback for legacy fixed Ekman-7 ordering.
        _set_penalty_if_valid(2, 1, penalty)  # felicidad -> enojo
        _set_penalty_if_valid(1, 2, penalty)  # enojo -> felicidad
        _set_penalty_if_valid(2, 4, penalty)  # felicidad -> neutral
        _set_penalty_if_valid(4, 2, penalty)  # neutral -> felicidad
        _set_penalty_if_valid(1, 4, penalty)  # enojo -> neutral
        _set_penalty_if_valid(4, 1, penalty)  # neutral -> enojo

    if custom_penalties:
        if label_map is None:
            print("[penalty] WARNING: --confusion_penalty_matrix provided but label_map is None — skipping custom penalties.")
        else:
            _aliases = {
                "feli": "felicidad", "hap": "felicidad", "happiness": "felicidad",
                "ang": "enojo", "anger": "enojo",
                "dis": "asco", "disgust": "asco",
                "neu": "neutral", "others": "neutral",
                "sor": "sorpresa", "surprise": "sorpresa",
                "sad": "tristeza", "sadness": "tristeza",
                "fea": "miedo", "fear": "miedo",
            }
            def _resolve(name: str) -> int | None:
                name = name.strip().lower()
                name = _aliases.get(name, name)
                return label_map.get(name, None)
            for entry in custom_penalties.split(","):
                parts = entry.strip().split(":")
                if len(parts) != 3:
                    print(f"[penalty] WARNING: invalid entry '{entry}' — expected 'true_cls:pred_cls:value'")
                    continue
                true_cls, pred_cls, val_str = parts
                true_idx = _resolve(true_cls)
                pred_idx = _resolve(pred_cls)
                try:
                    val = float(val_str)
                except ValueError:
                    print(f"[penalty] WARNING: invalid value '{val_str}' for entry '{entry}'")
                    continue
                if true_idx is None or pred_idx is None:
                    print(f"[penalty] WARNING: class not found for entry '{entry}' (true={true_cls}, pred={pred_cls})")
                    continue
                penalty_matrix[true_idx, pred_idx] = val
                print(f"[penalty] custom: {true_cls}({true_idx}) → {pred_cls}({pred_idx}) = {val}")
    return penalty_matrix

    # Función robusta para obtener out_features del clasificador
def get_classifier_out_features(model):
    # Si el modelo tiene 'classifier' (Sequential), busca el último Linear
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        if isinstance(classifier, torch.nn.Sequential):
            for layer in reversed(classifier):
                if hasattr(layer, 'out_features'):
                    return layer.out_features
        elif hasattr(classifier, 'out_features'):
            return classifier.out_features
    # Si tiene 'fc' (como en ResNet)
    if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
        return model.fc.out_features
    raise AttributeError("No se pudo encontrar 'out_features' en el modelo.")
# Función de pérdida personalizada
def custom_cross_entropy_with_penalty(logits, targets, penalty_matrix):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    preds = logits.argmax(dim=1)
    penalties = penalty_matrix.to(logits.device)[targets, preds]
    return (ce_loss * penalties).mean()
"""
train_densenet_flow.py
----------------------
Sequence-level micro-expression classifier using DenseNet121 as frame encoder.

Pipeline per sample:
    .npy (N, H, W, 3) float32 optical flow
        → uniform frame sampling  → T frames (H, W, 3)
        → resize 224×224 + normalize (ImageNet-adapted, 3-channel)
        → DenseNet121 features    (B*T, 3, 224, 224) → (B*T, 1024)
    → reshape + mean-pool over T  → (B, 1024)
    → Linear classifier          → (B, num_classes)

Changes vs. train_beit_flow.py:
    - BEiT / Hugging Face replaced with DenseNet121 (torchvision)
    - Input keeps the original 3-channel optical flow [dx, dy, mag]
    - Preprocessing uses torchvision transforms instead of BEiT processor

Run example:
  python scripts/train_densenet_flow.py \
      --data_dir  path/to/extracted_npy   \
      --csv_index path/to/extraction_index.csv \
      --epochs    30                          \
      --batch_size 8
"""


import argparse
import copy
import csv
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# Center Loss: penaliza la distancia intra-clase en el embedding.
class CenterLoss(nn.Module):
    """
    Center Loss: penaliza la distancia intra-clase en el embedding.
    Referencia: Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition" (ECCV 2016)
    """
    def __init__(self, num_classes, feat_dim, device=None, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.device = device or torch.device('cpu')
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=self.device))

    def forward(self, features, labels):
        # features: (batch, feat_dim)
        # labels: (batch,)
        batch_size = features.size(0)
        centers_batch = self.centers[labels]  # (batch, feat_dim)
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        return loss
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls together embeddings of the same class and pushes apart
    embeddings of different classes using all pairs in the batch.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """If domain_labels is provided, only form positive pairs within the same domain."""
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        features = F.normalize(features, dim=1)
        sim = torch.mm(features, features.T) / self.temperature
        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.T).float()
        # Domain mask: only same-domain pairs are positives
        if domain_labels is not None:
            domain_col = domain_labels.view(-1, 1)
            domain_mask = (domain_col == domain_col.T).float()
            pos_mask = pos_mask * domain_mask
        self_mask = torch.eye(B, device=features.device)
        pos_mask = pos_mask - self_mask
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        loss = -(pos_mask * log_prob).sum(dim=1)
        loss = loss[valid] / n_pos[valid]
        return loss.mean()


from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split, TensorDataset
from torchvision import models, transforms

# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # Data
    input_path:  str  = "Microexpresiones/outputs_dataset_index"
    data_dir:    str  = input_path
    csv_index:   str | None = str(Path(input_path) / "extraction_index.csv")
    onset_csv:   str | None = None

    # Frame sampling
    num_frames:  int  = 16

    # Backbone fine-tuning
    unfreeze_epoch:  int  = 2

    # Training
    epochs:      int  = 30
    batch_size:  int  = 8
    lr:          float = 1e-4
    weight_decay: float = 1e-2
    val_split:   float = 0.2
    split_mode:  str  = "subject"
    balanced_sampler: bool = True
    seed:        int  = 42
    num_workers: int  = 0
    early_stop_patience: int = 0
    early_stop_min_epochs: int = 0
    ema_decay: float = 0.0
    grad_clip_norm: float = 0.0
    freeze_bn_after_unfreeze: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# LABEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

EMOTION_TO_IDX: dict[str, int] = {
    "happiness": 0, "hap": 0, "happy": 0,
    "felicidad": 0,
    "surprise":  1, "sur": 1,
    "sorpresa": 1,
    "disgust":   2, "dis": 2,
    "asco": 2,
    "repression":3, "rep": 3,
    "represion": 3, "represión": 3,
    "others":    4, "oth": 4, "neutral": 4,
    "otros": 4,
    "sadness":   5, "sad": 5,
    "tristeza": 5,
    "fear":      6, "fea": 6,
    "miedo": 6,
    "contempt":  7, "con": 7,
    "anger":     8, "ang": 8, "enojo": 8,
    "tense":     9, "ten": 9,
}

# Canonical Ekman-7 set used in this project.
# Neutral can be injected from a dedicated source (e.g., MEME v2 neutral-only CSV).
EKMAN7_CANONICAL: tuple[str, ...] = (
    "asco",
    "enojo",
    "felicidad",
    "miedo",
    "neutral",
    "sorpresa",
    "tristeza",
)

# Backward-compatible Ekman-6 view (without neutral).
EKMAN6_CANONICAL: tuple[str, ...] = (
    "asco",
    "enojo",
    "felicidad",
    "miedo",
    "sorpresa",
    "tristeza",
)


def canonicalize_ekman7(emotion: str) -> str | None:
    """Map raw emotion labels to Ekman-7 canonical labels, or None if out-of-set."""
    e = emotion.strip().lower()
    aliases: dict[str, str] = {
        "disgust": "asco",
        "dis": "asco",
        "asco": "asco",
        "anger": "enojo",
        "ang": "enojo",
        "enojo": "enojo",
        "happiness": "felicidad",
        "happy": "felicidad",
        "hap": "felicidad",
        "felicidad": "felicidad",
        "fear": "miedo",
        "fea": "miedo",
        "miedo": "miedo",
        "surprise": "sorpresa",
        "sur": "sorpresa",
        "sorpresa": "sorpresa",
        "sadness": "tristeza",
        "sad": "tristeza",
        "tristeza": "tristeza",
        "neutral": "neutral",
        "others": "neutral",
        "oth": "neutral",
        "otros": "neutral",
        "repression": "neutral",
        "represion": "neutral",
        "represión": "neutral",
        "rep": "neutral",
    }
    return aliases.get(e)


def canonicalize_ekman6(emotion: str) -> str | None:
    """Map raw emotion labels to Ekman-6 canonical labels, excluding neutral."""
    c = canonicalize_ekman7(emotion)
    if c == "neutral":
        return None
    return c


# Backward-compatible alias for previous name used in this file/history.
canonicalize_ekman = canonicalize_ekman7


def build_label_map(labels_found: set[str]) -> dict[str, int]:
    sorted_names = sorted(labels_found)
    return {name: idx for idx, name in enumerate(sorted_names)}


# ══════════════════════════════════════════════════════════════════════════════
# FRAME SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_frames(
    sequence: np.ndarray,
    num_frames: int,
    onset: int | None = None,
    offset: int | None = None,
) -> np.ndarray:
    """
    Return exactly `num_frames` frames from `sequence` (N, H, W).

    If onset/offset are provided, restrict sampling to that range.
    If the available range is shorter than num_frames, indices are
    repeated (tile) to ensure a fixed-size output.
    """
    n = sequence.shape[0]
    start = max(0, onset  if onset  is not None else 0)
    end   = min(n, offset if offset is not None else n)
    if end <= start:
        start, end = 0, n

    available = end - start

    if available >= num_frames:
        indices = np.linspace(start, end - 1, num_frames, dtype=int)
    else:
        base = np.arange(start, end)
        reps = (num_frames // available) + 1
        indices = np.tile(base, reps)[:num_frames]

    return sequence[indices]  # (num_frames, H, W)


def _extract_onset_apex_offset(
    sequence: np.ndarray,
    onset: int,
    apex: int | None,
    offset: int,
    num_frames: int,
) -> np.ndarray:
    """
    Build the analysis window by concatenating two temporal segments:
        part1 = seq[onset : apex]      (onset-inclusive, apex-exclusive)
        part2 = seq[apex  : offset+1]  (apex-inclusive, offset-inclusive)

    This ensures the apex frame appears exactly once (at the start of part2).
    Falls back to the full onset-offset range when apex is unavailable.
    """
    n = sequence.shape[0]
    onset  = max(0, onset)
    offset = min(n - 1, offset)

    if apex is None:
        # No apex info: fall back to uniform sampling over onset-offset range
        return sample_frames(sequence, num_frames, onset, offset + 1)

    # Clamp apex to the valid window
    apex = max(onset, min(apex, offset))

    part1 = sequence[onset:apex]        # onset → apex (apex NOT duplicated)
    part2 = sequence[apex:offset + 1]   # apex → offset (apex included here)

    # Handle edge cases: empty part1 (apex==onset) or empty part2 (apex==offset+1)
    if part1.shape[0] == 0 and part2.shape[0] == 0:
        combined = sequence[onset:offset + 1]
    elif part1.shape[0] == 0:
        combined = part2
    elif part2.shape[0] == 0:
        combined = part1
    else:
        combined = np.concatenate([part1, part2], axis=0)

    # Final guard: if still empty, fall back to full sequence
    if combined.shape[0] == 0:
        combined = sequence

    return sample_frames(combined, num_frames)


def _extract_dual_phase(
    sequence: np.ndarray,
    onset: int,
    apex: int | None,
    offset: int,
    num_frames: int,
) -> np.ndarray:
    """
    Dual-phase frame extraction: allocate exactly num_frames//2 to each phase.

        Phase 1 (onset → apex)  : captures motion build-up / expression onset.
        Phase 2 (apex  → offset): captures motion relaxation / expression offset.

    The two halves are concatenated in temporal order so the model always sees
    [phase1_frames | phase2_frames]. When apex is unavailable, falls back to
    the standard _extract_onset_apex_offset logic.

    Based on the insight from FMANet (arXiv 2510.07810, 2025): both phases carry
    complementary, symmetric action-unit activations that improve recognition.
    """
    n = sequence.shape[0]
    if n == 0:
        # Empty sequence: return empty array
        return sequence

    onset  = max(0, onset)
    offset = min(n - 1, offset)

    if apex is None:
        # No apex available: use original sampling
        return _extract_onset_apex_offset(sequence, onset, None, offset, num_frames)

    apex = max(onset, min(apex, offset))

    half = num_frames // 2
    remainder = num_frames - 2 * half  # 1 when num_frames is odd, else 0

    part1 = sequence[onset:apex]        # onset → apex  (apex NOT in part1)
    part2 = sequence[apex:offset + 1]   # apex → offset (apex IS in part2)

    # Fallback: if both parts are empty, use full sequence
    if part1.shape[0] == 0 and part2.shape[0] == 0:
        return sample_frames(sequence, num_frames)

    # Sample each phase independently; tile if shorter than target.
    if part1.shape[0] == 0:
        # apex == onset: both halves come from part2
        frames1 = sample_frames(part2, half)
        frames2 = sample_frames(part2, half + remainder)
    elif part2.shape[0] == 0:
        # apex > offset (shouldn't happen after clamp, but be safe): both from part1
        frames1 = sample_frames(part1, half)
        frames2 = sample_frames(part1, half + remainder)
    else:
        frames1 = sample_frames(part1, half)
        frames2 = sample_frames(part2, half + remainder)

    return np.concatenate([frames1, frames2], axis=0)  # (num_frames, H, W, 3)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def build_transforms() -> transforms.Compose:
    """
        3-channel preprocessing pipeline for optical flow:
            - Resize to 224×224
            - ToTensor  → (3, 224, 224) float [0, 1]
            - Normalize to [-1, 1] range using flow-friendly stats
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_train_aug_transforms(magnitude: float = 1.5) -> transforms.Compose:
    """Spatial augmentation transforms safe for optical flow tensors (C, H, W).

    All operations preserve channel semantics of [dx, dy, mag] optical flow.
    Magnitude controls the intensity: 0.5=light, 1.0=moderate, 1.5=aggressive.
    """
    m = float(np.clip(magnitude, 0.1, 2.0))
    return transforms.Compose([
        # Larger rotation and translation for stronger augmentation
        transforms.RandomAffine(
            degrees=int(15 * m),               # ±15° rotation at magnitude=1.5
            translate=(0.10 * m, 0.10 * m),   # ±10% translation
            scale=(max(0.75, 1.0 - 0.2 * m), min(1.25, 1.0 + 0.2 * m)),
            fill=0,
        ),
        # Stronger Gaussian blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0 * m)),
        # More aggressive random erasing
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.15 * m),
            ratio=(0.2, 4.0),
            value=0,
        ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FlowSequenceDataset(Dataset):
    """
        PyTorch Dataset for pre-extracted optical-flow sequences (.npy).

        Each .npy file has shape (N, H, W, 3) float32 with [dx, dy, mag].

    Each __getitem__ returns:
            frames : Tensor (T, 3, 224, 224) float32 — DenseNet-ready
      label  : int — emotion class index
    """

    def __init__(
        self,
        data_dir:   str | Path,
        transform,                           # torchvision transform pipeline
        num_frames: int  = 8,
        csv_index:  str | Path | None = None,
        extra_csv_index: str | Path | None = None,
        smic_csv_index: str | Path | None = None,
        meme_csv_index: str | Path | None = None,
        meme_csv_index_2: str | Path | None = None,
        neutral_csv_index: str | Path | None = None,
        onset_csv:  str | Path | None = None,
        label_map:  dict[str, int] | None = None,
        temporal_aug_prob: float = 0.5,
        extra_data_dir: str | Path | None = None,
        smic_only_surprise: bool = False,
        oversample_minority: bool = False,
        min_samples_per_class: int = 20,
        dual_phase: bool = False,
        casme3_classes: set[str] | None = None,
        happiness_limit: int | None = None,  # Nuevo argumento opcional
    ) -> None:
        self.samples = []
        self.domain_labels = []
        self.onsets = {}
        self.data_dir   = Path(data_dir)
        self.transform  = transform
        self.num_frames = num_frames
        self.temporal_aug_prob = float(np.clip(temporal_aug_prob, 0.0, 1.0))
        self.dual_phase = dual_phase
        self.casme3_classes = casme3_classes
        self.happiness_limit = happiness_limit

        # Stores (onset, apex, offset) per sequence stem; apex may be None
        self.onsets: dict[str, tuple[int, int | None, int]] = {}

        # ── Load primary CSV o scan data_dir ─────────────────────────────
        if csv_index is not None:
            before = len(self.samples)
            self._build_from_csv(Path(csv_index), domain="primary")
            after = len(self.samples)
            print(f"[data]  CASME2 csv '{csv_index}': {after - before} secuencias cargadas")
        else:
            self._build_from_dir()

        # ── Load SMIC CSV (optional, surprise-only filter) ────────────────
        if smic_csv_index is not None:
            before = len(self.samples)
            # Permitir ambas variantes: 'sorpresa' y 'surprise'
            # Permitir variantes: 'sorpresa', 'surprise', 'Sorpresa', 'SURPRISE', etc. (sin acentos)
            import unicodedata
            def _normalize_emo(e):
                return unicodedata.normalize('NFKD', e).encode('ascii', 'ignore').decode('ascii').lower()
            _smic_allowed_raw = {"sorpresa", "surprise"}
            _smic_allowed = set(_normalize_emo(e) for e in _smic_allowed_raw) if smic_only_surprise else None
            # Parchear _build_from_csv para normalizar emociones en SMIC
            def _build_from_csv_smic(csv_path, domain="primary", allowed_emotions=None):
                import csv
                import unicodedata
                def _normalize_emo(e):
                    return unicodedata.normalize('NFKD', e).encode('ascii', 'ignore').decode('ascii').lower()
                base = csv_path.parent
                with csv_path.open("r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        rel  = row.get("archivo", "").strip()
                        emo  = row.get("emocion", "").strip()
                        emo_norm = _normalize_emo(emo)
                        subj = (
                            row.get("sujeto", "").strip()
                            or row.get("subject", "").strip()
                            or row.get("Subject", "").strip()
                            or row.get("SUBJECT", "").strip()
                            or row.get("sub", "").strip()
                            or row.get("participant", "").strip()
                        )
                        if allowed_emotions is not None and emo_norm not in allowed_emotions:
                            continue
                        if not rel or emo_norm not in set(_normalize_emo(e) for e in EMOTION_TO_IDX.keys()):
                            continue
                        if not subj:
                            rel_parts = Path(rel.replace("\\", "/")).parts
                            if len(rel_parts) >= 2:
                                subj = rel_parts[0]
                        if not subj:
                            subj = "unknown_subject"
                        npy_path = base / Path(rel.replace("\\", "/"))
                        if npy_path.exists():
                            self.samples.append((npy_path, emo_norm, subj))
                            self.domain_labels.append(domain)
                            # Onsets opcional
                            try:
                                _onset  = int(row["onset_frame"])  if row.get("onset_frame")  else None
                                _apex   = int(row["apex_frame"])   if row.get("apex_frame")   else None
                                _offset = int(row["offset_frame"]) if row.get("offset_frame") else None
                                if _onset is not None and _offset is not None:
                                    self.onsets[npy_path.stem] = (_onset, _apex, _offset)
                            except (ValueError, KeyError):
                                pass
            if smic_only_surprise:
                _build_from_csv_smic(Path(smic_csv_index), domain="smic", allowed_emotions=_smic_allowed)
            else:
                self._build_from_csv(
                    Path(smic_csv_index),
                    domain="smic",
                    allowed_emotions=_smic_allowed,
                )
            added = len(self.samples) - before
            print(f"[data]  SMIC csv '{smic_csv_index}': {added} sequences loaded"
                  + (" (surprise-only filter)" if smic_only_surprise else ""))

        # ── Load MEME CSV (optional) ──────────────────────────────────────
        if meme_csv_index is not None:
            before = len(self.samples)
            self._build_from_csv(Path(meme_csv_index), domain="meme")
            added = len(self.samples) - before
            print(f"[data]  MEME csv '{meme_csv_index}': {added} sequences loaded")

        # ── Load MEME CSV #2 (optional) ───────────────────────────────────
        if meme_csv_index_2 is not None:
            before = len(self.samples)
            self._build_from_csv(Path(meme_csv_index_2), domain="meme_2")
            added = len(self.samples) - before
            print(f"[data]  MEME csv #2 '{meme_csv_index_2}': {added} sequences loaded")

        # ── Load neutral-only CSV (optional, neutral class only) ──────────
        if neutral_csv_index is not None:
            before = len(self.samples)
            self._build_from_csv(
                Path(neutral_csv_index),
                domain="meme_v2_neutral",
                allowed_emotions={"neutral"},
            )
            added = len(self.samples) - before
            print(f"[data]  neutral csv '{neutral_csv_index}': {added} neutral sequences loaded")

        # ── Load extra CSV (always loaded when provided) ──────────────────
        if extra_csv_index is not None:
            before = len(self.samples)
            self._build_from_csv(
                Path(extra_csv_index),
                domain="extra",
                casme3_classes=self.casme3_classes,
                happiness_limit=self.happiness_limit
            )
            added = len(self.samples) - before
            if self.casme3_classes:
                print(f"[data]  extra csv '{extra_csv_index}': {added} sequences loaded (CASME3 class filter: {self.casme3_classes})")
            else:
                print(f"[data]  extra csv '{extra_csv_index}': {added} sequences loaded")

        # ── Load extra data directory (optional) ──────────────────────────
        if extra_data_dir is not None:
            self._build_from_dir_extra(Path(extra_data_dir))

        # ── Load onset/apex/offset CSV (optional) ─────────────────────────
        if onset_csv is not None:
            self._load_onset_csv(Path(onset_csv))

        if label_map is not None:
            self.label_map = label_map
        else:
            found = {emotion for _, emotion, _subject in self.samples}
            self.label_map = build_label_map(found)

        # oversample_minority is applied AFTER splitting (train-only), not here

    # ── Index builders ─────────────────────────────────────────────────────

    def _build_from_dir(self) -> None:
        for npy_path in sorted(self.data_dir.rglob("*.npy")):
            parts = npy_path.relative_to(self.data_dir).parts
            if len(parts) < 3:
                continue
            subject = parts[0]
            emotion = parts[1].lower()
            if emotion in EMOTION_TO_IDX:
                self.samples.append((npy_path, emotion, subject))
                self.domain_labels.append("primary")

    def _build_from_dir_extra(self, extra_dir: Path) -> None:
        """Scan extracted_flow-style structure: subject/camera/emotion.npy.
        
        Each .npy file represents one sequence; the emotion is the file stem.
        Example: p025/camaraWeb/Sorpresa.npy -> subject=p025, emotion=sorpresa
        """
        CAMERAS = {"camaraweb", "camarainfrarroja"}
        added = 0
        skipped_cam = 0
        skipped_emo = 0
        for npy_path in sorted(extra_dir.rglob("*.npy")):
            parts = npy_path.relative_to(extra_dir).parts
            # expected: subject / camera / emotion.npy  (exactly 3 parts)
            if len(parts) != 3:
                continue
            subject = parts[0]
            camera  = parts[1].lower()
            emotion = npy_path.stem.lower()
            if camera not in CAMERAS:
                skipped_cam += 1
                continue
            if emotion not in EMOTION_TO_IDX:
                skipped_emo += 1
                continue
            self.samples.append((npy_path, emotion, subject))
            self.domain_labels.append("extra")
            added += 1
        print(
            f"[data]  extra dataset '{extra_dir}': {added} sequences loaded "
            f"(skipped: {skipped_cam} unknown camera, {skipped_emo} unknown emotion)"
        )

    def _build_from_csv(
        self,
        csv_path: Path,
        domain: str = "primary",
        allowed_emotions: set[str] | None = None,
        casme3_classes: set[str] | None = None,
        happiness_limit: int | None = None,
    ) -> None:
        # If this is CASME3 and casme3_classes is provided, filter to those classes only
        is_casme3 = "casme3" in str(csv_path).lower()
        if is_casme3 and casme3_classes is not None:
            allowed_emotions = set(c.strip().lower() for c in casme3_classes)
        base = csv_path.parent
        happiness_count = 0
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel  = row.get("archivo", "").strip()
                emo  = row.get("emocion", "").strip().lower()
                subj = (
                    row.get("sujeto", "").strip()
                    or row.get("subject", "").strip()
                    or row.get("Subject", "").strip()
                    or row.get("SUBJECT", "").strip()
                    or row.get("sub", "").strip()
                    or row.get("participant", "").strip()
                )
                if allowed_emotions is not None and emo not in allowed_emotions:
                    continue
                if not rel or emo not in EMOTION_TO_IDX:
                    continue
                if not subj:
                    # Fallback: infer subject from the first path component.
                    rel_parts = Path(rel.replace("\\", "/")).parts
                    if len(rel_parts) >= 2:
                        subj = rel_parts[0]
                if not subj:
                    subj = "unknown_subject"
                # Limitar felicidad solo para CASME3 extra
                if (
                    happiness_limit is not None
                    and "casme3" in str(csv_path).lower()
                    and emo in {"felicidad", "happiness", "happy", "hap"}
                    and domain == "extra"
                ):
                    if happiness_count >= happiness_limit:
                        continue
                    happiness_count += 1
                npy_path = base / Path(rel.replace("\\", "/"))
                if npy_path.exists():
                    self.samples.append((npy_path, emo, subj))
                    self.domain_labels.append(domain)
                    # Populate onsets dict from CSV columns when available
                    try:
                        _onset  = int(row["onset_frame"])  if row.get("onset_frame")  else None
                        _apex   = int(row["apex_frame"])   if row.get("apex_frame")   else None
                        _offset = int(row["offset_frame"]) if row.get("offset_frame") else None
                        if _onset is not None and _offset is not None:
                            self.onsets[npy_path.stem] = (_onset, _apex, _offset)
                    except (ValueError, KeyError):
                        pass

    def _load_onset_csv(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key    = Path(row.get("filename", "")).stem
                onset  = int(row["onset"])  if row.get("onset")  else None
                apex   = int(row["apex"])   if row.get("apex")   else None
                offset = int(row["offset"]) if row.get("offset") else None
                if key and onset is not None and offset is not None:
                    self.onsets[key] = (onset, apex, offset)

    # ── PyTorch interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, emotion, _subject = self.samples[idx]
        label = self.label_map[emotion]

        # 1. Load full sequence
        sequence = np.load(str(npy_path))

        # 2. Extract frames using onset→apex and apex→offset segments
        onset_info = self.onsets.get(npy_path.stem)
        if onset_info is not None:
            _onset, _apex, _offset = onset_info
            if self.dual_phase:
                frames = _extract_dual_phase(
                    sequence, _onset, _apex, _offset, self.num_frames
                )
            else:
                frames = _extract_onset_apex_offset(
                    sequence, _onset, _apex, _offset, self.num_frames
                )
        else:
            # No temporal metadata: sample uniformly over the full sequence
            frames = sample_frames(sequence, self.num_frames)
        # frames: (T, H, W, 3) float32

        # 3. Apply temporal augmentation (if enabled)
        frames = self._augment_temporal(frames, p=self.temporal_aug_prob)

        # 4. Convert frames to (T, H, W, 3) uint8 pseudo-RGB.
        #    Flow channels are [dx, dy, mag] with ranges [-1,1], [-1,1], [0,1].
        if frames.ndim == 4:
            # Keep directional information instead of collapsing to grayscale.
            f0 = np.clip((frames[..., 0] + 1.0) / 2.0, 0.0, 1.0)
            f1 = np.clip((frames[..., 1] + 1.0) / 2.0, 0.0, 1.0)
            f2 = np.clip(frames[..., 2], 0.0, 1.0)
            rgb = np.stack([f0, f1, f2], axis=-1)
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = np.repeat(frames[..., None], 3, axis=-1)

        # 5. Apply per-frame transforms: resize → tensor → normalize
        tensor_frames = torch.stack([
            self.transform(Image.fromarray(frame, mode="RGB"))
            for frame in rgb
        ])
        # tensor_frames: (T, 3, 224, 224) float32

        return tensor_frames, label

    def _augment_temporal(self, frames: np.ndarray, p: float = 0.5) -> np.ndarray:
        """
        Temporal augmentation for optical flow sequences.
        Randomly applies: frame dropping, temporal jitter, horizontal flip with dx correction.
        
        Args:
            frames: (T, H, W, 3) float32 with [dx, dy, mag]
            p: probability of applying augmentation
        
        Returns:
            augmented frames: (T, H, W, 3) float32
        """
        if np.random.rand() > p:
            return frames  # No augmentation
        
        T = frames.shape[0]
        
        # Random choice of augmentations
        aug_type = np.random.choice(["drop", "jitter", "flip", "noise", "none"])
        
        if aug_type == "drop" and T > 2:
            # Drop 1-2 random frames and interpolate
            num_drop = np.random.randint(1, min(3, T // 2))
            drop_indices = np.random.choice(range(1, T - 1), size=num_drop, replace=False)
            frames = np.delete(frames, drop_indices, axis=0)
            # Repeat last frame to maintain sequence length
            if len(frames) < T:
                repeat_frame = frames[-1:]
                frames = np.vstack([frames, np.repeat(repeat_frame, T - len(frames), axis=0)])
        
        elif aug_type == "jitter" and T > 2:
            # Temporal jitter: permute with neighboring frames
            jitter_indices = np.arange(T)
            # Shuffle with small neighborhood
            for i in range(1, T - 1):
                if np.random.rand() < 0.3:
                    swap_with = np.random.choice([i - 1, i, i + 1])
                    jitter_indices[i], jitter_indices[swap_with] = jitter_indices[swap_with], jitter_indices[i]
            frames = frames[jitter_indices]
        
        elif aug_type == "flip":
            # Horizontal flip with dx sign correction.
            frames = np.flip(frames, axis=2).copy()
            frames[..., 0] *= -1.0

            # Low-probability vertical flip for consistency with dy sign.
            if np.random.rand() < 0.2:
                frames = np.flip(frames, axis=1).copy()
                frames[..., 1] *= -1.0
        
        elif aug_type == "noise":
            # Add small noise to magnitude and small jitter to flow
            noise_scale = 0.05
            frames = frames.copy().astype(np.float32)
            frames[..., 0] += np.random.normal(0, noise_scale, frames[..., 0].shape)  # dx noise
            frames[..., 1] += np.random.normal(0, noise_scale, frames[..., 1].shape)  # dy noise
            frames[..., 2] *= (1.0 + np.random.normal(0, noise_scale, frames[..., 2].shape))  # mag noise
            frames[..., 0] = np.clip(frames[..., 0], -1.0, 1.0)
            frames[..., 1] = np.clip(frames[..., 1], -1.0, 1.0)
            frames[..., 2] = np.clip(frames[..., 2], 0.0, 1.0)
        
        return frames

    def get_num_classes(self) -> int:
        return len(self.label_map)

    def get_label_map(self) -> dict[str, int]:
        return dict(self.label_map)

    def get_subjects(self) -> list[str]:
        return sorted({subject for _, _, subject in self.samples})

    def get_subject_indices(self, subjects: set[str]) -> list[int]:
        return [i for i, (_, _, subject) in enumerate(self.samples) if subject in subjects]


def split_subject_wise(
    dataset: FlowSequenceDataset,
    val_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    subjects = dataset.get_subjects()
    if len(subjects) < 2:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        return random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    target_val_subjects = max(1, int(round(len(subjects) * val_split)))
    target_val_subjects = min(target_val_subjects, len(subjects) - 1)
    all_labels = {dataset.label_map[emotion] for _, emotion, _ in dataset.samples}

    best_train_idx: list[int] | None = None
    best_val_idx: list[int] | None = None
    best_score: tuple[int, int, int] | None = None

    for attempt in range(64):
        shuffled = subjects.copy()
        random.Random(seed + attempt).shuffle(shuffled)

        val_subjects = set(shuffled[:target_val_subjects])
        train_subjects = set(shuffled[target_val_subjects:])
        train_idx = dataset.get_subject_indices(train_subjects)
        val_idx = dataset.get_subject_indices(val_subjects)

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        train_labels = {dataset.label_map[dataset.samples[i][1]] for i in train_idx}
        val_labels = {dataset.label_map[dataset.samples[i][1]] for i in val_idx}
        score = (
            len(train_labels & all_labels) + len(val_labels & all_labels),
            len(val_labels),
            -abs(len(val_idx) - round(len(dataset) * val_split)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_train_idx = train_idx
            best_val_idx = val_idx

    train_idx = best_train_idx or []
    val_idx = best_val_idx or []

    if len(train_idx) == 0 or len(val_idx) == 0:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        return random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_stratified_random(
    dataset: FlowSequenceDataset,
    val_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Random split preserving per-class proportions as much as possible."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, (_path, emotion, _subject) in enumerate(dataset.samples):
        y = dataset.label_map[emotion]
        by_label.setdefault(y, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for _, indices in by_label.items():
        idxs = indices.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx.extend(idxs)
            continue

        n_val = int(round(n * val_split))
        n_val = max(1, min(n_val, n - 1))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    if len(train_idx) == 0 or len(val_idx) == 0:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        return random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_loso_folds(dataset: FlowSequenceDataset) -> list[tuple[Subset, Subset, str]]:
    """Build Leave-One-Subject-Out folds: each subject is val once."""
    folds: list[tuple[Subset, Subset, str]] = []
    subjects = dataset.get_subjects()
    for subject in subjects:
        val_idx = dataset.get_subject_indices({subject})
        train_idx = [i for i in range(len(dataset)) if i not in set(val_idx)]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx), f"loso_{subject}"))
    return folds


def build_subject_kfolds(
    dataset: FlowSequenceDataset,
    num_folds: int,
    seed: int,
) -> list[tuple[Subset, Subset, str]]:
    """Build K folds splitting by subject identity."""
    subjects = dataset.get_subjects()
    if len(subjects) < 2:
        return []

    k = max(2, min(num_folds, len(subjects)))
    shuffled = subjects.copy()
    random.Random(seed).shuffle(shuffled)
    buckets = [shuffled[i::k] for i in range(k)]

    folds: list[tuple[Subset, Subset, str]] = []
    for i in range(k):
        val_subjects = set(buckets[i])
        train_subjects = set(s for j, b in enumerate(buckets) if j != i for s in b)

        val_idx = dataset.get_subject_indices(val_subjects)
        train_idx = dataset.get_subject_indices(train_subjects)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx), f"kfold_{i+1}"))
    return folds


def split_indices_subject_wise(
    base_dataset: FlowSequenceDataset,
    indices: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split `indices` into train/val by subject identity (no subject in both sets).

    Uses multiple deterministic shuffles and keeps the split with best class
    coverage on validation. This reduces the chance that rare classes (e.g. enojo)
    disappear from val when splitting by subject.
    """
    subject_to_indices: dict[str, list[int]] = {}
    for idx in indices:
        _, _, subject = base_dataset.samples[idx]
        subject_to_indices.setdefault(subject, []).append(idx)

    subjects = sorted(subject_to_indices.keys())
    if len(subjects) < 2:
        rng = random.Random(seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        return shuffled[n_val:], shuffled[:n_val]

    target_val = max(1, min(int(round(len(subjects) * val_split)), len(subjects) - 1))
    class_count = len(base_dataset.label_map)
    available_classes = {
        base_dataset.label_map[base_dataset.samples[idx][1]]
        for idx in indices
    }
    subject_classes: dict[str, set[int]] = {}
    for s, idxs in subject_to_indices.items():
        subject_classes[s] = {
            base_dataset.label_map[base_dataset.samples[idx][1]]
            for idx in idxs
        }
    global_counts = np.zeros(class_count, dtype=np.int64)
    for idx in indices:
        c = base_dataset.label_map[base_dataset.samples[idx][1]]
        global_counts[c] += 1

    def _score_val(val_idx: list[int]) -> tuple[int, float, int]:
        val_counts = np.zeros(class_count, dtype=np.int64)
        for idx in val_idx:
            c = base_dataset.label_map[base_dataset.samples[idx][1]]
            val_counts[c] += 1
        coverage = sum(1 for c in available_classes if val_counts[c] > 0)
        rarity_bonus = float(
            sum((1.0 / max(1, int(global_counts[c]))) for c in available_classes if val_counts[c] > 0)
        )
        # Prefer better coverage, then better rare-class presence, then larger val size.
        return coverage, rarity_bonus, len(val_idx)

    best_train: list[int] | None = None
    best_val: list[int] | None = None
    best_score: tuple[int, float, int] = (-1, -1.0, -1)

    num_trials = max(16, min(512, len(subjects) * 32))
    for t in range(num_trials):
        shuffled_subs = subjects.copy()
        random.Random(seed + t).shuffle(shuffled_subs)

        # Start from target size, then greedily add subjects that cover
        # missing classes (can exceed target_val to improve class coverage).
        val_subjects_list: list[str] = list(shuffled_subs[:target_val])
        val_subjects_set = set(val_subjects_list)

        covered: set[int] = set()
        for s in val_subjects_list:
            covered.update(subject_classes[s])
        missing = set(available_classes) - covered

        if missing:
            for s in shuffled_subs[target_val:]:
                if len(val_subjects_list) >= len(subjects) - 1:
                    break
                gain = subject_classes[s] & missing
                if not gain:
                    continue
                val_subjects_list.append(s)
                val_subjects_set.add(s)
                covered.update(subject_classes[s])
                missing = set(available_classes) - covered
                if not missing:
                    break

        train_idx = [idx for s, idxs in subject_to_indices.items() if s not in val_subjects_set for idx in idxs]
        val_idx = [idx for s, idxs in subject_to_indices.items() if s in val_subjects_set for idx in idxs]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        score = _score_val(val_idx)
        if score > best_score:
            best_score = score
            best_train = train_idx
            best_val = val_idx
            if score[0] == len(available_classes):
                break

    if best_train is not None and best_val is not None:
        return best_train, best_val

    # Safe fallback
    shuffled_subs = subjects.copy()
    random.Random(seed).shuffle(shuffled_subs)
    val_subjects = set(shuffled_subs[:target_val])
    train_idx = [idx for s, idxs in subject_to_indices.items() if s not in val_subjects for idx in idxs]
    val_idx = [idx for s, idxs in subject_to_indices.items() if s in val_subjects for idx in idxs]
    return train_idx, val_idx


def split_stratified_by_class(
    base_dataset: FlowSequenceDataset,
    indices: list[int],
    val_split: float,
    seed: int,
    val_split_overrides: dict[int, float] | None = None,
    val_per_class: int | None = None,
    train_only_indices: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Split `indices` into train/val stratified by class.
    
    Ensures that each class is proportionally represented in both train and val.
    This is the stratified_class_split option: simpler than subject-wise but better
    class distribution than random split.

    If `val_per_class` is set (int > 0), exactly that many samples per class are
    reserved for validation (or as many as possible while keeping ≥1 in train),
    ignoring `val_split` and `val_split_overrides`.

    If `train_only_indices` is provided, those indices are forced into train and
    excluded from the stratified split entirely (they never appear in val).
    """
    rng = random.Random(seed)

    # Separate train-only indices from the stratifiable pool
    _train_only_set: set[int] = set(train_only_indices) if train_only_indices else set()
    _splittable_indices = [i for i in indices if i not in _train_only_set]

    # Group indices by class (only splittable ones)
    class_to_indices: dict[int, list[int]] = {}
    for idx in _splittable_indices:
        class_idx = base_dataset.label_map[base_dataset.samples[idx][1]]
        class_to_indices.setdefault(class_idx, []).append(idx)
    
    # Start train_idx pre-populated with the forced-train-only indices
    train_idx = list(_train_only_set)
    val_idx = []
    
    # For each class, split independently then combine
    for cls_idx in sorted(class_to_indices.keys()):
        cls_indices = class_to_indices[cls_idx]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        if n == 1:
            # Can't split: keep in train so val is not contaminated
            train_idx.extend(cls_indices)
            continue

        if val_per_class is not None and val_per_class > 0:
            # Absolute count mode: reserve exactly val_per_class samples for val,
            # but always keep at least 1 sample in train.
            n_val = min(val_per_class, n - 1)
        else:
            cls_val_split = val_split
            if val_split_overrides and cls_idx in val_split_overrides:
                cls_val_split = float(val_split_overrides[cls_idx])
            # Keep at least 1 sample in both train and val for each class (if possible).
            n_val = int(round(n * cls_val_split))
            n_val = max(1, min(n_val, n - 1))

        val_idx.extend(cls_indices[:n_val])
        train_idx.extend(cls_indices[n_val:])
    
    # Shuffle the combined train/val to avoid class ordering bias
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    
    return train_idx, val_idx


def undersample_train_subset(
    train_subset: Subset,
    max_samples_per_class: int,
    seed: int = 42,
    auto_target: bool = False,
    target_percentile: float = 75.0,
    min_keep: int = 1,
) -> Subset:
    """Undersample majority classes in the TRAIN subset only.

    Strategy:
    - Classes with <= target samples: unchanged.
    - Classes with > target samples: randomly keep only `target` real samples.

    The validation subset is never touched.
    """
    base: FlowSequenceDataset = train_subset.dataset  # type: ignore[assignment]
    indices = list(train_subset.indices)

    by_emotion: dict[str, list[int]] = {}
    for idx in indices:
        _, emotion, _ = base.samples[idx]
        by_emotion.setdefault(emotion, []).append(idx)

    counts_by_emotion: dict[str, int] = {
        emotion: len(idxs)
        for emotion, idxs in by_emotion.items()
    }

    if not counts_by_emotion:
        return train_subset

    keep_floor = max(1, int(min_keep))
    target = int(max_samples_per_class)
    if auto_target:
        _counts = np.array(list(counts_by_emotion.values()), dtype=np.float32)
        _p = float(np.clip(target_percentile, 10.0, 90.0))
        target = int(np.ceil(np.percentile(_counts, _p)))
        print(
            f"[undersample-train] auto target enabled | percentile={_p:.1f} "
            f"target={target} min_keep={keep_floor}"
        )

    target = max(keep_floor, target)
    if target <= 0:
        return train_subset

    rng = random.Random(seed)
    reduced_indices: list[int] = []
    any_reduced = False

    for emotion, idxs in sorted(by_emotion.items()):
        # No reducir felicidad
        if emotion == 'felicidad':
            reduced_indices.extend(idxs)
            continue
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        keep = len(shuffled)
        if len(shuffled) > target:
            keep = max(keep_floor, min(target, len(shuffled)))
            any_reduced = True
            print(
                f"[undersample-train] '{emotion}': {len(shuffled)} -> {keep} "
                f"(-{len(shuffled) - keep} real sequences)"
            )
        reduced_indices.extend(shuffled[:keep])

    if not any_reduced:
        return train_subset

    rng.shuffle(reduced_indices)
    print(f"[undersample-train] total train samples after reduction: {len(reduced_indices)}")
    return Subset(base, reduced_indices)


def oversample_train_subset(
    train_subset: Subset,
    min_samples: int,
    num_frames: int,
    aug_strength: float,
    seed: int = 42,
    auto_target: bool = False,
    target_percentile: float = 50.0,
    max_multiplier: float = 1.0,
    global_aug_ratio: float = 0.5,
) -> Subset:
    """Oversample minority classes in the TRAIN subset only.

    Strategy:
    - Classes with >= min_samples: unchanged.
    - Classes with < min_samples: fill gap using AGGRESSIVE spatial+temporal
      augmentation to generate truly different variants from the few real samples.

    For each real sample in a rare class, generate multiple synthetic versions
    applying random combinations of:
      - Spatial: rotation, flip, crop, additive noise, elastic deformation
      - Temporal: frame jitter, stretch/compress, dropout

    The validation subset is never touched.
    """
    base: FlowSequenceDataset = train_subset.dataset  # type: ignore[assignment]
    indices = list(train_subset.indices)

    # Group train indices by emotion label
    by_emotion: dict[str, list[int]] = {}
    for idx in indices:
        _, emotion, _ = base.samples[idx]
        by_emotion.setdefault(emotion, []).append(idx)

    counts_by_emotion: dict[str, int] = {
        emotion: len(idxs)
        for emotion, idxs in by_emotion.items()
    }

    # Compute per-class augmentation targets.
    # Auto mode uses class-count percentile as target and caps each class by max_multiplier.
    target_by_emotion: dict[str, int] = {}
    if auto_target and counts_by_emotion:
        _counts = np.array(list(counts_by_emotion.values()), dtype=np.float32)
        _p = float(np.clip(target_percentile, 10.0, 90.0))
        _base_target = int(np.ceil(np.percentile(_counts, _p)))
        _base_target = max(_base_target, 1)
        _max_mult = float(max(0.0, max_multiplier))

        for emotion, count in counts_by_emotion.items():
            _class_cap = int(np.floor(count * (1.0 + _max_mult)))
            _target = min(_base_target, _class_cap)
            # Si es enojo, duplicar el target
            if emotion == 'enojo':
                _target = max(_target, int(_base_target * 2))
            target_by_emotion[emotion] = max(count, _target)

        print(
            f"[oversample-train] auto target enabled | percentile={_p:.1f} "
            f"base_target={_base_target} max_multiplier={_max_mult:.2f}"
        )
    else:
        for emotion, count in counts_by_emotion.items():
            # Si es enojo, duplicar el target
            if emotion == 'enojo':
                target_by_emotion[emotion] = max(count, int(min_samples) * 2)
            else:
                target_by_emotion[emotion] = max(count, int(min_samples))

    needed_by_emotion: dict[str, int] = {
        emotion: max(0, target_by_emotion[emotion] - count)
        for emotion, count in counts_by_emotion.items()
    }

    total_needed = int(sum(needed_by_emotion.values()))
    if total_needed <= 0:
        return train_subset

    # Global cap avoids exploding synthetic data: max added samples as a ratio of original train size.
    _global_ratio = float(max(0.0, global_aug_ratio))
    max_extra = int(np.floor(len(indices) * _global_ratio))
    if max_extra <= 0:
        print("[oversample-train] global_aug_ratio=0 -> skipping oversampling")
        return train_subset

    if total_needed > max_extra:
        scale = max_extra / float(total_needed)
        scaled: dict[str, int] = {
            emotion: int(np.floor(need * scale))
            for emotion, need in needed_by_emotion.items()
        }
        remaining = max_extra - int(sum(scaled.values()))
        for emotion, _ in sorted(needed_by_emotion.items(), key=lambda item: item[1], reverse=True):
            if remaining <= 0:
                break
            if scaled[emotion] < needed_by_emotion[emotion]:
                scaled[emotion] += 1
                remaining -= 1
        needed_by_emotion = scaled
        print(
            f"[oversample-train] capped by global_aug_ratio={_global_ratio:.2f}: "
            f"requested={total_needed}, applied={sum(needed_by_emotion.values())}"
        )

    rng = np.random.default_rng(seed)
    extra_arrays: list[np.ndarray] = []  # augmented flow arrays (in memory)
    extra_labels: list[int] = []
    extra_emotions: list[str] = []

    for emotion, idxs in by_emotion.items():
        count = len(idxs)
        needed = int(needed_by_emotion.get(emotion, 0))
        if needed <= 0:
            continue
        label_idx = base.label_map[emotion]
        print(
            f"[oversample-train] '{emotion}': {count} → {count + needed} "
            f"(+{needed} augmented sequences, strength={aug_strength:.2f})"
        )
        # Cycle through available samples and generate K augmented versions per sample
        for k in range(needed):
            source_idx = idxs[k % len(idxs)]
            seq = np.load(str(base.samples[source_idx][0])).astype(np.float32)
            
            # Uniform sample to num_frames
            n = seq.shape[0]
            idx_arr = np.linspace(0, n - 1, num_frames, dtype=int)
            sampled = seq[idx_arr]  # (T, H, W, 3)
            
            # Apply aggressive random augmentation
            augmented = _aggressive_augment_flow(sampled, rng, strength=aug_strength)
            
            extra_arrays.append(augmented)
            extra_labels.append(label_idx)
            extra_emotions.append(emotion)

    if not extra_arrays:
        return train_subset

    print(f"[oversample-train] total train samples after augmentation: {len(indices) + len(extra_arrays)}")

    # Wrap augmented sequences in a lightweight in-memory dataset
    class _AugmentedDataset(Dataset):
        def __init__(self, arrays: list[np.ndarray], labels: list[int], transform) -> None:
            self._arrays = arrays
            self._labels = labels
            self._transform = transform

        def __len__(self) -> int:
            return len(self._arrays)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            frames = self._arrays[idx]  # (T, H, W, 3) float32
            # Normalize flow channels to [0,1] → uint8 RGB
            f0 = np.clip((frames[..., 0] + 1.0) / 2.0, 0.0, 1.0)
            f1 = np.clip((frames[..., 1] + 1.0) / 2.0, 0.0, 1.0)
            f2 = np.clip(frames[..., 2], 0.0, 1.0)
            rgb = (np.stack([f0, f1, f2], axis=-1) * 255).clip(0, 255).astype(np.uint8)
            tensor_frames = torch.stack([
                self._transform(Image.fromarray(frame, mode="RGB"))
                for frame in rgb
            ])
            return tensor_frames, self._labels[idx]

    aug_ds = _AugmentedDataset(extra_arrays, extra_labels, base.transform)

    # ConcatDataset of original train subset + augmented samples
    from torch.utils.data import ConcatDataset
    return ConcatDataset([train_subset, aug_ds])  # type: ignore[return-value]


def _aggressive_augment_flow(
    frames: np.ndarray,  # (T, H, W, 3) float32 [dx, dy, mag]
    rng: np.random.Generator,
    strength: float = 0.6,
) -> np.ndarray:
    """Apply random aggressive spatial+temporal augmentation to optical flow sequence.
    
    Combines multiple techniques to generate truly different variants:
    - Spatial: rotation, flips, crops, additive noise, scale
    - Temporal: jitter, stretch/compress, frame dropout
    
    Returns augmented sequence with same shape (T, H, W, 3).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    T, H, W, C = frames.shape
    aug = frames.copy()

    # Controls augmentation intensity in a smooth way (0.0 = very mild, 1.0 = strong).
    prob_scale = 0.35 + 0.9 * strength

    def _p(base: float) -> bool:
        return rng.random() < min(0.95, max(0.0, base * prob_scale))
    
    # ─── Temporal augmentation ────────────────────────────────────────────
    # 1. Temporal jitter: shuffle frames slightly (preserves local order)
    if _p(0.5) and T > 3:
        jitter = rng.integers(-1, 2, size=T)  # -1, 0, +1
        new_idx = np.clip(np.arange(T) + jitter, 0, T - 1)
        aug = aug[new_idx]
    
    # 2. Temporal stretch/compress: resample at different speed
    if _p(0.4):
        stretch = 0.05 + 0.25 * strength
        scale = rng.uniform(1.0 - stretch, 1.0 + stretch)
        new_T = int(T * scale)
        new_T = max(T // 2, min(new_T, T * 2))
        idx_stretch = np.linspace(0, T - 1, new_T, dtype=int)
        aug_stretched = aug[idx_stretch]
        # Resample back to T frames
        idx_final = np.linspace(0, len(aug_stretched) - 1, T, dtype=int)
        aug = aug_stretched[idx_final]
    
    # 3. Frame dropout: randomly drop and repeat frames
    if _p(0.3) and T > 4:
        drop_mask = rng.random(T) > 0.2  # keep 80%
        if drop_mask.sum() >= T // 2:
            kept = aug[drop_mask]
            # Tile to fill T frames
            reps = (T // len(kept)) + 1
            aug = np.tile(kept, (reps, 1, 1, 1))[:T]
    
    # ─── Spatial augmentation ─────────────────────────────────────────────
    # 4. Horizontal flip (flip dx sign)
    if _p(0.5):
        aug = np.flip(aug, axis=2).copy()  # flip W
        aug[..., 0] *= -1.0  # flip dx
    
    # 5. Vertical flip (flip dy sign)
    if _p(0.3):
        aug = np.flip(aug, axis=1).copy()  # flip H
        aug[..., 1] *= -1.0  # flip dy
    
    # 6. Rotation (small angle): rotate dx/dy vectors
    if _p(0.4):
        max_angle = 5.0 + 12.0 * strength
        angle = rng.uniform(-max_angle, max_angle)  # degrees
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        dx_rot = aug[..., 0] * cos_a - aug[..., 1] * sin_a
        dy_rot = aug[..., 0] * sin_a + aug[..., 1] * cos_a
        aug[..., 0] = dx_rot
        aug[..., 1] = dy_rot
    
    # 7. Random crop (zoom in/out): scale spatial dimensions
    if _p(0.5):
        zoom = 0.05 + 0.20 * strength
        scale = rng.uniform(1.0 - zoom, 1.0 + zoom)
        new_H, new_W = int(H * scale), int(W * scale)
        # Simple resize approximation (bicubic-like via repeat/avg)
        if scale > 1.0:
            # Zoom out: average pool
            aug = aug[:, ::2, ::2, :]  # simple 2x downsample
            # Pad back to original size
            pad_h = (H - aug.shape[1]) // 2
            pad_w = (W - aug.shape[2]) // 2
            aug = np.pad(aug, ((0, 0), (pad_h, H - aug.shape[1] - pad_h),
                               (pad_w, W - aug.shape[2] - pad_w), (0, 0)),
                         mode='edge')
        else:
            # Zoom in: crop center
            crop_h, crop_w = int(H * scale), int(W * scale)
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            aug = aug[:, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
            # Tile to fill
            aug = np.pad(aug, ((0, 0), (0, H - crop_h), (0, W - crop_w), (0, 0)),
                         mode='edge')
    
    # 8. Additive noise to flow components
    if _p(0.6):
        noise_scale = rng.uniform(0.01, 0.02 + 0.08 * strength)
        noise_dx = rng.normal(0, noise_scale, (T, H, W))
        noise_dy = rng.normal(0, noise_scale, (T, H, W))
        noise_mag = rng.normal(0, noise_scale * 0.5, (T, H, W))
        aug[..., 0] += noise_dx
        aug[..., 1] += noise_dy
        aug[..., 2] += noise_mag
    
    # 9. Magnitude scaling (simulate intensity change)
    if _p(0.4):
        mag_jitter = 0.05 + 0.10 * strength
        scale = rng.uniform(1.0 - mag_jitter, 1.0 + mag_jitter)
        aug[..., 2] *= scale
    
    # 10. Gaussian blur (smooth spatial structure) - optional if scipy available
    if _p(0.3):
        try:
            from scipy.ndimage import gaussian_filter
            sigma = rng.uniform(0.3, 0.8 + 0.7 * strength)
            for t in range(T):
                for c in range(3):
                    aug[t, :, :, c] = gaussian_filter(aug[t, :, :, c], sigma=sigma)
        except ImportError:
            pass  # skip blur if scipy not available
    
    # Clamp flow values to valid ranges
    aug[..., 0] = np.clip(aug[..., 0], -1.0, 1.0)  # dx
    aug[..., 1] = np.clip(aug[..., 1], -1.0, 1.0)  # dy
    aug[..., 2] = np.clip(aug[..., 2], 0.0, 1.0)   # mag
    
    return aug


def get_subset_label_indices(
    subset_or_dataset: Subset | FlowSequenceDataset,
    base_dataset: FlowSequenceDataset,
) -> list[int]:
    if isinstance(subset_or_dataset, Subset):
        indices = subset_or_dataset.indices
    else:
        indices = range(len(base_dataset))

    return [
        base_dataset.label_map[base_dataset.samples[i][1]]
        for i in indices
    ]


def label_counts(labels: list[int], num_classes: int) -> list[int]:
    return np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(int).tolist()


class TrainAugSubset(Dataset):
    """Wraps a Subset and applies additional per-frame spatial augmentation at train time.

    The base dataset already handles resize+normalize inside __getitem__. This wrapper
    applies tensor-level augmentations (rotation, blur, erasing) to each frame AFTER
    the base transform, so augmentation is training-only and never touches val_ds.
    """

    def __init__(self, subset: Subset, aug_transform: transforms.Compose) -> None:
        self.subset = subset
        self.aug_transform = aug_transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        frames, label = self.subset[idx]
        # frames: (T, 3, 224, 224) — apply aug to each frame independently
        frames = torch.stack([self.aug_transform(f) for f in frames])
        return frames, label

    # Expose .dataset and .indices so downstream code (sampler, label extraction) works
    @property
    def dataset(self):
        return self.subset.dataset

    @property
    def indices(self):
        return self.subset.indices


def build_balanced_sampler_from_labels(labels: list[int]) -> WeightedRandomSampler:
    counts = np.bincount(np.array(labels, dtype=np.int64))
    counts = np.maximum(counts, 1)
    class_weights  = 1.0 / counts
    sample_weights = class_weights[np.array(labels, dtype=np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_domain_balanced_sampler(domain_labels: list[str]) -> WeightedRandomSampler:
    """Build a sampler that draws equally from each domain.

    Each domain contributes equally regardless of its sample count.
    Within a domain all samples have the same probability.
    Falls back to uniform sampling when only one domain is present.
    """
    domains = sorted(set(domain_labels))
    if len(domains) <= 1:
        weights = torch.ones(len(domain_labels), dtype=torch.double)
        return WeightedRandomSampler(weights=weights, num_samples=len(domain_labels), replacement=True)
    domain_counts = {d: sum(1 for x in domain_labels if x == d) for d in domains}
    domain_weights = {d: 1.0 / domain_counts[d] for d in domains}
    sample_weights = [domain_weights[d] for d in domain_labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights_from_labels(
    labels: list[int],
    num_classes: int,
    cap: float | None = None,
) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.where(counts <= 0.0, 1.0, counts)
    weights = (len(labels) / counts).astype(np.float32)
    weights = weights / max(weights.mean(), 1e-8)
    if cap is not None and cap > 0:
        weights = np.minimum(weights, float(cap))
    return torch.as_tensor(weights, dtype=torch.float32)


def confusion_matrix_np(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    f1_scores: list[float] = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def macro_f1_present_from_confusion(cm: np.ndarray) -> float:
    """Macro-F1 over classes present in ground truth (row sum > 0)."""
    present_classes = [c for c in range(cm.shape[0]) if cm[c, :].sum() > 0]
    if not present_classes:
        return 0.0
    f1_scores: list[float] = []
    for c in present_classes:
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def macro_precision_recall_from_confusion(cm: np.ndarray) -> tuple[float, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    return macro_precision, macro_recall


def adaptive_class_weights_from_confusion(
    cm: np.ndarray,
    base_weights: np.ndarray,
    recall_ema_prev: np.ndarray,
    *,
    ema_beta: float,
    target_recall: float,
    alpha: float,
    max_multiplier: float,
    min_val_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Update class weights using validation confusion matrix.

    Classes with recall below target_recall receive a multiplicative boost.
    Recalls are smoothed with EMA to avoid reacting to single noisy epochs.
    """
    num_classes = cm.shape[0]
    row_sum = cm.sum(axis=1).astype(np.float32)
    tp = np.diag(cm).astype(np.float32)
    recalls = np.divide(tp, np.maximum(row_sum, 1.0), dtype=np.float32)

    # Ignore very-low-support classes in this epoch to reduce noise.
    support_mask = row_sum >= float(max(1, min_val_support))
    recalls_for_ema = recall_ema_prev.copy()
    recalls_for_ema[support_mask] = recalls[support_mask]

    beta = float(np.clip(ema_beta, 0.0, 0.99))
    recall_ema = beta * recall_ema_prev + (1.0 - beta) * recalls_for_ema

    tgt = float(np.clip(target_recall, 1e-3, 1.0))
    deficit = np.maximum(0.0, tgt - recall_ema) / tgt

    mult = 1.0 + float(max(0.0, alpha)) * deficit
    mult = np.clip(mult, 1.0, float(max(1.0, max_multiplier)))

    new_w = base_weights.astype(np.float32) * mult.astype(np.float32)
    # Keep average weight around 1.0 for stable loss scale.
    mean_w = float(np.mean(new_w)) if new_w.size > 0 else 1.0
    if mean_w > 0:
        new_w = new_w / mean_w

    if new_w.shape[0] != num_classes:
        raise ValueError("adaptive weights size mismatch with confusion matrix")

    return new_w, recall_ema


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focuses on hard negatives and rebalances class importance.
    
    Args:
        alpha: class weights (e.g., from imbalanced data) or scalar
        gamma: focusing parameter (0 = CrossEntropyLoss, 2.0 = recommended)
    """
    def __init__(self, alpha: torch.Tensor | float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (B, C) logits
        targets: (B,) class indices
        """
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)  # probability of the true class
        focal_weight = (1 - pt) ** self.gamma
        
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha
        
        focal_loss = alpha_t * focal_weight * ce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT REVERSAL  (Domain-Adversarial Neural Network)
# ══════════════════════════════════════════════════════════════════════════════

class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Multiplies gradients by -lambda_ during back-propagation (no-op in forward)."""
    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, lambda_)


class _DomainLabeledDataset(Dataset):
    """Wraps a Subset/Dataset and appends a domain id (0=primary, 1=extra) per item."""
    def __init__(self, subset, domain_ids: list[int]) -> None:
        self.subset = subset
        self.domain_ids = domain_ids

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        frames, label = self.subset[idx]
        return frames, label, self.domain_ids[idx]


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL AGGREGATION MODULES
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAttentionPool(nn.Module):
    """
    Multi-head self-attention over the T frame features, then mean-pool the output.

    Architecture:
        (B, T, D) → LayerNorm → MultiHeadAttention (T→T) → residual
                  → LayerNorm → FFN (D→4D→D)              → residual
                  → mean over T → (B, D)

    This allows the model to learn which frames (onset / apex / offset) are most
    informative for the classification decision, instead of using a fixed linear ramp.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Ensure num_heads divides dim; fall back to 1 if needed
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, D)"""
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        # FFN block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        # Pool over time dimension
        return x.mean(dim=1)  # (B, D)


class LSTMTemporalPool(nn.Module):
    """
    Bidirectional LSTM over T frame features → concatenated final hidden states.

    Architecture:
        (B, T, D) → BiLSTM(hidden=256, layers=2) → cat(h_fwd, h_bwd) → Dropout → (B, 512)

    Captures temporal order (onset→apex→offset) explicitly, unlike attention which
    treats frames as an unordered set.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2  # bidirectional
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, hidden_dim*2)"""
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, B, hidden_dim) — last dim of num_layers*2 is the final layer
        # forward direction: index -2, backward direction: index -1
        h_fwd = h_n[-2]                            # (B, hidden_dim)
        h_bwd = h_n[-1]                            # (B, hidden_dim)
        h = torch.cat([h_fwd, h_bwd], dim=-1)      # (B, hidden_dim*2)
        return self.drop(h)


# ══════════════════════════════════════════════════════════════════════════════
# GEM POOLING
# ══════════════════════════════════════════════════════════════════════════════

class GeM(nn.Module):
    """
    Generalized Mean (GeM) pooling.

    Instead of plain average-pooling, raises each activation to the power p,
    averages, then takes the 1/p-th root.  p=1 → average pool; p→∞ → max pool.
    p is a *learnable* scalar initialised at 3, which lets the model tune how
    "selective" the spatial aggregation is.  For texture-like micro-expression
    optical-flow maps this usually helps over AdaptiveAvgPool2d.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL GRU
# ══════════════════════════════════════════════════════════════════════════════

class TemporalGRU(nn.Module):
    """
    Bidirectional GRU over T frame features → concatenated final hidden states.

    Architecture:
        (B, T, D) → LayerNorm → BiGRU(hidden, layers) → cat(h_fwd, h_bwd)
                  → Dropout → (B, hidden*2)

    LayerNorm before the GRU stabilises training on small datasets where batch
    statistics are noisy.  GRU has fewer parameters than LSTM (no cell state),
    reducing overfitting without sacrificing sequence modelling capacity.
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if layers == 1 else dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim) → (B, hidden*2)"""
        x = self.norm(x)
        self.gru.flatten_parameters()
        _, h_n = self.gru(x)
        # h_n: (num_layers*2, B, hidden)
        h_forward = h_n[-2]                                # (B, hidden)
        h_backward = h_n[-1]                               # (B, hidden)
        h = torch.cat([h_forward, h_backward], dim=-1)    # (B, hidden*2)
        return self.drop(h)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL TCN
# ══════════════════════════════════════════════════════════════════════════════

class TemporalTCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) over T frame features.

    Uses stacked dilated 1-D convolutions to capture multi-scale temporal
    patterns (onset / apex / offset) without recurrence.

    Architecture:
        (B, T, D) → LayerNorm → proj(D→C, k=1)
                  → [dilated residual block × 3  (dil = 1, 2, 4)]
                  → AdaptiveAvgPool1d(1) → (B, C)

    Each residual block:
        x → Conv1d(C, C, k=3, dil=d, pad=d) → GELU → Dropout
          → Conv1d(C, C, k=1) → + x  (shortcut)

    Effective receptive field with 3 blocks: 1 + 2*(3-1)*(1+2+4) = 29 frames,
    which comfortably covers a full microexpression sequence.

    Advantages over GRU on small datasets:
    - Fully parallelisable (no sequential bottleneck)
    - Multi-scale context via dilation
    - Fewer parameters for the same channel width
    - No vanishing gradient through time
    """

    def __init__(
        self,
        input_dim: int = 512,
        channels: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        # Input projection: (B, D, T) → (B, C, T)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1),
            nn.GELU(),
        )
        # Three dilated residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=1),
            )
            for dilation in [1, 2, 4]
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, channels)"""
        x = self.norm(x)                    # (B, T, D)
        x = x.transpose(1, 2)              # (B, D, T) for Conv1d
        x = self.input_proj(x)             # (B, C, T)
        for block in self.blocks:
            x = x + block(x)               # residual connection
        x = self.pool(x).squeeze(-1)       # (B, C)
        return self.drop(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FlowClassifier(nn.Module):
    """
    ResNet18/34/50 (or DenseNet121) frame encoder + weighted temporal pooling + classifier.

    Forward input:  (B, T, 3, 224, 224)
    Forward output: (B, num_classes) logits

    Backbone options:
        resnet18  → feature dim 512  (default, best for small datasets)
        resnet34  → feature dim 512
        resnet50  → feature dim 2048
        densenet121 → feature dim 1024
    """

    # Feature dimensions per backbone
    _FEATURE_DIMS: dict[str, int] = {
        "resnet18":    512,
        "resnet34":    512,
        "resnet50":    2048,
        "densenet121": 1024,
    }

    def __init__(
        self,
        num_classes:      int,
        pretrained:       bool = True,
        freeze_backbone:  bool = True,
        backbone:         str  = "resnet18",
        use_dann:         bool = False,
        use_temporal_attn: bool = False,
        use_lstm:         bool = False,
        lstm_hidden:      int  = 128,
        lstm_layers:      int  = 1,
        use_gru:          bool = False,
        gru_hidden:       int  = 256,
        gru_layers:       int  = 1,
        use_tcn:          bool = False,
        tcn_channels:     int  = 256,
    ) -> None:
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        backbone = backbone.lower()

        if backbone == "resnet18":
            # ResNet18: 11M params, fastest, most stable on small datasets
            net = models.resnet18(weights=weights)
            net.fc = nn.Identity()          # remove final FC; output is (B, 512)
            self.backbone = net
        elif backbone == "resnet34":
            net = models.resnet34(weights=weights)
            net.fc = nn.Identity()          # output is (B, 512)
            self.backbone = net
        elif backbone == "resnet50":
            net = models.resnet50(weights=weights)
            net.fc = nn.Identity()          # output is (B, 2048)
            self.backbone = net
        elif backbone == "densenet121":
            net = models.densenet121(weights=weights)
            # Keep DenseNet feature extractor + manual pool for compatibility
            self._densenet_features = net.features
            self._densenet_pool     = GeM()
            self.backbone = None            # signals densenet path in forward
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose: resnet18, resnet34, resnet50, densenet121")

        self._backbone_name = backbone
        self.feature_dim = self._FEATURE_DIMS[backbone]

        # ── Freeze backbone ────────────────────────────────────────────────
        if freeze_backbone:
            params = (
                self._densenet_features.parameters()
                if self.backbone is None
                else self.backbone.parameters()
            )
            for param in params:
                param.requires_grad = False

        # ── Temporal aggregation ─────────────────────────────────────────
        self.use_temporal_attn = use_temporal_attn
        self.use_lstm = use_lstm
        self.use_gru = use_gru
        self.use_tcn = use_tcn
        if use_temporal_attn:
            self.temporal_pool = TemporalAttentionPool(
                dim=self.feature_dim, num_heads=4, dropout=0.1
            )
            self.lstm_pool = None  # type: ignore[assignment]
            self.gru_pool  = None  # type: ignore[assignment]
            self.tcn_pool  = None  # type: ignore[assignment]
        if use_lstm:
            self.lstm_pool = LSTMTemporalPool(
                input_dim=self.feature_dim,
                hidden_dim=lstm_hidden,
                num_layers=lstm_layers,
                dropout=0.5,
            )
            self.temporal_pool = None  # type: ignore[assignment]
            self.gru_pool      = None  # type: ignore[assignment]
            self.tcn_pool      = None  # type: ignore[assignment]
        elif use_tcn:
            self.tcn_pool = TemporalTCN(
                input_dim=self.feature_dim,
                channels=tcn_channels,
                dropout=0.2,
            )
            self.temporal_pool = None  # type: ignore[assignment]
            self.lstm_pool     = None  # type: ignore[assignment]
            self.gru_pool      = None  # type: ignore[assignment]
        else:
            self.temporal_pool = None  # type: ignore[assignment]
            self.lstm_pool     = None  # type: ignore[assignment]
            self.gru_pool      = None  # type: ignore[assignment]
            self.tcn_pool      = None  # type: ignore[assignment]

        # Head input dim depends on pooling mode
        if use_lstm:
            head_dim = lstm_hidden * 2
        elif use_gru:
            head_dim = gru_hidden * 2
        elif use_tcn:
            head_dim = tcn_channels
        else:
            head_dim = self.feature_dim

        # ── Classifier head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(head_dim, num_classes),
        )

        # ── Domain classifier (DANN) ───────────────────────────────────────
        self.use_dann = use_dann
        if use_dann:
            self._grl = GradientReversalLayer()
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            )
        else:
            self._grl = None
            self.domain_classifier = None

    def forward(
        self,
        x: torch.Tensor,
        dann_lambda: float = 1.0,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """
        x: (B, T, 3, 224, 224)
        Returns class logits, or (class_logits, domain_logits) when use_dann and training.
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)           # (B*T, 3, 224, 224)

        # ── Feature extraction ─────────────────────────────────────────────
        if self.backbone is not None:
            # ResNet path: fc=Identity, so output is already (B*T, 512)
            feat = self.backbone(x_flat)
        else:
            # DenseNet path
            feat = self._densenet_features(x_flat)
            feat = F.relu(feat, inplace=True)
            feat = self._densenet_pool(feat).flatten(1)  # (B*T, 1024)

        # ── Temporal aggregation ─────────────────────────────────────────
        feat = feat.view(B, T, -1)                 # (B, T, feature_dim)
        if self.use_temporal_attn and self.temporal_pool is not None:
            feat = self.temporal_pool(feat)        # (B, feature_dim)
        elif self.use_lstm and self.lstm_pool is not None:
            feat = self.lstm_pool(feat)            # (B, hidden_dim*2)
        elif self.use_gru and self.gru_pool is not None:
            feat = self.gru_pool(feat)             # (B, gru_hidden*2)
        elif self.use_tcn and self.tcn_pool is not None:
            feat = self.tcn_pool(feat)             # (B, tcn_channels)
        else:
            # Weighted mean-pool (linear ramp: later frames weighted more)
            w = torch.linspace(0.5, 1.5, steps=T, device=feat.device, dtype=feat.dtype)
            w = w / w.sum()
            feat = (feat * w.view(1, T, 1)).sum(dim=1)  # (B, feature_dim)

        class_logits = self.classifier(feat)

        if self.use_dann and self.training and self._grl is not None:
            rev_feat = self._grl(feat, dann_lambda)
            domain_logits = self.domain_classifier(rev_feat)  # type: ignore[misc]
            return class_logits, domain_logits

        return class_logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the pre-classifier embedding (B, feat_dim) for a batch of
        sequences x: (B, T, 3, 224, 224).
        Used by CenterLoss and SupConLoss.
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        if self.backbone is not None:
            feat = self.backbone(x_flat)
        else:
            feat = self._densenet_features(x_flat)
            feat = F.relu(feat, inplace=True)
            feat = self._densenet_pool(feat).flatten(1)

        feat = feat.view(B, T, -1)
        if self.use_temporal_attn and self.temporal_pool is not None:
            feat = self.temporal_pool(feat)
        elif self.use_lstm and self.lstm_pool is not None:
            feat = self.lstm_pool(feat)
        elif self.use_gru and self.gru_pool is not None:
            feat = self.gru_pool(feat)
        elif self.use_tcn and self.tcn_pool is not None:
            feat = self.tcn_pool(feat)
        else:
            w = torch.linspace(0.5, 1.5, steps=T, device=feat.device, dtype=feat.dtype)
            w = w / w.sum()
            feat = (feat * w.view(1, T, 1)).sum(dim=1)

        return feat

    def unfreeze_backbone(self) -> None:
        """Enable fine-tuning of the backbone."""
        params = (
            self._densenet_features.parameters()
            if self.backbone is None
            else self.backbone.parameters()
        )
        for param in params:
            param.requires_grad = True

    def freeze_backbone(self) -> None:
        """Disable backbone updates and keep training only temporal/head modules."""
        params = (
            self._densenet_features.parameters()
            if self.backbone is None
            else self.backbone.parameters()
        )
        for param in params:
            param.requires_grad = False

    def parameter_groups(self, head_lr: float, backbone_lr_mult: float) -> list[dict[str, object]]:
        """Return optimizer param groups with differential LR for backbone/head."""
        if self.backbone is None:
            backbone_iter = self._densenet_features.parameters()
        else:
            backbone_iter = self.backbone.parameters()

        backbone_params = [p for p in backbone_iter if p.requires_grad]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]
        if self.domain_classifier is not None:
            head_params += [p for p in self.domain_classifier.parameters() if p.requires_grad]
        if self.temporal_pool is not None:
            head_params += [p for p in self.temporal_pool.parameters() if p.requires_grad]
        if self.lstm_pool is not None:
            head_params += [p for p in self.lstm_pool.parameters() if p.requires_grad]
        if self.gru_pool is not None:
            head_params += [p for p in self.gru_pool.parameters() if p.requires_grad]
        if self.tcn_pool is not None:
            head_params += [p for p in self.tcn_pool.parameters() if p.requires_grad]

        groups: list[dict[str, object]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": head_lr * backbone_lr_mult})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups


# Keep alias for backward compatibility with saved checkpoints
DenseNetFlowClassifier = FlowClassifier


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sample lambda from Beta(alpha, alpha) and mix a random pair in the batch."""
    lam: float = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def set_batchnorm_eval(module: nn.Module) -> None:
    """Freeze BatchNorm running stats while keeping affine params trainable."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA update for parameters; copy buffers (e.g., BN stats) from current model."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b.detach())


def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    mixup_alpha: float = 0.0,
    grad_clip_norm: float = 0.0,
    freeze_bn: bool = False,
    ema_model: nn.Module | None = None,
    ema_decay: float = 0.0,
    center_loss: nn.Module = None,
    center_loss_weight: float = 0.0,
    supcon_loss: nn.Module = None,
    supcon_loss_weight: float = 0.0,
    penalty_matrix: torch.Tensor | None = None,
    supcon_domain_aware: bool = False,
    domain_label_map: dict[int, str] | None = None,
) -> tuple[float, float]:
    model.train()
    if freeze_bn:
        set_batchnorm_eval(model)
    total_loss, correct, total = 0.0, 0, 0
    total_center_loss = 0.0
    total_supcon_loss = 0.0

    if penalty_matrix is None:
        penalty_matrix = get_confusion_penalty_matrix(get_classifier_out_features(model))
    for batch in loader:
        # Support both (frames, labels) and (frames, labels, domain_id) from DANN loader
        if len(batch) == 3:
            frames, labels, domain_ids = batch
        else:
            frames, labels = batch
            domain_ids = None
        frames = frames.to(device, non_blocking=True)   # (B, T, 3, 224, 224)
        labels = labels.to(device, non_blocking=True)   # (B,)

        if mixup_alpha > 0.0 and frames.size(0) > 1:
            frames, y_a, y_b, lam = mixup_data(frames, labels, mixup_alpha)
            optimizer.zero_grad()
            logits = model(frames)
            # Usar la función de pérdida personalizada con mixup
            loss   = lam * custom_cross_entropy_with_penalty(logits, y_a, penalty_matrix) + (1.0 - lam) * custom_cross_entropy_with_penalty(logits, y_b, penalty_matrix)
            center_loss_val = 0.0
        else:
            optimizer.zero_grad()
            logits = model(frames)
            loss   = custom_cross_entropy_with_penalty(logits, labels, penalty_matrix)
            center_loss_val = 0.0
            if center_loss is not None and center_loss_weight > 0.0:
                if hasattr(model, 'get_embedding'):
                    emb = model.get_embedding(frames)
                    center_loss_val = center_loss(emb, labels)
                    loss = loss + center_loss_weight * center_loss_val
            supcon_loss_val = 0.0
            if supcon_loss is not None and supcon_loss_weight > 0.0:
                if hasattr(model, 'get_embedding'):
                    emb = model.get_embedding(frames) if center_loss is None or center_loss_weight <= 0.0 else emb
                    # Domain mask: only same-domain pairs if enabled
                    domain_tensor = None
                    if supcon_domain_aware and domain_ids is not None:
                        domain_tensor = domain_ids.to(device, non_blocking=True)
                    supcon_val = supcon_loss(emb, labels, domain_labels=domain_tensor)
                    loss = loss + supcon_loss_weight * supcon_val
                    supcon_loss_val = supcon_val.item() if hasattr(supcon_val, 'item') else float(supcon_val)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()
        if ema_model is not None and ema_decay > 0.0:
            update_ema_model(ema_model, model, ema_decay)

        total_loss += loss.item() * labels.size(0)
        total_center_loss += float(center_loss_val) * labels.size(0)
        total_supcon_loss += float(supcon_loss_val) * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    if total > 0 and total_center_loss > 0.0:
        print(f"[center_loss] Promedio epoch: {total_center_loss / total:.4f}")
    if total > 0 and total_supcon_loss > 0.0:
        print(f"[supcon_loss] Promedio epoch: {total_supcon_loss / total:.4f}")
    return total_loss / total, correct / total


def train_one_epoch_dann(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    dann_lambda: float,
    mixup_alpha: float = 0.0,
    grad_clip_norm: float = 0.0,
    freeze_bn: bool = False,
    ema_model: nn.Module | None = None,
    ema_decay: float = 0.0,
) -> tuple[float, float]:
    """Training step with Domain-Adversarial loss.

    Loader must yield (frames, class_labels, domain_ids) where domain_ids
    are 0 for primary (CASME2) and 1 for extra (MEME).
    """
    model.train()
    if freeze_bn:
        set_batchnorm_eval(model)
    total_cls_loss, correct, total = 0.0, 0, 0

    for frames, labels, domain_ids in loader:
        frames     = frames.to(device, non_blocking=True)
        labels     = labels.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        if mixup_alpha > 0.0 and frames.size(0) > 1:
            frames, y_a, y_b, lam = mixup_data(frames, labels, mixup_alpha)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad()
        result = model(frames, dann_lambda=dann_lambda)
        if isinstance(result, tuple):
            class_logits, domain_logits = result
        else:
            class_logits = result
            domain_logits = None

        cls_loss = mixup_criterion(criterion, class_logits, y_a, y_b, lam)
        if domain_logits is not None:
            dom_loss = F.cross_entropy(domain_logits, domain_ids)
            # GRL already scales gradients with dann_lambda.
            loss = cls_loss + dom_loss
        else:
            loss = cls_loss

        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()
        if ema_model is not None and ema_decay > 0.0:
            update_ema_model(ema_model, model, ema_decay)

        total_cls_loss += cls_loss.item() * labels.size(0)
        correct        += (class_logits.argmax(dim=1) == labels).sum().item()
        total          += labels.size(0)

    return total_cls_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    num_classes: int,
) -> tuple[float, float, float, float, float, float, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    penalty_matrix = get_confusion_penalty_matrix(model.classifier.out_features if hasattr(model, 'classifier') else model.fc.out_features)
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames)
        loss   = custom_cross_entropy_with_penalty(logits, labels, penalty_matrix)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    cm       = confusion_matrix_np(y_true, y_pred, num_classes)
    macro_f1 = macro_f1_from_confusion(cm)
    macro_f1_present = macro_f1_present_from_confusion(cm)
    macro_precision, macro_recall = macro_precision_recall_from_confusion(cm)
    return total_loss / total, correct / total, macro_f1, macro_f1_present, macro_precision, macro_recall, cm


@torch.no_grad()
def evaluate_with_tta(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    num_classes: int,
    tta_passes: int = 4,
    feli_threshold: float = 0.0,
    feli_class_idx: int | None = None,
) -> tuple[float, float, float, float, float, float, np.ndarray]:
    """Evaluate with Test-Time Augmentation.

    For passes 1-4: deterministic views (original + flips).
    For passes > 4: the first 4 deterministic views are kept, then
    (tta_passes - 4) additional stochastic augmentation passes are averaged in,
    reducing prediction variance by ~1/sqrt(tta_passes).

    If feli_threshold > 0 and feli_class_idx is set, prediction for felicidad is
    overridden: if softmax[feli_class_idx] >= feli_threshold → predict feli.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    det_views = max(1, min(tta_passes, 4))
    stoch_passes = max(0, tta_passes - 4)

    def _deterministic_views(frames: torch.Tensor) -> list[torch.Tensor]:
        views: list[torch.Tensor] = [frames]
        if det_views >= 2:
            hf = frames.flip(-1).clone()
            hf[:, :, 0] = -hf[:, :, 0]
            views.append(hf)
        if det_views >= 3:
            vf = frames.flip(-2).clone()
            vf[:, :, 1] = -vf[:, :, 1]
            views.append(vf)
        if det_views >= 4:
            hvf = frames.flip(-1).flip(-2).clone()
            hvf[:, :, 0] = -hvf[:, :, 0]
            hvf[:, :, 1] = -hvf[:, :, 1]
            views.append(hvf)
        return views

    _stoch_aug = build_train_aug_transforms(magnitude=0.5)  # leve para TTA

    def _stoch_view(frames: torch.Tensor) -> torch.Tensor:
        """Apply light stochastic augmentation per frame (CPU clone)."""
        B, T, _, _, _ = frames.shape
        out = frames.clone().cpu()
        for b in range(B):
            for t in range(T):
                # Keep Tensor path to avoid PIL->Tensor assignment mismatch.
                out[b, t] = _stoch_aug(out[b, t])
        return out.to(frames.device)

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        avg_probs = torch.zeros(labels.size(0), num_classes, device=device)
        batch_loss = 0.0
        n_total_views = 0

        for i, view in enumerate(_deterministic_views(frames)):
            logits = model(view)
            avg_probs += torch.softmax(logits, dim=1)
            n_total_views += 1
            if i == 0:
                batch_loss = criterion(logits, labels).item()

        for _ in range(stoch_passes):
            sv = _stoch_view(frames)
            logits = model(sv)
            avg_probs += torch.softmax(logits, dim=1)
            n_total_views += 1

        avg_probs /= n_total_views

        if feli_threshold > 0.0 and feli_class_idx is not None:
            preds = avg_probs.argmax(dim=1)
            feli_mask = avg_probs[:, feli_class_idx] >= feli_threshold
            preds[feli_mask] = feli_class_idx
        else:
            preds = avg_probs.argmax(dim=1)

        total_loss += batch_loss * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    cm       = confusion_matrix_np(y_true, y_pred, num_classes)
    macro_f1 = macro_f1_from_confusion(cm)
    macro_f1_present = macro_f1_present_from_confusion(cm)
    macro_precision, macro_recall = macro_precision_recall_from_confusion(cm)
    return total_loss / total, correct / total, macro_f1, macro_f1_present, macro_precision, macro_recall, cm


@torch.no_grad()
def evaluate_with_tta_threshold_search(
    model:           nn.Module,
    loader:          DataLoader,
    criterion:       nn.Module,
    device:          torch.device,
    num_classes:     int,
    tta_passes:      int,
    feli_class_idx:  int,
    thresholds:      list[float] | None = None,
) -> float:
    """Grid-search the optimal felicidad threshold on the val set.

    Returns the best threshold found (maximizes macro-F1-present).
    Prints a table of threshold → F1 for transparency.
    """
    if thresholds is None:
        thresholds = [round(x * 0.05, 2) for x in range(5, 11)]  # 0.25 .. 0.50

    best_thresh = 0.0
    best_f1 = -1.0
    print("[feli_thresh_search] grid-searching felicidad threshold on val set ...")
    for thresh in thresholds:
        _, _, _, f1_p, _, _, _ = evaluate_with_tta(
            model, loader, criterion, device, num_classes,
            tta_passes=tta_passes,
            feli_threshold=thresh,
            feli_class_idx=feli_class_idx,
        )
        marker = " ←" if f1_p > best_f1 else ""
        print(f"[feli_thresh_search]  thresh={thresh:.2f}  macro-f1-present={f1_p:.4f}{marker}")
        if f1_p > best_f1:
            best_f1 = f1_p
            best_thresh = thresh
    print(f"[feli_thresh_search] best threshold={best_thresh:.2f} (macro-f1-present={best_f1:.4f})")
    return best_thresh


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DenseNet121 + temporal mean-pool on optical-flow sequences."
    )
    parser.add_argument('--use_center_loss', action='store_true', help='Añade center loss auxiliar para separar clases en el embedding')
    parser.add_argument('--center_loss_weight', type=float, default=0.1, help='Peso de la center loss respecto a la loss principal')
    parser.add_argument('--use_supcon_loss', action='store_true',
                        help='Supervised Contrastive Loss para mejorar separabilidad entre clases confundidas.')
    parser.add_argument('--supcon_loss_weight', type=float, default=0.3,
                        help='Peso de SupCon loss (default: 0.3).')
    parser.add_argument('--supcon_temperature', type=float, default=0.07,
                        help='Temperatura del SupCon loss (default: 0.07).')
    parser.add_argument(
        "--casme3_classes",
        type=str,
        default=None,
        help="Comma-separated list of class names to include from CASME3 extra CSV. If not set, all classes in the CSV are loaded. Example: 'enojo,miedo,tristeza'.",
    )
    parser.add_argument("--data_dir",    default=Config.data_dir)
    parser.add_argument("--csv_index",   default=Config.csv_index)
    parser.add_argument("--onset_csv",   default=Config.onset_csv)
    parser.add_argument("--num_frames",  type=int,   default=Config.num_frames)
    parser.add_argument("--epochs",      type=int,   default=Config.epochs)
    parser.add_argument("--batch_size",  type=int,   default=Config.batch_size)
    parser.add_argument("--lr",          type=float, default=Config.lr)
    parser.add_argument("--weight_decay",type=float, default=Config.weight_decay)
    parser.add_argument("--val_split",   type=float, default=Config.val_split)
    parser.add_argument(
        "--val_per_class",
        type=int,
        default=None,
        help=(
            "If set, reserve exactly this many samples per class for validation "
            "(used with --stratified_class_split). Overrides --val_split for the "
            "stratified split. Classes with fewer samples keep at least 1 in val "
            "and at least 1 in train."
        ),
    )
    parser.add_argument(
        "--val_split_by_class",
        type=str,
        default="",
        help="Optional class-specific val split overrides (only with --stratified_class_split). "
             "Format: 'class:ratio,class:ratio' with ratio in (0,1). "
             "Example: 'enojo:0.44,miedo:0.46,tristeza:0.45'.",
    )
    parser.add_argument(
        "--split_mode",
        choices=["subject", "random", "stratified_random", "loso", "subject_kfold"],
        default=Config.split_mode,
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds when --split_mode=subject_kfold.",
    )
    parser.add_argument(
        "--balanced_sampler",
        action="store_true",
        default=Config.balanced_sampler,
    )
    parser.add_argument(
        "--no_balanced_sampler",
        dest="balanced_sampler",
        action="store_false",
    )
    parser.add_argument("--seed",        type=int,   default=Config.seed)
    parser.add_argument("--num_workers", type=int,   default=Config.num_workers)
    parser.add_argument(
        "--input_path",
        default=Config.input_path,
        help="Base path with extracted data. Used when --data_dir/--csv_index are defaults.",
    )
    parser.add_argument(
        "--unfreeze_epoch", type=int, default=Config.unfreeze_epoch,
        help="Epoch at which to unfreeze DenseNet backbone (0 = never).",
    )
    parser.add_argument(
        "--auto_unfreeze_on_plateau",
        action="store_true",
        default=False,
        help="Automatically unfreeze backbone when validation monitor metric plateaus.",
    )
    parser.add_argument(
        "--auto_unfreeze_patience",
        type=int,
        default=6,
        help="Plateau patience (epochs without monitor improvement) before auto-unfreeze.",
    )
    parser.add_argument(
        "--auto_unfreeze_min_epochs",
        type=int,
        default=8,
        help="Minimum epoch before auto-unfreeze can trigger.",
    )
    parser.add_argument(
        "--auto_refreeze_on_plateau",
        action="store_true",
        default=False,
        help="Automatically refreeze backbone if monitor metric keeps plateauing after unfreeze.",
    )
    parser.add_argument(
        "--auto_refreeze_patience",
        type=int,
        default=8,
        help="Plateau patience (epochs without monitor improvement) before auto-refreeze.",
    )
    parser.add_argument(
        "--auto_refreeze_min_epochs_since_unfreeze",
        type=int,
        default=6,
        help="Minimum epochs to wait after unfreeze before auto-refreeze can trigger.",
    )
    parser.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=0.2,
        help="Backbone LR multiplier after unfreezing (final backbone LR = lr * multiplier).",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        default=False,
        help="Train backbone from scratch (no pretrained weights).",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50", "densenet121"],
        default="resnet18",
        help="Backbone architecture (default: resnet18).",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        default=False,
        help="Use Focal Loss instead of Cross-Entropy Loss (helps with imbalance).",
    )
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        default=False,
        help="Alias for --use_focal_loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss (higher = more focus on hard examples).",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        default=False,
        help="Disable class weighting in loss (useful with balanced sampler).",
    )
    parser.add_argument(
        "--class_weight_cap",
        type=float,
        default=2.8,
        help="Maximum per-class weight after normalization (<=0 disables cap).",
    )
    parser.add_argument(
        "--class_weight_boost",
        type=str,
        default="",
        help="Comma-separated class:multiplier pairs to boost specific class weights after auto-computation. "
             "E.g. 'enojo:2.0,miedo:2.5,asco:1.8'. Class names must match the dataset label map.",
    )
    parser.add_argument(
        "--adaptive_class_weights",
        action="store_true",
        default=False,
        help="Adapt class weights from validation confusion matrix (focus low-recall classes).",
    )
    parser.add_argument(
        "--adaptive_cw_warmup_epochs",
        type=int,
        default=8,
        help="Epochs before adaptive class weights start updating.",
    )
    parser.add_argument(
        "--adaptive_cw_update_every",
        type=int,
        default=1,
        help="Update adaptive class weights every N epochs.",
    )
    parser.add_argument(
        "--adaptive_cw_target_recall",
        type=float,
        default=0.45,
        help="Target per-class recall used to boost low-recall classes.",
    )
    parser.add_argument(
        "--adaptive_cw_alpha",
        type=float,
        default=0.8,
        help="Strength of adaptive class-weight boost.",
    )
    parser.add_argument(
        "--adaptive_cw_ema_beta",
        type=float,
        default=0.7,
        help="EMA beta for per-class recall smoothing (0=no smoothing, 0.7 recommended).",
    )
    parser.add_argument(
        "--adaptive_cw_max_multiplier",
        type=float,
        default=2.0,
        help="Maximum multiplier applied to any class weight by adaptive updates.",
    )
    parser.add_argument(
        "--adaptive_cw_min_val_support",
        type=int,
        default=4,
        help="Minimum validation samples required to trust a class recall in current epoch.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for CrossEntropyLoss (0.0 disables).",
    )
    parser.add_argument(
        "--temporal_aug_prob",
        type=float,
        default=0.5,
        help="Probability of applying temporal augmentation to sequences (0.0-1.0).",
    )
    parser.add_argument(
        "--dual_phase_flow",
        action="store_true",
        default=False,
        help=(
            "Enable dual-phase frame extraction: allocate num_frames//2 to onset→apex "
            "and num_frames//2 to apex→offset, preserving phase boundary. "
            "Requires apex_frame annotations in the CSV. Falls back to standard "
            "sampling when apex info is unavailable. Based on FMANet (2025)."
        ),
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=Config.early_stop_patience,
        help="Early stopping patience on val macro-f1 (0 disables early stopping).",
    )
    parser.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=Config.early_stop_min_epochs,
        help="Minimum epochs to run before early stopping can trigger.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=Config.ema_decay,
        help="EMA decay for model weights (0 disables EMA, typical: 0.999).",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=Config.grad_clip_norm,
        help="Gradient clipping max-norm (0 disables clipping).",
    )
    parser.add_argument(
        "--freeze_bn_after_unfreeze",
        action="store_true",
        default=Config.freeze_bn_after_unfreeze,
        help="Keep BatchNorm layers in eval mode after backbone unfreeze to improve stability with small batches.",
    )
    parser.add_argument(
        "--extra_data_dir",
        default=None,
        help="Path to extracted_flow-style dataset (subject/camera/emotion.npy). "
             "Appended to the main dataset before splitting.",
    )
    parser.add_argument(
        "--extra_csv_index",
        default=None,
        help="Optional CSV index for an extra dataset (same format as --csv_index). "
             "Rows are appended to the main dataset before splitting.",
    )
    parser.add_argument(
        "--smic_csv_index",
        default=None,
        help="Optional CSV index for SMIC data. Loaded as domain='smic'.",
    )
    parser.add_argument(
        "--smic_only_surprise",
        action="store_true",
        default=False,
        help="When used with --smic_csv_index, keeps only SMIC surprise class samples.",
    )
    parser.add_argument(
        "--meme_csv_index",
        default=None,
        help="Optional CSV index for MEME data. Loaded as domain='meme'. "
             "If set, it takes precedence over --extra_csv_index for MEME workflows.",
    )
    parser.add_argument(
        "--meme_csv_index_2",
        default=None,
        help="Optional second MEME CSV index (e.g., Eulerian version). "
             "Loaded as domain='meme_2' and appended to --meme_csv_index.",
    )
    parser.add_argument(
        "--neutral_csv_index",
        default=None,
        help="Optional CSV index loaded as neutral-only source (e.g., MEME v2 neutral).",
    )
    parser.add_argument(
        "--only_ekman7",
        action="store_true",
        default=True,
        help="Keep only Ekman-7 labels (includes neutral; default enabled).",
    )
    parser.add_argument(
        "--allow_non_ekman",
        dest="only_ekman7",
        action="store_false",
        help="Disable Ekman filtering and keep all detected labels.",
    )
    parser.add_argument(
        "--exclude_ekman7_classes",
        type=str,
        default="",
        help=(
            "Comma-separated Ekman-7 class names to exclude AFTER canonicalization. "
            "Example: 'miedo,tristeza'. Neutral cannot be excluded. "
            "Only applied when --only_ekman7 is enabled."
        ),
    )
    parser.add_argument(
        "--oversample_minority",
        action="store_true",
        default=False,
        help="Automatically duplicate minority class samples until each class "
             "reaches --min_samples_per_class (online augmentation via random __getitem__).",
    )
    parser.add_argument(
        "--undersample_majority",
        action="store_true",
        default=False,
        help="Randomly reduce majority classes in the TRAIN split before oversampling/sampling.",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="Absolute cap per class when --undersample_majority is enabled and auto target is off (0 disables absolute cap).",
    )
    parser.add_argument(
        "--auto_majority_target",
        action="store_true",
        default=False,
        help="Automatically compute an undersampling cap from the train class-count percentile.",
    )
    parser.add_argument(
        "--undersample_target_percentile",
        type=float,
        default=75.0,
        help="Percentile of train class counts used as auto undersampling cap (10-90).",
    )
    parser.add_argument(
        "--undersample_min_keep",
        type=int,
        default=1,
        help="Minimum real samples to keep per class when undersampling majority classes.",
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=20,
        help="Minimum samples per class when --oversample_minority is enabled (default: 20).",
    )
    parser.add_argument(
        "--minority_aug_strength",
        type=float,
        default=0.6,
        help="Augmentation intensity for synthetic minority samples (0.0-1.0).",
    )
    parser.add_argument(
        "--auto_minority_target",
        action="store_true",
        default=False,
        help="Automatically compute minority oversampling targets from train class distribution.",
    )
    parser.add_argument(
        "--oversample_target_percentile",
        type=float,
        default=50.0,
        help="Percentile of train class counts used as auto oversampling target (10-90).",
    )
    parser.add_argument(
        "--oversample_max_multiplier",
        type=float,
        default=1.0,
        help="Per-class cap for auto oversampling: max target = count * (1 + multiplier).",
    )
    parser.add_argument(
        "--oversample_global_aug_ratio",
        type=float,
        default=0.5,
        help="Global cap: maximum synthetic samples as ratio of original train size.",
    )
    parser.add_argument(
        "--tta_passes",
        type=int,
        default=1,
        help="Test-Time Augmentation passes (1=disabled, 2+=enabled). "
             "Passes 2-4 use deterministic views (hflip, vflip, both). "
             "Passes >4 repeat stochastic augmentation N times and average softmax.",
    )
    parser.add_argument(
        "--confusion_penalty_matrix",
        type=str,
        default="",
        help="Custom confusion penalty entries: 'true_cls:pred_cls:value,...'. "
             "E.g. 'feli:neutral:2.0,asco:neutral:1.5,feli:asco:1.5'. "
             "Class aliases: feli=felicidad, ang=enojo, dis=asco, neu=neutral.",
    )
    parser.add_argument(
        "--per_class_weight_cap",
        type=str,
        default="",
        help="Per-class weight cap overrides: 'class:cap,...'. "
             "E.g. 'neutral:50.0,felicidad:150.0,asco:150.0'. "
             "Overrides --class_weight_cap for specific classes after auto-computation.",
    )
    parser.add_argument(
        "--feli_threshold",
        type=float,
        default=0.0,
        help="If >0, override the decision threshold for felicidad class during evaluation. "
             "If softmax[feli] >= feli_threshold, predict felicidad. Default=0 (uses argmax).",
    )
    parser.add_argument(
        "--feli_threshold_search",
        action="store_true",
        help="After training, grid-search the optimal felicidad threshold on val set [0.25..0.50].",
    )
    parser.add_argument(
        "--threshold_search_only",
        action="store_true",
        help="Run ONLY felicidad threshold search on validation from an existing checkpoint (no training).",
    )
    parser.add_argument(
        "--threshold_search_ckpt",
        type=str,
        default="",
        help="Checkpoint path used with --threshold_search_only. If empty, uses default best_{backbone}_flow_{fold_tag}.pth.",
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        default=False,
        help="Two-stage training: Stage 1 pretrain on extra (MEME) data, "
             "Stage 2 fine-tune on primary (CASME2) data with lr*0.3. "
             "Requires --extra_csv_index.",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=20,
        help="Epochs for Stage 1 pretraining on the extra dataset (used with --two_stage).",
    )
    parser.add_argument(
        "--domain_balanced_sampler",
        action="store_true",
        default=False,
        help="Balance batches equally across domains (primary vs extra) instead of by class.",
    )
    parser.add_argument(
        "--primary_only_val",
        action="store_true",
        default=False,
        help="Primary-driven validation mode. Splits CASME2 for train/val and "
             "injects only classes that are missing or scarce in val from MEME.",
    )
    parser.add_argument(
        "--hybrid_val_min_per_class",
        type=int,
        default=1,
        help="Minimum validation samples per class in --primary_only_val mode. "
             "If val has fewer than this threshold for a class, samples are added "
             "from extra-domain (MEME). Default=1 (only missing classes).",
    )
    parser.add_argument(
        "--domain_adversarial",
        action="store_true",
        default=False,
        help="Enable Domain-Adversarial Neural Network (DANN) training. "
             "Adds a gradient-reversal domain classifier so the backbone learns "
             "domain-invariant features. Requires both primary and extra data.",
    )
    parser.add_argument(
        "--dann_lambda",
        type=float,
        default=0.5,
        help="Max GRL lambda for DANN (default: 0.5). "
             "Higher = stronger domain alignment pressure.",
    )
    parser.add_argument(
        "--monitor_metric",
        choices=["macro_f1", "macro_f1_present"],
        default="macro_f1_present",
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--casme2_train_meme_val",
        action="store_true",
        default=False,
        help="Inverse split mode: train on all CASME2 (primary), validate on MEME (extra). "
             "Use --min_samples_per_class to augment rare CASME2 classes in training. "
             "Mutually exclusive with --primary_only_val.",
    )
    parser.add_argument(
        "--stratified_class_split",
        action="store_true",
        default=False,
        help="Use class-stratified split: split all data 80/20 (train/val) with each class "
             "proportionally represented in both train and val. "
             "Recommended for small datasets with class imbalance. "
             "Mutually exclusive with --casme2_train_meme_val.",
    )
    parser.add_argument(
        "--meme_val_min_per_class",
        type=int,
        default=1,
        help="Minimum validation samples per class in --casme2_train_meme_val mode. "
             "If MEME val has fewer samples for a class, pads from MEME train. Default=1.",
    )
    parser.add_argument(
        "--min_samples_per_class_casme2",
        type=int,
        default=0,
        help="Minimum samples per class in CASME2 training. If < this, complement with MEME samples. "
             "Only used with --casme2_train_meme_val. Default=0 (disabled).",
    )
    parser.add_argument(
        "--lr_scheduler",
        choices=["none", "cosine", "cosine_restarts", "step"],
        default="cosine",
        help="LR scheduler (default: cosine). 'cosine'=CosineAnnealingLR over all epochs, "
             "'cosine_restarts'=CosineAnnealingWarmRestarts (T_0=10), "
             "'step'=StepLR (decay x0.5 every 15 epochs), 'none'=flat LR.",
    )
    parser.add_argument(
        "--temporal_attn",
        action="store_true",
        default=False,
        help="Replace weighted mean-pool over frames with a self-attention temporal module. "
             "Learns which frames (onset/apex/offset) are most informative.",
    )
    parser.add_argument(
        "--use_lstm",
        action="store_true",
        default=False,
        help="Replace weighted mean-pool with a bidirectional LSTM temporal module. "
             "Captures onset→apex→offset temporal order. "
             "Mutually exclusive with --temporal_attn (LSTM takes priority if both set).",
    )
    parser.add_argument(
        "--lstm_hidden",
        type=int,
        default=128,
        help="Hidden size per direction for the bidirectional LSTM (default: 128). "
             "Head input = hidden*2. Smaller values reduce overfitting on small datasets.",
    )
    parser.add_argument(
        "--lstm_layers",
        type=int,
        default=1,
        help="Number of stacked LSTM layers (default: 1). "
             "Increase to 2 only if dataset is large enough (>1000 samples).",
    )
    parser.add_argument(
    "--happiness_limit",
    type=int,
    default=None,
    help="Límite máximo de muestras de felicidad (CASME3 extra). Si es None, no se limita."
    )
    parser.add_argument(
        "--use_gru",
        action="store_true",
        default=False,
        help="Replace weighted mean-pool with a bidirectional GRU temporal module. "
             "Mutually exclusive with --use_lstm and --temporal_attn (LSTM takes priority if both set).",
    )
    parser.add_argument(
        "--gru_hidden",
        type=int,
        default=256,
        help="Hidden size per direction for the bidirectional GRU (default: 256). "
             "Head input = hidden*2.",
    )
    parser.add_argument(
        "--gru_layers",
        type=int,
        default=1,
        help="Number of stacked GRU layers (default: 1).",
    )
    parser.add_argument(
        "--use_tcn",
        action="store_true",
        default=False,
        help="Replace weighted mean-pool with a dilated Temporal Convolutional Network (TCN). "
             "Uses 3 residual blocks with dilations 1/2/4 for multi-scale temporal context. "
             "Fewer parameters than GRU, no vanishing gradient. "
             "Mutually exclusive with --use_gru/--use_lstm/--temporal_attn.",
    )
    parser.add_argument(
        "--tcn_channels",
        type=int,
        default=256,
        help="Number of channels in the TCN residual blocks (default: 256). "
             "Head input dim = tcn_channels. Reduce to 128 to decrease overfitting.",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.0,
        help="Mixup alpha parameter (Beta distribution). 0=disabled. "
             "Recommended: 0.2-0.4 for small datasets.",
    )
    parser.add_argument(
        "--train_spatial_aug",
        action="store_true",
        default=False,
        help="Apply spatial augmentation (rotation, blur, erasing) to training frames. "
             "Augmentation is applied post-normalization on tensors. "
             "Safe for optical flow: no horizontal flip (avoids dx sign corruption).",
    )
    parser.add_argument(
        "--aug_magnitude",
        type=float,
        default=1.0,
        help="Augmentation intensity for --train_spatial_aug. "
             "0.5=light, 1.0=moderate (default), 1.5=aggressive.",
    )
    # ── Hierarchical Classification ────────────────────────────────────────
    parser.add_argument(
        "--hierarchical_classifier",
        action="store_true",
        default=False,
        help="Enable two-stage hierarchical classification: Stage 1 coarse grouping, "
             "Stage 2 fine-grained within groups. Improves separability for confused classes.",
    )
    parser.add_argument(
        "--hierarchy_strategy",
        choices=["predefined_emotions", "auto_from_confusion"],
        default="predefined_emotions",
        help="Strategy for hierarchy definition: 'predefined_emotions' uses hardcoded emotion groups, "
             "'auto_from_confusion' learns groups from validation confusion matrix.",
    )
    parser.add_argument(
        "--binary_refiners",
        action="store_true",
        default=False,
        help="Enable binary refiners for heavily confused class pairs. "
             "Detects confusion from validation matrix and trains specialized binaries.",
    )
    parser.add_argument(
        "--confusion_threshold",
        type=float,
        default=0.15,
        help="Confusion threshold for marking a pair as problematic (0.0-1.0). "
             "Higher=more strict (fewer pairs selected). Default: 0.15 (15% confusion rate).",
    )
    parser.add_argument(
        "--min_samples_binary_trainer",
        type=int,
        default=10,
        help="Minimum samples per class required to train a binary refiner. "
             "Skips pairs with fewer than this threshold per class. Default: 10.",
    )
    # ── Cascade Neutral ───────────────────────────────────────────────────
    parser.add_argument(
        "--cascade_neutral",
        action="store_true",
        default=False,
        help=(
            "Enable 2-stage cascade for the neutral class. "
            "Stage 1: binary neutral-vs-non-neutral detector trained on ALL data. "
            "Stage 2: Ekman-6 classifier trained on NON-NEUTRAL data only. "
            "At inference both stages are chained and evaluated on the val set with 7-class metrics. "
            "Must be used together with --stratified_class_split."
        ),
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL CLASSIFICATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def analyze_confusion_matrix(
    val_cm: np.ndarray,
    class_names: list[str],
    confusion_threshold: float = 0.15,
    min_samples: int = 10,
    class_counts: dict[str, int] | None = None,
) -> dict[str, list[tuple[str, str, float]]]:
    """
    Analyze validation confusion matrix to identify heavily confused class pairs.
    
    Returns dict mapping emotion -> list of (emotion1, emotion2, confusion_rate) tuples,
    ordered by confusion rate descending. Only includes pairs where confusion_rate >= threshold.
    
    confusion_rate = (mutual_confusion_count) / (total_predictions_for_class)
    """
    if val_cm.shape[0] == 0 or class_names is None or len(class_names) == 0:
        return {}
    
    confused_pairs: dict[str, list[tuple[str, str, float]]] = {name: [] for name in class_names}
    num_classes = len(class_names)
    
    # For each class, find classes it's confused with
    for i in range(num_classes):
        class_i = class_names[i]
        
        # Skip if insufficient training samples
        if class_counts and class_counts.get(class_i, 0) < min_samples:
            continue
        
        total_i = val_cm[i].sum()
        if total_i == 0:
            continue
        
        for j in range(num_classes):
            if i == j:
                continue
            class_j = class_names[j]
            
            # Skip if insufficient training samples
            if class_counts and class_counts.get(class_j, 0) < min_samples:
                continue
            
            # Count: times class_i was predicted as class_j
            mutual_confusion = val_cm[i, j] + val_cm[j, i]  # bidirectional confusion
            confusion_rate = float(mutual_confusion) / float(total_i) if total_i > 0 else 0.0
            
            if confusion_rate >= confusion_threshold:
                confused_pairs[class_i].append((class_i, class_j, confusion_rate))
    
    # Sort each class by confusion rate descending
    for class_i in confused_pairs:
        confused_pairs[class_i].sort(key=lambda x: x[2], reverse=True)
    
    return confused_pairs


def define_class_hierarchy(
    num_classes: int,
    class_names: list[str],
    strategy: str = "predefined_emotions",
) -> dict[str, list[str]]:
    """
    Define hierarchical grouping of emotion classes.
    
    Returns dict mapping group_name -> list of class_names in that group.
    
    Strategy 'predefined_emotions':
        - Positiva: [felicidad, sorpresa]
        - Negativa: [asco, miedo, enojo]
        - Neutral: [tristeza] (solo si existe)
    """
    hierarchy: dict[str, list[str]] = {}
    
    if strategy == "predefined_emotions":
        # Hardcoded emotion grouping based on valence/arousal.
        # NOTE: tristeza (sadness) is a NEGATIVE emotion — it must NOT be in the neutral group.
        # NOTE: neutral is its own semantic category, separate from all Ekman emotions.
        positiva_map = {"felicidad", "happiness", "happy", "hap", "sorpresa", "surprise", "sur"}
        negativa_map = {
            "asco", "disgust", "dis",
            "miedo", "fear", "fea",
            "enojo", "anger", "ang",
            "tristeza", "sadness", "sad",
        }
        neutral_map = {"neutral"}

        positiva = [c for c in class_names if c.lower() in positiva_map]
        negativa = [c for c in class_names if c.lower() in negativa_map]
        neutral = [c for c in class_names if c.lower() in neutral_map]

        if positiva:
            hierarchy["positiva"] = positiva
        if negativa:
            hierarchy["negativa"] = negativa
        if neutral:
            hierarchy["neutral"] = neutral

        # Fallback: anything still unmapped gets its own singleton group
        all_mapped = set(positiva) | set(negativa) | set(neutral)
        for c in class_names:
            if c not in all_mapped:
                hierarchy[f"other_{c}"] = [c]
    
    return hierarchy if hierarchy else {f"group_{i}": [class_names[i]] for i in range(num_classes)}


def train_binary_refiner(
    model: nn.Module,
    class_idx_1: int,
    class_idx_2: int,
    train_ds: Subset,
    val_ds: Subset,
    device: torch.device,
    num_epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 8,
    num_workers: int = 0,
) -> dict:
    """
    Train a binary classifier to distinguish between two specific classes.
    Returns dict with 'accuracy', 'f1', and 'loss' on the binary task.
    """
    # Filter train/val to only these two classes
    train_base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    val_base_ds = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
    
    train_indices = []
    for idx in (train_ds.indices if isinstance(train_ds, Subset) else range(len(train_ds))):
        label = train_base_ds.label_map[train_base_ds.samples[idx][1]]
        if label in (class_idx_1, class_idx_2):
            train_indices.append(idx)
    
    val_indices = []
    for idx in (val_ds.indices if isinstance(val_ds, Subset) else range(len(val_ds))):
        label = val_base_ds.label_map[val_base_ds.samples[idx][1]]
        if label in (class_idx_1, class_idx_2):
            val_indices.append(idx)
    
    if len(train_indices) < 5 or len(val_indices) < 2:
        return {"accuracy": 0.0, "f1": 0.0, "loss": 0.0, "skipped": True}
    
    binary_train_ds = Subset(train_base_ds, train_indices)
    binary_val_ds = Subset(val_base_ds, val_indices)
    
    binary_model = copy.deepcopy(model)
    binary_model.classifier = nn.Linear(binary_model.classifier.in_features, 2)
    binary_model = binary_model.to(device)
    
    binary_optimizer = torch.optim.AdamW(binary_model.parameters(), lr=lr, weight_decay=1e-2)
    binary_criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(binary_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(binary_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    best_f1 = 0.0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        binary_model.train()
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            # Map original labels to binary: class_idx_1->0, class_idx_2->1
            binary_labels = torch.where(
                labels == class_idx_1,
                torch.tensor(0, device=device),
                torch.tensor(1, device=device)
            )
            binary_optimizer.zero_grad()
            logits = binary_model(frames)
            loss = binary_criterion(logits, binary_labels)
            loss.backward()
            binary_optimizer.step()
        
        # Validate
        binary_model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                binary_labels = torch.where(
                    labels == class_idx_1,
                    torch.tensor(0, device=device),
                    torch.tensor(1, device=device)
                )
                logits = binary_model(frames)
                preds = logits.argmax(dim=1)
                y_true.extend(binary_labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, zero_division=0) if len(set(y_true)) > 1 else 0.0
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = copy.deepcopy(binary_model.state_dict())
    
    result = {"accuracy": float(acc), "f1": float(best_f1), "loss": 0.0, "skipped": False, "model": binary_model}
    return result


def apply_hierarchical_refinement(
    logits: torch.Tensor,
    hierarchy: dict[str, list[str]],
    class_names: list[str],
    binary_refiners: dict[tuple[int, int], nn.Module] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Apply hierarchical refinement to logits.
    
    If binary_refiners are available, uses them to refine predictions for confused pairs.
    Otherwise, just returns original logits.
    """
    if binary_refiners is None or len(binary_refiners) == 0:
        return logits
    
    # For each confused pair, apply binary refiner to refine the coarse prediction
    batch_size = logits.shape[0]
    refined_logits = logits.clone()
    
    for (class_idx_1, class_idx_2), binary_model in binary_refiners.items():
        binary_model.eval()
        # This requires the input frames, which we don't have here.
        # This is a placeholder; in practice, you'd need to pass frames through the pipeline.
        # For now, we skip this and just return original logits.
    
    return refined_logits


def run_single_split(
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
    train_ds: Subset,
    val_ds: Subset,
    fold_tag: str,
    full_dataset: FlowSequenceDataset = None,
) -> tuple[float, float]:
    """Train/evaluate one split and return best (acc, macro_f1)."""
    _train_start_wall = time.perf_counter()
    # ── Train-only undersampling ─────────────────────────────────────────
    # Apply before class-weight computation so weights/sampler reflect the
    # effective real train split, not the pre-balanced distribution.
    if args.undersample_majority and not args.two_stage:
        train_ds = undersample_train_subset(
            train_ds,
            max_samples_per_class=int(getattr(args, "max_samples_per_class", 0)),
            seed=args.seed,
            auto_target=bool(getattr(args, "auto_majority_target", False)),
            target_percentile=float(getattr(args, "undersample_target_percentile", 75.0)),
            min_keep=int(getattr(args, "undersample_min_keep", 1)),
        )

    # ── Compute class weights from the effective train split (before oversampling)
    # so that weights reflect the real class distribution, not the inflated one.
    train_labels_orig = get_subset_label_indices(train_ds, train_ds.dataset)  # type: ignore[arg-type]

    # ── Domain info extraction (before oversampling) ─────────────────────
    _base_ds = train_ds.dataset if isinstance(train_ds, Subset) else None
    if _base_ds is not None and hasattr(_base_ds, 'domain_labels') and _base_ds.domain_labels:
        _orig_indices = list(train_ds.indices)
        train_domain_labels = [_base_ds.domain_labels[i] for i in _orig_indices]
        extra_train_indices = [
            _orig_indices[k]
            for k, d in enumerate(train_domain_labels)
            if d != "primary"
        ]
        primary_train_indices = [_orig_indices[k] for k, d in enumerate(train_domain_labels) if d == "primary"]
    else:
        train_domain_labels = ["primary"] * len(train_ds)
        extra_train_indices = []
        primary_train_indices = list(train_ds.indices) if isinstance(train_ds, Subset) else list(range(len(train_ds)))

    # ── Train-only oversampling ───────────────────────────────────────────
    # Skip when --two_stage is on: oversampling is applied to primary-only subset in Stage 2.
    if args.oversample_minority and not args.two_stage:
        train_ds = oversample_train_subset(  # type: ignore[assignment]
            train_ds,
            min_samples=args.min_samples_per_class,
            num_frames=args.num_frames,
            aug_strength=args.minority_aug_strength,
            seed=args.seed,
            auto_target=bool(getattr(args, "auto_minority_target", False)),
            target_percentile=float(getattr(args, "oversample_target_percentile", 50.0)),
            max_multiplier=float(getattr(args, "oversample_max_multiplier", 1.0)),
            global_aug_ratio=float(getattr(args, "oversample_global_aug_ratio", 0.5)),
        )

    # ── Train-only spatial augmentation ──────────────────────────────────
    if getattr(args, "train_spatial_aug", False):
        _aug_magnitude = float(getattr(args, "aug_magnitude", 1.0))
        _aug_transform = build_train_aug_transforms(magnitude=_aug_magnitude)
        train_ds = TrainAugSubset(train_ds, _aug_transform)  # type: ignore[assignment]
        print(f"[aug]   train spatial augmentation enabled | magnitude={_aug_magnitude:.1f} "
              f"(rotation=±{int(8 * _aug_magnitude)}°, blur, erasing)")

    train_labels = train_labels_orig  # used for sampler / weight / logging

    if args.domain_balanced_sampler and len(set(train_domain_labels)) > 1:
        train_sampler = build_domain_balanced_sampler(train_domain_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print("[data]  domain-balanced sampler enabled")
    elif args.balanced_sampler:
        train_sampler = build_balanced_sampler_from_labels(train_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print("[data]  balanced sampler enabled")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[data]  train={len(train_ds)}  val={len(val_ds)}")
    val_labels = get_subset_label_indices(val_ds, val_ds.dataset)  # type: ignore[arg-type]
    print(f"[data]  train class counts: {label_counts(train_labels, num_classes)}")
    print(f"[data]  val   class counts: {label_counts(val_labels, num_classes)}")

    # Resolve label names in index order (idx -> class name) for hierarchy/binary analysis.
    if _base_ds is not None and hasattr(_base_ds, "label_map") and _base_ds.label_map:
        _inv_label_map = {idx: name for name, idx in _base_ds.label_map.items()}
        label_names = [_inv_label_map.get(i, f"class_{i}") for i in range(num_classes)]
    else:
        label_names = [f"class_{i}" for i in range(num_classes)]

    pretrained = not args.no_pretrained
    freeze_backbone = True
    print(
        f"[model] building FlowClassifier backbone={args.backbone} "
        f"(pretrained={pretrained}, freeze_backbone={freeze_backbone}) ..."
    )
    use_dann = getattr(args, "domain_adversarial", False) and len(set(train_domain_labels)) > 1
    use_temporal_attn = getattr(args, "temporal_attn", False)
    use_lstm = getattr(args, "use_lstm", False)
    use_gru  = getattr(args, "use_gru",  False)
    use_tcn  = getattr(args, "use_tcn",  False)
    lstm_hidden = int(getattr(args, "lstm_hidden", 128))
    lstm_layers = int(getattr(args, "lstm_layers", 1))
    gru_hidden  = int(getattr(args, "gru_hidden",  256))
    gru_layers  = int(getattr(args, "gru_layers",  1))
    tcn_channels = int(getattr(args, "tcn_channels", 256))
    # Mutual exclusion: LSTM takes priority if both flags are accidentally set
    if use_lstm and use_gru:
        print("[model] --use_lstm and --use_gru both set; LSTM takes priority")
        use_gru = False
    if use_lstm and use_tcn:
        print("[model] --use_lstm and --use_tcn both set; LSTM takes priority")
        use_tcn = False
    if use_gru and use_tcn:
        print("[model] --use_gru and --use_tcn both set; GRU takes priority")
        use_tcn = False
    if use_lstm and use_temporal_attn:
        print("[model] --use_lstm and --temporal_attn both set; LSTM takes priority")
        use_temporal_attn = False
    if use_gru and use_temporal_attn:
        print("[model] --use_gru and --temporal_attn both set; GRU takes priority")
        use_temporal_attn = False
    if use_tcn and use_temporal_attn:
        print("[model] --use_tcn and --temporal_attn both set; TCN takes priority")
        use_temporal_attn = False
    model = FlowClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        backbone=args.backbone,
        use_dann=use_dann,
        use_temporal_attn=use_temporal_attn,
        use_lstm=use_lstm,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        use_gru=use_gru,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        use_tcn=use_tcn,
        tcn_channels=tcn_channels,
    ).to(device)
    if use_dann:
        print(f"[model] DANN enabled (lambda={args.dann_lambda})")
    if args.backbone == "densenet121":
        print("[model] GeM pooling enabled (DenseNet path)")
    if use_temporal_attn:
        print(f"[model] temporal self-attention pool enabled")
    if use_lstm:
        print(f"[model] bidirectional LSTM temporal pool enabled (hidden={lstm_hidden}, layers={lstm_layers})")
    if use_gru:
        print(f"[model] bidirectional GRU temporal pool enabled (hidden={gru_hidden}, layers={gru_layers})")
    if use_tcn:
        print(f"[model] TCN temporal pool enabled (channels={tcn_channels}, dilations=1/2/4)")

    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema_model: nn.Module | None = None
    if ema_decay > 0.0:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"[train] EMA enabled | decay={ema_decay:.6f}")

    if getattr(args, "freeze_bn_after_unfreeze", False):
        print("[train] BatchNorm freeze after unfreeze enabled")
    if getattr(args, "grad_clip_norm", 0.0) > 0.0:
        print(f"[train] gradient clipping enabled | max_norm={float(args.grad_clip_norm):.3f}")

    class_weight_cap = None if args.class_weight_cap <= 0 else float(args.class_weight_cap)
    _raw_class_weights = build_class_weights_from_labels(
        train_labels,
        num_classes,
        cap=None,
    ).to(device)
    class_weights = build_class_weights_from_labels(
        train_labels,
        num_classes,
        cap=class_weight_cap,
    ).to(device)
    if args.no_class_weights:
        class_weights = torch.ones_like(class_weights)
    # Apply manual per-class weight boosts (e.g. 'enojo:2.0,miedo:2.5')
    if args.no_class_weights and getattr(args, "class_weight_boost", ""):
        print("[loss]  class_weight_boost ignored because --no_class_weights is enabled")
    elif getattr(args, "class_weight_boost", ""):
        _boost_map: dict[str, float] = {}
        for _pair in args.class_weight_boost.split(","):
            _pair = _pair.strip()
            if ":" in _pair:
                _cls, _mult = _pair.rsplit(":", 1)
                _boost_map[_cls.strip()] = float(_mult.strip())
        _cw_np = class_weights.cpu().numpy().copy()
        _ds_label_map = _base_ds.label_map if (_base_ds is not None and hasattr(_base_ds, "label_map")) else {}
        for _cls_name, _mult in _boost_map.items():
            _idx = _ds_label_map.get(_cls_name, -1)
            if _idx >= 0:
                _cw_np[_idx] = _cw_np[_idx] * _mult
                print(f"[loss]  class_weight_boost: '{_cls_name}' (idx={_idx}) × {_mult:.2f} → {_cw_np[_idx]:.3f}")
            else:
                print(f"[loss]  class_weight_boost: WARNING '{_cls_name}' not found in label_map, skipping.")
        class_weights = torch.as_tensor(_cw_np, dtype=torch.float32).to(device)

    # Apply per-class weight cap overrides (e.g. 'neutral:50.0,felicidad:150.0').
    # This overrides the global cap for selected classes while still behaving as a cap:
    # effective_weight = min(raw_uncapped_weight, per_class_cap).
    _per_cls_cap_str = getattr(args, "per_class_weight_cap", "")
    if _per_cls_cap_str and not args.no_class_weights:
        _ds_label_map_cap = _base_ds.label_map if (_base_ds is not None and hasattr(_base_ds, "label_map")) else {}
        _cw_cap_np = class_weights.cpu().numpy().copy()
        _raw_cw_np = _raw_class_weights.cpu().numpy().copy()
        for _entry in _per_cls_cap_str.split(","):
            _entry = _entry.strip()
            if ":" in _entry:
                _cap_cls, _cap_val = _entry.rsplit(":", 1)
                _cap_idx = _ds_label_map_cap.get(_cap_cls.strip(), -1)
                try:
                    _cap_val_f = float(_cap_val.strip())
                except ValueError:
                    continue
                if _cap_idx >= 0:
                    _effective = min(float(_raw_cw_np[_cap_idx]), _cap_val_f)
                    _cw_cap_np[_cap_idx] = _effective
                    print(
                        f"[loss]  per_class_weight_cap: '{_cap_cls}' (idx={_cap_idx}) "
                        f"cap={_cap_val_f:.3f} raw={float(_raw_cw_np[_cap_idx]):.3f} → {_effective:.3f}"
                    )
                else:
                    print(f"[loss]  per_class_weight_cap: WARNING '{_cap_cls}' not found in label_map, skipping.")
        class_weights = torch.as_tensor(_cw_cap_np, dtype=torch.float32).to(device)

    # Build (optionally customized) confusion penalty matrix
    _label_map_for_penalty = _base_ds.label_map if (_base_ds is not None and hasattr(_base_ds, "label_map")) else None
    _custom_penalty_str = getattr(args, "confusion_penalty_matrix", "")
    _penalty_matrix = get_confusion_penalty_matrix(
        num_classes,
        label_map=_label_map_for_penalty,
        custom_penalties=_custom_penalty_str if _custom_penalty_str else None,
    )

    # Felicidad class index for threshold calibration
    _feli_class_idx: int | None = None
    if _label_map_for_penalty is not None:
        for _fn in ("felicidad", "happiness", "happy", "hap"):
            if _fn in _label_map_for_penalty:
                _feli_class_idx = _label_map_for_penalty[_fn]
                break

    # Warn if combined class-weight × penalty can dominate optimization dynamics.
    if _label_map_for_penalty is not None and _feli_class_idx is not None and not args.no_class_weights:
        _neutral_idx = _label_map_for_penalty.get("neutral", -1)
        if _neutral_idx >= 0:
            _eff_mult = float(class_weights[_feli_class_idx].item()) * float(
                _penalty_matrix[_feli_class_idx, _neutral_idx].item()
            )
            if _eff_mult >= 200.0:
                print(
                    f"[warn] high effective loss multiplier for felicidad→neutral: {_eff_mult:.1f}. "
                    "Consider lowering confusion penalties (e.g., 1.3/1.2/1.2) or per-class caps (e.g., 120)."
                )

    # SupCon domain-aware flag
    _supcon_domain_aware = getattr(args, "use_supcon_loss", False) and (
        _base_ds is not None and hasattr(_base_ds, "domain_labels") and bool(_base_ds.domain_labels)
    )
    if _supcon_domain_aware:
        print("[loss]  SupCon domain-aware mask enabled (same-domain positive pairs only)")

    if args.use_focal_loss or getattr(args, "focal_loss", False):
        focal_alpha: torch.Tensor | float = 1.0 if args.no_class_weights else class_weights
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma, reduction="mean")
        print(
            f"[loss]  Focal Loss enabled | gamma={args.focal_gamma} | "
            f"class weights: {class_weights.detach().cpu().numpy().round(3).tolist()}"
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(np.clip(args.label_smoothing, 0.0, 0.2)),
        )
        print(
            f"[loss]  Cross-Entropy Loss | class weights: {class_weights.detach().cpu().numpy().round(3).tolist()} "
            f"| label_smoothing={float(np.clip(args.label_smoothing, 0.0, 0.2)):.3f}"
        )

    adaptive_cw_enabled = bool(getattr(args, "adaptive_class_weights", False)) and not args.no_class_weights
    base_class_weights_np = class_weights.detach().cpu().numpy().astype(np.float32).copy()
    adaptive_recall_ema = np.full((num_classes,), 0.5, dtype=np.float32)
    if bool(getattr(args, "adaptive_class_weights", False)) and args.no_class_weights:
        print("[loss]  adaptive class weights ignored because --no_class_weights is enabled")
    if adaptive_cw_enabled:
        print(
            f"[loss]  adaptive class weights enabled | target_recall={float(args.adaptive_cw_target_recall):.2f} "
            f"alpha={float(args.adaptive_cw_alpha):.2f} warmup={int(args.adaptive_cw_warmup_epochs)}"
        )

    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.lr, args.backbone_lr_mult),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────
    _scheduler_name = getattr(args, "lr_scheduler", "cosine")
    if _scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
        )
        print(f"[sched] CosineAnnealingLR | T_max={args.epochs}, eta_min={args.lr * 1e-2:.2e}")
    elif _scheduler_name == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=args.lr * 1e-2
        )
        print(f"[sched] CosineAnnealingWarmRestarts | T_0=10, T_mult=2")
    elif _scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        print(f"[sched] StepLR | step_size=15, gamma=0.5")
    else:
        scheduler = None
        print("[sched] no LR scheduler")

    # ── Two-stage training ────────────────────────────────────────────────
    # Stage 1: pretrain on extra (MEME) data.
    # Stage 2: fine-tune on primary (CASME2) data with lr*0.3.
    _skip_unfreeze = False  # set True when backbone is already unfrozen for Stage 2
    if args.two_stage and extra_train_indices and _base_ds is not None:
        _extra_subset = Subset(_base_ds, extra_train_indices)
        _extra_labels = [_base_ds.label_map[_base_ds.samples[i][1]] for i in extra_train_indices]
        _extra_sampler = build_balanced_sampler_from_labels(_extra_labels)
        _extra_loader = DataLoader(
            _extra_subset,
            batch_size=args.batch_size,
            sampler=_extra_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(
            f"\n[two-stage] Stage 1: pretraining on {len(extra_train_indices)} MEME samples "
            f"for {args.pretrain_epochs} epochs ..."
        )
        _pt_optimizer = torch.optim.AdamW(
            model.parameter_groups(args.lr, args.backbone_lr_mult),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        for _pt_epoch in range(1, args.pretrain_epochs + 1):
            if args.unfreeze_epoch > 0 and _pt_epoch == args.unfreeze_epoch:
                print(f"[pretrain epoch {_pt_epoch}] unfreezing {args.backbone} backbone")
                model.unfreeze_backbone()
                _pt_optimizer = torch.optim.AdamW(
                    model.parameter_groups(args.lr, args.backbone_lr_mult),
                    lr=args.lr, weight_decay=args.weight_decay,
                )
            _pt_loss, _pt_acc = train_one_epoch(model, _extra_loader, _pt_optimizer, criterion, device)
            print(f"[pretrain {_pt_epoch:03d}/{args.pretrain_epochs}] loss={_pt_loss:.4f} acc={_pt_acc:.3f}")

        # Stage 2: rebuild train_loader for primary (CASME2) data
        _s2_indices = primary_train_indices if primary_train_indices else (
            list(train_ds.indices) if isinstance(train_ds, Subset) else list(range(len(train_ds)))
        )
        print(
            f"\n[two-stage] Stage 2: fine-tuning on {len(_s2_indices)} CASME2 samples "
            f"for {args.epochs} epochs ..."
        )
        _primary_subset: Subset | object = Subset(_base_ds, _s2_indices)

        if args.undersample_majority:
            _primary_subset = undersample_train_subset(  # type: ignore[assignment]
                _primary_subset,  # type: ignore[arg-type]
                max_samples_per_class=int(getattr(args, "max_samples_per_class", 0)),
                seed=args.seed,
                auto_target=bool(getattr(args, "auto_majority_target", False)),
                target_percentile=float(getattr(args, "undersample_target_percentile", 75.0)),
                min_keep=int(getattr(args, "undersample_min_keep", 1)),
            )

        _primary_labels_s2 = get_subset_label_indices(_primary_subset, _base_ds)  # type: ignore[arg-type]

        if args.oversample_minority:
            _primary_subset = oversample_train_subset(  # type: ignore[assignment]
                _primary_subset,  # type: ignore[arg-type]
                min_samples=args.min_samples_per_class,
                num_frames=args.num_frames,
                aug_strength=args.minority_aug_strength,
                seed=args.seed,
                auto_target=bool(getattr(args, "auto_minority_target", False)),
                target_percentile=float(getattr(args, "oversample_target_percentile", 50.0)),
                max_multiplier=float(getattr(args, "oversample_max_multiplier", 1.0)),
                global_aug_ratio=float(getattr(args, "oversample_global_aug_ratio", 0.5)),
            )

        finetune_lr = args.lr * 0.3
        model.unfreeze_backbone()
        _skip_unfreeze = True  # do not re-run epoch-based unfreeze in the main loop
        optimizer = torch.optim.AdamW(
            model.parameter_groups(finetune_lr, args.backbone_lr_mult),
            lr=finetune_lr, weight_decay=args.weight_decay,
        )
        _ft_sampler = build_balanced_sampler_from_labels(_primary_labels_s2)
        train_loader = DataLoader(
            _primary_subset,
            batch_size=args.batch_size,
            sampler=_ft_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        # Recompute criterion with primary-only class distribution
        _primary_cw = build_class_weights_from_labels(
            _primary_labels_s2, num_classes, cap=class_weight_cap
        ).to(device)
        if args.no_class_weights:
            _primary_cw = torch.ones_like(_primary_cw)
        if args.use_focal_loss or getattr(args, "focal_loss", False):
            criterion = FocalLoss(
                alpha=(1.0 if args.no_class_weights else _primary_cw),
                gamma=args.focal_gamma, reduction="mean",
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=_primary_cw,
                label_smoothing=float(np.clip(args.label_smoothing, 0.0, 0.2)),
            )
        print(
            f"[two-stage] Stage 2 class weights: "
            f"{_primary_cw.detach().cpu().numpy().round(3).tolist()}"
        )
    elif args.two_stage:
        print("[two-stage] WARNING: no extra-domain samples found in train split; "
              "falling back to single-stage training.")

    # ── Wrap train_loader for DANN (adds domain_id as 3rd element) ────────
    _dann_train_loader = train_loader  # will be replaced below if DANN active
    if use_dann and not args.two_stage:
        _domain_ids = [0 if d == "primary" else 1 for d in train_domain_labels]
        # If oversampling extended the dataset, augmented samples are primary (0)
        n_orig = len(_domain_ids)
        n_total = len(train_ds)
        if n_total > n_orig:
            _domain_ids = _domain_ids + [0] * (n_total - n_orig)
        _dann_train_ds = _DomainLabeledDataset(train_ds, _domain_ids)
        _dann_train_loader = DataLoader(
            _dann_train_ds,
            batch_size=args.batch_size,
            sampler=train_loader.sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    # ── Auxiliary metric learning losses ─────────────────────────────────
    _center_loss = None
    _center_loss_weight = 0.0
    if getattr(args, "use_center_loss", False):
        with torch.no_grad():
            _dummy = torch.zeros(1, args.num_frames, 3, 224, 224, device=device)
            _feat_dim = model.get_embedding(_dummy).shape[-1]
        _center_loss = CenterLoss(num_classes=num_classes, feat_dim=_feat_dim, device=device).to(device)
        _center_loss_weight = float(getattr(args, "center_loss_weight", 0.1))
        print(f"[loss]  CenterLoss enabled | feat_dim={_feat_dim} | weight={_center_loss_weight}")

    _supcon_loss = None
    _supcon_loss_weight = 0.0
    if getattr(args, "use_supcon_loss", False):
        _supcon_temp = float(getattr(args, "supcon_temperature", 0.07))
        _supcon_loss = SupConLoss(temperature=_supcon_temp).to(device)
        _supcon_loss_weight = float(getattr(args, "supcon_loss_weight", 0.3))
        print(f"[loss]  SupConLoss enabled | temperature={_supcon_temp} | weight={_supcon_loss_weight}")

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_f1_present = 0.0
    best_monitor_score = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    saved = False
    ckpt = f"best_{args.backbone}_flow_{fold_tag}.pth"
    best_val_cm: np.ndarray | None = None
    early_stop_epoch: int | None = None
    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    val_f1_present_hist: list[float] = []
    epoch_time_hist_sec: list[float] = []
    monitor_name = "val macro-f1 (present classes only)" if args.monitor_metric == "macro_f1_present" else "val macro-f1"

    # Optional mode: threshold search only from an existing checkpoint (no training).
    if getattr(args, "threshold_search_only", False):
        if _feli_class_idx is None:
            print("[feli_thresh_search] felicidad class not found in label map; skipping.")
            return 0.0, 0.0

        _ts_ckpt = getattr(args, "threshold_search_ckpt", "").strip()
        if not _ts_ckpt:
            _ts_ckpt = f"best_{args.backbone}_flow_{fold_tag}.pth"

        if not os.path.exists(_ts_ckpt):
            print(f"[feli_thresh_search] checkpoint not found: {_ts_ckpt}")
            return 0.0, 0.0

        print(f"[feli_thresh_search] threshold-search only mode | loading '{_ts_ckpt}'")
        eval_model = ema_model if ema_model is not None else model
        eval_model.load_state_dict(torch.load(_ts_ckpt, map_location=device))

        _base_thresh = float(getattr(args, "feli_threshold", 0.0))
        if args.tta_passes > 1:
            val_loss, val_acc, val_f1, val_f1_present, val_precision, val_recall, val_cm = evaluate_with_tta(
                eval_model, val_loader, criterion, device, num_classes,
                tta_passes=args.tta_passes,
                feli_threshold=_base_thresh,
                feli_class_idx=_feli_class_idx,
            )
        else:
            val_loss, val_acc, val_f1, val_f1_present, val_precision, val_recall, val_cm = evaluate(
                eval_model, val_loader, criterion, device, num_classes
            )
        print(
            f"[feli_thresh_search] baseline | loss={val_loss:.4f} acc={val_acc:.3f} "
            f"f1={val_f1:.3f} f1_present={val_f1_present:.3f} "
            f"precision={val_precision:.3f} recall={val_recall:.3f}"
        )
        print("[feli_thresh_search] baseline confusion matrix:")
        print(val_cm)

        best_thresh = evaluate_with_tta_threshold_search(
            eval_model, val_loader, criterion, device, num_classes,
            tta_passes=max(args.tta_passes, 1),
            feli_class_idx=_feli_class_idx,
            thresholds=[round(x * 0.05, 2) for x in range(5, 11)],  # 0.25..0.50
        )
        print(f"[feli_thresh_search] recommended --feli_threshold {best_thresh:.2f}")
        return val_acc, val_f1

    if args.early_stop_patience > 0:
        print(
            f"[train] early stopping enabled | patience={args.early_stop_patience}, "
            f"min_epochs={args.early_stop_min_epochs} (monitor={monitor_name})"
        )

    if args.auto_unfreeze_on_plateau:
        print(
            f"[train] auto-unfreeze enabled | patience={args.auto_unfreeze_patience}, "
            f"min_epochs={args.auto_unfreeze_min_epochs} (monitor={monitor_name})"
        )

    if args.auto_refreeze_on_plateau:
        print(
            f"[train] auto-refreeze enabled | patience={args.auto_refreeze_patience}, "
            f"min_epochs_since_unfreeze={args.auto_refreeze_min_epochs_since_unfreeze} "
            f"(monitor={monitor_name})"
        )

    backbone_unfrozen = bool(_skip_unfreeze)
    last_unfreeze_epoch = 0
    refreeze_done = False
    skip_scheduler_step_once = False

    def _unfreeze_backbone_at_epoch(cur_epoch: int, reason: str, post_train_reset: bool = False) -> None:
        nonlocal model, optimizer, scheduler, backbone_unfrozen, last_unfreeze_epoch, skip_scheduler_step_once
        print(f"[epoch {cur_epoch}] unfreezing {args.backbone} backbone ({reason})")
        prev_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.lr)
        model.unfreeze_backbone()
        optimizer = torch.optim.AdamW(
            model.parameter_groups(prev_lr, args.backbone_lr_mult),
            lr=prev_lr,
            weight_decay=args.weight_decay,
        )
        remaining = args.epochs - cur_epoch + 1
        if _scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=prev_lr * 1e-2
            )
        elif _scheduler_name == "cosine_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=prev_lr * 1e-2
            )
        elif _scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        backbone_unfrozen = True
        last_unfreeze_epoch = cur_epoch
        if post_train_reset:
            skip_scheduler_step_once = True

    def _refreeze_backbone_at_epoch(cur_epoch: int, reason: str, post_train_reset: bool = False) -> None:
        nonlocal model, optimizer, scheduler, backbone_unfrozen, refreeze_done, skip_scheduler_step_once
        print(f"[epoch {cur_epoch}] refreezing {args.backbone} backbone ({reason})")
        prev_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.lr)
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            model.parameter_groups(prev_lr, args.backbone_lr_mult),
            lr=prev_lr,
            weight_decay=args.weight_decay,
        )
        remaining = args.epochs - cur_epoch + 1
        if _scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=prev_lr * 1e-2
            )
        elif _scheduler_name == "cosine_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=prev_lr * 1e-2
            )
        elif _scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        backbone_unfrozen = False
        refreeze_done = True
        if post_train_reset:
            skip_scheduler_step_once = True

    for epoch in range(1, args.epochs + 1):
        _epoch_t0 = time.perf_counter()
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch and not _skip_unfreeze and not backbone_unfrozen:
            _unfreeze_backbone_at_epoch(epoch, reason="fixed epoch")

        freeze_bn_now = bool(
            getattr(args, "freeze_bn_after_unfreeze", False)
            and (args.unfreeze_epoch <= 0 or _skip_unfreeze or epoch >= args.unfreeze_epoch)
        )
        if use_dann and not args.two_stage:
            # Compute progressive lambda: ramps from 0 → dann_lambda over all epochs
            p = (epoch - 1) / max(args.epochs - 1, 1)
            _lambda = args.dann_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            train_loss, train_acc = train_one_epoch_dann(
                model, _dann_train_loader, optimizer, criterion, device,
                dann_lambda=_lambda,
                mixup_alpha=getattr(args, "mixup_alpha", 0.0),
                grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
                freeze_bn=freeze_bn_now,
                ema_model=ema_model,
                ema_decay=ema_decay,
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                mixup_alpha=getattr(args, "mixup_alpha", 0.0),
                grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
                freeze_bn=freeze_bn_now,
                ema_model=ema_model,
                ema_decay=ema_decay,
                center_loss=_center_loss,
                center_loss_weight=_center_loss_weight,
                supcon_loss=_supcon_loss,
                supcon_loss_weight=_supcon_loss_weight,
                penalty_matrix=_penalty_matrix,
                supcon_domain_aware=_supcon_domain_aware,
            )

        _feli_thresh = float(getattr(args, "feli_threshold", 0.0))
        eval_model = ema_model if ema_model is not None else model
        if args.tta_passes > 1:
            val_loss, val_acc, val_f1, val_f1_present, val_precision, val_recall, val_cm = evaluate_with_tta(
                eval_model, val_loader, criterion, device, num_classes,
                tta_passes=args.tta_passes,
                feli_threshold=_feli_thresh,
                feli_class_idx=_feli_class_idx,
            )
        else:
            val_loss, val_acc, val_f1, val_f1_present, val_precision, val_recall, val_cm = evaluate(
                eval_model, val_loader, criterion, device, num_classes
            )

        monitor_score = val_f1_present if args.monitor_metric == "macro_f1_present" else val_f1
        flag = " ✓" if monitor_score > best_monitor_score else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        if val_f1_present > best_val_f1_present:
            best_val_f1_present = val_f1_present
        if monitor_score > best_monitor_score:
            best_monitor_score = monitor_score
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(eval_model.state_dict(), ckpt)
            saved = True
            best_val_cm = val_cm.copy()
        else:
            no_improve_epochs += 1

        if (
            args.auto_unfreeze_on_plateau
            and not _skip_unfreeze
            and not backbone_unfrozen
            and not refreeze_done
            and args.auto_unfreeze_patience > 0
            and epoch >= max(1, args.auto_unfreeze_min_epochs)
            and no_improve_epochs >= args.auto_unfreeze_patience
        ):
            _unfreeze_backbone_at_epoch(
                epoch,
                reason=f"{monitor_name} plateau {no_improve_epochs} epochs",
                post_train_reset=True,
            )
            # Give the unfrozen backbone runway before early stopping can stop immediately.
            no_improve_epochs = 0

        if (
            args.auto_refreeze_on_plateau
            and backbone_unfrozen
            and args.auto_refreeze_patience > 0
            and no_improve_epochs >= args.auto_refreeze_patience
            and (epoch - last_unfreeze_epoch) >= max(1, args.auto_refreeze_min_epochs_since_unfreeze)
        ):
            _refreeze_backbone_at_epoch(
                epoch,
                reason=f"{monitor_name} plateau {no_improve_epochs} epochs after unfreeze",
                post_train_reset=True,
            )
            # Reset counter so early-stop decision reflects post-refreeze behavior.
            no_improve_epochs = 0

        if (
            adaptive_cw_enabled
            and epoch >= max(1, int(getattr(args, "adaptive_cw_warmup_epochs", 1)))
            and (epoch % max(1, int(getattr(args, "adaptive_cw_update_every", 1))) == 0)
        ):
            _new_w_np, adaptive_recall_ema = adaptive_class_weights_from_confusion(
                val_cm,
                base_class_weights_np,
                adaptive_recall_ema,
                ema_beta=float(getattr(args, "adaptive_cw_ema_beta", 0.7)),
                target_recall=float(getattr(args, "adaptive_cw_target_recall", 0.45)),
                alpha=float(getattr(args, "adaptive_cw_alpha", 0.8)),
                max_multiplier=float(getattr(args, "adaptive_cw_max_multiplier", 2.0)),
                min_val_support=int(getattr(args, "adaptive_cw_min_val_support", 4)),
            )
            _new_w_t = torch.as_tensor(_new_w_np, dtype=torch.float32, device=device)
            if isinstance(criterion, nn.CrossEntropyLoss):
                criterion.weight = _new_w_t
            elif isinstance(criterion, FocalLoss) and isinstance(criterion.alpha, torch.Tensor):
                criterion.alpha = _new_w_t
            print(
                f"[loss]  adaptive class weights (epoch {epoch}) -> "
                f"{np.round(_new_w_np, 3).tolist()}"
            )

        if scheduler is not None:
            if skip_scheduler_step_once:
                print("[sched] skip step this epoch (optimizer/scheduler reset after training)")
                skip_scheduler_step_once = False
            else:
                scheduler.step()
        _cur_lr = optimizer.param_groups[0]["lr"]
        _epoch_dt = time.perf_counter() - _epoch_t0
        epoch_time_hist_sec.append(_epoch_dt)
        train_loss_hist.append(float(train_loss))
        val_loss_hist.append(float(val_loss))
        val_f1_present_hist.append(float(val_f1_present))
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val  loss={val_loss:.4f}  acc={val_acc:.3f} "
            f"f1={val_f1:.3f} f1_present={val_f1_present:.3f} "
            f"precision={val_precision:.3f} recall={val_recall:.3f} lr={_cur_lr:.2e} "
            f"epoch_time={_epoch_dt:.2f}s{flag}"
        )
        print("[val cm]")
        print(val_cm)

        # Hierarchical Analysis at best epoch
        if getattr(args, "hierarchical_classifier", False) and epoch == best_epoch:
            hierarchy = define_class_hierarchy(
                num_classes,
                label_names,
                strategy=getattr(args, "hierarchy_strategy", "predefined_emotions"),
            )
            print(f"[hierarchy] strategy={getattr(args, 'hierarchy_strategy', 'predefined_emotions')}")
            for group_name, classes in hierarchy.items():
                print(f"[hierarchy]   {group_name}: {classes}")
        
        if getattr(args, "binary_refiners", False) and epoch == best_epoch:
            confusion_threshold = float(getattr(args, "confusion_threshold", 0.15))
            min_samples_binary = int(getattr(args, "min_samples_binary_trainer", 10))

            train_class_counts: dict[str, int] = {name: 0 for name in label_names}
            for label_idx in train_labels:
                if 0 <= int(label_idx) < len(label_names):
                    _name = label_names[int(label_idx)]
                    train_class_counts[_name] = train_class_counts.get(_name, 0) + 1
            
            confused_pairs = analyze_confusion_matrix(
                val_cm,
                label_names,
                confusion_threshold=confusion_threshold,
                min_samples=min_samples_binary,
                class_counts=train_class_counts,
            )
            
            if any(confused_pairs.values()):
                print(f"[binary_refiners] detected confused pairs (threshold={confusion_threshold:.2f}):")
                for class_name, pairs in confused_pairs.items():
                    if pairs:
                        for _, other_class, confusion_rate in pairs:
                            print(
                                f"[binary_refiners]   {class_name} ↔ {other_class}: "
                                f"confusion_rate={confusion_rate:.3f}"
                            )
            else:
                print(f"[binary_refiners] no heavily confused pairs found (threshold={confusion_threshold:.2f})")

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and no_improve_epochs >= args.early_stop_patience
        ):
            early_stop_epoch = epoch
            print(
                f"[early-stop] no {monitor_name} improvement for {no_improve_epochs} epochs "
                f"(best={best_monitor_score:.3f} at epoch {best_epoch}). Stopping."
            )
            break

    if saved:
        _total_train_time_sec = time.perf_counter() - _train_start_wall
        print(
            f"\n[done:{fold_tag}] best val accuracy: {best_val_acc:.3f} | "
            f"best val macro-f1: {best_val_f1:.3f} | "
            f"best val macro-f1-present: {best_val_f1_present:.3f} "
            f"(monitor={monitor_name}: {best_monitor_score:.3f}) | "
            f"total_train_time={_total_train_time_sec:.2f}s"
        )
        eval_model = ema_model if ema_model is not None else model
        if os.path.exists(ckpt):
            eval_model.load_state_dict(torch.load(ckpt, map_location=device))
            eval_model.eval()

        # Guardar matriz de confusión y métricas clave para reporte
        cm = best_val_cm if best_val_cm is not None else val_cm
        cm_raw_path = f'confusion_matrix_{fold_tag}.png'
        cm_norm_path = f'confusion_matrix_normalized_{fold_tag}.png'
        if 'val_cm' in locals():
            save_confusion_matrix(cm, label_names, cm_raw_path, title='Validation Confusion Matrix (Counts)')
            save_confusion_matrix_normalized(cm, label_names, cm_norm_path, title='Validation Confusion Matrix (Normalized)')
            print(f"[mejor F1] Matriz de confusión (rows=true, cols=pred):\n{cm}\n")
            with open(f'metrics_report_{fold_tag}.txt', 'w') as f:
                f.write(f'Best val accuracy: {best_val_acc:.3f}\n')
                f.write(f'Best val macro-f1: {best_val_f1:.3f}\n')
                f.write(f'Best val macro-f1-present: {best_val_f1_present:.3f}\n')
                f.write(f'Total training time (s): {_total_train_time_sec:.2f}\n')
                f.write(f'Confusion matrix (rows=true, cols=pred):\n{cm}\n')
                f.write(f'Confusion matrix image (counts): {cm_raw_path}\n')
                f.write(f'Confusion matrix image (normalized): {cm_norm_path}\n')

        loss_curve_path, metric_curve_path = _plot_learning_curves(
            train_loss_hist,
            val_loss_hist,
            val_f1_present_hist,
            best_epoch,
            early_stop_epoch,
            fold_tag,
        )

        infer_latency_mean, infer_latency_std = _estimate_tta_latency(
            eval_model,
            val_loader,
            criterion,
            device,
            num_classes,
            tta_passes=30,
            repeats=3,
        )

        temporal_case_path = f"temporal_case_study_{fold_tag}.png"
        temporal_case_note = _save_temporal_case_study(
            eval_model,
            val_ds,
            label_names,
            device,
            temporal_case_path,
        )

        # Calcular y guardar curvas ROC y AUC para cada clase (one-vs-rest)
        report_path = ""
        try:
            # Obtener logits y labels verdaderos del mejor modelo en validación
            all_logits = []
            all_targets = []
            for batch in val_loader:
                x, y = batch[:2]
                x = x.to(device)
                with torch.no_grad():
                    logits = eval_model(x)
                all_logits.append(logits.cpu().numpy())
                all_targets.append(y.cpu().numpy())
            logits = np.concatenate(all_logits, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            # One-hot para targets
            targets_onehot = np.eye(num_classes)[targets]
            # ROC y AUC para cada clase
            aucs = []
            for i, class_name in enumerate(label_names):
                fpr, tpr, _ = roc_curve(targets_onehot[:, i], logits[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.figure()
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=1)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC curve - {class_name}')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(f'roc_curve_{fold_tag}_{class_name}.png')
                plt.close()
            # Macro/micro AUC
            macro_auc = np.mean(aucs)
            micro_auc = roc_auc_score(targets_onehot, logits, average='micro', multi_class='ovr')
            with open(f'metrics_report_{fold_tag}.txt', 'a') as f:
                f.write(f'Macro AUC: {macro_auc:.3f}\n')
                f.write(f'Micro AUC: {micro_auc:.3f}\n')
                for i, class_name in enumerate(label_names):
                    f.write(f'AUC {class_name}: {aucs[i]:.3f}\n')

            report_path = write_detailed_training_report(
                fold_tag=fold_tag,
                class_names=label_names,
                cm=cm,
                train_losses=train_loss_hist,
                val_losses=val_loss_hist,
                val_f1_present=val_f1_present_hist,
                epoch_times_sec=epoch_time_hist_sec,
                best_epoch=best_epoch,
                early_stop_epoch=early_stop_epoch,
                args=args,
                penalty_matrix=_penalty_matrix,
                class_weights=class_weights,
                loss_curve_path=loss_curve_path,
                metric_curve_path=metric_curve_path,
                cm_raw_path=cm_raw_path,
                cm_norm_path=cm_norm_path,
                temporal_case_path=temporal_case_path,
                temporal_case_note=temporal_case_note,
                infer_latency_mean=infer_latency_mean,
                infer_latency_std=infer_latency_std,
            )
            print(f"[report] Detailed report generated: {report_path}")
        except Exception as e:
            print(f"[warn] ROC/AUC computation failed: {e}")
            try:
                report_path = write_detailed_training_report(
                    fold_tag=fold_tag,
                    class_names=label_names,
                    cm=cm,
                    train_losses=train_loss_hist,
                    val_losses=val_loss_hist,
                    val_f1_present=val_f1_present_hist,
                    epoch_times_sec=epoch_time_hist_sec,
                    best_epoch=best_epoch,
                    early_stop_epoch=early_stop_epoch,
                    args=args,
                    penalty_matrix=_penalty_matrix,
                    class_weights=class_weights,
                    loss_curve_path=loss_curve_path,
                    metric_curve_path=metric_curve_path,
                    cm_raw_path=cm_raw_path,
                    cm_norm_path=cm_norm_path,
                    temporal_case_path=temporal_case_path,
                    temporal_case_note=temporal_case_note,
                    infer_latency_mean=infer_latency_mean,
                    infer_latency_std=infer_latency_std,
                )
                print(f"[report] Detailed report generated (without ROC/AUC): {report_path}")
            except Exception as e2:
                print(f"[warn] Detailed report generation failed: {e2}")
    else:
        _total_train_time_sec = time.perf_counter() - _train_start_wall
        print(
            f"\n[done:{fold_tag}] no improved checkpoint saved | "
            f"best val accuracy: {best_val_acc:.3f} | "
            f"best val macro-f1: {best_val_f1:.3f} | "
            f"best val macro-f1-present: {best_val_f1_present:.3f} | "
            f"total_train_time={_total_train_time_sec:.2f}s"
        )

    # ── Threshold calibration post-train (Cambio 4) ────────────────────
    if getattr(args, "feli_threshold_search", False) and _feli_class_idx is not None and saved:
        print(f"\n[feli_thresh_search] loading best checkpoint '{ckpt}' for threshold search ...")
        eval_model.load_state_dict(torch.load(ckpt, map_location=device))
        best_thresh = evaluate_with_tta_threshold_search(
            eval_model, val_loader, criterion, device, num_classes,
            tta_passes=max(args.tta_passes, 1),
            feli_class_idx=_feli_class_idx,
            thresholds=[round(x * 0.05, 2) for x in range(5, 11)],  # 0.25..0.50
        )
        print(f"[feli_thresh_search] rerun inference with --feli_threshold {best_thresh:.2f} for best results.")

    return best_val_acc, best_val_f1


# ══════════════════════════════════════════════════════════════════════════════
# CASCADE NEUTRAL: 2-stage binary + Ekman-6 pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_cascade_neutral(
    args: argparse.Namespace,
    device: torch.device,
    full_dataset: "FlowSequenceDataset",
    train_ds: "Subset",
    val_ds: "Subset",
) -> None:
    """
    Train and evaluate a 2-stage cascade for the neutral class:

      Stage 1 — Binary neutral detector
        · Input: ALL training data (neutral + 6 Ekman emotions)
        · Labels: 0 = non-neutral, 1 = neutral
        · Checkpoint: best_{backbone}_flow_cascade_stage1_binary.pth

      Stage 2 — Ekman-6 classifier
        · Input: ONLY non-neutral training data (neutral samples removed)
        · Labels: 0-5 (asco, enojo, felicidad, miedo, sorpresa, tristeza)
        · Checkpoint: best_{backbone}_flow_cascade_stage2_ekman6.pth

      Cascade evaluation
        · Each val sample passes through Stage 1 first.
        · If Stage 1 → neutral  : final prediction = neutral
        · If Stage 1 → non-neutral: final prediction = Stage 2 output
        · Reports 7-class accuracy and macro-F1.
    """
    import copy as _copy

    neutral_name = "neutral"
    neutral_full_idx = full_dataset.label_map.get(neutral_name)
    if neutral_full_idx is None:
        print(
            "[cascade] ERROR: 'neutral' class not found in the dataset label map. "
            "Make sure --neutral_csv_index is provided."
        )
        return

    # ── STAGE 1: Binary neutral detector ────────────────────────────────
    # Build a shallow copy of full_dataset whose .samples and .label_map are binary.
    # Shallow copy keeps .data_dir, .transform, .num_frames, .onsets, etc. identical.
    ds1 = _copy.copy(full_dataset)
    ds1.samples = [
        (path, neutral_name if emo == neutral_name else "non_neutral", subj)
        for path, emo, subj in full_dataset.samples
    ]
    ds1.label_map = {"non_neutral": 0, neutral_name: 1}
    # domain_labels is already shallow-copied and shares the same indices → correct

    train_ds1: Subset = Subset(ds1, list(train_ds.indices))
    val_ds1:   Subset = Subset(ds1, list(val_ds.indices))

    _n_train_neutral = sum(1 for i in train_ds.indices if full_dataset.samples[i][1] == neutral_name)
    _n_val_neutral   = sum(1 for i in val_ds.indices   if full_dataset.samples[i][1] == neutral_name)
    print("\n[cascade] ══════════ STAGE 1: Binary Neutral Detector ══════════")
    print(f"[cascade]   label map : {ds1.label_map}")
    print(f"[cascade]   train     : {len(train_ds1)} samples  (neutral={_n_train_neutral})")
    print(f"[cascade]   val       : {len(val_ds1)} samples  (neutral={_n_val_neutral})")

    run_single_split(args, device, 2, train_ds1, val_ds1, "cascade_stage1_binary")

    # ── STAGE 2: Ekman-6 (non-neutral only) ─────────────────────────────
    # Collect which original indices (into full_dataset) are non-neutral.
    non_neutral_orig_idx: list[int] = [
        i for i, (_, emo, _) in enumerate(full_dataset.samples)
        if emo != neutral_name
    ]
    # Build mapping: original index → ds2 index (0..M-1)
    orig_to_ds2: dict[int, int] = {orig: ds2_i for ds2_i, orig in enumerate(non_neutral_orig_idx)}

    ds2 = _copy.copy(full_dataset)
    ds2.samples = [full_dataset.samples[i] for i in non_neutral_orig_idx]
    ds2.label_map = build_label_map({emo for _, emo, _ in ds2.samples})
    # Remap domain_labels to the new (smaller) index space
    if hasattr(full_dataset, "domain_labels") and full_dataset.domain_labels:
        ds2.domain_labels = [full_dataset.domain_labels[i] for i in non_neutral_orig_idx]
    # onsets keys are file stems → shared reference is fine (read-only)

    train_idx_6: list[int] = [orig_to_ds2[i] for i in train_ds.indices if i in orig_to_ds2]
    val_idx_6:   list[int] = [orig_to_ds2[i] for i in val_ds.indices   if i in orig_to_ds2]

    train_ds2: Subset = Subset(ds2, train_idx_6)
    val_ds2:   Subset = Subset(ds2, val_idx_6)

    print(f"\n[cascade] ══════════ STAGE 2: Ekman-6 (non-neutral) ══════════")
    print(f"[cascade]   label map : {ds2.label_map}")
    print(f"[cascade]   train     : {len(train_ds2)} samples")
    print(f"[cascade]   val       : {len(val_ds2)} samples")

    run_single_split(args, device, len(ds2.label_map), train_ds2, val_ds2, "cascade_stage2_ekman6")

    # ── CASCADE EVALUATION ───────────────────────────────────────────────
    print("\n[cascade] ══════════ Cascade Evaluation (7-class) ══════════")
    ckpt1 = f"best_{args.backbone}_flow_cascade_stage1_binary.pth"
    ckpt2 = f"best_{args.backbone}_flow_cascade_stage2_ekman6.pth"

    for ckpt in (ckpt1, ckpt2):
        if not Path(ckpt).exists():
            print(f"[cascade] Checkpoint not found: {ckpt} — skipping cascade eval.")
            return

    # Rebuild both models and load checkpoints
    def _build_flow_classifier(n_classes: int) -> "FlowClassifier":
        return FlowClassifier(
            num_classes=n_classes,
            pretrained=False,
            freeze_backbone=False,
            backbone=args.backbone,
            use_dann=False,
            use_temporal_attn=bool(getattr(args, "temporal_attn", False)),
            use_lstm=bool(getattr(args, "use_lstm", False)),
            lstm_hidden=int(getattr(args, "lstm_hidden", 128)),
            lstm_layers=int(getattr(args, "lstm_layers", 1)),
            use_gru=bool(getattr(args, "use_gru", False)),
            gru_hidden=int(getattr(args, "gru_hidden", 256)),
            gru_layers=int(getattr(args, "gru_layers", 1)),
            use_tcn=bool(getattr(args, "use_tcn", False)),
            tcn_channels=int(getattr(args, "tcn_channels", 256)),
        ).to(device)

    model1 = _build_flow_classifier(2)
    model1.load_state_dict(torch.load(ckpt1, map_location=device, weights_only=True))
    model1.eval()

    model2 = _build_flow_classifier(len(ds2.label_map))
    model2.load_state_dict(torch.load(ckpt2, map_location=device, weights_only=True))
    model2.eval()

    # Stage 2 index → full 7-class index (both label maps are sorted alphabetically)
    stage2_to_full: dict[int, int] = {
        ds2_idx: full_dataset.label_map[emo]
        for emo, ds2_idx in ds2.label_map.items()
    }

    val_loader_7 = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    num_classes_full = len(full_dataset.label_map)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for frames, labels in val_loader_7:
            frames = frames.to(device)
            # Stage 1 prediction
            out1 = model1(frames)
            s1_logits = out1[0] if isinstance(out1, tuple) else out1
            s1_pred = s1_logits.argmax(dim=1)  # 0=non_neutral, 1=neutral
            # Stage 2 prediction (run on all, select conditionally)
            out2 = model2(frames)
            s2_logits = out2[0] if isinstance(out2, tuple) else out2
            s2_pred_raw = s2_logits.argmax(dim=1)

            for i in range(len(labels)):
                if s1_pred[i].item() == 1:   # Stage 1 → neutral
                    y_pred.append(neutral_full_idx)
                else:                          # Stage 1 → non-neutral, use Stage 2
                    y_pred.append(stage2_to_full[s2_pred_raw[i].item()])
                y_true.append(labels[i].item())

    cascade_cm  = confusion_matrix_np(y_true, y_pred, num_classes_full)
    cascade_acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    cascade_f1         = macro_f1_from_confusion(cascade_cm)
    cascade_f1_present = macro_f1_present_from_confusion(cascade_cm)
    cascade_prec, cascade_rec = macro_precision_recall_from_confusion(cascade_cm)

    label_names_full = [k for k, v in sorted(full_dataset.label_map.items(), key=lambda x: x[1])]
    print(f"[cascade] val accuracy  (7-class): {cascade_acc:.3f}")
    print(f"[cascade] val macro-F1  (7-class): {cascade_f1:.3f}")
    print(f"[cascade] val macro-F1-present:    {cascade_f1_present:.3f}")
    print(f"[cascade] val precision (7-class): {cascade_prec:.3f}")
    print(f"[cascade] val recall    (7-class): {cascade_rec:.3f}")
    print("[cascade] confusion matrix  (rows=true, cols=pred):")
    print(f"[cascade] classes: {label_names_full}")
    print(cascade_cm)


def main() -> None:
    args = parse_args()

    if args.data_dir == Config.data_dir:
        args.data_dir = args.input_path
    if args.csv_index == Config.csv_index:
        csv_candidate = Path(args.input_path) / "extraction_index.csv"
        args.csv_index = str(csv_candidate) if csv_candidate.exists() else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"[device] {device}")

    # ── Preprocessing transforms ──────────────────────────────────────────
    transform = build_transforms()

    # ── Build dataset ─────────────────────────────────────────────────────
    print(f"[data]  scanning '{args.data_dir}' ...")
    if args.extra_data_dir:
        print(f"[data]  extra dataset: '{args.extra_data_dir}'")
    if args.extra_csv_index:
        print(f"[data]  extra csv: '{args.extra_csv_index}'")
    if args.smic_csv_index:
        print(f"[data]  smic csv: '{args.smic_csv_index}'")
        if args.smic_only_surprise:
            print("[data]  smic filter: surprise-only")
    if args.meme_csv_index:
        print(f"[data]  meme csv: '{args.meme_csv_index}'")
    if args.meme_csv_index_2:
        print(f"[data]  meme csv #2: '{args.meme_csv_index_2}'")
    if args.neutral_csv_index:
        print(f"[data]  neutral csv: '{args.neutral_csv_index}'")
    # Procesar casme3_classes como set si está presente
    casme3_classes_set = None
    if args.casme3_classes:
        casme3_classes_set = set(c.strip().lower() for c in args.casme3_classes.split(",") if c.strip())

    full_dataset = FlowSequenceDataset(
        data_dir   = args.data_dir,
        transform  = transform,
        num_frames = args.num_frames,
        csv_index  = args.csv_index,
        extra_csv_index=args.extra_csv_index,
        smic_csv_index=args.smic_csv_index,
        meme_csv_index=args.meme_csv_index,
        meme_csv_index_2=args.meme_csv_index_2,
        neutral_csv_index=args.neutral_csv_index,
        onset_csv  = args.onset_csv,
        temporal_aug_prob=args.temporal_aug_prob,
        extra_data_dir=args.extra_data_dir,
        smic_only_surprise=bool(getattr(args, "smic_only_surprise", False)),
        oversample_minority=args.oversample_minority,
        min_samples_per_class=args.min_samples_per_class,
        dual_phase=bool(getattr(args, "dual_phase_flow", False)),
        casme3_classes=casme3_classes_set,
        happiness_limit=args.happiness_limit,
    )
    
    # ── Keep only Ekman-7 classes (default; includes neutral when present) ─
    if args.only_ekman7:
        before = len(full_dataset.samples)
        filtered: list[tuple[Path, str, str]] = []
        filtered_domains: list[str] = []
        dropped = 0
        _src_domains = (
            full_dataset.domain_labels
            if len(full_dataset.domain_labels) == len(full_dataset.samples)
            else ["primary"] * len(full_dataset.samples)
        )
        for (path, emotion, subject), domain in zip(full_dataset.samples, _src_domains):
            canonical = canonicalize_ekman7(emotion)
            if canonical is None:
                dropped += 1
                continue
            # Neutral is only accepted from the dedicated neutral CSV (meme_v2_neutral).
            # Sequences from other domains whose raw label aliases to neutral
            # (e.g. "others", "rep", "repression") are excluded to avoid noise.
            if canonical == "neutral" and domain != "meme_v2_neutral":
                dropped += 1
                continue
            filtered.append((path, canonical, subject))
            filtered_domains.append(domain)

        full_dataset.samples = filtered
        full_dataset.domain_labels = filtered_domains
        full_dataset.label_map = build_label_map({emotion for _, emotion, _ in full_dataset.samples})

        print(
            f"[data]  Ekman-7 filter enabled: kept {len(full_dataset.samples)}/{before} "
            f"samples, dropped {dropped}."
        )
        print(f"[data]  Ekman-7 target classes: {EKMAN7_CANONICAL}")

    # Optional class exclusion after Ekman-7 canonicalization.
    _raw_excluded = (getattr(args, "exclude_ekman7_classes", "") or "").strip()
    if _raw_excluded:
        if not args.only_ekman7:
            print(
                "[data]  WARNING: --exclude_ekman7_classes requires --only_ekman7; ignoring exclusions."
            )
        else:
            _requested = [x.strip().lower() for x in _raw_excluded.split(",") if x.strip()]
            _exclude_set: set[str] = set()
            for _name in _requested:
                _canon = canonicalize_ekman7(_name)
                if _canon is None:
                    print(f"[data]  WARNING: unknown class in --exclude_ekman7_classes '{_name}', skipping")
                    continue
                if _canon == "neutral":
                    print("[data]  WARNING: 'neutral' cannot be excluded; skipping")
                    continue
                _exclude_set.add(_canon)

            if _exclude_set:
                _before_excl = len(full_dataset.samples)
                _excluded_samples = sum(1 for _, e, _ in full_dataset.samples if e in _exclude_set)
                _src_domains = (
                    full_dataset.domain_labels
                    if len(full_dataset.domain_labels) == len(full_dataset.samples)
                    else ["primary"] * len(full_dataset.samples)
                )
                _kept_samples: list[tuple[Path, str, str]] = []
                _kept_domains: list[str] = []
                for (path, emotion, subject), domain in zip(full_dataset.samples, _src_domains):
                    if emotion in _exclude_set:
                        continue
                    _kept_samples.append((path, emotion, subject))
                    _kept_domains.append(domain)

                full_dataset.samples = _kept_samples
                full_dataset.domain_labels = _kept_domains
                full_dataset.label_map = build_label_map({emotion for _, emotion, _ in full_dataset.samples})

                print(
                    f"[data]  excluded classes: {tuple(sorted(_exclude_set))} | "
                    f"dropped {_excluded_samples}/{_before_excl} samples"
                )
            else:
                print("[data]  no valid classes to exclude after validation")
    
    num_classes = full_dataset.get_num_classes()
    label_map   = full_dataset.get_label_map()
    print(f"[data]  {len(full_dataset)} sequences | {num_classes} classes: {label_map}")

    if len(full_dataset) == 0:
        raise RuntimeError(
            "No sequences found. Check --data_dir and --csv_index paths."
        )

    # ── Split strategy: single split or cross-validation ──────────────────
    # When --casme2_train_meme_val is set:
    # 1. Train on 100% CASME2 + SMIC(sorpresa only) + MEME supplementation for scarce/missing classes
    # 2. Validate on 100% MEME only (pure test domain)
    # This keeps MEME as the target validation domain while using SMIC surprise to strengthen CASME2 training.
    if getattr(args, "casme2_train_meme_val", False) and full_dataset.domain_labels:
        primary_idx = [i for i, d in enumerate(full_dataset.domain_labels) if d == "primary"]
        meme_domains = {"meme", "extra"}
        meme_idx = [i for i, d in enumerate(full_dataset.domain_labels) if d in meme_domains]
        smic_idx = [i for i, d in enumerate(full_dataset.domain_labels) if d == "smic"]
        
        if not primary_idx or not meme_idx:
            raise RuntimeError(
                "--casme2_train_meme_val requires both primary (CASME2) and MEME domain data."
            )

        # Use only SMIC surprise samples in training augmentation.
        surprise_label_idx = full_dataset.label_map.get("sorpresa")
        smic_surprise_idx: list[int] = []
        if surprise_label_idx is not None and smic_idx:
            smic_surprise_idx = [
                i for i in smic_idx
                if full_dataset.label_map[full_dataset.samples[i][1]] == surprise_label_idx
            ]
            print(
                f"[split] SMIC available: {len(smic_idx)} samples, "
                f"using {len(smic_surprise_idx)} 'sorpresa' samples for train"
            )
        elif smic_idx:
            print("[split] WARNING: class 'sorpresa' not found in label map, SMIC not used.")
        
        # Get class distribution in CASME2
        _casme2_labels = [full_dataset.label_map[full_dataset.samples[i][1]] for i in primary_idx]
        _casme2_classes = set(_casme2_labels)
        _all_classes = set(full_dataset.label_map.values())
        _missing_in_casme2 = _all_classes - _casme2_classes
        _inv_label_map = {v: k for k, v in full_dataset.label_map.items()}
        
        # Count CASME2 samples per class
        _casme2_counts = {cls: sum(1 for lbl in _casme2_labels if lbl == cls) for cls in _all_classes}
        
        print(f"[split] CASME2 class counts: {[_casme2_counts[c] for c in sorted(_all_classes)]}")
        print(f"[split] CASME2 class names: {sorted([_inv_label_map[c] for c in _casme2_classes])}")
        if _missing_in_casme2:
            print(f"[split] Classes MISSING in CASME2: {sorted([_inv_label_map[c] for c in _missing_in_casme2])}")
        else:
            print(f"[split] All classes present in CASME2")
        
        # Split MEME into train/val by subject
        meme_train, meme_val = split_indices_subject_wise(
            full_dataset, meme_idx, args.val_split, args.seed
        )
        
        # Determine which classes need MEME supplementation in training
        _min_casme2_per_class = max(0, int(getattr(args, "min_samples_per_class_casme2", 0)))
        _classes_needing_meme = set()
        
        if _min_casme2_per_class > 0:
            for cls in _all_classes:
                if _casme2_counts[cls] < _min_casme2_per_class:
                    _classes_needing_meme.add(cls)
                    print(
                        f"[split] class '{_inv_label_map[cls]}': only {_casme2_counts[cls]} in CASME2, "
                        f"needs {_min_casme2_per_class} → will complement with MEME"
                    )
        
        # Add missing classes to the needing-meme set
        _classes_needing_meme.update(_missing_in_casme2)
        
        # Add MEME train samples to the training set, but only for classes needing complement
        _meme_supplemented = []
        for cls_idx in _classes_needing_meme:
            cls_name = _inv_label_map[cls_idx]
            cls_meme_train = [
                i for i in meme_train
                if full_dataset.label_map[full_dataset.samples[i][1]] == cls_idx
            ]
            if cls_meme_train:
                # If min_samples_per_class_casme2 is set, take enough to reach the minimum
                if _min_casme2_per_class > 0 and cls_idx not in _missing_in_casme2:
                    need = _min_casme2_per_class - _casme2_counts[cls_idx]
                    take_n = min(need, len(cls_meme_train))
                else:
                    # For missing classes, take all available
                    take_n = len(cls_meme_train)
                
                _meme_supplemented.extend(cls_meme_train[:take_n])
                print(
                    f"[split] class '{cls_name}': "
                    f"adding {take_n} MEME train samples (need={_min_casme2_per_class if _min_casme2_per_class > 0 else 'all'})"
                )
            else:
                print(
                    f"[split] WARNING: class '{cls_name}' also absent in MEME train. "
                    "Using only MEME val samples for this class during training may cause issues."
                )
        
        # Build training set: 100% CASME2 + SMIC(sorpresa) + MEME supplementation
        train_indices = primary_idx + smic_surprise_idx + _meme_supplemented
        train_ds = Subset(full_dataset, train_indices)
        
        # Validation: 100% MEME (pure)
        val_ds = Subset(full_dataset, list(meme_val))
        
        _val_labels = [full_dataset.label_map[full_dataset.samples[i][1]] for i in meme_val]
        _val_classes = set(_val_labels)
        _missing_val = sorted(_all_classes - _val_classes)
        
        print(
            f"[split] --casme2_train_meme_val: "
            f"train={len(primary_idx)} CASME2 + {len(smic_surprise_idx)} SMIC(sorpresa) "
            f"+ {len(_meme_supplemented)} MEME (rare/missing classes) = {len(train_ds)} total | "
            f"val={len(val_ds)} MEME (pure)"
        )
        
        if _missing_val:
            _missing_names = [_inv_label_map[c] for c in _missing_val]
            print(
                f"[split] WARNING: validation missing classes: {_missing_names}. "
                "These may be absent in MEME entirely."
            )
            print(
                "[split] NOTE: with onset/apex/offset event labels, some classes can be absent in a domain split. "
                "Use monitor_metric=macro_f1_present for robust model selection in this setting."
            )
        
        run_single_split(args, device, num_classes, train_ds, val_ds, "casme2_train_meme_val")
        return
    
    # When --stratified_class_split is set:
    # Split ALL data (CASME2 + SMIC + MEME) with class-stratification
    # Each class is proportionally represented in train (80%) and val (20%)
    if getattr(args, "stratified_class_split", False):
        all_indices = list(range(len(full_dataset)))
        
        # Get class labels for all samples
        all_labels = [full_dataset.label_map[full_dataset.samples[i][1]] for i in all_indices]
        _inv_label_map = {v: k for k, v in full_dataset.label_map.items()}

        # ── CASME3 happiness → train-only guardrail ───────────────────────
        # Happiness samples from CASME3 (domain="extra") must NEVER go to val.
        # Validation for happiness class must only use MEME and CASME2 samples.
        _happiness_names = {"felicidad", "happiness", "happy", "hap"}
        _happiness_cls_indices = {
            v for k, v in full_dataset.label_map.items() if k.lower() in _happiness_names
        }
        _casme3_happiness_train_only: list[int] = []
        if full_dataset.domain_labels:
            for i in all_indices:
                _dom = full_dataset.domain_labels[i]
                _emo = full_dataset.samples[i][1].lower()
                _cls = full_dataset.label_map.get(full_dataset.samples[i][1])
                if _dom == "extra" and _cls in _happiness_cls_indices:
                    _casme3_happiness_train_only.append(i)
        if _casme3_happiness_train_only:
            print(
                f"[split] CASME3 happiness guardrail: {len(_casme3_happiness_train_only)} samples "
                "forced into train-only (will NEVER appear in val)."
            )
        # ─────────────────────────────────────────────────────────────────

        # Parse per-class validation split overrides (e.g. enojo/miedo/tristeza only).
        _class_val_overrides: dict[int, float] = {}
        _raw_overrides = (getattr(args, "val_split_by_class", "") or "").strip()
        if _raw_overrides:
            for _pair in _raw_overrides.split(","):
                _pair = _pair.strip()
                if not _pair:
                    continue
                if ":" not in _pair:
                    print(f"[split] WARNING: invalid val_split_by_class entry '{_pair}', expected class:ratio")
                    continue

                _cls_name, _ratio_str = _pair.rsplit(":", 1)
                _cls_name = _cls_name.strip()
                _ratio_str = _ratio_str.strip()

                if _cls_name not in full_dataset.label_map:
                    print(f"[split] WARNING: val_split_by_class unknown class '{_cls_name}', skipping")
                    continue

                try:
                    _ratio = float(_ratio_str)
                except ValueError:
                    print(f"[split] WARNING: val_split_by_class invalid ratio '{_ratio_str}' for '{_cls_name}', skipping")
                    continue

                if not (0.0 < _ratio < 1.0):
                    print(f"[split] WARNING: val_split_by_class ratio out of range for '{_cls_name}' ({_ratio}); expected (0,1), skipping")
                    continue

                _class_val_overrides[full_dataset.label_map[_cls_name]] = _ratio

        if _class_val_overrides:
            _override_msg = ", ".join(
                f"{_inv_label_map[_idx]}={_ratio:.2f}"
                for _idx, _ratio in sorted(_class_val_overrides.items())
            )
            print(f"[split] class-specific val_split overrides: {_override_msg}")
        
        # Perform stratified split; CASME3 happiness indices are excluded from the
        # splittable pool and go directly to train via train_only_indices.
        train_idx, val_idx = split_stratified_by_class(
            full_dataset,
            all_indices,
            args.val_split,
            args.seed,
            val_split_overrides=_class_val_overrides if _class_val_overrides else None,
            val_per_class=getattr(args, "val_per_class", None),
            train_only_indices=_casme3_happiness_train_only if _casme3_happiness_train_only else None,
        )

        # ── Explicit post-split verification ─────────────────────────────
        if _casme3_happiness_train_only and full_dataset.domain_labels:
            _casme3_hap_set = set(_casme3_happiness_train_only)
            _leaked = [i for i in val_idx if i in _casme3_hap_set]
            if _leaked:
                raise RuntimeError(
                    f"[split] DATA LEAKAGE: {len(_leaked)} CASME3 happiness sample(s) "
                    "found in val despite train_only_indices guardrail. "
                    "This should never happen — please report this bug."
                )
            else:
                print("[split] ✓ Verified: no CASME3 happiness samples leaked into val.")
        # ─────────────────────────────────────────────────────────────────
        
        train_ds = Subset(full_dataset, train_idx)
        val_ds = Subset(full_dataset, val_idx)
        
        # Report split statistics
        _train_labels = [full_dataset.label_map[full_dataset.samples[i][1]] for i in train_idx]
        _val_labels = [full_dataset.label_map[full_dataset.samples[i][1]] for i in val_idx]
        
        print(f"[split] --stratified_class_split (class-stratified, base val_split={args.val_split:.2f}):")
        print(f"[split]   train: {len(train_idx)} samples")
        for cls_idx in sorted(set(all_labels)):
            cls_name = _inv_label_map[cls_idx]
            train_count = sum(1 for lbl in _train_labels if lbl == cls_idx)
            val_count = sum(1 for lbl in _val_labels if lbl == cls_idx)
            total_count = train_count + val_count
            if total_count > 0:
                print(
                    f"[split]     {cls_name:10s}: train={train_count:3d} ({100*train_count/total_count:.1f}%) | "
                    f"val={val_count:3d} ({100*val_count/total_count:.1f}%)"
                )
            else:
                print(
                    f"[split]     {cls_name:10s}: train={train_count:3d} (  -%) | "
                    f"val={val_count:3d} (  -%)"
                )
        print(f"[split]   val: {len(val_idx)} samples")

        if getattr(args, "cascade_neutral", False):
            run_cascade_neutral(args, device, full_dataset, train_ds, val_ds)
        else:
            run_single_split(args, device, num_classes, train_ds, val_ds, "stratified_class_split", full_dataset)
        return
    
    # When --primary_only_val is set, split primary-domain (CASME2) subjects
    # for train/val. If some classes are absent/scarce in primary validation,
    # inject those classes from extra domain into val, while keeping the
    # remaining extra-domain samples in train.
    if getattr(args, "primary_only_val", False) and full_dataset.domain_labels:
        primary_idx = [i for i, d in enumerate(full_dataset.domain_labels) if d == "primary"]
        extra_idx   = [i for i, d in enumerate(full_dataset.domain_labels) if d != "primary"]
        _primary_labels = [full_dataset.label_map[full_dataset.samples[i][1]] for i in primary_idx]
        print(f"[split] primary-domain class counts: {label_counts(_primary_labels, num_classes)}")

        _p_train, _p_val = split_indices_subject_wise(
            full_dataset, primary_idx, args.val_split, args.seed
        )

        _min_val_per_class = max(1, int(getattr(args, "hybrid_val_min_per_class", 1)))
        _all_classes = set(full_dataset.label_map.values())
        _primary_classes = {
            full_dataset.label_map[full_dataset.samples[i][1]] for i in primary_idx
        }
        _inv_label_map = {v: k for k, v in full_dataset.label_map.items()}

        _val_primary_counts = np.zeros(num_classes, dtype=np.int64)
        for i in _p_val:
            _val_primary_counts[full_dataset.label_map[full_dataset.samples[i][1]]] += 1
        _val_primary_counts_list = [int(x) for x in _val_primary_counts.tolist()]
        print(
            f"[split] primary val class counts before hybrid inject: {_val_primary_counts_list} "
            f"(target min per class={_min_val_per_class})"
        )

        _needs_inject: dict[int, int] = {}
        for cls_idx in sorted(_all_classes):
            need = _min_val_per_class - int(_val_primary_counts[cls_idx])
            if need > 0:
                _needs_inject[cls_idx] = need

        _needed_names = [_inv_label_map[c] for c in sorted(_needs_inject.keys())]
        if _needed_names:
            print(f"[split] classes requiring extra-domain val top-up: {_needed_names}")
        else:
            print("[split] no extra-domain val top-up required")

        extra_val_missing: list[int] = []
        extra_train_missing: list[int] = []
        injected_classes_set = set(_needs_inject.keys())

        # ── CASME3 happiness → train-only guardrail (primary_only_val) ──────
        # For happiness class, only MEME/CASME2 samples may enter val.
        # Any extra-domain happiness sample that comes from CASME3 (domain="extra")
        # must be routed to train regardless of inject logic.
        _hap_names_pov = {"felicidad", "happiness", "happy", "hap"}
        _hap_cls_indices_pov = {
            v for k, v in full_dataset.label_map.items() if k.lower() in _hap_names_pov
        }
        _casme3_hap_extra = set()
        if full_dataset.domain_labels:
            for i in extra_idx:
                _dom_i = full_dataset.domain_labels[i]
                _cls_i = full_dataset.label_map.get(full_dataset.samples[i][1])
                if _dom_i == "extra" and _cls_i in _hap_cls_indices_pov:
                    _casme3_hap_extra.add(i)
        if _casme3_hap_extra:
            print(
                f"[split] primary_only_val: CASME3 happiness guardrail active — "
                f"{len(_casme3_hap_extra)} CASME3 happiness sample(s) will not be injected into val."
            )
        # ─────────────────────────────────────────────────────────────────

        for missing_cls, needed in _needs_inject.items():
            cls_extra_idx = [
                i for i in extra_idx
                if full_dataset.label_map[full_dataset.samples[i][1]] == missing_cls
                # Exclude CASME3 happiness from the inject pool for val
                and i not in _casme3_hap_extra
            ]
            if not cls_extra_idx:
                print(
                    f"[split] WARNING: class '{_inv_label_map[missing_cls]}' is missing "
                    "in primary and has no extra-domain samples to inject into validation."
                )
                continue

            cls_train, cls_val_initial = split_indices_subject_wise(
                full_dataset, cls_extra_idx, args.val_split, args.seed
            )

            # Keep subject-wise split first, then guarantee minimum support in val.
            cls_train = list(cls_train)
            cls_val = list(cls_val_initial)
            if len(cls_val) < needed and len(cls_train) > 0:
                rng = random.Random(args.seed + missing_cls)
                extra_pool = cls_train.copy()
                rng.shuffle(extra_pool)
                take_n = min(needed - len(cls_val), len(extra_pool))
                to_val = set(extra_pool[:take_n])
                cls_val.extend(list(to_val))
                cls_train = [idx for idx in cls_train if idx not in to_val]

            extra_train_missing.extend(cls_train)
            extra_val_missing.extend(cls_val)

            print(
                f"[split] inject class '{_inv_label_map[missing_cls]}' from extra: "
                f"need={needed}, train+={len(cls_train)}, val+={len(cls_val)}"
            )

        extra_non_missing = [
            i for i in extra_idx
            if full_dataset.label_map[full_dataset.samples[i][1]] not in injected_classes_set
        ]

        all_train_idx = _p_train + extra_non_missing + extra_train_missing
        all_val_idx = _p_val + extra_val_missing
        all_train_idx = sorted(set(all_train_idx))
        all_val_idx = sorted(set(all_val_idx))

        overlap = set(all_train_idx) & set(all_val_idx)
        if overlap:
            raise RuntimeError(
                "Data leakage detected in primary_only_val split: train/val overlap found."
            )

        # ── Post-split verification: CASME3 happiness must not be in val ──
        if _casme3_hap_extra:
            _hap_val_leaked = [i for i in all_val_idx if i in _casme3_hap_extra]
            if _hap_val_leaked:
                raise RuntimeError(
                    f"[split] DATA LEAKAGE: {len(_hap_val_leaked)} CASME3 happiness sample(s) "
                    "found in val in primary_only_val mode. "
                    "This should never happen — please report this bug."
                )
            else:
                print("[split] ✓ Verified: no CASME3 happiness samples leaked into val (primary_only_val).")
        # ─────────────────────────────────────────────────────────────────

        train_ds = Subset(full_dataset, all_train_idx)
        val_ds   = Subset(full_dataset, all_val_idx)

        print(
            f"[split] --primary_only_val hybrid: {len(primary_idx)} primary split + "
            f"{len(extra_non_missing)} extra non-missing→train + "
            f"{len(extra_train_missing)} injected-class extra→train + "
            f"{len(extra_val_missing)} injected-class extra→val"
        )

        _val_primary_classes = {
            full_dataset.label_map[full_dataset.samples[i][1]] for i in _p_val
        }
        _missing_primary_val = sorted(_primary_classes - _val_primary_classes)
        if _missing_primary_val:
            _missing_names = [_inv_label_map[c] for c in _missing_primary_val]
            print(
                f"[split] WARNING: primary-domain validation is missing classes: {_missing_names}. "
                "Consider adjusting --seed or --val_split if full primary coverage is required."
            )

        _val_classes = {
            full_dataset.label_map[full_dataset.samples[i][1]] for i in all_val_idx
        }
        _missing_val = sorted(_all_classes - _val_classes)
        if _missing_val:
            _missing_names = [_inv_label_map[c] for c in _missing_val]
            print(
                f"[split] WARNING: global validation is missing classes: {_missing_names}. "
                "Some classes may be too scarce in primary/extra for current split settings."
            )
        print(f"[split] primary_only_val_hybrid: train={len(train_ds)}, val={len(val_ds)}")
        run_single_split(args, device, num_classes, train_ds, val_ds, "primary_only_val_hybrid", full_dataset)
        return

    if args.split_mode == "subject":
        train_ds, val_ds = split_subject_wise(full_dataset, args.val_split, args.seed)
        print("[split] subject-wise split enabled")
        run_single_split(args, device, num_classes, train_ds, val_ds, "subject", full_dataset)
        return

    if args.split_mode == "random":
        val_size = max(1, int(len(full_dataset) * args.val_split))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print("[split] random split enabled")
        run_single_split(args, device, num_classes, train_ds, val_ds, "random", full_dataset)
        return

    if args.split_mode == "stratified_random":
        train_ds, val_ds = split_stratified_random(full_dataset, args.val_split, args.seed)
        print("[split] stratified random split enabled")
        run_single_split(args, device, num_classes, train_ds, val_ds, "stratified_random", full_dataset)
        return

    if args.split_mode == "loso":
        folds = build_loso_folds(full_dataset)
        print(f"[split] LOSO enabled | folds={len(folds)}")
    else:
        folds = build_subject_kfolds(full_dataset, args.num_folds, args.seed)
        print(f"[split] subject_kfold enabled | folds={len(folds)}")

    if len(folds) == 0:
        raise RuntimeError("No valid folds were created. Check data and split settings.")

    fold_accs: list[float] = []
    fold_f1s: list[float] = []

    for i, (train_ds, val_ds, fold_tag) in enumerate(folds, start=1):
        print("\n" + "=" * 80)
        print(f"[fold {i}/{len(folds)}] {fold_tag}")
        print("=" * 80)
        best_acc, best_f1 = run_single_split(args, device, num_classes, train_ds, val_ds, fold_tag, full_dataset)
        fold_accs.append(best_acc)
        fold_f1s.append(best_f1)

    acc_mean = float(np.mean(fold_accs))
    acc_std = float(np.std(fold_accs))
    f1_mean = float(np.mean(fold_f1s))
    f1_std = float(np.std(fold_f1s))

    print("\n" + "#" * 80)
    print("[cv summary]")
    print(f"best-val-acc per fold: {[round(x, 3) for x in fold_accs]}")
    print(f"best-val-f1  per fold: {[round(x, 3) for x in fold_f1s]}")
    print(f"acc mean±std: {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"f1  mean±std: {f1_mean:.3f} ± {f1_std:.3f}")
    print("#" * 80)


if __name__ == "__main__":
    main()