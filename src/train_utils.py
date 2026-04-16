from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class TrainConfig:
    out_dir: Path
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    seed: int = 42
    bidirectional: bool = False
    arch: str = "lstm"
    transformer_kwargs: dict | None = None
    balanced_sampler: bool = False
    device: str | None = None
    normalize: bool = True
    patience: int = 5
    clip: float = 1.0
    use_scheduler: bool = True
    validation_mode: str = "split"
    subject_col: str = "subject_id"
    label_col: str = "label"
    loss_type: str = "ce"
    focal_gamma: float = 2.0
    test_subject: str | None = None
    input_mode: str = "feature"
    pretrained_weights: str | None = None

    @classmethod
    def from_kwargs(cls, out_dir: str | Path, **kwargs) -> "TrainConfig":
        data = dict(kwargs)
        data["out_dir"] = Path(out_dir)
        return cls(**data)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class WarmupScheduler:
    def __init__(self, optimizer, base_lr: float, warmup_steps: int = 500):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            steps_after = self.step_num - self.warmup_steps
            lr = self.base_lr * (1.0 / (1.0 + 0.1 * steps_after ** 0.5))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


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
            x = (x - self.mean[None, :]) / (self.std[None, :] + 1e-8)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class SequenceContractDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_series: pd.Series):
        self.df = df.reset_index(drop=True)
        self.labels = label_series.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _load_array(path_value) -> np.ndarray:
        if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
            raise ValueError("Missing required array path")
        return np.load(str(path_value))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row.get("clip_path")
        landmarks_path = row.get("landmarks_path")
        if video_path is None or landmarks_path is None:
            raise ValueError(f"Faltan clip_path o landmarks_path para el índice {idx}")
        if isinstance(video_path, float) and np.isnan(video_path):
            raise ValueError(f"clip_path inválido para el índice {idx}")
        if isinstance(landmarks_path, float) and np.isnan(landmarks_path):
            raise ValueError(f"landmarks_path inválido para el índice {idx}")

        video = self._load_array(video_path).astype(np.float32)
        if video.ndim == 4 and video.shape[-1] in (1, 3):
            video = np.transpose(video, (0, 3, 1, 2))
        if video.max() > 1.0:
            video = video / 255.0

        landmarks = self._load_array(landmarks_path).astype(np.float32)
        if landmarks.ndim == 3 and landmarks.shape[-1] == 2:
            landmarks = landmarks.astype(np.float32)
        elif landmarks.ndim == 2 and landmarks.shape[1] == 136:
            landmarks = landmarks.reshape(landmarks.shape[0], 68, 2)
        else:
            raise ValueError(f"Formato de landmarks inválido para {landmarks_path}: {landmarks.shape}")

        label = int(self.labels.iloc[idx])
        return torch.from_numpy(video), torch.from_numpy(landmarks), torch.tensor(label, dtype=torch.long)


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
    xs, ys = zip(*batch)
    lengths = torch.tensor([int(x.shape[0]) for x in xs], dtype=torch.long)
    xs_padded = pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs_padded, ys, lengths


def collate_pad_multimodal(batch):
    videos, landmarks, ys = zip(*batch)

    target_h = 224
    target_w = 224
    normalized_videos = []
    normalized_landmarks = []
    lengths_list = []

    for video, landmark in zip(videos, landmarks):
        if video.ndim != 4:
            raise ValueError(f"Se esperaba video [T, C, H, W], recibido {tuple(video.shape)}")

        if video.shape[-2] != target_h or video.shape[-1] != target_w:
            video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)

        min_t = min(int(video.shape[0]), int(landmark.shape[0]))
        if min_t <= 0:
            raise ValueError("Secuencia multimodal vacía tras sincronizar video y landmarks")

        video = video[:min_t]
        landmark = landmark[:min_t]

        normalized_videos.append(video)
        normalized_landmarks.append(landmark)
        lengths_list.append(min_t)

    lengths = torch.tensor(lengths_list, dtype=torch.long)
    videos_padded = pad_sequence(normalized_videos, batch_first=True)
    landmarks_padded = pad_sequence(normalized_landmarks, batch_first=True)
    ys = torch.stack(ys)
    return videos_padded, landmarks_padded, ys, lengths


def build_loss_function(loss_type: str, class_weights: np.ndarray, device: torch.device, focal_gamma: float = 2.0):
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if loss_type.lower() == "focal":
        return FocalLoss(alpha=weights, gamma=focal_gamma)
    return torch.nn.CrossEntropyLoss(weight=weights)


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_train).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    class_weights = (len(y_train) / counts).astype(np.float32)
    return class_weights / np.mean(class_weights)


def build_datasets(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_te: np.ndarray,
    input_mode: str,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
):
    if input_mode == "multimodal":
        train_ds = SequenceContractDataset(df_tr, pd.Series(y_tr))
        val_ds = SequenceContractDataset(df_va, pd.Series(y_va))
        test_ds = SequenceContractDataset(df_te, pd.Series(y_te))
        collate_fn = collate_pad_multimodal
    else:
        train_ds = NpySeqDataset(df_tr["feature_path"].tolist(), y_tr, mean=mean, std=std)
        val_ds = NpySeqDataset(df_va["feature_path"].tolist(), y_va, mean=mean, std=std)
        test_ds = NpySeqDataset(df_te["feature_path"].tolist(), y_te, mean=mean, std=std)
        collate_fn = collate_pad
    return train_ds, val_ds, test_ds, collate_fn