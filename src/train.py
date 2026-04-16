from __future__ import annotations

from pathlib import Path

from .engine import run_training
from .loso_manager import (
    build_split_bundle,
    default_person_split,
    resolve_series,
    split_by_persona,
    split_by_subject,
    split_train_val_by_subjects,
)
from .train_utils import (
    NpySeqDataset,
    SequenceContractDataset,
    TrainConfig,
    WarmupScheduler,
    build_datasets,
    build_loss_function,
    collate_pad,
    collate_pad_multimodal,
    compute_class_weights,
    compute_mean_std,
    set_seed,
)

__all__ = [
    "NpySeqDataset",
    "SequenceContractDataset",
    "TrainConfig",
    "WarmupScheduler",
    "build_datasets",
    "build_loss_function",
    "build_split_bundle",
    "collate_pad",
    "collate_pad_multimodal",
    "compute_class_weights",
    "compute_mean_std",
    "default_person_split",
    "resolve_series",
    "set_seed",
    "split_by_persona",
    "split_by_subject",
    "split_train_val_by_subjects",
    "train_lstm",
]


def train_lstm(df_index, out_dir, **kwargs):
    config = TrainConfig.from_kwargs(out_dir, **kwargs)
    return run_training(df_index, config)