"""
train_mex.py
----------------------
Sequence-level micro-expression classifier using configurable CNN backbones
as frame encoders (ResNet18/34/50, DenseNet121).

Pipeline per sample:
    .npy (N, H, W, 3) float32 optical flow [dx, dy, mag]
        → uniform frame sampling  → T frames (H, W, 3)
        → resize 224×224 + normalize (flow-adapted, 3-channel)
        → CNN backbone features   (B*T, 3, 224, 224) → (B*T, feat_dim)
    → reshape + temporal pooling  → (B, feat_dim)
    → Linear classifier           → (B, num_classes)

CAMBIOS vs versión anterior:
  - LSTM desactivado por defecto y protegido con guarda explícita
  - Bucles de 64 intentos en split_subject_wise / split_indices_subject_wise
    reducidos a 16 con early-exit → evita colgar el servidor
  - num_workers default = 0 para evitar deadlocks en servidores con poca RAM
  - Se eliminan referencias a variables no definidas en run_single_split (two_stage paths)
  - Nuevo --cap_majority_ratio: undersampling de clases mayoritarias al tope
    de min_clase * ratio (ej: min=25, ratio=3.0 → tope=75)
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)
from torchvision import models, transforms

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# LABEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

EKMAN7_CANONICAL: tuple[str, ...] = (
    "asco",
    #"enojo",
    "felicidad",
    #"miedo",
    "neutral",
    "sorpresa",
    "tristeza",
)

# Mantener alias antiguo para compatibilidad
EKMAN6_CANONICAL = EKMAN7_CANONICAL

_EKMAN6_ALIASES: dict[str, str] = {
    "disgust":    "asco",     "dis":       "asco",     "asco":      "asco",
    "happiness":  "felicidad","happy":     "felicidad","hap":       "felicidad","felicidad": "felicidad",
    "neutral":    "neutral",  "neu":       "neutral",  "neutra":    "neutral",
    "surprise":   "sorpresa", "sur":       "sorpresa", "sorpresa":  "sorpresa",
    "sadness":    "tristeza", "sad":       "tristeza", "tristeza":  "tristeza",
    #"fear":       "miedo",    "mie":       "miedo",    "miedo":     "miedo",
    #"fearful":    "miedo",
    #"anger":      "enojo",    "ang":       "enojo",    "enojo":     "enojo",
    #"angry":      "enojo",
}


def canonicalize_ekman6(emotion: str) -> Optional[str]:
    """Map raw emotion label to Ekman-7 canonical name (incl. neutral), or None if out-of-set."""
    return _EKMAN6_ALIASES.get(emotion.strip().lower())


def build_label_map(labels_found: set[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(sorted(labels_found))}


# ══════════════════════════════════════════════════════════════════════════════
# FRAME SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_frames(
    sequence: np.ndarray,
    num_frames: int,
    onset: Optional[int] = None,
    offset: Optional[int] = None,
) -> np.ndarray:
    n = sequence.shape[0]
    start = max(0, onset if onset is not None else 0)
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

    return sequence[indices]


def _extract_onset_apex_offset(
    sequence: np.ndarray,
    onset: int,
    apex: Optional[int],
    offset: int,
    num_frames: int,
) -> np.ndarray:
    n = sequence.shape[0]
    onset  = max(0, onset)
    offset = min(n - 1, offset)

    if num_frames == 3 and apex is not None:
        apex = max(onset, min(apex, offset))
        onset_frame  = (onset + apex) // 2
        apex_frame   = apex
        offset_frame = (apex + offset) // 2 if offset > apex else apex
        indices = [
            max(0, min(onset_frame,  n - 1)),
            max(0, min(apex_frame,   n - 1)),
            max(0, min(offset_frame, n - 1)),
        ]
        return sequence[indices]

    if apex is None:
        return sample_frames(sequence, num_frames, onset, offset + 1)

    apex = max(onset, min(apex, offset))
    part1 = sequence[onset:apex]
    part2 = sequence[apex:offset + 1]

    if part1.shape[0] == 0 and part2.shape[0] == 0:
        combined = sequence[onset:offset + 1]
    elif part1.shape[0] == 0:
        combined = part2
    elif part2.shape[0] == 0:
        combined = part1
    else:
        combined = np.concatenate([part1, part2], axis=0)

    if combined.shape[0] == 0:
        combined = sequence

    return sample_frames(combined, num_frames)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def build_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def _augment_temporal(frames: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() > p:
        return frames

    T = frames.shape[0]
    aug_type = np.random.choice(["drop", "jitter", "flip", "noise", "none"])

    if aug_type == "drop" and T > 2 and T // 2 > 1:
        num_drop = np.random.randint(1, min(3, T // 2))
        drop_indices = np.random.choice(range(1, T - 1), size=num_drop, replace=False)
        frames = np.delete(frames, drop_indices, axis=0)
        if len(frames) < T:
            pad = np.repeat(frames[-1:], T - len(frames), axis=0)
            frames = np.vstack([frames, pad])

    elif aug_type == "jitter" and T > 2:
        idx = np.arange(T)
        for i in range(1, T - 1):
            if np.random.rand() < 0.3:
                swap = np.random.choice([i - 1, i, i + 1])
                idx[i], idx[swap] = idx[swap], idx[i]
        frames = frames[idx]

    elif aug_type == "flip":
        frames = np.flip(frames, axis=2).copy()
        frames[..., 0] *= -1.0
        if np.random.rand() < 0.2:
            frames = np.flip(frames, axis=1).copy()
            frames[..., 1] *= -1.0

    elif aug_type == "noise":
        frames = frames.copy().astype(np.float32)
        scale = 0.05
        frames[..., 0] = np.clip(frames[..., 0] + np.random.normal(0, scale, frames[..., 0].shape), -1.0, 1.0)
        frames[..., 1] = np.clip(frames[..., 1] + np.random.normal(0, scale, frames[..., 1].shape), -1.0, 1.0)
        frames[..., 2] = np.clip(frames[..., 2] * (1.0 + np.random.normal(0, scale, frames[..., 2].shape)), 0.0, 1.0)

    return frames

def _aggressive_augment_flow(frames: np.ndarray, rng: np.random.Generator, strength: float = 0.6) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    T, H, W, C = frames.shape
    aug = frames.copy()
    prob_scale = 0.35 + 0.9 * strength
    def _p(base: float) -> bool: return rng.random() < min(0.95, max(0.0, base * prob_scale))

    if _p(0.5) and T > 3:
        jitter = rng.integers(-1, 2, size=T)
        new_idx = np.clip(np.arange(T) + jitter, 0, T - 1)
        aug = aug[new_idx]
    if _p(0.4):
        stretch = 0.05 + 0.25 * strength
        scale = rng.uniform(1.0 - stretch, 1.0 + stretch)
        new_T = max(T // 2, min(int(T * scale), T * 2))
        idx_stretch = np.linspace(0, T - 1, new_T, dtype=int)
        stretched = aug[idx_stretch]
        idx_final = np.linspace(0, len(stretched) - 1, T, dtype=int)
        aug = stretched[idx_final]
    if _p(0.3) and T > 4:
        keep_mask = rng.random(T) > 0.2
        if keep_mask.sum() >= T // 2:
            kept = aug[keep_mask]
            reps = (T // len(kept)) + 1
            aug = np.tile(kept, (reps, 1, 1, 1))[:T]
    if _p(0.5):
        aug = np.flip(aug, axis=2).copy()
        aug[..., 0] *= -1.0
    if _p(0.3):
        aug = np.flip(aug, axis=1).copy()
        aug[..., 1] *= -1.0
    if _p(0.4):
        max_angle = 5.0 + 12.0 * strength
        angle = rng.uniform(-max_angle, max_angle)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        try:
            from scipy.ndimage import rotate as ndimage_rotate
            rotated = np.stack([ndimage_rotate(aug[t], angle, axes=(0, 1), reshape=False, order=1) for t in range(T)])
            dx_rot = rotated[..., 0] * cos_a - rotated[..., 1] * sin_a
            dy_rot = rotated[..., 0] * sin_a + rotated[..., 1] * cos_a
            aug = rotated.copy()
            aug[..., 0], aug[..., 1] = dx_rot, dy_rot
        except ImportError:
            dx_rot = aug[..., 0] * cos_a - aug[..., 1] * sin_a
            dy_rot = aug[..., 0] * sin_a + aug[..., 1] * cos_a
            aug[..., 0], aug[..., 1] = dx_rot, dy_rot
    if _p(0.5):
        zoom = 0.05 + 0.20 * strength
        scale = rng.uniform(1.0 - zoom, 1.0 + zoom)
        crop_h = max(1, min(int(H / scale if scale > 1.0 else H * scale), H))
        crop_w = max(1, min(int(W / scale if scale > 1.0 else W * scale), W))
        start_h, start_w = (H - crop_h) // 2, (W - crop_w) // 2
        cropped = aug[:, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
        aug = np.pad(cropped, ((0, 0), (0, H - crop_h), (0, W - crop_w), (0, 0)), mode="edge")
    if _p(0.6):
        noise_scale = rng.uniform(0.01, 0.02 + 0.08 * strength)
        aug[..., 0] += rng.normal(0, noise_scale, (T, H, W))
        aug[..., 1] += rng.normal(0, noise_scale, (T, H, W))
        aug[..., 2] += rng.normal(0, noise_scale * 0.5, (T, H, W))
    if _p(0.4):
        mag_jitter = 0.05 + 0.10 * strength
        aug[..., 2] *= rng.uniform(1.0 - mag_jitter, 1.0 + mag_jitter)
    if _p(0.3):
        try:
            from scipy.ndimage import gaussian_filter
            sigma = rng.uniform(0.3, 0.8 + 0.7 * strength)
            for t in range(T):
                for c in range(3):
                    aug[t, :, :, c] = gaussian_filter(aug[t, :, :, c], sigma=sigma)
        except ImportError:
            pass

    aug[..., 0] = np.clip(aug[..., 0], -1.0, 1.0)
    aug[..., 1] = np.clip(aug[..., 1], -1.0, 1.0)
    aug[..., 2] = np.clip(aug[..., 2], 0.0, 1.0)
    return aug


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _flow_to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    f0 = np.clip((frames[..., 0] + 1.0) / 2.0, 0.0, 1.0)
    f1 = np.clip((frames[..., 1] + 1.0) / 2.0, 0.0, 1.0)
    f2 = np.clip(frames[..., 2], 0.0, 1.0)
    rgb = np.stack([f0, f1, f2], axis=-1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


class FlowSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir:   str | Path,
        transform:  transforms.Compose,
        num_frames: int = 8,
        csv_index:  Optional[str | Path] = None,
        extra_csv_index: Optional[str | Path] = None,
        smic_csv_index:  Optional[str | Path] = None,
        meme_csv_index:  Optional[str | Path] = None,
        onset_csv:  Optional[str | Path] = None,
        label_map:  Optional[dict[str, int]] = None,
        temporal_aug_prob: float = 0.5,
        extra_data_dir: Optional[str | Path] = None,
        smic_data_dir:  Optional[str | Path] = None,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.transform  = transform
        self.num_frames = num_frames
        self.temporal_aug_prob = float(np.clip(temporal_aug_prob, 0.0, 1.0))
        self.onsets: dict[str, tuple[int, Optional[int], int]] = {}
        if onset_csv:
            self._load_onset_csv(Path(onset_csv))

        self.samples: list[tuple[Path, str, str]] = []
        self.domain_labels: list[str] = []

        if csv_index:
            self._build_from_csv(Path(csv_index))
        elif self.data_dir.exists():
            self._build_from_dir()

        for path, domain_tag in [(smic_csv_index, "smic"), (meme_csv_index, "meme")]:
            if path:
                before = len(self.samples)
                self._build_from_csv(Path(path), domain=domain_tag)
                log.info(f"{domain_tag.upper()} csv '{path}': {len(self.samples) - before} sequences")

        if extra_csv_index and not meme_csv_index:
            before = len(self.samples)
            self._build_from_csv(Path(extra_csv_index), domain="extra")
            log.info(f"extra csv '{extra_csv_index}': {len(self.samples) - before} sequences")

        if extra_data_dir is not None:
            self._build_from_dir_meme(Path(extra_data_dir))
        if smic_data_dir is not None:
            self._build_from_dir_smic(Path(smic_data_dir))

        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = build_label_map({emotion for _, emotion, _ in self.samples})

    _KNOWN_EMOTIONS: frozenset[str] = frozenset(_EKMAN6_ALIASES.keys())

    def _build_from_dir(self) -> None:
        for npy_path in sorted(self.data_dir.rglob("*.npy")):
            parts = npy_path.relative_to(self.data_dir).parts
            if len(parts) < 3:
                continue
            subject, emotion = parts[0], parts[1].lower()
            if emotion in self._KNOWN_EMOTIONS:
                self.samples.append((npy_path, emotion, subject))
                self.domain_labels.append("primary")

    def _build_from_dir_meme(self, meme_dir: Path) -> None:
        SKIP_CAMERAS = {"camarainfrarroja"}
        added = skipped_cam = skipped_emo = 0
        for npy_path in sorted(meme_dir.rglob("*.npy")):
            parts = npy_path.relative_to(meme_dir).parts
            if len(parts) != 3:
                continue
            subject, emotion = parts[0], parts[1].lower()
            if any(skip in npy_path.stem.lower() for skip in SKIP_CAMERAS):
                skipped_cam += 1
                continue
            canonical = canonicalize_ekman6(emotion)
            if canonical is None:
                skipped_emo += 1
                continue
            self.samples.append((npy_path, canonical, subject))
            self.domain_labels.append("meme")
            added += 1
        log.info(f"MEME dataset '{meme_dir}': {added} loaded (skipped: {skipped_cam} infrared camera, {skipped_emo} unknown emotion)")

    def _build_from_dir_smic(self, smic_dir: Path) -> None:
        SURPRISE_ALIASES = {"surprise", "sur", "sorpresa"}
        added = skipped = 0
        for npy_path in sorted(smic_dir.rglob("*.npy")):
            parts = npy_path.relative_to(smic_dir).parts
            if len(parts) < 3:
                continue
            subject, emotion = parts[0], parts[1].lower()
            if emotion not in SURPRISE_ALIASES:
                skipped += 1
                continue
            self.samples.append((npy_path, "sorpresa", subject))
            self.domain_labels.append("smic")
            added += 1
        log.info(f"SMIC dataset '{smic_dir}': {added} surprise sequences loaded (skipped {skipped} non-surprise sequences)")

    def _build_from_csv(self, csv_path: Path, domain: str = "primary") -> None:
        base = csv_path.parent
        skipped_emo = skipped_path = loaded = 0
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel  = row.get("archivo", "").strip()
                emo  = row.get("emocion", "").strip().lower()
                subj = (
                    row.get("sujeto") or row.get("subject") or row.get("Subject")
                    or row.get("SUBJECT") or row.get("sub") or row.get("participant") or ""
                ).strip()
                if not rel or emo not in self._KNOWN_EMOTIONS:
                    skipped_emo += 1
                    continue
                if not subj:
                    rel_parts = Path(rel.replace("\\", "/")).parts
                    subj = rel_parts[0] if len(rel_parts) >= 2 else "unknown_subject"
                npy_path = base / Path(rel.replace("\\", "/"))
                if not npy_path.exists():
                    skipped_path += 1
                    continue
                self.samples.append((npy_path, emo, subj))
                self.domain_labels.append(domain)
                loaded += 1
                try:
                    _onset  = int(row["onset_frame"])  if row.get("onset_frame")  else None
                    _apex   = int(row["apex_frame"])   if row.get("apex_frame")   else None
                    _offset = int(row["offset_frame"]) if row.get("offset_frame") else None
                    if _onset is not None and _offset is not None:
                        self.onsets[npy_path.stem] = (_onset, _apex, _offset)
                except (ValueError, KeyError):
                    pass
        log.info(f"[csv:{domain}] '{csv_path.name}': {loaded} cargadas | "
                 f"{skipped_emo} emoción desconocida | {skipped_path} archivo no encontrado")

    def _load_onset_csv(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key    = Path(row.get("filename", "")).stem
                onset  = int(row["onset"])  if row.get("onset")  else None
                apex   = int(row["apex"])   if row.get("apex")   else None
                offset = int(row["offset"]) if row.get("offset") else None
                if key and onset is not None and offset is not None:
                    self.onsets[key] = (onset, apex, offset)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, emotion, _ = self.samples[idx]
        label = self.label_map[emotion]
        sequence = np.load(str(npy_path))

        onset_info = self.onsets.get(npy_path.stem)
        if onset_info is not None:
            _onset, _apex, _offset = onset_info
            frames = _extract_onset_apex_offset(sequence, _onset, _apex, _offset, self.num_frames)
        else:
            frames = sample_frames(sequence, self.num_frames)

        frames = _augment_temporal(frames, p=self.temporal_aug_prob)

        if frames.ndim == 4:
            rgb = _flow_to_uint8_rgb(frames)
        else:
            rgb = np.repeat(frames[..., None], 3, axis=-1)

        tensor_frames = torch.stack([self.transform(Image.fromarray(frame, mode="RGB")) for frame in rgb])
        return tensor_frames, label

    def get_num_classes(self) -> int:
        return len(self.label_map)

    def get_label_map(self) -> dict[str, int]:
        return dict(self.label_map)

    def get_subjects(self) -> list[str]:
        return sorted({subject for _, _, subject in self.samples})

    def get_subject_indices(self, subjects: set[str]) -> list[int]:
        return [i for i, (_, _, subject) in enumerate(self.samples) if subject in subjects]


class _AugmentedFlowDataset(Dataset):
    def __init__(self, arrays: list[np.ndarray], labels: list[int], transform: transforms.Compose) -> None:
        self._arrays, self._labels, self._transform = arrays, labels, transform

    def __len__(self) -> int:
        return len(self._arrays)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rgb = _flow_to_uint8_rgb(self._arrays[idx])
        tensor_frames = torch.stack([self._transform(Image.fromarray(frame, mode="RGB")) for frame in rgb])
        return tensor_frames, self._labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT UTILITIES
# FIX: Bucles reducidos de 64 → 16 intentos con early-exit para evitar
#      bloquear el servidor en datasets grandes.
# ══════════════════════════════════════════════════════════════════════════════

def split_subject_wise(dataset: FlowSequenceDataset, val_split: float, seed: int) -> tuple[Subset, Subset]:
    subjects = dataset.get_subjects()
    if len(subjects) < 2:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    target_val_subjects = max(1, min(int(round(len(subjects) * val_split)), len(subjects) - 1))
    all_labels = {dataset.label_map[emotion] for _, emotion, _ in dataset.samples}
    best_train_idx, best_val_idx, best_score = None, None, None

    # FIX: 64 → 16 intentos con early-exit cuando se alcanza cobertura completa
    for attempt in range(16):
        shuffled = subjects.copy()
        random.Random(seed + attempt).shuffle(shuffled)
        val_subjects   = set(shuffled[:target_val_subjects])
        train_subjects = set(shuffled[target_val_subjects:])
        train_idx = dataset.get_subject_indices(train_subjects)
        val_idx   = dataset.get_subject_indices(val_subjects)
        if not train_idx or not val_idx:
            continue
        train_labels = {dataset.label_map[dataset.samples[i][1]] for i in train_idx}
        val_labels   = {dataset.label_map[dataset.samples[i][1]] for i in val_idx}
        score = (
            len(train_labels & all_labels) + len(val_labels & all_labels),
            len(val_labels),
            -abs(len(val_idx) - round(len(dataset) * val_split)),
        )
        if best_score is None or score > best_score:
            best_score, best_train_idx, best_val_idx = score, train_idx, val_idx
            # Early-exit: si ya tenemos cobertura completa en ambos splits no hay mejor solución
            if len(train_labels & all_labels) == len(all_labels) and len(val_labels & all_labels) == len(all_labels):
                break

    train_idx, val_idx = best_train_idx or [], best_val_idx or []
    if not train_idx or not val_idx:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_domain_aware(
    dataset: FlowSequenceDataset,
    val_split: float,
    seed: int,
    min_per_class: int = 1,
) -> tuple[Subset, Subset]:
    """
    Split subject-wise corregido:
    - PRIMARY: Se divide subject-wise (80/20) entre train y val.
    - SMIC/MEME: El 100% de los datos se van a TRAIN para robustecer el modelo.
    - VAL: Queda puro con datos del dataset primario.
    """
    domains = dataset.domain_labels
    inv_map = {v: k for k, v in dataset.label_map.items()}

    # Agrupar índices por dominio (normalizar "extra" → "meme")
    domain_buckets: dict[str, list[int]] = {}
    for i, d in enumerate(domains):
        key = "meme" if d in ("extra", "meme") else d
        domain_buckets.setdefault(key, []).append(i)

    def _subject_split(indices: list[int], tag: str) -> tuple[list[int], list[int]]:
        if not indices:
            return [], []
        subj_to_idx: dict[str, list[int]] = {}
        for i in indices:
            subj_to_idx.setdefault(dataset.samples[i][2], []).append(i)
        subjects = sorted(subj_to_idx.keys())

        if len(subjects) < 2:
            rng = random.Random(seed)
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_split))
            return shuffled[n_val:], shuffled[:n_val]

        target_val_subjs = max(1, min(int(round(len(subjects) * val_split)), len(subjects) - 1))
        all_lbls = {dataset.label_map[dataset.samples[i][1]] for i in indices}
        best_train, best_val, best_score = None, None, None

        for attempt in range(32):
            shuffled = subjects.copy()
            random.Random(seed + attempt).shuffle(shuffled)
            val_subjs  = set(shuffled[:target_val_subjs])
            val_cand   = [i for s in val_subjs for i in subj_to_idx[s]]
            train_cand = [i for s, idxs in subj_to_idx.items() if s not in val_subjs for i in idxs]
            
            if not val_cand or not train_cand:
                continue
                
            val_lbl_counts: dict[int, int] = {}
            for i in val_cand:
                l = dataset.label_map[dataset.samples[i][1]]
                val_lbl_counts[l] = val_lbl_counts.get(l, 0) + 1
            
            min_in_val = min(val_lbl_counts.values()) if val_lbl_counts else 0
            coverage   = len(set(val_lbl_counts) & all_lbls)
            score = (coverage, min_in_val >= min_per_class, min_in_val,
                     -abs(len(val_cand) - round(len(indices) * val_split)))
            
            if best_score is None or score > best_score:
                best_score, best_train, best_val = score, train_cand, val_cand
                if coverage == len(all_lbls) and min_in_val >= min_per_class:
                    break

        if best_train is None:
            rng = random.Random(seed)
            shuffled_i = indices.copy()
            rng.shuffle(shuffled_i)
            n_val = max(1, int(len(shuffled_i) * val_split))
            return shuffled_i[n_val:], shuffled_i[:n_val]

        # Tras el loop de búsqueda, antes del log final:
        # Garantizar min_per_class para tristeza redistribuyendo si es necesario
        if best_val is not None and best_train is not None:
            val_counts_check: dict[int, int] = {}
            for i in best_val:
                l = dataset.label_map[dataset.samples[i][1]]
                val_counts_check[l] = val_counts_check.get(l, 0) + 1
            
            for lbl, cnt in val_counts_check.items():
                if cnt < min_per_class:
                    # Mover muestras de train a val para esta clase
                    train_candidates = [i for i in best_train 
                                       if dataset.label_map[dataset.samples[i][1]] == lbl]
                    needed = min_per_class - cnt
                    to_move = train_candidates[:needed]
                    best_val   = best_val + to_move
                    best_train = [i for i in best_train if i not in set(to_move)]
                    log.info(f"[split:{tag}] clase {lbl}: movidas {len(to_move)} muestras train→val para garantizar min={min_per_class}")

        return best_train, best_val

    # --- Lógica de ensamble de Train y Val ---
    primary_idxs = domain_buckets.get("primary", [])
    meme_idxs    = domain_buckets.get("meme", [])
    smic_idxs    = domain_buckets.get("smic", [])

    # Clases presentes en primary
    primary_classes = {dataset.samples[i][1] for i in primary_idxs}

    # MEME: separar clases exclusivas (no están en primary) de las compartidas
    meme_exclusive = [i for i in meme_idxs if dataset.samples[i][1] not in primary_classes]
    meme_shared    = [i for i in meme_idxs if dataset.samples[i][1] in primary_classes]

    log.info(f"[split] primary_classes={primary_classes}")
    log.info(f"[split] meme_shared={len(meme_shared)} | meme_exclusive={len(meme_exclusive)} "
            f"(clases: {sorted({dataset.samples[i][1] for i in meme_exclusive})})")

    # Primary y meme_exclusive se dividen subject-wise 80/20
    tr_p,     va_p     = _subject_split(primary_idxs,  "primary")
    tr_m_excl, va_m_excl = _subject_split(meme_exclusive, "meme_exclusive")

    # meme_shared y smic van completos a train (refuerzan clases ya en primary)
    all_train = tr_p + meme_shared + smic_idxs + tr_m_excl
    all_val   = va_p + va_m_excl

    def _count(indices: list[int]) -> dict[str, int]:
        c: dict[int, int] = {}
        for i in indices:
            l = dataset.label_map[dataset.samples[i][1]]
            c[l] = c.get(l, 0) + 1
        return {inv_map.get(l, str(l)): cnt for l, cnt in sorted(c.items())}

    log.info(f"[split] TOTAL → train={len(all_train)} (P:{len(tr_p)}+M:{len(meme_idxs)}+S:{len(smic_idxs)}) | val={len(all_val)}")
    log.info(f"[split] train por clase: {_count(all_train)}")
    log.info(f"[split] val por clase: {_count(all_val)}")

    return Subset(dataset, all_train), Subset(dataset, all_val)

def split_stratified_random(dataset: FlowSequenceDataset, val_split: float, seed: int) -> tuple[Subset, Subset]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, (_, emotion, _) in enumerate(dataset.samples):
        by_label.setdefault(dataset.label_map[emotion], []).append(i)
    train_idx, val_idx = [], []
    for _, indices in by_label.items():
        idxs = indices.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx.extend(idxs)
            continue
        n_val = max(1, min(int(round(n * val_split)), n - 1))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    if not train_idx or not val_idx:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_loso_folds(dataset: FlowSequenceDataset) -> list[tuple[Subset, Subset, str]]:
    folds: list[tuple[Subset, Subset, str]] = []
    for subject in dataset.get_subjects():
        val_idx   = dataset.get_subject_indices({subject})
        train_idx = [i for i in range(len(dataset)) if i not in set(val_idx)]
        if train_idx and val_idx:
            folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx), f"loso_{subject}"))
    return folds


def build_subject_kfolds(dataset: FlowSequenceDataset, num_folds: int, seed: int) -> list[tuple[Subset, Subset, str]]:
    subjects = dataset.get_subjects()
    if len(subjects) < 2:
        return []
    k = max(2, min(num_folds, len(subjects)))
    shuffled = subjects.copy()
    random.Random(seed).shuffle(shuffled)
    buckets = [shuffled[i::k] for i in range(k)]
    folds: list[tuple[Subset, Subset, str]] = []
    for i in range(k):
        val_subjects   = set(buckets[i])
        train_subjects = set(s for j, b in enumerate(buckets) if j != i for s in b)
        val_idx   = dataset.get_subject_indices(val_subjects)
        train_idx = dataset.get_subject_indices(train_subjects)
        if train_idx and val_idx:
            folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx), f"kfold_{i+1}"))
    return folds


def split_indices_subject_wise(
    base_dataset: FlowSequenceDataset,
    indices: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    subject_to_indices: dict[str, list[int]] = {}
    for idx in indices:
        subject_to_indices.setdefault(base_dataset.samples[idx][2], []).append(idx)
    subjects = sorted(subject_to_indices.keys())

    if len(subjects) < 2:
        shuffled = list(indices)
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        return shuffled[n_val:], shuffled[:n_val]

    target_val = max(1, min(int(round(len(subjects) * val_split)), len(subjects) - 1))
    available_classes = {base_dataset.label_map[base_dataset.samples[idx][1]] for idx in indices}
    subject_classes = {
        s: {base_dataset.label_map[base_dataset.samples[idx][1]] for idx in idxs}
        for s, idxs in subject_to_indices.items()
    }
    best_train, best_val, best_coverage = None, None, -1

    # FIX: 64 → 16 intentos con early-exit
    for t in range(16):
        shuffled_subs = subjects.copy()
        random.Random(seed + t).shuffle(shuffled_subs)
        val_subs = list(shuffled_subs[:target_val])
        val_set  = set(val_subs)
        covered  = set().union(*[subject_classes[s] for s in val_subs])
        missing  = available_classes - covered
        for s in shuffled_subs[target_val:]:
            if not missing or len(val_set) >= len(subjects) - 1:
                break
            gain = subject_classes[s] & missing
            if gain:
                val_subs.append(s)
                val_set.add(s)
                covered.update(gain)
                missing -= gain
        train_idx = [idx for s, idxs in subject_to_indices.items() if s not in val_set for idx in idxs]
        val_idx   = [idx for s, idxs in subject_to_indices.items() if s in val_set for idx in idxs]
        if not train_idx or not val_idx:
            continue
        coverage = len(covered & available_classes)
        if coverage > best_coverage:
            best_coverage = coverage
            best_train, best_val = train_idx, val_idx
            if coverage == len(available_classes):
                break  # Early-exit: cobertura completa alcanzada

    if best_train is not None and best_val is not None:
        return best_train, best_val

    # Fallback
    shuffled_subs = subjects.copy()
    random.Random(seed).shuffle(shuffled_subs)
    val_set = set(shuffled_subs[:target_val])
    return (
        [idx for s, idxs in subject_to_indices.items() if s not in val_set for idx in idxs],
        [idx for s, idxs in subject_to_indices.items() if s in val_set for idx in idxs],
    )


# ══════════════════════════════════════════════════════════════════════════════
# OVERSAMPLING & WEIGHT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def oversample_train_subset(
    train_subset: Subset,
    min_samples: int,
    num_frames: int,
    aug_strength: float,
    seed: int = 42,
) -> ConcatDataset:
    base: FlowSequenceDataset = train_subset.dataset  # type: ignore
    indices = list(train_subset.indices)
    by_emotion: dict[str, list[int]] = {}
    for idx in indices:
        by_emotion.setdefault(base.samples[idx][1], []).append(idx)
    rng = np.random.default_rng(seed)
    extra_arrays, extra_labels = [], []
    for emotion, idxs in by_emotion.items():
        count = len(idxs)
        if count >= min_samples:
            continue
        needed = min_samples - count
        label_idx = base.label_map[emotion]
        log.info(f"[oversample] '{emotion}': {count} → {count + needed} (+{needed} synthetic, strength={aug_strength:.2f})")
        for k in range(needed):
            source_idx = idxs[k % len(idxs)]
            seq = np.load(str(base.samples[source_idx][0])).astype(np.float32)
            frame_idx = np.linspace(0, seq.shape[0] - 1, num_frames, dtype=int)
            extra_arrays.append(_aggressive_augment_flow(seq[frame_idx], rng, strength=aug_strength))
            extra_labels.append(label_idx)
    if not extra_arrays:
        return ConcatDataset([train_subset])
    log.info(f"[oversample] total train size after augmentation: {len(indices) + len(extra_arrays)}")
    return ConcatDataset([train_subset, _AugmentedFlowDataset(extra_arrays, extra_labels, base.transform)])


def undersample_majority_subset(
    train_subset: Subset,
    cap_ratio: float,
    seed: int = 42,
) -> Subset:
    """Undersampling de clases mayoritarias.

    Calcula el mínimo de muestras entre todas las clases presentes y
    limita cada clase a min_count * cap_ratio muestras, elegidas al azar.

    Ejemplo: clase mínima=25, cap_ratio=3.0 → ninguna clase puede tener >75.

    Args:
        train_subset: Subset de entrenamiento original.
        cap_ratio:    Multiplicador sobre la clase mínima (e.g. 3.0).
        seed:         Semilla para reproducibilidad del shuffle.

    Returns:
        Nuevo Subset con índices filtrados.
    """
    if cap_ratio <= 0:
        return train_subset

    base: FlowSequenceDataset = train_subset.dataset  # type: ignore
    indices = list(train_subset.indices)

    by_label: dict[int, list[int]] = {}
    for idx in indices:
        label = base.label_map[base.samples[idx][1]]
        by_label.setdefault(label, []).append(idx)

    min_count = min(len(v) for v in by_label.values())
    cap = max(1, int(round(min_count * cap_ratio)))

    rng = random.Random(seed)
    kept: list[int] = []
    for label, idxs in by_label.items():
        if len(idxs) > cap:
            chosen = rng.sample(idxs, cap)
            log.info(f"[undersample] clase {label}: {len(idxs)} → {cap} (cap={min_count}×{cap_ratio}={cap})")
        else:
            chosen = idxs
        kept.extend(chosen)

    log.info(f"[undersample] total train tras undersampling: {len(kept)} (era {len(indices)})")
    return Subset(base, kept)


def get_subset_label_indices(subset: Subset, base_dataset: FlowSequenceDataset) -> list[int]:
    return [base_dataset.label_map[base_dataset.samples[i][1]] for i in subset.indices]


def label_counts(labels: list[int], num_classes: int) -> list[int]:
    return np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(int).tolist()


def build_balanced_sampler_from_labels(labels: list[int]) -> WeightedRandomSampler:
    counts = np.maximum(np.bincount(np.array(labels, dtype=np.int64)), 1)
    sample_weights = (1.0 / counts)[np.array(labels, dtype=np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_domain_balanced_sampler(domain_labels: list[str]) -> WeightedRandomSampler:
    domains = sorted(set(domain_labels))
    if len(domains) <= 1:
        return WeightedRandomSampler(
            weights=torch.ones(len(domain_labels), dtype=torch.double),
            num_samples=len(domain_labels),
            replacement=True,
        )
    domain_counts = {d: sum(1 for x in domain_labels if x == d) for d in domains}
    sample_weights = [1.0 / domain_counts[d] for d in domain_labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights_from_labels(
    labels: list[int],
    num_classes: int,
    cap: Optional[float] = None,
) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.where(counts <= 0.0, 1.0, counts)
    weights = (len(labels) / counts).astype(np.float32)
    weights = weights / max(weights.mean(), 1e-8)
    if cap is not None and cap > 0:
        weights = np.minimum(weights, float(cap))
    return torch.as_tensor(weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS & LOSS
# ══════════════════════════════════════════════════════════════════════════════

def confusion_matrix_np(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _per_class_f1(cm: np.ndarray, classes: list[int]) -> list[float]:
    f1s = []
    for c in classes:
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return f1s


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    return float(np.mean(_per_class_f1(cm, list(range(cm.shape[0]))))) if cm.shape[0] > 0 else 0.0


def macro_f1_present_from_confusion(cm: np.ndarray) -> float:
    present = [c for c in range(cm.shape[0]) if cm[c, :].sum() > 0]
    return float(np.mean(_per_class_f1(cm, present))) if present else 0.0


def macro_precision_recall_from_confusion(cm: np.ndarray) -> tuple[float, float]:
    precs, recs = [], []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        precs.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recs.append(tp  / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(precs)), float(np.mean(recs))


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | float = 1.0, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
        loss = alpha_t * focal_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class _GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lambda_ * grad, None


class GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return _GradRevFn.apply(x, lambda_)


class _DomainLabeledDataset(Dataset):
    def __init__(self, subset: Dataset, domain_ids: list[int]) -> None:
        self.subset, self.domain_ids = subset, domain_ids

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        frames, label = self.subset[idx]
        return frames, label, self.domain_ids[idx]


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL AGGREGATION
# NOTA: LSTMTemporalPool y TemporalAttentionPool se definen pero NO se
#       instancian a menos que se pasen --temporal_attn o --use_lstm
#       explícitamente. Por defecto ambos son False.
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + r
        r = x
        x = self.norm2(x)
        return self.ffn(x) + r


class LSTMTemporalPool(nn.Module):
    """LSTM temporal pooling. Desactivado por defecto; usar --use_lstm para activar."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.drop(torch.cat([h_n[-2], h_n[-1]], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FlowClassifier(nn.Module):
    _FEATURE_DIMS: dict[str, int] = {
        "resnet18":    512,
        "resnet32":    512,   # mapeado a resnet34
        "resnet34":    512,
        "resnet50":    2048,
        "densenet121": 1024,
    }

    def __init__(
        self,
        num_classes:       int,
        pretrained:        bool = True,
        freeze_backbone:   bool = True,
        backbone:          str  = "resnet18",
        use_dann:          bool = False,
        use_temporal_attn: bool = False,
        # FIX: use_lstm desactivado por defecto y protegido con guarda
        use_lstm:          bool = False,
        lstm_hidden:       int  = 128,
        lstm_layers:       int  = 1,
    ) -> None:
        super().__init__()

        # FIX: guarda explícita — LSTM requiere flag y num_layers >= 1
        if use_lstm and lstm_layers < 1:
            log.warning("[model] lstm_layers < 1 → forzando a 1")
            lstm_layers = 1

        weights = "DEFAULT" if pretrained else None
        backbone = backbone.lower()
        self._backbone_name = backbone
        self.feature_dim = self._FEATURE_DIMS[backbone]

        if backbone == "resnet18":
            net = models.resnet18(weights=weights)
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone in ["resnet34", "resnet32"]:
            if backbone == "resnet32":
                log.warning("PyTorch no tiene ResNet32 nativo para 224x224. Usando ResNet34.")
            net = models.resnet34(weights=weights)
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone == "resnet50":
            net = models.resnet50(weights=weights)
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone == "densenet121":
            net = models.densenet121(weights=weights)
            self._densenet_features = net.features
            self._densenet_pool     = nn.AdaptiveAvgPool2d(1)
            self.backbone = None
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose: resnet18, resnet32, resnet34, resnet50, densenet121"
            )

        if freeze_backbone:
            for param in self._backbone_params():
                param.requires_grad = False

        self.use_temporal_attn = use_temporal_attn
        self.use_lstm          = use_lstm

        # FIX: instanciación condicional — por defecto ninguno se crea
        if use_lstm:
            self.lstm_pool     = LSTMTemporalPool(self.feature_dim, lstm_hidden, lstm_layers, dropout=0.3)
            self.temporal_pool = None
            head_dim = lstm_hidden * 2
        elif use_temporal_attn:
            self.temporal_pool = TemporalAttentionPool(self.feature_dim, num_heads=4, dropout=0.1)
            self.lstm_pool     = None
            head_dim = self.feature_dim
        else:
            # Modo por defecto: weighted mean con campana centrada en el ápice
            self.temporal_pool = None
            self.lstm_pool     = None
            head_dim = self.feature_dim

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(head_dim, num_classes))

        self.use_dann = use_dann
        if use_dann:
            self._grl = GradientReversalLayer()
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
            )
        else:
            self._grl = None
            self.domain_classifier = None

    def _backbone_params(self):
        if self.backbone is None:
            return self._densenet_features.parameters()
        return self.backbone.parameters()

    def forward(
        self, x: torch.Tensor, dann_lambda: float = 1.0
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        if self.backbone is not None:
            feat = self.backbone(x_flat)
        else:
            feat = self._densenet_features(x_flat)
            feat = F.relu(feat, inplace=True)
            feat = self._densenet_pool(feat).flatten(1)

        feat = feat.view(B, T, -1)

        if self.use_lstm and self.lstm_pool is not None:
            feat = self.lstm_pool(feat)
        elif self.use_temporal_attn and self.temporal_pool is not None:
            feat = self.temporal_pool(feat)
            feat = feat.mean(dim=1)
        # En FlowClassifier.forward(), reemplaza el bloque "else":
        else:
            if T == 3:
                # onset=0.3, apex=1.0, offset=0.3 — el apex domina
                w = torch.tensor([0.3, 1.0, 0.3], device=feat.device, dtype=feat.dtype)
            else:
                half = torch.linspace(0.5, 1.0, steps=(T + 1) // 2,
                                    device=feat.device, dtype=feat.dtype)
                if T % 2 == 0:
                    w = torch.cat([half, half.flip(0)])
                else:
                    w = torch.cat([half, half[:-1].flip(0)])
            w = w / w.sum()
            feat = (feat * w.view(1, T, 1)).sum(dim=1)

        class_logits = self.classifier(feat)

        if self.use_dann and self.training and self._grl is not None:
            rev_feat = self._grl(feat, dann_lambda)
            domain_logits = self.domain_classifier(rev_feat)
            return class_logits, domain_logits

        return class_logits

    def unfreeze_backbone(self) -> None:
        for param in self._backbone_params():
            param.requires_grad = True

    def parameter_groups(self, head_lr: float, backbone_lr_mult: float) -> list[dict]:
        backbone_params = [p for p in self._backbone_params() if p.requires_grad]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]
        for mod in [self.domain_classifier, self.temporal_pool, self.lstm_pool]:
            if mod is not None:
                head_params += [p for p in mod.parameters() if p.requires_grad]
        groups: list[dict] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": head_lr * backbone_lr_mult})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups


DenseNetFlowClassifier = FlowClassifier


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam: float = float(np.random.beta(alpha, alpha)) if alpha > 0.0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def set_batchnorm_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b.detach())


def train_one_epoch(
    model, loader, optimizer, criterion, device,
    mixup_alpha=0.0, grad_clip=0.0, freeze_bn=False,
    ema_model=None, ema_decay=0.0,
):
    model.train()
    if freeze_bn:
        set_batchnorm_eval(model)
    total_loss = correct = total = 0
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_alpha > 0.0 and frames.size(0) > 1:
            frames, y_a, y_b, lam = mixup_data(frames, labels, mixup_alpha)
            optimizer.zero_grad()
            logits = model(frames)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            logits = model(frames)
            loss   = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema_model is not None and ema_decay > 0.0:
            update_ema(ema_model, model, ema_decay)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def train_one_epoch_dann(
    model, loader, optimizer, criterion, device, dann_lambda,
    mixup_alpha=0.0, grad_clip=0.0, freeze_bn=False,
    ema_model=None, ema_decay=0.0,
):
    model.train()
    if freeze_bn:
        set_batchnorm_eval(model)
    total_cls_loss = correct = total = 0
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
            class_logits, domain_logits = result, None
        cls_loss = mixup_criterion(criterion, class_logits, y_a, y_b, lam)
        loss = cls_loss + (F.cross_entropy(domain_logits, domain_ids) if domain_logits is not None else 0.0)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema_model is not None and ema_decay > 0.0:
            update_ema(ema_model, model, ema_decay)
        total_cls_loss += cls_loss.item() * labels.size(0)
        correct        += (class_logits.argmax(dim=1) == labels).sum().item()
        total          += labels.size(0)
    return total_cls_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = correct = total = 0
    y_true, y_pred = [], []
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frames)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    return (
        total_loss / total, correct / total,
        macro_f1_from_confusion(cm), macro_f1_present_from_confusion(cm),
        *macro_precision_recall_from_confusion(cm), cm,
    )


@torch.no_grad()
def evaluate_with_tta(model, loader, criterion, device, num_classes, tta_passes=4):
    model.eval()
    n_views = max(1, min(tta_passes, 4))
    total_loss = correct = total = 0
    y_true, y_pred = [], []

    def _tta_views(f):
        views = [f]
        if n_views >= 2:
            hf = f.flip(-1).clone(); hf[:, :, 0] *= -1; views.append(hf)
        if n_views >= 3:
            vf = f.flip(-2).clone(); vf[:, :, 1] *= -1; views.append(vf)
        if n_views >= 4:
            hvf = f.flip(-1).flip(-2).clone(); hvf[:, :, 0] *= -1; hvf[:, :, 1] *= -1; views.append(hvf)
        return views

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        avg_probs = torch.zeros(labels.size(0), num_classes, device=device)
        for i, view in enumerate(_tta_views(frames)):
            logits = model(view)
            avg_probs += torch.softmax(logits, dim=1)
            if i == 0:
                total_loss += criterion(logits, labels).item() * labels.size(0)
        avg_probs /= n_views
        preds = avg_probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    return (
        total_loss / total, correct / total,
        macro_f1_from_confusion(cm), macro_f1_present_from_confusion(cm),
        *macro_precision_recall_from_confusion(cm), cm,
    )


def _build_criterion(args, train_labels, num_classes, device, label_map=None):
    cap = None if args.class_weight_cap <= 0 else float(args.class_weight_cap)
    cw  = build_class_weights_from_labels(train_labels, num_classes, cap=cap)
    if args.no_class_weights:
        cw = torch.ones_like(cw)
    if getattr(args, "class_weight_boost", "") and label_map:
        cw_np = cw.numpy().copy()
        for pair in args.class_weight_boost.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            cls_name, mult_str = pair.rsplit(":", 1)
            idx = label_map.get(cls_name.strip(), -1)
            if idx >= 0:
                cw_np[idx] *= float(mult_str.strip())
                log.info(f"[loss] boost '{cls_name}' (idx={idx}) × {mult_str} → {cw_np[idx]:.3f}")
        cw = torch.as_tensor(cw_np, dtype=torch.float32)
    cw = cw.to(device)
    smoothing = float(np.clip(getattr(args, "label_smoothing", 0.0), 0.0, 0.2))
    if args.use_focal_loss:
        alpha = 1.0 if args.no_class_weights else cw
        crit  = FocalLoss(alpha=alpha, gamma=args.focal_gamma, reduction="mean")
        log.info(f"[loss] Focal Loss (gamma={args.focal_gamma}) | weights={cw.cpu().numpy().round(3).tolist()}")
    else:
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=smoothing)
        log.info(f"[loss] CrossEntropy | weights={cw.cpu().numpy().round(3).tolist()} | smoothing={smoothing:.3f}")
    return crit


def _build_scheduler(args, optimizer, n_epochs):
    name = getattr(args, "lr_scheduler", "cosine")
    if name == "cosine":
        log.info(f"[sched] CosineAnnealingLR | T_max={n_epochs}")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.lr * 1e-2)
    if name == "cosine_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=args.lr * 1e-2)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    return None


def _build_train_loader(args, train_ds, train_labels, train_domain_labels, device):
    pin = device.type == "cuda"
    if args.domain_balanced_sampler and len(set(train_domain_labels)) > 1:
        sampler = build_domain_balanced_sampler(train_domain_labels)
    elif args.balanced_sampler:
        sampler = build_balanced_sampler_from_labels(train_labels)
    else:
        return DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin,
        )
    return DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=pin,
    )


def run_single_split(args, device, num_classes, train_ds, val_ds, fold_tag):
    # ── Resolver base_ds e índices de forma robusta ───────────────────────────
    # train_ds puede ser Subset(FlowSequenceDataset, indices) en todos los casos
    # (split_subject_wise, split_primary_only_val, etc.)
    base_ds: FlowSequenceDataset = train_ds.dataset  # type: ignore
    orig_indices = list(train_ds.indices)

    if hasattr(base_ds, "domain_labels") and len(base_ds.domain_labels) == len(base_ds.samples):
        train_domain_labels = [base_ds.domain_labels[i] for i in orig_indices]
    else:
        train_domain_labels = ["primary"] * len(orig_indices)

    # ── Mostrar distribución de clases antes de balancear ────────────────────
    raw_counts: dict[int, int] = {}
    for i in orig_indices:
        lbl = base_ds.label_map[base_ds.samples[i][1]]
        raw_counts[lbl] = raw_counts.get(lbl, 0) + 1
    inv_map = {v: k for k, v in base_ds.label_map.items()}
    log.info("[balance] distribución original train: " +
             " | ".join(f"{inv_map.get(l, l)}={c}" for l, c in sorted(raw_counts.items())))

    # ── Undersampling primero (reduce mayorías) ───────────────────────────────
    cap_ratio = float(getattr(args, "cap_majority_ratio", 0.0))
    if cap_ratio > 0:
        # undersample_majority_subset necesita un Subset simple con .dataset = FlowSequenceDataset
        # orig_indices ya apunta al FlowSequenceDataset correcto → construimos Subset fresco
        train_subset_for_under = Subset(base_ds, orig_indices)
        train_ds    = undersample_majority_subset(train_subset_for_under, cap_ratio=cap_ratio, seed=args.seed)
        orig_indices = list(train_ds.indices)
        # Actualizar domain labels tras el recorte
        train_domain_labels = [base_ds.domain_labels[i] for i in orig_indices] \
            if hasattr(base_ds, "domain_labels") and len(base_ds.domain_labels) == len(base_ds.samples) \
            else ["primary"] * len(orig_indices)

    # ── Oversampling (sube minorías con augmentación sintética) ───────────────
    if args.oversample_minority:
        # oversample_train_subset necesita Subset simple también
        train_subset_for_over = Subset(base_ds, orig_indices) if not isinstance(train_ds, Subset) else train_ds
        train_ds = oversample_train_subset(
            train_subset_for_over,
            min_samples=args.min_samples_per_class,
            num_frames=args.num_frames,
            aug_strength=args.minority_aug_strength,
            seed=args.seed,
        )

    # train_labels_orig se calcula sobre los índices DESPUÉS del undersampling
    # pero ANTES del oversample sintético (que no tiene índices en base_ds)
    train_labels_orig = [base_ds.label_map[base_ds.samples[i][1]] for i in orig_indices]
    train_loader      = _build_train_loader(args, train_ds, train_labels_orig, train_domain_labels, device)

    pin = device.type == "cuda"
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    log.info(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    use_dann          = getattr(args, "domain_adversarial", False) and len(set(train_domain_labels)) > 1
    use_temporal_attn = getattr(args, "temporal_attn", False)
    # FIX: use_lstm siempre False salvo que se pase explícitamente --use_lstm
    use_lstm          = getattr(args, "use_lstm", False)

    model = FlowClassifier(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
        backbone=args.backbone,
        use_dann=use_dann,
        use_temporal_attn=use_temporal_attn,
        use_lstm=use_lstm,
        lstm_hidden=int(getattr(args, "lstm_hidden", 128)),
        lstm_layers=int(getattr(args, "lstm_layers", 1)),
    ).to(device)

    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema_model = None
    if ema_decay > 0.0:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    criterion = _build_criterion(args, train_labels_orig, num_classes, device, base_ds.label_map)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.lr, args.backbone_lr_mult),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(args, optimizer, args.epochs)

    dann_train_loader = train_loader
    if use_dann:
        _domain_ids = [0 if d == "primary" else 1 for d in train_domain_labels]
        n_extra = len(train_ds) - len(_domain_ids)
        if n_extra > 0:
            _domain_ids += [0] * n_extra
        dann_ds = _DomainLabeledDataset(train_ds, _domain_ids)
        dann_train_loader = DataLoader(
            dann_ds, batch_size=args.batch_size, sampler=train_loader.sampler,
            num_workers=args.num_workers, pin_memory=pin,
        )

    monitor_name = "macro_f1_present" if args.monitor_metric == "macro_f1_present" else "macro_f1"
    best_monitor = best_val_acc = best_val_f1 = 0.0
    best_epoch = no_improve = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = f"best_{args.backbone}_{fold_tag}_{ts}.pth"

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameter_groups(args.lr, args.backbone_lr_mult),
                lr=args.lr, weight_decay=args.weight_decay,
            )
            scheduler = _build_scheduler(args, optimizer, args.epochs - epoch + 1)

        freeze_bn = bool(
            getattr(args, "freeze_bn_after_unfreeze", False)
            and (args.unfreeze_epoch <= 0 or epoch >= args.unfreeze_epoch)
        )
        grad_clip = float(getattr(args, "grad_clip_norm", 0.0))
        mixup_a   = float(getattr(args, "mixup_alpha", 0.0))

        if use_dann:
            p = (epoch - 1) / max(args.epochs - 1, 1)
            _lambda = args.dann_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            train_loss, train_acc = train_one_epoch_dann(
                model, dann_train_loader, optimizer, criterion, device,
                dann_lambda=_lambda, mixup_alpha=mixup_a, grad_clip=grad_clip,
                freeze_bn=freeze_bn, ema_model=ema_model, ema_decay=ema_decay,
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                mixup_alpha=mixup_a, grad_clip=grad_clip, freeze_bn=freeze_bn,
                ema_model=ema_model, ema_decay=ema_decay,
            )

        eval_model = ema_model if ema_model is not None else model
        if args.tta_passes > 1:
            val_loss, val_acc, val_f1, val_f1_present, val_prec, val_rec, val_cm = evaluate_with_tta(
                eval_model, val_loader, criterion, device, num_classes, tta_passes=args.tta_passes
            )
        else:
            val_loss, val_acc, val_f1, val_f1_present, val_prec, val_rec, val_cm = evaluate(
                eval_model, val_loader, criterion, device, num_classes
            )

        monitor_score = val_f1_present if args.monitor_metric == "macro_f1_present" else val_f1
        improved = monitor_score > best_monitor
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        if improved:
            best_monitor, best_epoch, no_improve = monitor_score, epoch, 0
            torch.save(eval_model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        flag = " ✓" if improved else ""
        log.info(
            f"[{epoch:03d}/{args.epochs}] train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f} "
            f"f1_p={val_f1_present:.3f} lr={cur_lr:.2e}{flag}"
        )

        # ── Matriz de confusión por época ──────────────────────────────────
        inv_map = {v: k for k, v in base_ds.label_map.items()}
        class_names = [inv_map.get(i, str(i)) for i in range(num_classes)]
        col_w = max(len(n) for n in class_names) + 2
        header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in class_names)
        log.info(f"[CM epoch {epoch:03d}]\n{header}")
        for row_i, row_name in enumerate(class_names):
            row_str = f"{row_name:>{col_w}}" + "".join(
                f"{int(val_cm[row_i, col_i]):>{col_w}}" for col_i in range(num_classes)
            )
            log.info(row_str)
        # Per-class recall en una línea
        recalls = []
        for i in range(num_classes):
            total_true = val_cm[i, :].sum()
            recalls.append(f"{class_names[i]}={val_cm[i,i]/total_true:.2f}" if total_true > 0 else f"{class_names[i]}=N/A")
        log.info(f"[recall] {' | '.join(recalls)}")

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and no_improve >= args.early_stop_patience
        ):
            log.info(f"[early stop] no mejora en {no_improve} épocas → deteniendo.")
            break

    log.info(
        f"\n[done:{fold_tag}] best val acc={best_val_acc:.3f} | "
        f"best val macro-f1={best_val_f1:.3f} | checkpoint: {ckpt_path}"
    )
    return best_val_acc, best_val_f1


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Micro-expression sequence classifier.")
    p.add_argument("--data_dir",    default="Microexpresiones/outputs_dataset_index")
    p.add_argument("--csv_index",   default=None)
    p.add_argument("--onset_csv",   default=None)
    p.add_argument("--extra_data_dir",  default=None)
    p.add_argument("--smic_data_dir",   default=None)
    p.add_argument("--extra_csv_index", default=None)
    p.add_argument("--smic_csv_index",  default=None)
    p.add_argument("--meme_csv_index",  default=None)
    p.add_argument("--input_path",  default="Microexpresiones/outputs_dataset_index")
    p.add_argument("--num_frames",  type=int,   default=16)
    p.add_argument("--temporal_aug_prob", type=float, default=0.5)
    p.add_argument(
        "--backbone",
        choices=["resnet18", "resnet32", "resnet34", "resnet50", "densenet121"],
        default="resnet34",
    )
    p.add_argument("--no_pretrained", action="store_true", default=False)
    p.add_argument("--unfreeze_epoch", type=int, default=2)
    p.add_argument("--backbone_lr_mult", type=float, default=0.2)
    p.add_argument("--temporal_attn", action="store_true", default=False)
    # FIX: use_lstm desactivado por defecto — agregar --use_lstm para activarlo
    p.add_argument("--use_lstm",      action="store_true", default=False)
    p.add_argument("--lstm_hidden",   type=int, default=128)
    p.add_argument("--lstm_layers",   type=int, default=1)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--mixup_alpha",  type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)
    p.add_argument("--ema_decay",    type=float, default=0.0)
    p.add_argument(
        "--lr_scheduler",
        choices=["none", "cosine", "cosine_restarts", "step"],
        default="cosine",
    )
    p.add_argument("--freeze_bn_after_unfreeze", action="store_true", default=False)
    p.add_argument("--use_focal_loss",   action="store_true", default=False)
    p.add_argument("--focal_gamma",      type=float, default=2.0)
    p.add_argument("--no_class_weights", action="store_true", default=False)
    p.add_argument("--class_weight_cap", type=float, default=2.8)
    p.add_argument("--class_weight_boost", type=str, default="")
    p.add_argument("--label_smoothing",  type=float, default=0.0)
    p.add_argument("--val_split",   type=float, default=0.2)
    p.add_argument(
        "--split_mode",
        choices=["subject", "random", "stratified_random", "loso", "subject_kfold"],
        default="subject",
    )
    p.add_argument("--num_folds",   type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    # FIX: num_workers=0 por defecto — evita deadlocks con DataLoader en servidores
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--balanced_sampler",    action="store_true",  default=True)
    p.add_argument("--no_balanced_sampler", dest="balanced_sampler", action="store_false")
    p.add_argument("--domain_balanced_sampler", action="store_true", default=False)
    p.add_argument("--early_stop_patience",   type=int, default=0)
    p.add_argument("--early_stop_min_epochs", type=int, default=0)
    p.add_argument("--only_ekman7",    action="store_true",  default=True)
    p.add_argument("--allow_non_ekman", dest="only_ekman7",  action="store_false")
    p.add_argument("--oversample_minority",     action="store_true", default=False)
    p.add_argument("--min_samples_per_class",   type=int,   default=20)
    p.add_argument("--minority_aug_strength",   type=float, default=0.6)
    p.add_argument(
        "--cap_majority_ratio", type=float, default=0.0,
        help=(
            "Undersampling de clases mayoritarias. Si > 0, ninguna clase puede tener más de "
            "min_clase * cap_majority_ratio muestras. Ej: min=25, ratio=3.0 → tope=75. "
            "0 = desactivado (default)."
        ),
    )
    p.add_argument(
        "--merge_classes", type=str, default="",
        help=(
            "Fusionar clases en el entrenamiento. Formato: 'origen:destino,...'. "
            "Ej: 'miedo:tristeza' o 'miedo:tristeza,enojo:tristeza'. "
            "Se aplica ANTES del filtro only_ekman7."
        ),
    )
    p.add_argument("--tta_passes", type=int, default=1)
    p.add_argument("--domain_adversarial", action="store_true", default=False)
    p.add_argument("--dann_lambda",        type=float, default=0.5)
    p.add_argument(
        "--primary_only_val", action="store_true", default=False,
        help="Val usa solo muestras del dataset primario. Extra/MEME/SMIC van siempre a train.",
    )
    p.add_argument(
        "--hybrid_val_min_per_class", type=int, default=1,
        help="Mínimo de muestras por clase en val cuando se usa --primary_only_val.",
    )
    p.add_argument(
        "--monitor_metric",
        choices=["macro_f1", "macro_f1_present"],
        default="macro_f1_present",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tta_passes < 1 or args.tta_passes > 4:
        raise ValueError("--tta_passes must be between 1 and 4.")
    if args.val_split <= 0 or args.val_split >= 1:
        raise ValueError("--val_split must be in (0, 1).")
    if args.num_frames < 1:
        raise ValueError("--num_frames must be >= 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.csv_index:
        csv_candidate = Path(args.input_path) / "extraction_index.csv"
        args.csv_index = str(csv_candidate) if csv_candidate.exists() else None

    device = torch.device("cpu")
    set_seed(args.seed)
    transform = build_transforms()

    full_dataset = FlowSequenceDataset(
        data_dir          = args.data_dir,
        transform         = transform,
        num_frames        = args.num_frames,
        csv_index         = args.csv_index,
        extra_csv_index   = args.extra_csv_index,
        smic_csv_index    = args.smic_csv_index,
        meme_csv_index    = args.meme_csv_index,
        onset_csv         = args.onset_csv,
        temporal_aug_prob = args.temporal_aug_prob,
        extra_data_dir    = args.extra_data_dir,
        smic_data_dir     = getattr(args, "smic_data_dir", None),
    )

    # ── Fusión de clases configurable ────────────────────────────────────────────
    if getattr(args, "merge_classes", ""):
        merge_map: dict[str, str] = {}
        for pair in args.merge_classes.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            src, dst = pair.split(":", 1)
            merge_map[src.strip().lower()] = dst.strip().lower()

        if merge_map:
            log.info(f"[merge] Fusionando clases: {merge_map}")
            merged_samples = []
            for path, emotion, subject in full_dataset.samples:
                new_emotion = merge_map.get(emotion.lower(), emotion)
                merged_samples.append((path, new_emotion, subject))
            full_dataset.samples   = merged_samples
            full_dataset.label_map = build_label_map({e for _, e, _ in full_dataset.samples})
            # Mostrar resultado de la fusión
            from collections import Counter
            counts = Counter(e for _, e, _ in full_dataset.samples)
            log.info(f"[merge] Distribución después de fusión: {dict(sorted(counts.items()))}")

    if args.only_ekman7:
        before = len(full_dataset.samples)
        filtered, filtered_d = [], []
        domains = (
            full_dataset.domain_labels
            if len(full_dataset.domain_labels) == before
            else ["primary"] * before
        )
        for (path, emotion, subject), domain in zip(full_dataset.samples, domains):
            canonical = canonicalize_ekman6(emotion)
            if canonical is None:
                continue
            filtered.append((path, canonical, subject))
            filtered_d.append(domain)
        full_dataset.samples      = filtered
        full_dataset.domain_labels = filtered_d
        full_dataset.label_map    = build_label_map({e for _, e, _ in full_dataset.samples})

    num_classes = full_dataset.get_num_classes()
    if len(full_dataset) == 0:
        raise RuntimeError("No sequences found.")

    log.info(f"[dataset] {len(full_dataset)} sequences | {num_classes} classes | device={device}")
    log.info(f"[model]   backbone={args.backbone} | lstm={args.use_lstm} | attn={args.temporal_attn}")

    # --primary_only_val tiene prioridad sobre split_mode
    if args.primary_only_val:
        train_ds, val_ds = split_domain_aware(
            full_dataset,
            val_split=args.val_split,
            seed=args.seed,
            min_per_class=args.hybrid_val_min_per_class,
        )
        run_single_split(args, device, num_classes, train_ds, val_ds, "primary_val")
        return

    if args.split_mode == "subject":
        train_ds, val_ds = split_subject_wise(full_dataset, args.val_split, args.seed)
        run_single_split(args, device, num_classes, train_ds, val_ds, "subject")
        return

    if args.split_mode == "stratified_random":
        train_ds, val_ds = split_stratified_random(full_dataset, args.val_split, args.seed)
        run_single_split(args, device, num_classes, train_ds, val_ds, "stratified")
        return

    if args.split_mode == "loso":
        folds = build_loso_folds(full_dataset)
        accs, f1s = [], []
        for train_ds, val_ds, tag in folds:
            acc, f1 = run_single_split(args, device, num_classes, train_ds, val_ds, tag)
            accs.append(acc); f1s.append(f1)
        log.info(f"[LOSO] mean acc={np.mean(accs):.3f} | mean f1={np.mean(f1s):.3f}")
        return

    if args.split_mode == "subject_kfold":
        folds = build_subject_kfolds(full_dataset, args.num_folds, args.seed)
        accs, f1s = [], []
        for train_ds, val_ds, tag in folds:
            acc, f1 = run_single_split(args, device, num_classes, train_ds, val_ds, tag)
            accs.append(acc); f1s.append(f1)
        log.info(f"[KFold] mean acc={np.mean(accs):.3f} | mean f1={np.mean(f1s):.3f}")
        return

    # Fallback: random split
    val_size   = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    run_single_split(args, device, num_classes, train_ds, val_ds, "random")


if __name__ == "__main__":
    main()