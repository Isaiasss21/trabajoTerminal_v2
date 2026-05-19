"""
train_mobilenet_flow.py
-----------------------
Sequence-level micro-expression classifier using MobileNetV3 as frame encoder.

Pipeline per sample:
  .npy (N, 64, 64, 3) float32
    → uniform frame sampling  → T frames (64, 64, 3)
    → flow-to-pseudo-RGB mapping + resize 224×224 + ImageNet normalization
    → MobileNetV3  (B*T, 3, 224, 224) → (B*T, D)
    → reshape + mean-pool over T  → (B, D)
    → Linear classifier          → (B, num_classes)

Run example:
  python scripts/train_mobilenet_flow.py \
      --data_dir   output_extraction_meme_v2                \
      --csv_index  output_extraction_meme_v2/extraction_index.csv \
      --epochs     30                                     \
      --batch_size 16
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

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
    data_dir:   str       = "output_extraction_meme_v2"
    csv_index:  str | None = "output_extraction_meme_v2/extraction_index.csv"
    onset_csv:  str | None = None        # optional CSV with onset/offset columns

    # Frame sampling
    num_frames: int  = 16                # fixed T per sample

    # MobileNetV3
    mobilenet_variant: str  = "small"    # "small" (D=576) or "large" (D=960)
    freeze_backbone:   bool = True       # freeze feature extractor at start

    # Training
    num_classes:   int   = 7            # updated automatically from data
    epochs:        int   = 30
    batch_size:    int   = 16
    lr:            float = 1e-4
    weight_decay:  float = 1e-2
    val_split:     float = 0.2
    split_mode:    str   = "subject"    # "subject" | "random"
    balanced_sampler: bool = True
    unfreeze_epoch: int  = 5            # epoch at which backbone is unfrozen (0 = never)
    seed:          int   = 42
    num_workers:   int   = 0            # 0 is safe on Windows; increase on Linux


# ══════════════════════════════════════════════════════════════════════════════
# LABEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Canonicalized emotion vocabulary (Spanish names from datasetVideosTT)
EMOTION_TO_IDX: dict[str, int] = {
    "happiness": 0, "hap": 0, "happy": 0,
    "felicidad": 0,
    "surprise":  1, "sur": 1,
    "sorpresa":  1,
    "disgust":   2, "dis": 2,
    "asco":      2,
    "repression":3, "rep": 3,
    "represion": 3, "represión": 3,
    "others":    4, "oth": 4, "otros": 4,
    "neutral":   5,
    "sadness":   5, "sad": 5,
    "tristeza":  5,
    "fear":      6, "fea": 6,
    "miedo":     6,
    "contempt":  7, "con": 7,
    "anger":     8, "ang": 8,
    "enojo":     8,
    "tense":     9, "ten": 9,
}


def build_label_map(labels_found: set[str]) -> dict[str, int]:
    """Compact 0-based map over only the emotions present in the dataset."""
    sorted_names = sorted(labels_found)
    return {name: idx for idx, name in enumerate(sorted_names)}


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def flow_to_pseudo_rgb(frames: np.ndarray) -> np.ndarray:
    """
    Map optical-flow channels to [0, 1] pseudo-RGB.

    Input:  (T, H, W, 3) float32  channels = [dx ∈ [-1,1], dy ∈ [-1,1], mag ∈ [0,1]]
    Output: (T, H, W, 3) float32  all channels in [0, 1]
    """
    out = np.empty_like(frames)
    out[..., 0] = np.clip((frames[..., 0] + 1.0) / 2.0, 0.0, 1.0)  # dx → [0,1]
    out[..., 1] = np.clip((frames[..., 1] + 1.0) / 2.0, 0.0, 1.0)  # dy → [0,1]
    out[..., 2] = np.clip(frames[..., 2], 0.0, 1.0)                 # mag already [0,1]
    return out


def preprocess_frame(frame_uint8: np.ndarray) -> torch.Tensor:
    """
    (H, W, 3) uint8 → (3, 224, 224) float32 normalized with ImageNet stats.
    """
    img = torch.from_numpy(frame_uint8).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    img = TF.resize(img, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    img = TF.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# FRAME SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_frames(
    sequence:   np.ndarray,
    num_frames: int,
    onset:      int | None = None,
    offset:     int | None = None,
) -> np.ndarray:
    """
    Return exactly `num_frames` frames from `sequence` (N, H, W, 3).
    Repeats frames (tile) if the available range is shorter than num_frames.
    """
    n     = sequence.shape[0]
    start = max(0, onset  if onset  is not None else 0)
    end   = min(n, offset if offset is not None else n)
    if end <= start:
        start, end = 0, n

    available = end - start

    if available >= num_frames:
        indices = np.linspace(start, end - 1, num_frames, dtype=int)
    else:
        base    = np.arange(start, end)
        reps    = (num_frames // available) + 1
        indices = np.tile(base, reps)[:num_frames]

    return sequence[indices]  # (num_frames, H, W, 3)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FlowSequenceDataset(Dataset):
    """
    PyTorch Dataset for pre-extracted optical-flow sequences (.npy).

    Each __getitem__ returns:
      frames : Tensor (T, 3, 224, 224) float32 — MobileNet-ready
      label  : int — emotion class index
    """

    def __init__(
        self,
        data_dir:   str | Path,
        num_frames: int  = 16,
        csv_index:  str | Path | None = None,
        onset_csv:  str | Path | None = None,
        label_map:  dict[str, int] | None = None,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.num_frames = num_frames

        # onset/offset lookup: {npy_stem: (onset, offset)}
        self.onsets: dict[str, tuple[int, int]] = {}
        if onset_csv:
            self._load_onset_csv(Path(onset_csv))

        # Sample list: [(npy_path, emotion_canonical, subject_id), ...]
        self.samples: list[tuple[Path, str, str]] = []
        if csv_index:
            self._build_from_csv(Path(csv_index))
        else:
            self._build_from_dir()

        if label_map is not None:
            self.label_map = label_map
        else:
            found = {emo for _, emo, _ in self.samples}
            self.label_map = build_label_map(found)

    # ── Index builders ─────────────────────────────────────────────────────

    def _build_from_dir(self) -> None:
        """Scan data_dir/.../**.npy — expects subject/camera/emotion/seq.npy tree."""
        for npy_path in sorted(self.data_dir.rglob("*.npy")):
            parts = npy_path.relative_to(self.data_dir).parts
            if len(parts) < 3:
                continue
            subject = parts[0]
            # Try different depth positions for emotion label
            for part in parts[1:]:
                emo = part.lower()
                if emo in EMOTION_TO_IDX:
                    self.samples.append((npy_path, emo, subject))
                    break

    def _build_from_csv(self, csv_path: Path) -> None:
        """
        Build index from extraction_index.csv.
        'archivo' column contains paths relative to data_dir.
        """
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel  = row.get("archivo", "").strip()
                emo  = row.get("emocion", "").strip().lower()
                subj = (
                    row.get("persona", "").strip()
                    or row.get("sujeto", "").strip()
                    or "unknown_subject"
                )
                if not rel or emo not in EMOTION_TO_IDX:
                    continue
                # Paths in CSV may use Windows separators even on Linux
                npy_path = self.data_dir / Path(rel.replace("\\", "/"))
                if npy_path.exists():
                    self.samples.append((npy_path, emo, subj))

    def _load_onset_csv(self, path: Path) -> None:
        """Load onset/offset annotations. Expected columns: filename, onset, offset."""
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key    = Path(row.get("filename", "")).stem
                onset  = int(row["onset"])  if row.get("onset")  else None
                offset = int(row["offset"]) if row.get("offset") else None
                if key and onset is not None and offset is not None:
                    self.onsets[key] = (onset, offset)

    # ── PyTorch interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, emotion, _subject = self.samples[idx]
        label = self.label_map[emotion]

        # 1. Load full sequence  (N, H, W, 3) float32
        sequence = np.load(str(npy_path))

        # 2. Sample T frames
        onset, offset = self.onsets.get(npy_path.stem, (None, None))
        frames = sample_frames(sequence, self.num_frames, onset, offset)
        # frames: (T, H, W, 3)

        # 3. Map flow channels → pseudo-RGB [0, 1]
        frames = flow_to_pseudo_rgb(frames)

        # 4. Convert to uint8 then preprocess each frame
        frames_uint8 = (frames * 255).clip(0, 255).astype(np.uint8)
        pixel_values = torch.stack([preprocess_frame(f) for f in frames_uint8])
        # pixel_values: (T, 3, 224, 224)

        return pixel_values, label

    # ── Helpers ─────────────────────────────────────────────────────────

    def get_num_classes(self) -> int:
        return len(self.label_map)

    def get_label_map(self) -> dict[str, int]:
        return dict(self.label_map)

    def get_subjects(self) -> list[str]:
        return sorted({subject for _, _, subject in self.samples})

    def get_subject_indices(self, subjects: set[str]) -> list[int]:
        return [i for i, (_, _, subj) in enumerate(self.samples) if subj in subjects]


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def split_subject_wise(
    dataset:   FlowSequenceDataset,
    val_split: float,
    seed:      int,
) -> tuple[Subset, Subset]:
    """Split by subject identity to prevent data leakage."""
    subjects = dataset.get_subjects()
    if len(subjects) < 2:
        return _random_fallback(dataset, val_split, seed)

    rng = random.Random(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_split)))
    n_val = min(n_val, len(shuffled) - 1)
    val_subjects   = set(shuffled[:n_val])
    train_subjects = set(shuffled[n_val:])

    train_idx = dataset.get_subject_indices(train_subjects)
    val_idx   = dataset.get_subject_indices(val_subjects)

    if not train_idx or not val_idx:
        return _random_fallback(dataset, val_split, seed)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _random_fallback(
    dataset: FlowSequenceDataset, val_split: float, seed: int
) -> tuple[Subset, Subset]:
    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def get_subset_label_indices(
    subset_or_dataset: Subset | FlowSequenceDataset,
    base_dataset:      FlowSequenceDataset,
) -> list[int]:
    """Numeric labels for a Subset without loading frames."""
    indices = (
        subset_or_dataset.indices
        if isinstance(subset_or_dataset, Subset)
        else range(len(base_dataset))
    )
    return [base_dataset.label_map[base_dataset.samples[i][1]] for i in indices]


def build_balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts         = np.maximum(np.bincount(np.asarray(labels, dtype=np.int64)), 1)
    sample_weights = (1.0 / counts)[np.asarray(labels, dtype=np.int64)]
    return WeightedRandomSampler(
        weights     = torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples = len(sample_weights),
        replacement = True,
    )


def build_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts  = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts  = np.where(counts <= 0.0, 1.0, counts)
    weights = (len(labels) / counts).astype(np.float32)
    weights = weights / max(weights.mean(), 1e-8)
    return torch.as_tensor(weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def confusion_matrix_np(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> np.ndarray:
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
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

_MOBILENET_HIDDEN = {"small": 576, "large": 960}


class MobileNetFlowClassifier(nn.Module):
    """
    MobileNetV3 frame encoder + temporal mean pooling + linear classifier.

    Forward input:  (B, T, 3, 224, 224)
    Forward output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes:       int,
        mobilenet_variant: str  = "small",
        freeze_backbone:   bool = True,
    ) -> None:
        super().__init__()

        variant = mobilenet_variant.lower()
        if variant == "large":
            base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        else:
            base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        hidden_dim = _MOBILENET_HIDDEN[variant]

        # Keep only the convolutional feature extractor and adaptive pooling
        self.features = base.features   # outputs (B, C, 7, 7) for 224×224 input
        self.avgpool  = base.avgpool    # AdaptiveAvgPool2d → (B, C, 1, 1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, 224, 224)"""
        B, T, C, H, W = x.shape

        # Encode all frames in a single batch pass
        x_flat = x.view(B * T, C, H, W)          # (B*T, 3, 224, 224)
        feats   = self.features(x_flat)           # (B*T, C', 7, 7)
        feats   = self.avgpool(feats)             # (B*T, C', 1, 1)
        feats   = feats.flatten(1)                # (B*T, hidden_dim)

        # Temporal mean pooling
        feats      = feats.view(B, T, -1)         # (B, T, hidden_dim)
        seq_embed  = feats.mean(dim=1)            # (B, hidden_dim)

        return self.classifier(seq_embed)         # (B, num_classes)

    def unfreeze_backbone(self) -> None:
        """Enable fine-tuning of the MobileNetV3 feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)   # (B, T, 3, 224, 224)
        labels = labels.to(device, non_blocking=True)   # (B,)

        optimizer.zero_grad()
        logits = model(frames)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    num_classes: int,
    device:      torch.device,
) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    cm       = confusion_matrix_np(y_true, y_pred, num_classes)
    macro_f1 = macro_f1_from_confusion(cm)
    return total_loss / total, correct / total, macro_f1, cm


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MobileNetV3 + temporal mean-pool on optical-flow sequences."
    )
    p.add_argument("--data_dir",          default=Config.data_dir,
                   help="Root directory containing the extracted .npy files.")
    p.add_argument("--csv_index",         default=Config.csv_index,
                   help="Path to extraction_index.csv.")
    p.add_argument("--onset_csv",         default=Config.onset_csv)
    p.add_argument("--num_frames",        type=int,   default=Config.num_frames)
    p.add_argument("--mobilenet_variant", choices=["small", "large"],
                   default=Config.mobilenet_variant,
                   help="MobileNetV3 variant: 'small' (D=576) or 'large' (D=960).")
    p.add_argument("--epochs",            type=int,   default=Config.epochs)
    p.add_argument("--batch_size",        type=int,   default=Config.batch_size)
    p.add_argument("--lr",                type=float, default=Config.lr)
    p.add_argument("--weight_decay",      type=float, default=Config.weight_decay)
    p.add_argument("--val_split",         type=float, default=Config.val_split)
    p.add_argument("--split_mode",        choices=["subject", "random"],
                   default=Config.split_mode)
    p.add_argument("--balanced_sampler",  action="store_true",
                   default=Config.balanced_sampler)
    p.add_argument("--no_balanced_sampler", dest="balanced_sampler",
                   action="store_false")
    p.add_argument("--unfreeze_epoch",    type=int,   default=Config.unfreeze_epoch,
                   help="Epoch to unfreeze backbone (0 = keep frozen for all epochs).")
    p.add_argument("--seed",              type=int,   default=Config.seed)
    p.add_argument("--num_workers",       type=int,   default=Config.num_workers)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"[device] {device}")
    print(f"[model]  MobileNetV3-{args.mobilenet_variant}  T={args.num_frames} frames")

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"[data]   scanning '{args.data_dir}' ...")
    full_dataset = FlowSequenceDataset(
        data_dir   = args.data_dir,
        num_frames = args.num_frames,
        csv_index  = args.csv_index,
        onset_csv  = args.onset_csv,
    )
    num_classes = full_dataset.get_num_classes()
    label_map   = full_dataset.get_label_map()
    print(f"[data]   {len(full_dataset)} sequences | {num_classes} classes: {label_map}")

    if len(full_dataset) == 0:
        raise RuntimeError(
            "No sequences found. Check --data_dir and --csv_index paths.\n"
            f"  data_dir  = {args.data_dir}\n"
            f"  csv_index = {args.csv_index}"
        )

    # ── Train / val split ─────────────────────────────────────────────────
    if args.split_mode == "subject":
        train_ds, val_ds = split_subject_wise(full_dataset, args.val_split, args.seed)
        print("[split]  subject-wise")
    else:
        val_size   = max(1, int(len(full_dataset) * args.val_split))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print("[split]  random")
    print(f"[data]   train={len(train_ds)}  val={len(val_ds)}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_labels = get_subset_label_indices(train_ds, full_dataset)

    if args.balanced_sampler:
        sampler      = build_balanced_sampler(train_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size  = args.batch_size,
            sampler     = sampler,
            num_workers = args.num_workers,
            pin_memory  = device.type == "cuda",
        )
        print("[data]   balanced sampler ON")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = args.num_workers,
            pin_memory  = device.type == "cuda",
        )

    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = MobileNetFlowClassifier(
        num_classes       = num_classes,
        mobilenet_variant = args.mobilenet_variant,
        freeze_backbone   = True,
    ).to(device)

    # ── Loss & optimizer ──────────────────────────────────────────────────
    class_weights = build_class_weights(train_labels, num_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[loss]   class weights: {class_weights.cpu().numpy().round(3).tolist()}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_val_f1  = 0.0

    for epoch in range(1, args.epochs + 1):

        # Backbone unfreezing
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            print(f"[epoch {epoch:03d}] unfreezing MobileNetV3 backbone")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr           = args.lr * 0.1,   # lower LR for fine-tuning
                weight_decay = args.weight_decay,
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, val_cm = evaluate(
            model, val_loader, criterion, num_classes, device
        )

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_val_f1  = val_f1
            torch.save(model.state_dict(), "best_mobilenet_flow.pth")

        flag = " ✓" if improved else ""
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val  loss={val_loss:.4f}  acc={val_acc:.3f} f1={val_f1:.3f}{flag}"
        )
        print("[val cm]")
        print(val_cm)

    print(
        f"\n[done] best val  acc={best_val_acc:.3f}  macro-f1={best_val_f1:.3f}"
        f"  → saved to best_mobilenet_flow.pth"
    )


if __name__ == "__main__":
    main()
