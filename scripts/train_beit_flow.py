"""
train_beit_flow.py
------------------
Sequence-level micro-expression classifier using BEiT as frame encoder.

Pipeline per sample:
  .npy (N, 64, 64, 3) float32
    → uniform frame sampling  → T frames (64, 64, 3)
    → flow-to-pseudo-RGB mapping + resize 224×224 + BEiT processor
    → BeitModel  (B*T, 3, 224, 224) → (B*T, D)
    → reshape + mean-pool over T  → (B, D)
    → Linear classifier          → (B, num_classes)

Run example:
  python scripts/train_beit_flow.py \
      --data_dir  path/to/extracted_npy   \
      --csv_index path/to/extraction_index.csv \
      --epochs    20                          \
      --batch_size 8
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import resize as tv_resize
from transformers import BeitModel, AutoImageProcessor

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
    data_dir:    str  = "outputs"          # root with subject/emotion/*.npy
    csv_index:   str | None = None         # optional extraction_index.csv
    onset_csv:   str | None = None         # optional CSV with onset/offset cols

    # Frame sampling
    num_frames:  int  = 8                  # fixed T per sample

    # BEiT
    beit_name:   str  = "microsoft/beit-base-patch16-224-pt22k-ft22k"
    freeze_beit: bool = True               # freeze backbone at start

    # Training
    num_classes: int  = 5                  # updated automatically from data
    epochs:      int  = 20
    batch_size:  int  = 8
    lr:          float = 1e-4
    weight_decay: float = 1e-2
    val_split:   float = 0.2
    seed:        int  = 42
    num_workers: int  = 0                  # set >0 on Linux; 0 is safer on Windows


# ══════════════════════════════════════════════════════════════════════════════
# LABEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Unified CASME II emotion vocabulary (same as casme_preextracted_dataset.py)
EMOTION_TO_IDX: dict[str, int] = {
    "happiness": 0, "hap": 0, "happy": 0,
    "surprise":  1, "sur": 1,
    "disgust":   2, "dis": 2,
    "repression":3, "rep": 3,
    "others":    4, "oth": 4, "neutral": 4,
    "sadness":   5, "sad": 5,
    "fear":      6, "fea": 6,
    "contempt":  7, "con": 7,
    "anger":     8, "ang": 8,
    "tense":     9, "ten": 9,
}

IDX_TO_EMOTION: dict[int, str] = {
    0: "happiness", 1: "surprise", 2: "disgust",  3: "repression",
    4: "others",    5: "sadness",  6: "fear",      7: "contempt",
    8: "anger",     9: "tense",
}


def build_label_map(labels_found: set[str]) -> dict[str, int]:
    """
    Build a compact 0-based label map from the emotions actually present
    in the dataset. This avoids sparse label tensors when only a subset
    of the 10 emotions are available.
    """
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
    Return exactly `num_frames` frames from `sequence` (N, H, W, 3).

    If onset/offset are provided, restrict sampling to that range.
    If the available range is shorter than num_frames, indices are
    repeated (tile) to ensure a fixed-size output.
    """
    n = sequence.shape[0]
    start = max(0, onset  if onset  is not None else 0)
    end   = min(n, offset if offset is not None else n)
    if end <= start:        # fallback: use full sequence
        start, end = 0, n

    available = end - start  # number of frames in the valid range

    if available >= num_frames:
        # Uniform sampling without replacement
        indices = np.linspace(start, end - 1, num_frames, dtype=int)
    else:
        # Tile indices to fill num_frames (handles very short sequences)
        base = np.arange(start, end)
        reps = (num_frames // available) + 1
        indices = np.tile(base, reps)[:num_frames]

    return sequence[indices]  # (num_frames, H, W, 3)


# ══════════════════════════════════════════════════════════════════════════════
# FLOW → PSEUDO-RGB PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def flow_to_pseudo_rgb(frames: np.ndarray) -> np.ndarray:
    """
    Map optical-flow channels to a [0, 1] pseudo-RGB image.

    Input:  (T, H, W, 3) float32 with channels [dx, dy, mag]
              dx  ∈ [-1, 1]
              dy  ∈ [-1, 1]
              mag ∈ [ 0, 1]
    Output: (T, H, W, 3) float32 in [0, 1] — compatible with BEiT processor.
    """
    out = np.empty_like(frames)
    out[..., 0] = np.clip((frames[..., 0] + 1.0) / 2.0, 0.0, 1.0)  # dx → [0,1]
    out[..., 1] = np.clip((frames[..., 1] + 1.0) / 2.0, 0.0, 1.0)  # dy → [0,1]
    out[..., 2] = np.clip(frames[..., 2], 0.0, 1.0)                 # mag (already [0,1])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FlowSequenceDataset(Dataset):
    """
    PyTorch Dataset for pre-extracted optical-flow sequences (.npy).

    Each __getitem__ returns:
      frames : Tensor (T, 3, 224, 224) float32 — BEiT-ready pseudo-RGB
      label  : int — emotion class index
    """

    def __init__(
        self,
        data_dir:   str | Path,
        processor,                       # HF BEiT image processor
        num_frames: int  = 8,
        csv_index:  str | Path | None = None,
        onset_csv:  str | Path | None = None,
        label_map:  dict[str, int] | None = None,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.processor  = processor
        self.num_frames = num_frames

        # onset/offset lookup: {seq_name: (onset, offset)}
        self.onsets: dict[str, tuple[int, int]] = {}
        if onset_csv:
            self._load_onset_csv(Path(onset_csv))

        # Build sample list: [(npy_path, emotion_str), ...]
        self.samples: list[tuple[Path, str]] = []
        if csv_index:
            self._build_from_csv(Path(csv_index))
        else:
            self._build_from_dir()

        # Compact label map (only classes present in this split)
        if label_map is not None:
            self.label_map = label_map
        else:
            found = {emotion for _, emotion in self.samples}
            self.label_map = build_label_map(found)

    # ── Index builders ─────────────────────────────────────────────────────

    def _build_from_dir(self) -> None:
        """Scan data_dir for subject/emotion/sequence.npy structure."""
        for npy_path in sorted(self.data_dir.rglob("*.npy")):
            parts = npy_path.relative_to(self.data_dir).parts
            if len(parts) < 3:
                continue
            emotion = parts[1].lower()
            if emotion in EMOTION_TO_IDX:
                self.samples.append((npy_path, emotion))

    def _build_from_csv(self, csv_path: Path) -> None:
        """Build index from extraction_index.csv."""
        base = csv_path.parent
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel  = row.get("archivo", "").strip()
                emo  = row.get("emocion", "").strip().lower()
                if not rel or emo not in EMOTION_TO_IDX:
                    continue
                npy_path = base / rel
                if npy_path.exists():
                    self.samples.append((npy_path, emo))

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
        npy_path, emotion = self.samples[idx]
        label = self.label_map[emotion]

        # 1. Load full sequence
        sequence = np.load(str(npy_path))  # (N, 64, 64, 3) float32

        # 2. Sample T frames, respecting onset/offset if available
        onset, offset = self.onsets.get(npy_path.stem, (None, None))
        frames = sample_frames(sequence, self.num_frames, onset, offset)
        # frames: (T, 64, 64, 3) float32

        # 3. Map flow channels to pseudo-RGB [0, 1]
        frames = flow_to_pseudo_rgb(frames)  # (T, 64, 64, 3) float32

        # 4. Resize each frame to 224×224 and apply BEiT processor normalization.
        #    processor accepts a list of PIL-like uint8 images or float arrays.
        #    We convert to uint8 [0,255] so the processor handles normalization
        #    consistently with its pretrained statistics.
        frames_uint8 = (frames * 255).clip(0, 255).astype(np.uint8)
        # processor returns dict with "pixel_values": (T, 3, 224, 224) float32
        processed = self.processor(
            images=list(frames_uint8),        # list of T (H, W, 3) uint8 arrays
            return_tensors="pt",
            do_resize=True,
            size={"height": 224, "width": 224},
        )
        pixel_values = processed["pixel_values"]  # (T, 3, 224, 224)

        return pixel_values, label

    def get_num_classes(self) -> int:
        return len(self.label_map)

    def get_label_map(self) -> dict[str, int]:
        return dict(self.label_map)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class BeitFlowClassifier(nn.Module):
    """
    BEiT frame encoder + temporal mean pooling + linear classifier.

    Forward input:  (B, T, 3, 224, 224)
    Forward output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int, beit_name: str, freeze_beit: bool = True) -> None:
        super().__init__()

        # ── BEiT backbone ─────────────────────────────────────────────────
        self.beit = BeitModel.from_pretrained(beit_name, add_pooling_layer=False)
        hidden_dim = self.beit.config.hidden_size  # 768 for base

        if freeze_beit:
            for param in self.beit.parameters():
                param.requires_grad = False

        # ── Classifier head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 3, 224, 224)
        """
        B, T, C, H, W = x.shape

        # Flatten temporal dimension for a single efficient forward pass
        x_flat = x.view(B * T, C, H, W)                  # (B*T, 3, 224, 224)

        # BEiT returns last_hidden_state: (B*T, num_patches+1, D)
        beit_out = self.beit(pixel_values=x_flat)
        tokens   = beit_out.last_hidden_state              # (B*T, S, D)

        # Use the [CLS] token (index 0) as the frame-level embedding
        cls_tokens = tokens[:, 0, :]                       # (B*T, D)

        # Reshape back to (B, T, D) and mean-pool over T
        cls_tokens = cls_tokens.view(B, T, -1)             # (B, T, D)
        seq_embed  = cls_tokens.mean(dim=1)                # (B, D)

        return self.classifier(seq_embed)                  # (B, num_classes)

    def unfreeze_beit(self) -> None:
        """Call this to enable fine-tuning of the BEiT backbone."""
        for param in self.beit.parameters():
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
    """One full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)    # (B, T, 3, 224, 224)
        labels = labels.to(device, non_blocking=True)    # (B,)

        optimizer.zero_grad()
        logits = model(frames)                           # (B, num_classes)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Validation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BEiT + temporal mean-pool on optical-flow sequences."
    )
    parser.add_argument("--data_dir",    default=Config.data_dir)
    parser.add_argument("--csv_index",   default=Config.csv_index)
    parser.add_argument("--onset_csv",   default=Config.onset_csv)
    parser.add_argument("--num_frames",  type=int,   default=Config.num_frames)
    parser.add_argument("--beit_name",   default=Config.beit_name)
    parser.add_argument("--epochs",      type=int,   default=Config.epochs)
    parser.add_argument("--batch_size",  type=int,   default=Config.batch_size)
    parser.add_argument("--lr",          type=float, default=Config.lr)
    parser.add_argument("--val_split",   type=float, default=Config.val_split)
    parser.add_argument("--seed",        type=int,   default=Config.seed)
    parser.add_argument("--num_workers", type=int,   default=Config.num_workers)
    parser.add_argument(
        "--unfreeze_epoch", type=int, default=0,
        help="Epoch at which to unfreeze BEiT (0 = never unfreeze during training).",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"[device] {device}")

    # ── Load BEiT processor ───────────────────────────────────────────────
    print(f"[BEiT]  loading processor from '{args.beit_name}' ...")
    processor = AutoImageProcessor.from_pretrained(args.beit_name)

    # ── Build dataset ─────────────────────────────────────────────────────
    print(f"[data]  scanning '{args.data_dir}' ...")
    full_dataset = FlowSequenceDataset(
        data_dir   = args.data_dir,
        processor  = processor,
        num_frames = args.num_frames,
        csv_index  = args.csv_index,
        onset_csv  = args.onset_csv,
    )
    num_classes = full_dataset.get_num_classes()
    label_map   = full_dataset.get_label_map()
    print(f"[data]  {len(full_dataset)} sequences | {num_classes} classes: {label_map}")

    if len(full_dataset) == 0:
        raise RuntimeError(
            "No sequences found. Check --data_dir and --csv_index paths."
        )

    # ── Train / validation split ──────────────────────────────────────────
    val_size   = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

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
    print(f"[data]  train={train_size}  val={val_size}")

    # ── Build model ───────────────────────────────────────────────────────
    print(f"[model] building BeitFlowClassifier (freeze_beit={True}) ...")
    model = BeitFlowClassifier(
        num_classes = num_classes,
        beit_name   = args.beit_name,
        freeze_beit = True,
    ).to(device)

    # ── Loss and optimizer ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):

        # Optional backbone unfreezing at a given epoch
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            print(f"[epoch {epoch}] unfreezing BEiT backbone")
            model.unfreeze_beit()
            # Reinitialize optimizer to include newly unfrozen params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr           = args.lr * 0.1,   # lower LR for fine-tuning
                weight_decay = args.weight_decay,
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        flag = " ✓" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_beit_flow.pth")

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val  loss={val_loss:.4f}  acc={val_acc:.3f}{flag}"
        )

    print(f"\n[done] best val accuracy: {best_val_acc:.3f}  → saved to best_beit_flow.pth")


if __name__ == "__main__":
    main()
