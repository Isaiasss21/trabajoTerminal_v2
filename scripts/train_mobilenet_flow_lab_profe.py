import numpy as np
print("numpy OK")

import torch
print("torch OK")

from torchvision.models import mobilenet_v3_small
print("torchvision OK")

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

ROOT_DIR = Path("../")


DATA_DIR = ROOT_DIR / 'extracted_flow'
'''CSV_PATH = ROOT_DIR / 'outputs_dataset_index' / 'extraction_index.csv'''
CSV_PATH = ROOT_DIR / 'extracted_flow' / 'extraction_index_newMemeUni.csv'

OUTPUT_DIR = ROOT_DIR / 'Mobile' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('Data dir:', DATA_DIR.resolve())

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def flow_to_pseudo_rgb(frames):
    out = np.empty_like(frames)
    out[..., 0] = (frames[..., 0] + 1) / 2
    out[..., 1] = (frames[..., 1] + 1) / 2
    out[..., 2] = frames[..., 2]
    return np.clip(out, 0, 1)

def preprocess_frame(frame):
    img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    img = TF.resize(img, [224, 224])
    img = TF.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
    return img

def sample_frames(sequence, num_frames=16):
    n = sequence.shape[0]
    if n >= num_frames:
        idx = np.linspace(0, n - 1, num_frames).astype(int)
    else:
        idx = np.tile(np.arange(n), (num_frames // n) + 1)[:num_frames]
    return sequence[idx]

import csv

EMOTION_TO_IDX = {
    'happiness': 0, 'felicidad': 0,
    'surprise': 1, 'sorpresa': 1,
    'disgust': 2, 'asco': 2,
    'repression': 3,
    'others': 4, 'neutral': 4,
    'sadness': 5, 'tristeza': 5,
    'fear': 6, 'miedo': 6,
}

class FlowDataset(Dataset):
    def __init__(self, data_dir, csv_path, num_frames=16):
        self.samples = []
        self.num_frames = num_frames
        self.data_dir = Path(data_dir)

        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                archivo_rel = row['archivo'].replace("\\", "/")
                path = self.data_dir / archivo_rel
                emo = row['emocion'].lower()
                if emo in EMOTION_TO_IDX and path.exists():
                    self.samples.append((path, emo))

        labels = sorted(set(e for _, e in self.samples))
        self.label_map = {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, emo = self.samples[idx]
        label = self.label_map[emo]

        seq = np.load(path)
        frames = sample_frames(seq, self.num_frames)
        frames = flow_to_pseudo_rgb(frames)

        frames = (frames * 255).astype(np.uint8)
        frames = torch.stack([preprocess_frame(f) for f in frames])

        return frames, label

    def num_classes(self):
        return len(self.label_map)

class MobileNetFlow(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        self.features = base.features
        self.pool = base.avgpool

        self.features = base.features
        # congelar primeras capas solamente
        for i, layer in enumerate(self.features):
            if i < 8:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # clasificador con regularización
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(576, num_classes)
        )
        self.fc = nn.Linear(576, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = x.view(B, T, -1).mean(1)
        return self.fc(x)

def train(model, loader, optimizer, loss_fn):
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(out, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # -----------------------------
    # Calcular métricas manualmente
    # -----------------------------
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(loader)

    # Accuracy manual
    acc = np.mean(all_preds == all_labels)

    # F1 macro manual
    classes = np.unique(all_labels)
    f1_scores = []

    for c in classes:
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if (precision + recall) > 0:
            f1_class = 2 * (precision * recall) / (precision + recall)
        else:
            f1_class = 0

        f1_scores.append(f1_class)

    f1 = np.mean(f1_scores)

    return avg_loss, acc, f1

from torch.utils.data import random_split, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random

# =========================
# TRAIN FUNCTION
# =========================
def train(model, loader, optimizer, loss_fn):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(out, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(loader)

    # Accuracy manual
    acc = np.mean(all_preds == all_labels)

    # F1 macro manual
    classes = np.unique(all_labels)
    f1_scores = []

    for c in classes:
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            f1_class = 2 * precision * recall / (precision + recall)
        else:
            f1_class = 0

        f1_scores.append(f1_class)

    f1 = np.mean(f1_scores)

    return avg_loss, acc, f1


# =========================
# VALIDATION FUNCTION
# =========================
def validate(model, loader, loss_fn):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            total_loss += loss.item()

            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(loader)

    # Accuracy manual
    acc = np.mean(all_preds == all_labels)

    # F1 macro manual
    classes = np.unique(all_labels)
    f1_scores = []

    for c in classes:
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            f1_class = 2 * precision * recall / (precision + recall)
        else:
            f1_class = 0

        f1_scores.append(f1_class)

    f1 = np.mean(f1_scores)

    return avg_loss, acc, f1


# =========================
# DATASET SPLIT
# =========================
dataset = FlowDataset(DATA_DIR, CSV_PATH)

# =========================
# OBTENER SUJETOS ÚNICOS
# =========================
subjects = []

for path, emo in dataset.samples:
    # path ejemplo:
    # extracted_flow/p034/camaraWeb/Asco.npy
    subject = path.parts[-3]   # obtiene p034
    subjects.append(subject)

subjects = list(set(subjects))

print("Sujetos encontrados:")
print(subjects)

# mezclar sujetos
random.seed(42)
random.shuffle(subjects)

# =========================
# SPLIT 80/20 POR SUJETO
# =========================
split_idx = int(0.8 * len(subjects))

train_subjects = subjects[:split_idx]
val_subjects = subjects[split_idx:]

print("\nSujetos train:")
print(train_subjects)

print("\nSujetos validation:")
print(val_subjects)

# =========================
# OBTENER ÍNDICES
# =========================
train_indices = []
val_indices = []

for i, (path, emo) in enumerate(dataset.samples):
    subject = path.parts[-3]

    if subject in train_subjects:
        train_indices.append(i)
    else:
        val_indices.append(i)

# crear subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

# =========================
# INFO FINAL
# =========================
print("\nResumen split:")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# =========================
# MODEL
# =========================
model = MobileNetFlow(dataset.num_classes()).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.CrossEntropyLoss()

EPOCHS = 20


# =========================
# RESULTS STORAGE
# =========================
results = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "train_f1": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": []
}


# =========================
# TRAINING LOOP
# =========================
best_val_f1 = 0

for epoch in range(EPOCHS):

    train_loss, train_acc, train_f1 = train(
        model,
        train_loader,
        optimizer,
        loss_fn
    )

    val_loss, val_acc, val_f1 = validate(
        model,
        val_loader,
        loss_fn
    )

    print(
        f"Epoch {epoch+1}: "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_acc:.4f} | "
        f"train_f1={train_f1:.4f} || "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f} | "
        f"val_f1={val_f1:.4f}"
    )

    # guardar solo mejor modelo
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            OUTPUT_DIR / "best_model.pth"
        )

        print("✅ Mejor modelo guardado")

# =========================
# FINAL TABLE
# =========================
print("\nRESUMEN FINAL\n")

print(
    f"{'Epoch':<8}"
    f"{'Train Loss':<15}"
    f"{'Train Acc':<15}"
    f"{'Train F1':<15}"
    f"{'Val Loss':<15}"
    f"{'Val Acc':<15}"
    f"{'Val F1':<15}"
)

for i in range(len(results["epoch"])):
    print(
        f"{results['epoch'][i]:<8}"
        f"{results['train_loss'][i]:<15.4f}"
        f"{results['train_acc'][i]:<15.4f}"
        f"{results['train_f1'][i]:<15.4f}"
        f"{results['val_loss'][i]:<15.4f}"
        f"{results['val_acc'][i]:<15.4f}"
        f"{results['val_f1'][i]:<15.4f}"
    )

from collections import Counter

class_counter = Counter()

for path, emotion in dataset.samples:
    class_counter[emotion] += 1

print("Distribución de clases:")
for emotion, count in class_counter.items():
    print(f"{emotion}: {count}")
