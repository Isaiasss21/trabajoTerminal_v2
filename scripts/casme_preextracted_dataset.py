"""
casme_preextracted_dataset.py
------------------------------
Dataset de PyTorch para cargar secuencias de flujo óptico pre-extraídas del CASME II.

Los archivos .npy generados por extract_sequences.py tienen forma (N, 64, 64, 3) float32
con canales [dx, dy, mag]. __getitem__ los devuelve ya transpuestos al formato PyTorch
estándar: (N, 3, 64, 64).

Estructura de directorio esperada (salida de extract_sequences.py):
  <root>/
    <sujeto>/
      <emocion>/
        <secuencia>.npy
  extraction_index.csv   (opcional — permite carga más rápida que el escaneo)

Uso básico
----------
    from casme_preextracted_dataset import CASMEPreextractedDataset, casme_collate_fn
    from torch.utils.data import DataLoader

    ds = CASMEPreextractedDataset(
        root_dir="ruta/a/extracciones",
        csv_index="ruta/a/extracciones/extraction_index.csv",  # opcional
        min_frames=5,
    )
    print(ds)  # muestra distribución de clases

    loader = DataLoader(
        ds, batch_size=8, shuffle=True, collate_fn=casme_collate_fn
    )
    for tensors, labels, lengths in loader:
        # tensors: (B, N_max, 3, 64, 64) float32 — con zero-padding
        # labels:  (B,) int64
        # lengths: (B,) int64 — número real de frames por muestra
        ...
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Mapeo unificado de emociones CASME II ────────────────────────────────────
# Soporta nombres cortos y completos de distintas versiones del dataset.
EMOTION_TO_IDX: dict[str, int] = {
    # happiness
    "happiness": 0, "hap": 0, "happy": 0,
    # surprise
    "surprise": 1, "sur": 1,
    # disgust
    "disgust": 2, "dis": 2,
    # repression
    "repression": 3, "rep": 3,
    # others / neutral
    "others": 4, "oth": 4, "neutral": 4,
    # sadness
    "sadness": 5, "sad": 5,
    # fear
    "fear": 6, "fea": 6,
    # contempt
    "contempt": 7, "con": 7,
    # anger
    "anger": 8, "ang": 8,
    # tense (CASME I y algunas versiones de CASME II)
    "tense": 9, "ten": 9,
}

IDX_TO_EMOTION: dict[int, str] = {
    0: "happiness",
    1: "surprise",
    2: "disgust",
    3: "repression",
    4: "others",
    5: "sadness",
    6: "fear",
    7: "contempt",
    8: "anger",
    9: "tense",
}


class CASMEPreextractedDataset(Dataset):
    """
    Dataset de PyTorch para secuencias de flujo óptico del CASME II pre-extraídas.

    Parámetros
    ----------
    root_dir : str | Path
        Directorio raíz con la estructura <sujeto>/<emocion>/<secuencia>.npy
        (salida de extract_sequences.py).
    csv_index : str | Path | None
        Ruta al CSV generado por extract_sequences.py.  Si se proporciona,
        el índice se construye desde el CSV (más rápido que escanear el árbol).
        Las columnas esperadas son: sujeto, emocion, secuencia, archivo.
    transform : Callable | None
        Transformación opcional aplicada sobre el tensor (N, 3, 64, 64).
    emotion_to_idx : dict[str, int] | None
        Diccionario personalizado de mapeo emoción → índice.
        Por defecto usa EMOTION_TO_IDX.
    min_frames : int
        Descarta secuencias con menos frames que este umbral. Por defecto 1.
    subjects : list[str] | None
        Si se especifica, filtra sólo los sujetos indicados (útil para LOSO).
    emotions : list[str] | None
        Si se especifica, filtra sólo las emociones indicadas.
    """

    def __init__(
        self,
        root_dir: str | Path,
        csv_index: str | Path | None = None,
        transform: Optional[Callable] = None,
        emotion_to_idx: Optional[dict[str, int]] = None,
        min_frames: int = 1,
        subjects: Optional[list[str]] = None,
        emotions: Optional[list[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.emotion_to_idx = emotion_to_idx if emotion_to_idx is not None else EMOTION_TO_IDX
        self.min_frames = min_frames
        self._subject_filter = set(subjects) if subjects else None
        self._emotion_filter  = {e.lower() for e in emotions} if emotions else None

        # Cada entrada: (npy_path, label_idx, subject, emotion, seq_name)
        self.samples: list[tuple[Path, int, str, str, str]] = []

        if csv_index is not None:
            self._build_index_from_csv(Path(csv_index))
        else:
            self._build_index_from_dir()

    # ── Construcción del índice ────────────────────────────────────────────────

    def _build_index_from_dir(self) -> None:
        """Escanea recursivamente el directorio raíz buscando archivos .npy."""
        for npy_path in sorted(self.root_dir.rglob("*.npy")):
            # Estructura esperada: root / subject / emotion / sequence.npy
            parts = npy_path.relative_to(self.root_dir).parts
            if len(parts) < 3:
                continue
            subject  = parts[0]
            emotion  = parts[1]
            seq_name = Path(parts[2]).stem
            self._try_add(npy_path, subject, emotion, seq_name)

    def _build_index_from_csv(self, csv_path: Path) -> None:
        """Construye el índice a partir del CSV generado por extract_sequences.py."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV de índice no encontrado: {csv_path}")

        base = csv_path.parent  # directorio base para resolver rutas relativas

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject  = row.get("sujeto",    "").strip()
                emotion  = row.get("emocion",   "").strip()
                seq_name = row.get("secuencia", "").strip()
                rel_path = row.get("archivo",   "").strip()
                if not rel_path:
                    continue
                npy_path = base / rel_path
                self._try_add(npy_path, subject, emotion, seq_name)

    def _try_add(
        self, npy_path: Path, subject: str, emotion: str, seq_name: str
    ) -> None:
        """Valida la muestra y la agrega al índice si supera los filtros."""
        if not npy_path.exists():
            return

        emotion_lower = emotion.lower()

        if self._subject_filter and subject not in self._subject_filter:
            return
        if self._emotion_filter and emotion_lower not in self._emotion_filter:
            return

        label = self.emotion_to_idx.get(emotion_lower)
        if label is None:
            return  # emoción no reconocida — omitir

        # Verificar min_frames sin cargar el volumen completo en memoria
        try:
            arr = np.load(str(npy_path), mmap_mode="r")
            if arr.shape[0] < self.min_frames:
                return
        except Exception:
            return

        self.samples.append((npy_path, label, subject, emotion, seq_name))

    # ── Interfaz Dataset ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Devuelve (tensor, label):
          tensor : float32 de forma (N, 3, 64, 64)  — (frames, canales, H, W)
          label  : int — índice numérico de la emoción
        """
        npy_path, label, _subject, _emotion, _seq = self.samples[idx]

        volume = np.load(str(npy_path))          # (N, H, W, C) = (N, 64, 64, 3)
        tensor = (
            torch.from_numpy(volume.copy())      # copia para evitar problemas con mmap
            .permute(0, 3, 1, 2)                 # (N, 64, 64, 3) → (N, 3, 64, 64)
            .float()
        )

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    # ── Utilidades del dataset ────────────────────────────────────────────────

    def get_subject(self, idx: int) -> str:
        """Nombre del sujeto de la muestra i-ésima."""
        return self.samples[idx][2]

    def get_emotion_name(self, idx: int) -> str:
        """Nombre de la emoción de la muestra i-ésima."""
        return self.samples[idx][3]

    def get_sequence_name(self, idx: int) -> str:
        """Nombre de la secuencia de la muestra i-ésima."""
        return self.samples[idx][4]

    def get_labels(self) -> list[int]:
        """Lista completa de etiquetas (útil para WeightedRandomSampler)."""
        return [s[1] for s in self.samples]

    def get_subjects(self) -> list[str]:
        """Lista de sujetos únicos presentes en el dataset."""
        return sorted({s[2] for s in self.samples})

    def get_class_distribution(self) -> dict[str, int]:
        """Conteo de muestras por nombre de emoción."""
        dist: dict[str, int] = {}
        for _, _, _, emotion, _ in self.samples:
            dist[emotion] = dist.get(emotion, 0) + 1
        return dict(sorted(dist.items()))

    def get_num_classes(self) -> int:
        """Número de clases únicas presentes en el dataset."""
        return len({s[1] for s in self.samples})

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Pesos inversamente proporcionales a la frecuencia de cada clase.
        Útil para inicializar nn.CrossEntropyLoss(weight=...) o
        WeightedRandomSampler.

        Devuelve tensor float32 de forma (num_clases_únicas,) ordenado por índice.
        El índice 0 del tensor corresponde al label 0, etc.
        """
        labels = self.get_labels()
        unique_labels = sorted(set(labels))
        counts = {lbl: labels.count(lbl) for lbl in unique_labels}
        total = sum(counts.values())
        # Peso = total / (num_clases * count_clase)  — normalizado
        n_cls = len(unique_labels)
        weights = torch.tensor(
            [total / (n_cls * counts[lbl]) for lbl in unique_labels],
            dtype=torch.float32,
        )
        return weights

    def split_by_subject(
        self, test_subjects: list[str]
    ) -> tuple["CASMEPreextractedDataset", "CASMEPreextractedDataset"]:
        """
        Divide el dataset en train/test según los sujetos de prueba indicados.
        Útil para validación cruzada Leave-One-Subject-Out (LOSO).

        Devuelve (train_dataset, test_dataset) como instancias vacías
        con el índice ya filtrado.
        """
        test_set  = set(test_subjects)
        train_smp = [s for s in self.samples if s[2] not in test_set]
        test_smp  = [s for s in self.samples if s[2] in test_set]

        train_ds = CASMEPreextractedDataset.__new__(CASMEPreextractedDataset)
        train_ds.root_dir       = self.root_dir
        train_ds.transform      = self.transform
        train_ds.emotion_to_idx = self.emotion_to_idx
        train_ds.min_frames     = self.min_frames
        train_ds._subject_filter = None
        train_ds._emotion_filter  = None
        train_ds.samples        = train_smp

        test_ds = CASMEPreextractedDataset.__new__(CASMEPreextractedDataset)
        test_ds.root_dir        = self.root_dir
        test_ds.transform       = self.transform
        test_ds.emotion_to_idx  = self.emotion_to_idx
        test_ds.min_frames      = self.min_frames
        test_ds._subject_filter = None
        test_ds._emotion_filter  = None
        test_ds.samples         = test_smp

        return train_ds, test_ds

    def __repr__(self) -> str:
        dist  = self.get_class_distribution()
        lines = [
            "CASMEPreextractedDataset(",
            f"  root_dir   = {self.root_dir}",
            f"  samples    = {len(self.samples)}",
            f"  classes    = {self.get_num_classes()}",
            f"  min_frames = {self.min_frames}",
            f"  distribución de clases:",
        ]
        for emotion, count in dist.items():
            lines.append(f"    {emotion:<16} {count:>4} muestras")
        lines.append(")")
        return "\n".join(lines)


# ── Función de collation para secuencias de longitud variable ─────────────────

def casme_collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Función de collation para DataLoader que maneja secuencias de longitud variable.

    Las secuencias se rellenan con ceros (zero-padding) hasta la longitud máxima
    del batch para formar un tensor uniforme.

    Parámetros
    ----------
    batch : lista de (tensor, label) donde tensor tiene forma (N_i, 3, 64, 64).

    Retorna
    -------
    tensors : (B, N_max, 3, 64, 64) float32 — secuencias con padding
    labels  : (B,) int64
    lengths : (B,) int64 — longitud real de cada secuencia antes del padding
    """
    tensors, labels = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    n_max   = int(lengths.max().item())
    _, c, h, w = tensors[0].shape

    padded = torch.zeros(len(tensors), n_max, c, h, w, dtype=torch.float32)
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t

    return padded, torch.tensor(labels, dtype=torch.long), lengths
