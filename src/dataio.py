from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SEQUENCE_REQUIRED_COLUMNS = ["persona", "camara", "emocion", "paths_joined"]
FEATURE_REQUIRED_COLUMNS = ["persona", "camara", "emocion", "feature_path"]


def _ensure_required_columns(df: pd.DataFrame, required_columns: Sequence[str], source: str | Path) -> pd.DataFrame:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en {source}")
    return df


def read_sequences_csv(sequences_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(sequences_csv)
    return _ensure_required_columns(df, SEQUENCE_REQUIRED_COLUMNS, sequences_csv)

def parse_paths_joined(paths_joined: str) -> List[str]:
    return [p for p in str(paths_joined).split("|") if p]

@dataclass
class SequenceRecord:
    persona: str
    camara: str
    emocion: str
    paths_joined: str
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    apex_idx: Optional[int] = None
    num_frames: Optional[int] = None
    clip_path: Optional[str] = None


@dataclass
class FeatureRecord:
    persona: str
    camara: str
    emocion: str
    feature_path: str  # .npy


def save_sequence_index(records: List[SequenceRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "persona",
            "camara",
            "emocion",
            "paths_joined",
            "frame_start",
            "frame_end",
            "apex_idx",
            "num_frames",
            "clip_path",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({
                "persona": r.persona,
                "camara": r.camara,
                "emocion": r.emocion,
                "paths_joined": r.paths_joined,
                "frame_start": r.frame_start,
                "frame_end": r.frame_end,
                "apex_idx": r.apex_idx,
                "num_frames": r.num_frames,
                "clip_path": r.clip_path,
            })

def save_feature_index(records: List[FeatureRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "camara", "emocion", "feature_path"])
        for r in records:
            w.writerow([r.persona, r.camara, r.emocion, r.feature_path])

def load_feature_index(index_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    return _ensure_required_columns(df, FEATURE_REQUIRED_COLUMNS, index_csv)


def load_sequence_index(index_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    return _ensure_required_columns(df, SEQUENCE_REQUIRED_COLUMNS, index_csv)


def iter_loso_splits(df: pd.DataFrame, subject_col: str = "persona") -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    if subject_col not in df.columns:
        raise ValueError(f"Falta columna '{subject_col}' para generar splits LOSO")

    subjects = sorted(df[subject_col].dropna().astype(str).unique().tolist())
    for subject in subjects:
        test_mask = df[subject_col].astype(str) == subject
        test_df = df.loc[test_mask].reset_index(drop=True)
        train_df = df.loc[~test_mask].reset_index(drop=True)
        yield subject, train_df, test_df


def normalize_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with expected identifiers normalized to string dtype."""
    out = df.copy()
    for col in ["persona", "camara", "emocion"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out
