"dataio.py"
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def read_sequences_csv(sequences_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(sequences_csv)
    # expected: persona, camara, emocion, paths_joined
    for col in ["persona", "camara", "emocion", "paths_joined"]:
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}' en {sequences_csv}")
    return df

def parse_paths_joined(paths_joined: str) -> List[str]:
    return [p for p in str(paths_joined).split("|") if p]

@dataclass
class FeatureRecord:
    persona: str
    camara: str
    emocion: str
    feature_path: str  # .npy

def save_feature_index(records: List[FeatureRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "camara", "emocion", "feature_path"])
        for r in records:
            w.writerow([r.persona, r.camara, r.emocion, r.feature_path])

def load_feature_index(index_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    for col in ["persona", "camara", "emocion", "feature_path"]:
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}' en {index_csv}")
    return df
