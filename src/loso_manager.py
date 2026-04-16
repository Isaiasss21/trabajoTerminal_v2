from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]


def resolve_series(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in df.columns:
        return df[preferred].astype(str)
    if fallback in df.columns:
        return df[fallback].astype(str)
    raise ValueError(f"Faltan columnas '{preferred}' y '{fallback}' en el índice")


def default_person_split(personas: List[str], n_train: int = 10, n_val: int = 3) -> Tuple[List[str], List[str], List[str]]:
    personas = list(personas)
    if not personas:
        return [], [], []
    # Stable shuffle so the split remains reproducible.
    import random

    random.seed(42)
    random.shuffle(personas)
    train = personas[:n_train]
    val = personas[n_train : n_train + n_val]
    test = personas[n_train + n_val :]
    return train, val, test


def split_by_persona(df: pd.DataFrame, train_personas: List[str], val_personas: List[str], test_personas: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["persona"].isin(train_personas)].reset_index(drop=True)
    va = df[df["persona"].isin(val_personas)].reset_index(drop=True)
    te = df[df["persona"].isin(test_personas)].reset_index(drop=True)
    return tr, va, te


def split_by_subject(df: pd.DataFrame, subject_col: str, test_subject: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subject_series = resolve_series(df, subject_col, "persona")
    test_mask = subject_series == str(test_subject)
    train_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    return train_df, test_df


def split_train_val_by_subjects(df: pd.DataFrame, subject_col: str, val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subject_series = resolve_series(df, subject_col, "persona")
    subjects = sorted(subject_series.unique().tolist())
    if not subjects:
        raise ValueError("No hay sujetos disponibles para el split de validación")
    n_val = max(1, int(len(subjects) * val_fraction))
    val_subjects = set(subjects[:n_val])
    train_mask = ~subject_series.isin(val_subjects)
    val_mask = subject_series.isin(val_subjects)
    return df.loc[train_mask].reset_index(drop=True), df.loc[val_mask].reset_index(drop=True)


def build_split_bundle(
    df_index: pd.DataFrame,
    validation_mode: str = "split",
    subject_col: str = "subject_id",
    test_subject: str | None = None,
    val_fraction: float = 0.2,
) -> SplitBundle:
    subject_series = resolve_series(df_index, subject_col, "persona")
    subjects = sorted(subject_series.unique().tolist())
    if not subjects:
        raise ValueError("No hay sujetos disponibles para construir splits")

    if validation_mode.lower() == "loso":
        chosen_test_subject = str(test_subject) if test_subject is not None else subjects[-1]
        train_val_df, test_df = split_by_subject(df_index, subject_col, chosen_test_subject)
        train_val_subjects = sorted(resolve_series(train_val_df, subject_col, "persona").unique().tolist())
        train_subjects, val_subjects, _ = default_person_split(
            train_val_subjects,
            n_train=max(1, int(len(train_val_subjects) * (1.0 - val_fraction))),
            n_val=max(1, int(len(train_val_subjects) * val_fraction)),
        )
        train_df = train_val_df[resolve_series(train_val_df, subject_col, "persona").isin(train_subjects)].reset_index(drop=True)
        val_df = train_val_df[resolve_series(train_val_df, subject_col, "persona").isin(val_subjects)].reset_index(drop=True)
        return SplitBundle(train_df, val_df, test_df, train_subjects, val_subjects, [chosen_test_subject])

    train_subjects, val_subjects, test_subjects = default_person_split(
        subjects,
        n_train=max(1, int(len(subjects) * (1.0 - val_fraction - 0.13))),
        n_val=max(1, int(len(subjects) * val_fraction)),
    )
    train_df, val_df, test_df = split_by_persona(df_index, train_subjects, val_subjects, test_subjects)
    return SplitBundle(train_df, val_df, test_df, train_subjects, val_subjects, test_subjects)