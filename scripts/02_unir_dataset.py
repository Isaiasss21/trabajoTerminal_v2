import pandas as pd
from pathlib import Path


MASTER_CONTRACT_COLUMNS = ["clip_path", "subject_id", "label", "apex_frame", "landmarks_path"]


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path).copy()

    if "subject_id" not in df.columns:
        if "persona" in df.columns:
            df["subject_id"] = df["persona"].astype(str)
        else:
            df["subject_id"] = None

    if "label" not in df.columns:
        if "emocion" in df.columns:
            df["label"] = df["emocion"].astype(str)
        else:
            df["label"] = None

    if "apex_frame" not in df.columns:
        if "apex_idx" in df.columns:
            df["apex_frame"] = df["apex_idx"]
        else:
            df["apex_frame"] = None

    if "clip_path" not in df.columns:
        df["clip_path"] = None

    if "landmarks_path" not in df.columns:
        df["landmarks_path"] = None

    return df


def _reorder_master_contract(df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = MASTER_CONTRACT_COLUMNS + [col for col in df.columns if col not in MASTER_CONTRACT_COLUMNS]
    return df.reindex(columns=ordered_columns)


def unir_csvs():
    print("[INFO] Iniciando fusión de datasets base (CASME II + SMIC)...")

    csv_casme = Path("pipeline_out_resnet_casme/features_index_resnet.csv")
    csv_smic = Path("pipeline_out_resnet_smic/features_index_resnet.csv")

    out_dir = Path("pipeline_out_resnet_maestro")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "features_index_maestro.csv"

    dfs = []

    for name, csv_path in [("CASME II", csv_casme), ("SMIC", csv_smic)]:
        loaded = _load_csv(csv_path)
        if loaded is None:
            print(f"[WARN] No se encontró {name} en: {csv_path}")
            continue
        print(f"[OK] {name} encontrado: {csv_path}")
        dfs.append(loaded)

    if not dfs:
        print("\n[ERROR] Fusión abortada. Faltan los archivos de extracción del Script 01.")
        return

    all_columns = []
    for df in dfs:
        for col in df.columns:
            if col not in all_columns:
                all_columns.append(col)

    normalized = [df.reindex(columns=all_columns) for df in dfs]
    df_maestro = pd.concat(normalized, ignore_index=True, sort=False)
    df_maestro = _reorder_master_contract(df_maestro)
    df_maestro.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"\n[ÉXITO] Fusión completada. {len(df_maestro)} secuencias totales.")
    print(f"[SAVE] Archivo maestro: {out_csv}")
    print("-" * 40)
    print("Resumen de emociones en el dataset maestro (Antes de agrupar):")
    if "label" in df_maestro.columns:
        print(df_maestro["label"].value_counts())
    elif "emocion" in df_maestro.columns:
        print(df_maestro["emocion"].value_counts())
    print("-" * 40)


if __name__ == "__main__":
    unir_csvs()