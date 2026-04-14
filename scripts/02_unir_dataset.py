import pandas as pd
from pathlib import Path

def unir_csvs():
    print("[INFO] Iniciando fusión de datasets base (CASME II + SMIC)...")
    
    # Rutas esperadas generadas por el Script 01
    csv_casme = Path("pipeline_out_resnet_casme/features_index_resnet.csv")
    csv_smic = Path("pipeline_out_resnet_smic/features_index_resnet.csv")
    
    out_dir = Path("pipeline_out_resnet_maestro")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "features_index_maestro.csv"

    dfs = []
    
    if csv_casme.exists():
        print(f"[OK] CASME II encontrado: {csv_casme}")
        dfs.append(pd.read_csv(csv_casme))
    else:
        print(f"[WARN] No se encontró CASME II en: {csv_casme}")

    if csv_smic.exists():
        print(f"[OK] SMIC encontrado: {csv_smic}")
        dfs.append(pd.read_csv(csv_smic))
    else:
        print(f"[WARN] No se encontró SMIC en: {csv_smic}")

    if dfs:
        df_maestro = pd.concat(dfs, ignore_index=True)
        df_maestro.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"\n[ÉXITO] Fusión completada. {len(df_maestro)} secuencias totales.")
        print(f"[SAVE] Archivo maestro: {out_csv}")
        print("-" * 40)
        print("Resumen de emociones en el dataset maestro (Antes de agrupar):")
        print(df_maestro['emocion'].value_counts())
        print("-" * 40)
    else:
        print("\n[ERROR] Fusión abortada. Faltan los archivos de extracción del Script 01.")

if __name__ == "__main__":
    unir_csvs()