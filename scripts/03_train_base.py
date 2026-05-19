"""
03_train_base.py  —  EmotionFrameLSTM con ResNet18 CONGELADO permanentemente.

Cambio clave vs versión anterior:
  freeze_encoder_epochs = 999  →  el ResNet18 nunca se desbloquea.
  Solo entrenan: LSTM + cabeza de clasificación.

Uso:
    python scripts/03_train_base.py
    python scripts/03_train_base.py --epochs 50 --dropout 0.6 --hidden_dim 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_frame_lstm  # noqa: E402


DEFAULTS = {
    "csv":        PROJECT_ROOT / "outputs_dataset_index" / "extraction_index.csv",
    "output_dir": PROJECT_ROOT / "outputs" / "run_frozen",

    # Modelo — reducido para evitar overfitting con ~190 muestras
    "frame_embedding_dim": 256,
    "hidden_dim":          64,    # ← bajado de 128
    "num_layers":          1,
    "dropout":             0.6,   # ← subido de 0.5
    "bidirectional":       True,
    "baseline_subtract":   True,
    "baseline_frames":     3,

    # Entrenamiento
    "epochs":               60,
    "batch_size":           8,
    "lr":                   5e-4,
    "weight_decay":         1e-3,  # ← más regularización
    "patience":             12,
    "clip":                 1.0,
    "seed":                 42,

    # CLAVE: ResNet18 nunca se descongela
    "freeze_encoder_epochs": 999,

    "balanced_sampler": True,
    "use_scheduler":    True,
    "device":           "auto",
    "min_frames":       5,
}


def load_and_adapt_csv(csv_path: Path, npy_root: Path, min_frames: int = 1) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = {"persona", "emocion", "archivo"} - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")
    df["feature_path"] = df["archivo"].apply(
        lambda rel: str((npy_root / Path(rel.replace("\\", "/"))).resolve())
    )
    if "num_frames" in df.columns and min_frames > 1:
        antes = len(df)
        df = df[df["num_frames"] >= min_frames].reset_index(drop=True)
        if (d := antes - len(df)) > 0:
            print(f"[INFO] {d} secuencias descartadas por < {min_frames} frames.")
    existe = df["feature_path"].apply(lambda p: Path(p).exists())
    if (f := (~existe).sum()) > 0:
        print(f"[WARN] {f} archivos .npy no encontrados. Se omitirán.")
        df = df[existe].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No quedaron secuencias válidas.")
    return df[["persona", "emocion", "feature_path"]].copy()


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "═" * 55)
    print("  DATASET")
    print("═" * 55)
    print(f"  Total: {len(df)}  |  Personas: {df['persona'].nunique()}  |  Clases: {df['emocion'].nunique()}")
    dist = df["emocion"].value_counts()
    for em, c in dist.items():
        bar = "█" * max(1, c * 20 // dist.max())
        print(f"    {em:<20} {c:>4}  {bar}")
    print("═" * 55 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",        type=Path, default=DEFAULTS["csv"])
    p.add_argument("--output_dir", type=Path, default=DEFAULTS["output_dir"])
    p.add_argument("--frame_embedding_dim", type=int,   default=DEFAULTS["frame_embedding_dim"])
    p.add_argument("--hidden_dim",          type=int,   default=DEFAULTS["hidden_dim"])
    p.add_argument("--num_layers",          type=int,   default=DEFAULTS["num_layers"])
    p.add_argument("--dropout",             type=float, default=DEFAULTS["dropout"])
    p.add_argument("--epochs",              type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch_size",          type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",                  type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay",        type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--patience",            type=int,   default=DEFAULTS["patience"])
    p.add_argument("--seed",                type=int,   default=DEFAULTS["seed"])
    p.add_argument("--device",              type=str,   default=DEFAULTS["device"])
    p.add_argument("--min_frames",          type=int,   default=DEFAULTS["min_frames"])
    p.add_argument("--freeze_encoder_epochs", type=int, default=DEFAULTS["freeze_encoder_epochs"],
                   help="999 = nunca descongelar (recomendado con dataset pequeño)")
    p.add_argument("--no_balanced_sampler", dest="balanced_sampler",
                   action="store_false", default=DEFAULTS["balanced_sampler"])
    p.add_argument("--affectnet_weights_path", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    npy_root = csv_path.parent

    print(f"\n[INFO] Modelo  : EmotionFrameLSTM — ResNet18 CONGELADO + BiLSTM + Atención")
    print(f"[INFO] Salida  : {args.output_dir}")
    print(f"[INFO] Freeze  : {args.freeze_encoder_epochs} épocas (999 = permanente)")
    print(f"[INFO] hidden  : {args.hidden_dim}  dropout: {args.dropout}  lr: {args.lr}")

    df = load_and_adapt_csv(csv_path, npy_root, min_frames=args.min_frames)
    print_summary(df)

    meta = train_frame_lstm(
        df_index              = df,
        out_dir               = args.output_dir,
        epochs                = args.epochs,
        batch_size            = args.batch_size,
        lr                    = args.lr,
        weight_decay          = args.weight_decay,
        frame_embedding_dim   = args.frame_embedding_dim,
        hidden_dim            = args.hidden_dim,
        num_layers            = args.num_layers,
        dropout               = args.dropout,
        seed                  = args.seed,
        bidirectional         = True,
        balanced_sampler      = args.balanced_sampler,
        device                = args.device,
        patience              = args.patience,
        clip                  = 1.0,
        use_scheduler         = True,
        baseline_subtract     = True,
        baseline_frames       = 3,
        freeze_encoder_epochs = args.freeze_encoder_epochs,
        affectnet_weights_path= args.affectnet_weights_path,
    )

    history  = meta.get("history", {})
    val_accs = history.get("val_acc", [])
    best_acc = max(val_accs) if val_accs else 0.0

    print("\n" + "═" * 55)
    print("  RESULTADO")
    print("═" * 55)
    print(f"  Mejor val_acc : {best_acc:.4f}  ({best_acc*100:.1f}%)")
    print(f"  Test acc      : {meta.get('test_acc', 0.0):.4f}")
    print(f"  Épocas        : {len(val_accs)}")
    print(f"  Checkpoint    : {meta.get('best_checkpoint')}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()