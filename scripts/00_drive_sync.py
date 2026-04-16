import argparse
import csv
from pathlib import Path
import re
import sys


DEFAULT_DATASET_FOLDER_URL = "https://drive.google.com/drive/folders/1HvrEL14eyqlYacr6MtqjSzCW2phhPf5t?usp=sharing"


def _require_gdown():
    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "No se encontro gdown. Instala con: pip install gdown"
        ) from exc
    return gdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga indices/dataset desde Google Drive y corrige rutas en CSV de secuencias"
    )
    parser.add_argument("--casme_csv_url", default=None, help="URL de Drive para sequences_casme2.csv")
    parser.add_argument("--smic_csv_url", default=None, help="URL de Drive para sequences_smic.csv")
    parser.add_argument("--dataset_file_url", default=None, help="URL de Drive para dataset (zip/tar/archivo)")
    parser.add_argument(
        "--dataset_folder_url",
        default=DEFAULT_DATASET_FOLDER_URL,
        help="URL de Drive para carpeta de dataset (por defecto: carpeta compartida CASME/SMIC/MEME-Uni)",
    )
    parser.add_argument("--outputs_index_dir", default="outputs_dataset_index", help="Carpeta de salida de CSV de secuencias")
    parser.add_argument("--dataset_out_dir", default="datasets", help="Carpeta donde descargar dataset")
    parser.add_argument(
        "--replace_from",
        default="auto",
        help="Prefijo original en paths_joined. Usa 'auto' para inferirlo desde el CSV",
    )
    parser.add_argument(
        "--replace_to",
        default=None,
        help="Prefijo nuevo en paths_joined. Si no se pasa, intenta usar subcarpeta por dataset y si no dataset_out_dir",
    )
    parser.add_argument("--skip_rewrite", action="store_true", help="No reescribir paths_joined")
    parser.add_argument("--force", action="store_true", help="Forzar descarga aunque exista archivo")
    return parser.parse_args()


def _download_file(gdown_module, url: str, output_path: Path, force: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"[SKIP] Ya existe: {output_path}")
        return output_path

    print(f"[DL] Descargando archivo: {url}")
    result = gdown_module.download(url=url, output=str(output_path), quiet=False, fuzzy=True)
    if not result:
        raise RuntimeError(f"Fallo descarga de archivo: {url}")

    resolved = Path(result)
    print(f"[OK] Archivo descargado: {resolved}")
    return resolved


def _download_folder(gdown_module, url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DL] Descargando carpeta: {url}")
    result = gdown_module.download_folder(url=url, output=str(output_dir), quiet=False, remaining_ok=True)
    if not result:
        raise RuntimeError(f"Fallo descarga de carpeta: {url}")

    print(f"[OK] Carpeta descargada en: {output_dir}")
    return output_dir


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _dataset_hint_from_csv_name(csv_path: Path) -> str | None:
    name = csv_path.stem.lower()
    if "casme" in name:
        return "casme"
    if "smic" in name:
        return "smic"
    if "meme" in name:
        return "meme"
    return None


def _find_dataset_subdir(dataset_out_dir: Path, hint: str | None) -> Path | None:
    if hint is None or not dataset_out_dir.exists():
        return None

    candidates = [p for p in dataset_out_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    hint_key = _normalize_name(hint)
    for c in candidates:
        key = _normalize_name(c.name)
        if hint_key in key or key in hint_key:
            return c.resolve()

    return None


def _infer_prefix_from_csv(csv_path: Path, max_rows: int = 200) -> str | None:
    if not csv_path.exists():
        return None

    pattern = re.compile(r"^[A-Za-z]:[\\/][^\\/]+")
    counts: dict[str, int] = {}
    seen = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joined = str(row.get("paths_joined", "") or "")
            if not joined:
                continue
            first = joined.split("|")[0].strip()
            if not first:
                continue
            match = pattern.match(first)
            if match:
                prefix = match.group(0)
                counts[prefix] = counts.get(prefix, 0) + 1
                seen += 1
            if seen >= max_rows:
                break

    if not counts:
        return None

    return max(counts.items(), key=lambda x: x[1])[0]


def _rewrite_paths_in_csv(csv_path: Path, from_prefix: str, to_prefix: str) -> None:
    if not csv_path.exists():
        print(f"[WARN] No existe CSV para reescritura: {csv_path}")
        return

    rows = []
    changed = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV sin encabezado: {csv_path}")
        fieldnames = reader.fieldnames

        for row in reader:
            original = row.get("paths_joined", "")
            if original and from_prefix in original:
                row["paths_joined"] = original.replace(from_prefix, to_prefix)
                changed += 1
            rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Reescrito {csv_path.name}: filas_modificadas={changed}")


def _resolve_replace_to(args: argparse.Namespace, csv_path: Path, dataset_out_dir: Path) -> str:
    if args.replace_to:
        return str(Path(args.replace_to).resolve())

    hint = _dataset_hint_from_csv_name(csv_path)
    subdir = _find_dataset_subdir(dataset_out_dir, hint)
    if subdir is not None:
        return str(subdir)

    return str(dataset_out_dir.resolve())


def main() -> None:
    args = parse_args()
    gdown = _require_gdown()

    outputs_index_dir = Path(args.outputs_index_dir)
    dataset_out_dir = Path(args.dataset_out_dir)

    outputs_index_dir.mkdir(parents=True, exist_ok=True)
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    casme_csv_path = outputs_index_dir / "sequences_casme2.csv"
    smic_csv_path = outputs_index_dir / "sequences_smic.csv"

    if args.casme_csv_url:
        _download_file(gdown, args.casme_csv_url, casme_csv_path, args.force)
    else:
        print("[INFO] --casme_csv_url no especificado")

    if args.smic_csv_url:
        _download_file(gdown, args.smic_csv_url, smic_csv_path, args.force)
    else:
        print("[INFO] --smic_csv_url no especificado")

    if args.dataset_file_url:
        target_file = dataset_out_dir / "dataset_from_drive"
        _download_file(gdown, args.dataset_file_url, target_file, args.force)
    elif args.dataset_folder_url:
        _download_folder(gdown, args.dataset_folder_url, dataset_out_dir)
    else:
        print("[INFO] No se especifico URL de dataset (archivo o carpeta)")

    if args.skip_rewrite:
        print("[INFO] Reescritura de paths_joined omitida por --skip_rewrite")
        return

    for csv_path in [casme_csv_path, smic_csv_path]:
        from_prefix = args.replace_from
        if from_prefix.lower() == "auto":
            inferred = _infer_prefix_from_csv(csv_path)
            if inferred is None:
                print(f"[WARN] No se pudo inferir prefijo en {csv_path.name}; se omite reescritura")
                continue
            from_prefix = inferred
            print(f"[INFO] Prefijo inferido para {csv_path.name}: {from_prefix}")

        to_prefix = _resolve_replace_to(args, csv_path, dataset_out_dir)
        print(f"[INFO] Reescribiendo {csv_path.name}: '{from_prefix}' -> '{to_prefix}'")
        _rewrite_paths_in_csv(csv_path, from_prefix, to_prefix)

    print("[DONE] Sincronizacion con Drive completada")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
