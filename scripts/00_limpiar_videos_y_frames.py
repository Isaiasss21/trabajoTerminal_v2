"""
00_limpiar_videos_y_frames.py
------------------------------
Limpia un árbol de carpetas de grabaciones:
  1. Borra todos los archivos de video (.mp4, .avi, .mov, .mkv, .wmv, .mpeg, .mpg).
  2. En cada subcarpeta de imágenes, conserva solo 1 de cada N frames
     (por defecto N=5), borrando el resto.

Uso (dry-run, solo muestra lo que haría):
    python scripts/00_limpiar_videos_y_frames.py --root D:\datasetVideosTT\03_grabaciones\p008

Uso (aplicar cambios reales):
    python scripts/00_limpiar_videos_y_frames.py --root D:\datasetVideosTT\03_grabaciones\p008 --apply

Opciones:
    --root        Carpeta raíz a procesar (obligatorio).
    --keep-every  Conservar 1 de cada N imágenes (default: 5).
    --apply       Sin esta bandera el script solo simula (dry-run).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

VIDEO_EXTENSIONS  = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp"}


def _natural_key(path: Path) -> list:
    """Ordena rutas de forma natural (img2 < img10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def scan_tree(root: Path) -> tuple[list[Path], dict[Path, list[Path]]]:
    """
    Recorre el árbol una sola vez.
    Devuelve:
      videos       — lista de archivos de video encontrados
      img_by_dir   — dict {carpeta: [imágenes ordenadas naturalmente]}
    """
    videos: list[Path] = []
    img_by_dir: dict[Path, list[Path]] = {}

    for dirpath, _dirs, files in os.walk(root):
        folder = Path(dirpath)
        imgs: list[Path] = []
        for name in files:
            p = folder / name
            ext = p.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                videos.append(p)
            elif ext in IMAGE_EXTENSIONS:
                imgs.append(p)
        if imgs:
            img_by_dir[folder] = sorted(imgs, key=_natural_key)

    return videos, img_by_dir


def select_images_to_delete(images: list[Path], keep_every: int) -> list[Path]:
    """Devuelve los índices a BORRAR (conserva índices 0, N, 2N, ...)."""
    return [p for i, p in enumerate(images) if i % keep_every != 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Borra videos y adelgaza frames (conserva 1 de cada N).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root",       required=True, type=Path,
                        help="Carpeta raíz a procesar.")
    parser.add_argument("--keep-every", default=5, type=int,
                        help="Conservar 1 de cada N imágenes.")
    parser.add_argument("--apply",      action="store_true",
                        help="Aplicar borrados reales (sin esta bandera: dry-run).")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    keep_every: int = args.keep_every
    apply: bool = args.apply

    if not root.is_dir():
        print(f"[ERROR] No existe el directorio: {root}")
        return

    mode = "APLICANDO" if apply else "DRY-RUN"
    print(f"[{mode}] Raíz: {root}  |  keep_every={keep_every}")

    videos, img_by_dir = scan_tree(root)

    # ── Videos ────────────────────────────────────────────────────────────────
    print(f"\nVideos encontrados: {len(videos)}")
    for vp in videos:
        print(f"  DEL video  {vp}")
        if apply:
            vp.unlink()

    # ── Imágenes ───────────────────────────────────────────────────────────────
    total_imgs   = sum(len(v) for v in img_by_dir.values())
    to_delete    = []
    for imgs in img_by_dir.values():
        to_delete.extend(select_images_to_delete(imgs, keep_every))

    kept = total_imgs - len(to_delete)
    print(f"\nImágenes totales : {total_imgs}")
    print(f"Se conservarán   : {kept}")
    print(f"Se borrarán      : {len(to_delete)}")

    if apply:
        for p in to_delete:
            p.unlink()
        print("\n[OK] Borrado completado.")
    else:
        print("\n[DRY-RUN] Nada fue borrado. Usa --apply para aplicar.")


if __name__ == "__main__":
    main()
