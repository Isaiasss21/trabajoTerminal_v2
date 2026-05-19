"""
extract_sequences.py
--------------------
Extracción en lote de características de flujo óptico del dataset CASME II.

Estructura de entrada esperada:
  <dataset_root>/
    <sujeto>/
      <emocion>/
        <secuencia>/
          img001.jpg   (o .png / .bmp)
          img002.jpg
          ...

Estructura de salida generada:
  <output_dir>/
    <sujeto>/
      <emocion>/
        <secuencia>.npy   -- tensor (N, 64, 64, 3) float32 [dx, dy, mag]
  <output_dir>/extraction_index.csv
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np


# ── Parámetros del pipeline (idénticos a gui_app.py) ──────────────────────────
FACE_SIZE = (64, 64)
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5, "levels": 3, "winsize": 15,
    "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
}
REGIONS = {
    "left_eyebrow":  ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46],                    1.0),
    "right_eyebrow": ([300, 293, 334, 296, 336, 285, 295, 282, 283, 276],             1.0),
    "mouth":         ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                       291, 409, 270, 269, 267, 0, 37, 39, 40],                       0.9),
    "left_eye":      ([33, 160, 158, 133, 153, 144],                                  0.25),
    "right_eye":     ([362, 385, 387, 263, 373, 380],                                 0.25),
    "nose":          ([8, 193, 114, 129, 219, 2, 439, 358, 343, 417],                 0.85),
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_CSV_NAME = "extraction_index.csv"
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# ── Paleta de colores (mismo tema que gui_app.py) ─────────────────────────────
_C_BG      = "#1a1a2e"
_C_SURFACE = "#16213e"
_C_BORDER  = "#0f3460"
_C_GREEN   = "#22c55e"
_C_GREEN_H = "#16a34a"
_C_RED     = "#ef4444"
_C_RED_H   = "#dc2626"
_C_AMBER   = "#f59e0b"
_C_TEXT    = "#e2e8f0"
_C_SUBTEXT = "#64748b"
_C_DISBLD  = "#374151"


# ── Funciones del pipeline (misma lógica que gui_app.py) ──────────────────────

def _init_landmarker(model_path: Path):
    """Carga el detector de landmarks de MediaPipe. Retorna None si el modelo no existe."""
    if not model_path.exists():
        return None
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _extract_face_and_landmarks(image: np.ndarray, face_landmarks) -> tuple:
    """Extrae la ROI en gris 64×64 y mapea los landmarks al espacio de la ROI."""
    h, w = image.shape[:2]
    px = np.array([[int(pt.x * w), int(pt.y * h)] for pt in face_landmarks])
    x_min, y_min = np.min(px, axis=0)
    x_max, y_max = np.max(px, axis=0)
    mx = int((x_max - x_min) * 0.1)
    my = int((y_max - y_min) * 0.1)
    x1 = max(0, x_min - mx);  y1 = max(0, y_min - my)
    x2 = min(w, x_max + mx);  y2 = min(h, y_max + my)
    if x2 <= x1 or y2 <= y1:
        return None, None, None
    roi_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, FACE_SIZE)
    sx = FACE_SIZE[0] / (x2 - x1)
    sy = FACE_SIZE[1] / (y2 - y1)
    mapped = [[(lx - x1) * sx, (ly - y1) * sy] for lx, ly in px]
    return roi_resized, np.array(mapped), (x1, y1, x2, y2)


def _create_roi_mask(mapped_landmarks: np.ndarray) -> np.ndarray:
    """Construye la máscara de pesos 64×64 basada en regiones faciales."""
    mask = np.zeros(FACE_SIZE, dtype=np.float32)
    for _name, (indices, weight) in REGIONS.items():
        pts = np.array([mapped_landmarks[i] for i in indices], dtype=np.int32)
        layer = np.zeros(FACE_SIZE, dtype=np.float32)
        cv2.fillConvexPoly(layer, cv2.convexHull(pts), 1.0)
        mask = np.maximum(mask, layer * weight)
    return cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5)


def _compute_flow_tensor(
    prev_gray: np.ndarray, curr_gray: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Calcula flujo óptico enmascarado.
    Devuelve solo el tensor CNN (64, 64, 3) float32 con canales [dx, dy, mag].
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)
    flow_f = flow * mask[..., np.newaxis]
    magnitude, _ = cv2.cartToPolar(flow_f[..., 0], flow_f[..., 1])
    dx  = cv2.normalize(flow_f[..., 0], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dy  = cv2.normalize(flow_f[..., 1], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mag = cv2.normalize(magnitude,       None,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.stack([dx, dy, mag], axis=-1)


# ── Funciones de escaneo y extracción ─────────────────────────────────────────

def _get_frame_paths(sequence_dir: Path) -> list[Path]:
    """Devuelve los frames de una secuencia ordenados por nombre."""
    return sorted(
        p for p in sequence_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _collect_sequences(dataset_root: Path) -> list[tuple[Path, str, str, str]]:
    """
    Escanea el dataset y retorna lista de (sequence_dir, subject, emotion, seq_name).
    Estructura esperada: <root>/<sujeto>/<emocion>/<secuencia>/
    """
    sequences: list[tuple[Path, str, str, str]] = []
    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for emotion_dir in sorted(subject_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue
            for seq_dir in sorted(emotion_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                sequences.append(
                    (seq_dir, subject_dir.name, emotion_dir.name, seq_dir.name)
                )
    return sequences


def extract_sequence(sequence_dir: Path, detector) -> np.ndarray | None:
    """
    Procesa una secuencia de imágenes con el pipeline completo de landmarks + flujo.
    Devuelve tensor (N, 64, 64, 3) float32, o None si no se generaron frames válidos.
    """
    frame_paths = _get_frame_paths(sequence_dir)
    if not frame_paths:
        return None

    tensors: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None

    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)

        if not result.face_landmarks:
            prev_gray = None
            continue

        curr_gray, mapped_lms, _ = _extract_face_and_landmarks(
            frame, result.face_landmarks[0]
        )
        if curr_gray is None:
            prev_gray = None
            continue

        if prev_gray is not None:
            mask = _create_roi_mask(mapped_lms)
            flow_tensor = _compute_flow_tensor(prev_gray, curr_gray, mask)
            tensors.append(flow_tensor)

        prev_gray = curr_gray

    if not tensors:
        return None

    return np.stack(tensors, axis=0).astype(np.float32)


def _append_csv_row(csv_path: Path, row: dict) -> None:
    """Agrega una fila al CSV índice; crea la cabecera si el archivo es nuevo."""
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ── GUI ────────────────────────────────────────────────────────────────────────

class ExtractionApp:
    """Interfaz gráfica para la extracción en lote del dataset CASME II."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Extracción Batch — CASME II")
        self.root.geometry("780x660")
        self.root.configure(bg=_C_BG)
        self.root.resizable(True, True)
        self.root.minsize(680, 560)

        self._stop_flag = threading.Event()
        self._worker_thread: threading.Thread | None = None

        self._build_ui()

    # ── Construcción de la UI ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Green.Horizontal.TProgressbar",
            troughcolor=_C_SURFACE,
            background=_C_GREEN,
            bordercolor=_C_BORDER,
            lightcolor=_C_GREEN,
            darkcolor=_C_GREEN_H,
        )

        container = tk.Frame(self.root, bg=_C_BG, padx=20, pady=16)
        container.pack(fill="both", expand=True)

        # ── Título ────────────────────────────────────────────────────────────
        tk.Label(
            container,
            text="Extracción Batch de Flujo Óptico — CASME II",
            bg=_C_BG, fg=_C_TEXT, font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 14))

        # ── Rutas ─────────────────────────────────────────────────────────────
        self._dataset_var = tk.StringVar()
        self._output_var  = tk.StringVar()
        self._model_var   = tk.StringVar(value=str(_MODEL_PATH))

        paths_frame = tk.Frame(container, bg=_C_SURFACE, padx=12, pady=10)
        paths_frame.pack(fill="x", pady=(0, 12))

        for label_text, var, browse_cmd in [
            ("Dataset raíz:",        self._dataset_var, self._browse_dataset),
            ("Directorio de salida:", self._output_var,  self._browse_output),
            ("Modelo MediaPipe:",    self._model_var,   self._browse_model),
        ]:
            row = tk.Frame(paths_frame, bg=_C_SURFACE)
            row.pack(fill="x", pady=4)
            tk.Label(
                row, text=label_text,
                bg=_C_SURFACE, fg=_C_SUBTEXT, font=("Segoe UI", 9), width=22, anchor="w",
            ).pack(side="left")
            tk.Entry(
                row, textvariable=var,
                bg="#0f172a", fg=_C_TEXT, insertbackground=_C_TEXT,
                relief="flat", font=("Segoe UI", 9), width=46,
            ).pack(side="left", padx=(4, 6))
            tk.Button(
                row, text="...", command=browse_cmd,
                bg=_C_BORDER, fg=_C_TEXT, relief="flat",
                padx=8, pady=2, font=("Segoe UI", 9), cursor="hand2",
            ).pack(side="left")

        # ── Botones de control ────────────────────────────────────────────────
        btn_frame = tk.Frame(container, bg=_C_BG)
        btn_frame.pack(fill="x", pady=(0, 12))

        self.start_btn = tk.Button(
            btn_frame, text="▶  Iniciar Extracción",
            bg=_C_GREEN, fg="white",
            activebackground=_C_GREEN_H, activeforeground="white",
            font=("Segoe UI", 10, "bold"), relief="flat",
            padx=18, pady=8, cursor="hand2",
            command=self._start_extraction,
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(
            btn_frame, text="■  Detener",
            bg=_C_DISBLD, fg=_C_SUBTEXT,
            activebackground=_C_RED_H, activeforeground="white",
            font=("Segoe UI", 10, "bold"), relief="flat",
            padx=18, pady=8, cursor="hand2", state="disabled",
            command=self._stop_extraction,
        )
        self.stop_btn.pack(side="left")

        # ── Progreso ──────────────────────────────────────────────────────────
        prog_frame = tk.Frame(container, bg=_C_SURFACE, padx=12, pady=10)
        prog_frame.pack(fill="x", pady=(0, 8))

        self._progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            prog_frame,
            variable=self._progress_var,
            maximum=100,
            mode="determinate",
            style="Green.Horizontal.TProgressbar",
        ).pack(fill="x", pady=(0, 8))

        stats_row = tk.Frame(prog_frame, bg=_C_SURFACE)
        stats_row.pack(fill="x")

        self._pct_label = tk.Label(
            stats_row, text="0%",
            bg=_C_SURFACE, fg=_C_TEXT, font=("Segoe UI", 10, "bold"),
        )
        self._pct_label.pack(side="left")

        self._stats_label = tk.Label(
            stats_row, text="Total: 0  |  Procesadas: 0  |  ✓ 0  |  ✗ 0",
            bg=_C_SURFACE, fg=_C_SUBTEXT, font=("Segoe UI", 9),
        )
        self._stats_label.pack(side="right")

        # ── Log ───────────────────────────────────────────────────────────────
        log_outer = tk.Frame(container, bg=_C_BORDER, padx=1, pady=1)
        log_outer.pack(fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_outer,
            bg="#0a0f1e", fg=_C_TEXT,
            insertbackground=_C_TEXT, relief="flat",
            font=("Consolas", 9), state="disabled", wrap="word",
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("ok",   foreground=_C_GREEN)
        self.log_text.tag_config("fail", foreground=_C_RED)
        self.log_text.tag_config("info", foreground=_C_SUBTEXT)
        self.log_text.tag_config("warn", foreground=_C_AMBER)
        self.log_text.tag_config("done", foreground=_C_TEXT)

    # ── Diálogos de selección de ruta ─────────────────────────────────────────

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory(title="Selecciona la carpeta raíz del dataset CASME II")
        if path:
            self._dataset_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Selecciona el directorio de salida para los .npy")
        if path:
            self._output_var.set(path)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecciona el modelo face_landmarker.task",
            filetypes=[("MediaPipe Task", "*.task"), ("Todos los archivos", "*.*")],
        )
        if path:
            self._model_var.set(path)

    # ── Helpers de log y progreso ──────────────────────────────────────────────

    def _log(self, message: str, tag: str = "info") -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_progress(
        self, total: int, processed: int, success: int, failed: int
    ) -> None:
        pct = (processed / total * 100) if total > 0 else 0.0
        self._progress_var.set(pct)
        self._pct_label.configure(text=f"{pct:.1f}%")
        self._stats_label.configure(
            text=f"Total: {total}  |  Procesadas: {processed}  |  ✓ {success}  |  ✗ {failed}"
        )

    # ── Control de extracción ──────────────────────────────────────────────────

    def _start_extraction(self) -> None:
        dataset_root = Path(self._dataset_var.get().strip())
        output_dir   = Path(self._output_var.get().strip())
        model_path   = Path(self._model_var.get().strip())

        if not self._dataset_var.get().strip() or not dataset_root.is_dir():
            self._log("✗ El directorio del dataset no existe o no fue especificado.", "fail")
            return
        if not self._output_var.get().strip():
            self._log("✗ Especifica un directorio de salida.", "fail")
            return
        if not model_path.exists():
            self._log(f"✗ Modelo no encontrado: {model_path}", "fail")
            return

        # Limpiar estado previo
        self._stop_flag.clear()
        self._progress_var.set(0.0)
        self._pct_label.configure(text="0%")
        self._stats_label.configure(
            text="Total: 0  |  Procesadas: 0  |  ✓ 0  |  ✗ 0"
        )
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self.start_btn.configure(state="disabled", bg=_C_DISBLD, fg=_C_SUBTEXT)
        self.stop_btn.configure(state="normal", bg=_C_RED, fg="white")

        self._worker_thread = threading.Thread(
            target=self._extraction_worker,
            args=(dataset_root, output_dir, model_path),
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_extraction(self) -> None:
        self._stop_flag.set()
        self.stop_btn.configure(state="disabled", bg=_C_DISBLD, fg=_C_SUBTEXT)
        self.root.after(0, self._log, "⏹  Deteniendo — esperando el frame actual...", "warn")

    # ── Worker de extracción (hilo de fondo) ───────────────────────────────────

    def _extraction_worker(
        self, dataset_root: Path, output_dir: Path, model_path: Path
    ) -> None:
        self.root.after(0, self._log, f"Escaneando dataset en: {dataset_root}", "info")

        sequences = _collect_sequences(dataset_root)
        total = len(sequences)

        if total == 0:
            self.root.after(
                0, self._log,
                "✗ No se encontraron secuencias. Verifica que la estructura sea:\n"
                "  <root>/<sujeto>/<emocion>/<secuencia>/<frames>",
                "fail",
            )
            self.root.after(0, self._finish_worker)
            return

        self.root.after(0, self._log, f"Secuencias encontradas: {total}", "info")
        self.root.after(0, self._log, "Cargando modelo MediaPipe...", "info")

        detector = _init_landmarker(model_path)
        if detector is None:
            self.root.after(
                0, self._log, "✗ No se pudo cargar el modelo de landmarks.", "fail"
            )
            self.root.after(0, self._finish_worker)
            return

        self.root.after(0, self._log, "Modelo cargado. Iniciando extracción...\n", "info")

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / OUTPUT_CSV_NAME

        success = 0
        failed  = 0

        for idx, (seq_dir, subject, emotion, seq_name) in enumerate(sequences, start=1):
            if self._stop_flag.is_set():
                self.root.after(
                    0, self._log,
                    f"\n⏹  Extracción detenida por el usuario en {idx - 1}/{total}.",
                    "warn",
                )
                break

            try:
                volume = extract_sequence(seq_dir, detector)

                if volume is None or volume.shape[0] == 0:
                    raise ValueError("No se generaron tensores válidos.")

                # Guardar espejando la estructura del dataset original
                out_dir = output_dir / subject / emotion
                out_dir.mkdir(parents=True, exist_ok=True)
                npy_path = out_dir / f"{seq_name}.npy"
                np.save(str(npy_path), volume)

                n, h, w, c = volume.shape
                _append_csv_row(csv_path, {
                    "sujeto":     subject,
                    "emocion":    emotion,
                    "secuencia":  seq_name,
                    "archivo":    str(npy_path.relative_to(output_dir)),
                    "num_frames": n,
                    "alto":       h,
                    "ancho":      w,
                    "canales":    c,
                    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                success += 1
                msg = f"[{idx:>4}/{total}] ✓  {subject}/{emotion}/{seq_name}  ({n} frames)"
                self.root.after(0, self._log, msg, "ok")

            except Exception as exc:
                failed += 1
                msg = f"[{idx:>4}/{total}] ✗  {subject}/{emotion}/{seq_name}  — {exc}"
                self.root.after(0, self._log, msg, "fail")

            self.root.after(
                0, self._update_progress, total, idx, success, failed
            )

        if hasattr(detector, "close"):
            detector.close()

        summary = (
            f"\n{'─' * 50}\n"
            f"Extracción finalizada.\n"
            f"  ✓ Exitosas:  {success}\n"
            f"  ✗ Fallidas:  {failed}\n"
            f"  Total:       {total}\n"
            f"  CSV índice:  {output_dir / OUTPUT_CSV_NAME}"
        )
        self.root.after(0, self._log, summary, "done")
        self.root.after(0, self._finish_worker)

    def _finish_worker(self) -> None:
        """Restablece los botones al terminar o detener el worker."""
        self.start_btn.configure(state="normal", bg=_C_GREEN, fg="white")
        self.stop_btn.configure(state="disabled", bg=_C_DISBLD, fg=_C_SUBTEXT)


def main() -> None:
    root = tk.Tk()
    ExtractionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
