"""
gui_app.py
----------
Aplicación de escritorio para capturar y guardar secuencias de frames.

Características principales:
  - Detección de cámaras disponibles
  - Vista previa en vivo
  - Inicio y fin de secuencias desde la GUI
  - Guardado de frames en carpetas con timestamp

La app está pensada como paso previo a futuros módulos CNN + LSTM.
"""

from __future__ import annotations

import threading
import csv
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
import numpy as np


SEQUENCES_DIR = Path("sequences")
SEQUENCE_CSV_NAME = "sequence_info.csv"
PREVIEW_SIZE    = (800, 600)
SIDE_PANEL_SIZE = (204, 204)  # Tamaño de los paneles de flujo y máscara
MAX_CAMERA_INDEX = 5

# ─── Paleta de colores (tema oscuro moderno) ─────────────────────────────────
_C_BG      = "#1a1a2e"   # Fondo principal
_C_SURFACE = "#16213e"   # Superficie de paneles
_C_BORDER  = "#0f3460"   # Borde del recuadro de cámara
_C_GREEN   = "#22c55e"   # Iniciar / éxito
_C_GREEN_H = "#16a34a"   # Verde hover
_C_RED     = "#ef4444"   # Detener / error
_C_RED_H   = "#dc2626"   # Rojo hover
_C_AMBER   = "#f59e0b"   # Grabando / advertencia
_C_TEXT    = "#e2e8f0"   # Texto primario
_C_SUBTEXT = "#64748b"   # Texto secundario
_C_DISBLD  = "#374151"   # Botón desactivado

# ─── Configuración del pipeline de landmarks ─────────────────────────────────
FACE_SIZE = (64, 64)
SEQUENCE_LENGTH = 15
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5, "levels": 3, "winsize": 15,
    "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
}
REGIONS = {
    "left_eyebrow":  ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  1.0),
    "right_eyebrow": ([300, 293, 334, 296, 336, 285, 295, 282, 283, 276], 1.0),
    "mouth":         ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40], 0.9),
    "left_eye":      ([33, 160, 158, 133, 153, 144], 0.25),
    "right_eye":     ([362, 385, 387, 263, 373, 380], 0.25),
    "nose":          ([8, 193, 114, 129, 219, 2, 439, 358, 343, 417], 0.85),
}
# Ruta al modelo de MediaPipe (misma convención que real_time_face_detection.py)
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def _init_landmarker():
    """Carga el detector de landmarks de MediaPipe. Retorna None si el modelo no existe."""
    if not _MODEL_PATH.exists():
        print(f"[ADVERTENCIA] Modelo no encontrado: {_MODEL_PATH}")
        print("  Descárgalo desde: https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        return None
    base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
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


def _compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray,
                  mask: np.ndarray) -> tuple:
    """Calcula flujo óptico enmascarado. Devuelve (tensor CNN, imagen HSV visual)."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)
    flow_f = flow * mask[..., np.newaxis]
    magnitude, angle = cv2.cartToPolar(flow_f[..., 0], flow_f[..., 1])
    dx  = cv2.normalize(flow_f[..., 0], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dy  = cv2.normalize(flow_f[..., 1], None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mag = cv2.normalize(magnitude,      None,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    hsv = np.zeros((*FACE_SIZE, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.stack([dx, dy, mag], axis=-1), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def get_available_cameras(max_index: int = MAX_CAMERA_INDEX) -> list[int]:
    """Detecta cámaras disponibles probando índices consecutivos."""
    available: list[int] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _frame = cap.read()
            if ok:
                available.append(index)
        cap.release()
    return available


def _append_to_csv(csv_path: Path, row: dict) -> None:
    """Agrega una fila al CSV índice; crea el archivo con cabecera si no existe."""
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _sanitize_sequence_name(name: str) -> str:
    """Normaliza el nombre de la secuencia para usarlo como carpeta."""
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name.strip())
    return cleaned.strip("_")


def save_sequence(
    frames: list[np.ndarray],
    sequence_name: str,
    subject_name: str = "default",
    base_dir: Path = SEQUENCES_DIR,
) -> Path:
    """
    Guarda la secuencia de tensores de flujo óptico dentro de una carpeta propia:
      - archivo .npy con forma (N, 64, 64, 3) dtype float32  [dx, dy, mag]
      - CSV local con metadatos de la secuencia
    Devuelve la ruta de la carpeta generada.
    """
    sanitized_name = _sanitize_sequence_name(sequence_name)
    sanitized_subject = _sanitize_sequence_name(subject_name) or "default"
    if not sanitized_name:
        raise ValueError("El nombre de la secuencia no es válido.")

    base_dir.mkdir(parents=True, exist_ok=True)

    # Crea una carpeta por secuencia; si ya existe, añade timestamp
    sequence_dir = base_dir / sanitized_name
    if sequence_dir.exists():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sequence_dir = base_dir / f"{sanitized_name}_{timestamp}"

    sequence_dir.mkdir(parents=True, exist_ok=False)

    npy_path = sequence_dir / f"{sanitized_subject}_{sanitized_name}.npy"
    csv_path = sequence_dir / SEQUENCE_CSV_NAME

    # Apila tensores de flujo en (N, 64, 64, 3) float32 y guarda
    volume = np.stack(frames, axis=0).astype(np.float32)  # (N, H, W, 3) [dx,dy,mag]
    np.save(str(npy_path), volume)

    # Registra metadatos en el CSV local de la secuencia
    n, h, w, c = volume.shape
    _append_to_csv(
        csv_path,
        {
            "sujeto":     subject_name or "default",
            "nombre":     sequence_name,
            "archivo":    npy_path.name,
            "tipo":       "flujo_filtrado_float32",
            "num_frames": n,
            "alto":       h,
            "ancho":      w,
            "canales":    c,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    return sequence_dir


class SequenceCaptureApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Captura de secuencias")
        self.root.geometry("980x760")
        self.root.configure(bg=_C_BG)

        self.available_cameras = get_available_cameras()
        self.selected_camera = tk.IntVar(value=self.available_cameras[0] if self.available_cameras else -1)
        self.status_label: tk.Label | None = None

        self.cap: cv2.VideoCapture | None = None
        self.current_frame: np.ndarray | None = None
        self.preview_image: tk.PhotoImage | None = None
        self.recording = False
        self.saving = False
        self.frames_buffer: list[np.ndarray] = []      # frames crudos (ya no se guardan)
        self.flow_tensors_buffer: list[np.ndarray] = []  # tensores de flujo filtrado (64,64,3) float32
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.preview_job: str | None = None
        self.flow_image: tk.PhotoImage | None = None
        self.mask_image: tk.PhotoImage | None = None
        self.flow_label: tk.Label | None = None
        self.mask_label: tk.Label | None = None

        # Estado del pipeline de landmarks / flujo óptico
        self.mp_detector = _init_landmarker()
        self.prev_gray_face: np.ndarray | None = None
        self.flow_buffer: deque = deque(maxlen=SEQUENCE_LENGTH)

        self._build_ui()
        self._open_selected_camera()
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        """Construye la interfaz con tema oscuro moderno."""
        # ── Estilos ttk (solo para Combobox que no admite tk.Widget fácilmente) ──
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
            fieldbackground=_C_SURFACE, background=_C_BORDER,
            foreground=_C_TEXT, arrowcolor=_C_TEXT,
            selectforeground=_C_TEXT, selectbackground=_C_SURFACE,
            padding=(6, 4),
        )
        style.map("TCombobox",
            fieldbackground=[("readonly", _C_SURFACE), ("disabled", _C_DISBLD)],
            selectbackground=[("readonly", _C_SURFACE)],
            selectforeground=[("readonly", _C_TEXT)],
            foreground=[("disabled", _C_SUBTEXT)],
        )

        # ── Contenedor principal ──────────────────────────────────────────────
        container = tk.Frame(self.root, bg=_C_BG, padx=16, pady=14)
        container.pack(fill="both", expand=True)

        # ── Cabecera: título + selector de cámara ─────────────────────────────
        header = tk.Frame(container, bg=_C_BG)
        header.pack(fill="x", pady=(0, 12))

        tk.Label(
            header, text="Captura de Secuencias",
            bg=_C_BG, fg=_C_TEXT, font=("Segoe UI", 14, "bold"),
        ).pack(side="left")

        cam_frame = tk.Frame(header, bg=_C_BG)
        cam_frame.pack(side="right")
        tk.Label(
            cam_frame, text="Cámara:",
            bg=_C_BG, fg=_C_SUBTEXT, font=("Segoe UI", 10),
        ).pack(side="left", padx=(0, 6))

        camera_values = self.available_cameras if self.available_cameras else ["Sin cámaras"]
        self.camera_combo = ttk.Combobox(
            cam_frame, values=camera_values, width=10, state="readonly",
        )
        if self.available_cameras:
            self.camera_combo.set(str(self.selected_camera.get()))
        else:
            self.camera_combo.set("Sin cámaras")
            self.camera_combo.state(["disabled"])
        self.camera_combo.pack(side="left")
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_changed)

        # ── Botones de control ────────────────────────────────────────────────
        controls = tk.Frame(container, bg=_C_BG)
        controls.pack(fill="x", pady=(0, 10))

        self.start_button = tk.Button(
            controls, text="▶  Iniciar Captura",
            bg=_C_GREEN, fg="white",
            activebackground=_C_GREEN_H, activeforeground="white",
            disabledforeground=_C_SUBTEXT,
            font=("Segoe UI", 10, "bold"), relief="flat",
            padx=18, pady=8, cursor="hand2",
            command=self.start_capture,
        )
        self.start_button.pack(side="left", padx=(0, 10))

        self.stop_button = tk.Button(
            controls, text="■  Detener Captura",
            bg=_C_DISBLD, fg=_C_SUBTEXT,
            activebackground=_C_RED_H, activeforeground="white",
            disabledforeground=_C_SUBTEXT,
            font=("Segoe UI", 10, "bold"), relief="flat",
            padx=18, pady=8, cursor="hand2",
            state="disabled",
            command=self.stop_capture,
        )
        self.stop_button.pack(side="left")

        # ── Área de contenido: cámara (izq) + paneles laterales (der) ──────────
        content = tk.Frame(container, bg=_C_BG)
        content.pack(fill="both", expand=True)

        # Webcam (izquierda, expandible)
        preview_outer = tk.Frame(content, bg=_C_BORDER, padx=2, pady=2)
        preview_outer.pack(side="left", fill="both", expand=True)
        preview_inner = tk.Frame(preview_outer, bg=_C_SURFACE)
        preview_inner.pack(fill="both", expand=True)
        self.preview_label = tk.Label(
            preview_inner, bg=_C_SURFACE, fg=_C_SUBTEXT,
            text="Inicializando cámara...", anchor="center",
            font=("Segoe UI", 11),
        )
        self.preview_label.pack(fill="both", expand=True)

        # Paneles laterales (derecha, ancho fijo)
        sidebar = tk.Frame(content, bg=_C_BG, width=220)
        sidebar.pack(side="left", fill="y", padx=(8, 0))
        sidebar.pack_propagate(False)

        flow_panel = tk.Frame(sidebar, bg=_C_SURFACE)
        flow_panel.pack(fill="x", pady=(0, 8))
        tk.Label(
            flow_panel, text="Flujo Filtrado",
            bg=_C_SURFACE, fg=_C_TEXT, font=("Segoe UI", 9, "bold"),
        ).pack(pady=(6, 2))
        self.flow_label = tk.Label(
            flow_panel, bg="#000000",
            width=SIDE_PANEL_SIZE[0], height=SIDE_PANEL_SIZE[1],
        )
        self.flow_label.pack(padx=6, pady=(0, 6))

        mask_panel = tk.Frame(sidebar, bg=_C_SURFACE)
        mask_panel.pack(fill="x")
        tk.Label(
            mask_panel, text="Máscara de Pesos",
            bg=_C_SURFACE, fg=_C_TEXT, font=("Segoe UI", 9, "bold"),
        ).pack(pady=(6, 2))
        self.mask_label = tk.Label(
            mask_panel, bg="#000000",
            width=SIDE_PANEL_SIZE[0], height=SIDE_PANEL_SIZE[1],
        )
        self.mask_label.pack(padx=6, pady=(0, 6))

        # ── Barra de estado DEBAJO del preview ────────────────────────────────
        status_bar = tk.Frame(container, bg=_C_SURFACE, height=36)
        status_bar.pack(fill="x", pady=(4, 0))
        status_bar.pack_propagate(False)

        self.status_label = tk.Label(
            status_bar, bg=_C_SURFACE, fg=_C_SUBTEXT,
            text="", font=("Segoe UI", 10), anchor="w", padx=12,
        )
        self.status_label.pack(fill="both", expand=True)

    def _set_status(self, message: str, color: str = _C_SUBTEXT) -> None:
        """Actualiza el texto y color de la barra de estado."""
        if self.status_label is not None:
            self.status_label.configure(text=message, fg=color)

    def on_camera_changed(self, _event=None) -> None:
        """Cambia la cámara activa desde el selector."""
        if self.recording or self.saving:
            return

        selected = self.camera_combo.get()
        if not selected.isdigit():
            return

        self.selected_camera.set(int(selected))
        self._open_selected_camera()

    def _open_selected_camera(self) -> None:
        """Abre la cámara seleccionada y reinicia el estado de preview."""
        self._release_camera()

        camera_index = self.selected_camera.get()
        if camera_index < 0:
            self._set_status("")
            self.preview_label.configure(text="No se detectaron cámaras.")
            return

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._set_status("")
            self.preview_label.configure(text=f"No se pudo abrir la cámara {camera_index}.")
            return

        self.cap = cap
        self._set_status("")
        # Reinicia el estado del pipeline al cambiar de cámara
        self.prev_gray_face = None
        self.flow_buffer.clear()

    def start_capture(self) -> None:
        """Inicia la grabación de una nueva secuencia."""
        if self.cap is None or not self.cap.isOpened() or self.recording or self.saving:
            return

        self.frames_buffer = []
        self.flow_tensors_buffer = []
        self.recording = True
        self._set_status("●  Grabando...", _C_AMBER)
        self.start_button.configure(state=tk.DISABLED, bg=_C_DISBLD, fg=_C_SUBTEXT)
        self.stop_button.configure(state=tk.NORMAL, bg=_C_RED, fg="white")
        self.camera_combo.state(["disabled"])

    def _center_dialog(self, dialog: tk.Toplevel) -> None:
        """Centra un diálogo con respecto a la ventana principal."""
        self.root.update_idletasks()
        dialog.update_idletasks()

        parent_x = self.root.winfo_rootx()
        parent_y = self.root.winfo_rooty()
        parent_w = self.root.winfo_width()
        parent_h = self.root.winfo_height()

        dlg_w = dialog.winfo_width()
        dlg_h = dialog.winfo_height()

        x = parent_x + max(0, (parent_w - dlg_w) // 2)
        y = parent_y + max(0, (parent_h - dlg_h) // 2)
        dialog.geometry(f"+{x}+{y}")

    def _ask_subject_name(self) -> str | None:
        """Muestra un diálogo para capturar el nombre del sujeto con opción default."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Nombre del sujeto")
        dialog.configure(bg=_C_SURFACE)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        container = tk.Frame(dialog, bg=_C_SURFACE, padx=14, pady=12)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Escribe el nombre del sujeto solamente",
            bg=_C_SURFACE,
            fg=_C_TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        subject_var = tk.StringVar(value="")
        use_default_var = tk.BooleanVar(value=False)
        result: dict[str, str | None] = {"value": None}

        entry = tk.Entry(
            container,
            textvariable=subject_var,
            bg="#0f172a",
            fg=_C_TEXT,
            insertbackground=_C_TEXT,
            relief="flat",
            font=("Segoe UI", 10),
            width=30,
            disabledbackground="#0f172a",
            disabledforeground=_C_SUBTEXT,
        )
        entry.pack(fill="x", pady=(0, 8))

        def _toggle_default() -> None:
            if use_default_var.get():
                entry.configure(state=tk.DISABLED)
            else:
                entry.configure(state=tk.NORMAL)
                entry.focus_set()

        def _confirm() -> None:
            if use_default_var.get():
                result["value"] = "default"
            else:
                result["value"] = subject_var.get().strip() or "default"
            dialog.destroy()

        def _cancel() -> None:
            result["value"] = None
            dialog.destroy()

        tk.Checkbutton(
            container,
            text="Usar default",
            variable=use_default_var,
            command=_toggle_default,
            bg=_C_SURFACE,
            fg=_C_SUBTEXT,
            activebackground=_C_SURFACE,
            activeforeground=_C_TEXT,
            selectcolor=_C_SURFACE,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", pady=(0, 10))

        actions = tk.Frame(container, bg=_C_SURFACE)
        actions.pack(fill="x")
        tk.Button(
            actions,
            text="Cancelar",
            command=_cancel,
            bg=_C_DISBLD,
            fg=_C_TEXT,
            relief="flat",
            padx=10,
            pady=6,
        ).pack(side="right", padx=(8, 0))
        tk.Button(
            actions,
            text="Aceptar",
            command=_confirm,
            bg=_C_GREEN,
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
        ).pack(side="right")

        _toggle_default()
        self._center_dialog(dialog)
        dialog.protocol("WM_DELETE_WINDOW", _cancel)
        self.root.wait_window(dialog)
        return result["value"]

    def _ask_sequence_name(self) -> str | None:
        """Muestra un diálogo para capturar el nombre de la secuencia con opción default."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Nombre de secuencia")
        dialog.configure(bg=_C_SURFACE)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        container = tk.Frame(dialog, bg=_C_SURFACE, padx=14, pady=12)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Escribe el nombre de la secuencia solamente",
            bg=_C_SURFACE,
            fg=_C_TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        sequence_var = tk.StringVar(value="")
        use_default_var = tk.BooleanVar(value=False)
        result: dict[str, str | None] = {"value": None}

        entry = tk.Entry(
            container,
            textvariable=sequence_var,
            bg="#0f172a",
            fg=_C_TEXT,
            insertbackground=_C_TEXT,
            relief="flat",
            font=("Segoe UI", 10),
            width=30,
            disabledbackground="#0f172a",
            disabledforeground=_C_SUBTEXT,
        )
        entry.pack(fill="x", pady=(0, 8))

        def _toggle_default() -> None:
            if use_default_var.get():
                entry.configure(state=tk.DISABLED)
            else:
                entry.configure(state=tk.NORMAL)
                entry.focus_set()

        def _confirm() -> None:
            if use_default_var.get():
                result["value"] = "default"
            else:
                result["value"] = sequence_var.get().strip() or "default"
            dialog.destroy()

        def _cancel() -> None:
            result["value"] = None
            dialog.destroy()

        tk.Checkbutton(
            container,
            text="Usar default",
            variable=use_default_var,
            command=_toggle_default,
            bg=_C_SURFACE,
            fg=_C_SUBTEXT,
            activebackground=_C_SURFACE,
            activeforeground=_C_TEXT,
            selectcolor=_C_SURFACE,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", pady=(0, 10))

        actions = tk.Frame(container, bg=_C_SURFACE)
        actions.pack(fill="x")
        tk.Button(
            actions,
            text="Cancelar",
            command=_cancel,
            bg=_C_DISBLD,
            fg=_C_TEXT,
            relief="flat",
            padx=10,
            pady=6,
        ).pack(side="right", padx=(8, 0))
        tk.Button(
            actions,
            text="Aceptar",
            command=_confirm,
            bg=_C_GREEN,
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
        ).pack(side="right")

        _toggle_default()
        self._center_dialog(dialog)
        dialog.protocol("WM_DELETE_WINDOW", _cancel)
        self.root.wait_window(dialog)
        return result["value"]

    def stop_capture(self) -> None:
        """Detiene la grabación y guarda la secuencia sin congelar la UI."""
        if not self.recording or self.saving:
            return

        sequence_name = self._ask_sequence_name()
        if sequence_name is None:
            return

        subject_name = self._ask_subject_name()
        if subject_name is None:
            return

        self.recording = False
        self.saving = True
        self._set_status("Guardando secuencia...", _C_AMBER)
        self.stop_button.configure(state=tk.DISABLED, bg=_C_DISBLD, fg=_C_SUBTEXT)

        frames_to_save = list(self.flow_tensors_buffer)   # tensores (64,64,3) float32
        self.flow_tensors_buffer = []
        self.frames_buffer = []

        worker = threading.Thread(
            target=self._save_sequence_worker,
            args=(frames_to_save, sequence_name, subject_name),
            daemon=True,
        )
        worker.start()

    def _save_sequence_worker(
        self,
        frames: list[np.ndarray],
        sequence_name: str,
        subject_name: str,
    ) -> None:
        """Guarda la secuencia en segundo plano y actualiza la UI al terminar."""
        try:
            if not frames:
                message = "No se capturaron frames para guardar."
                color = _C_AMBER
            else:
                sequence_dir = save_sequence(frames, sequence_name, subject_name)
                message = f"✓  Guardado en carpeta: {sequence_dir.name}"
                color = _C_GREEN
        except Exception as exc:
            message = f"✗  Error: {exc}"
            color = _C_RED

        self.root.after(0, self._finish_saving, message, color)

    def _finish_saving(self, status_message: str, color: str = _C_SUBTEXT) -> None:
        """Restablece la UI después de guardar la secuencia."""
        self.saving = False
        self._set_status(status_message, color)
        self.start_button.configure(state=tk.NORMAL, bg=_C_GREEN, fg="white")
        self.stop_button.configure(state=tk.DISABLED, bg=_C_DISBLD, fg=_C_SUBTEXT)
        self.camera_combo.state(["readonly"])

    def _process_pipeline(self, frame: np.ndarray) -> tuple:
        """
        Ejecuta detección → landmarks → flujo óptico → máscara.
        Devuelve (display_frame, flow_bgr | None, mask_bgr | None, flow_tensor | None).
        display_frame solo tiene el bounding box; el flujo y la máscara
        se muestran en paneles separados, fuera del recuadro de cámara.
        """
        display = frame.copy()

        if self.mp_detector is None:
            return display, None, None, None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.mp_detector.detect(mp_image)

        if not result.face_landmarks:
            self.prev_gray_face = None
            self.flow_buffer.clear()
            return display, None, None, None

        curr_gray, mapped_lms, bbox = _extract_face_and_landmarks(
            frame, result.face_landmarks[0]
        )
        if curr_gray is None:
            self.prev_gray_face = None
            return display, None, None, None

        # Bounding box del rostro (única anotación sobre la cámara)
        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        flow_out = None
        mask_out = None

        if self.prev_gray_face is not None:
            mask = _create_roi_mask(mapped_lms)
            flow_tensor, flow_visual = _compute_flow(self.prev_gray_face, curr_gray, mask)
            self.flow_buffer.append(flow_tensor)
            flow_out = flow_visual
            mask_out = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.prev_gray_face = curr_gray
            return display, flow_out, mask_out, flow_tensor

        self.prev_gray_face = curr_gray
        return display, None, None, None

    def update_frame(self) -> None:
        """Lee el frame de cámara, ejecuta el pipeline y actualiza los tres paneles."""
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                self.current_frame = frame

                display, flow_bgr, mask_bgr, flow_tensor = self._process_pipeline(frame)

                # Acumula tensores de flujo filtrado mientras se graba
                if self.recording and flow_tensor is not None:
                    self.flow_tensors_buffer.append(flow_tensor)

                # Panel principal: cámara con bounding box
                self.preview_image = self._prepare_preview(display)
                self.preview_label.configure(image=self.preview_image, text="")

                # Panel lateral izquierdo: flujo filtrado
                if self.flow_label is not None:
                    self.flow_image = self._encode_side_panel(flow_bgr)
                    self.flow_label.configure(image=self.flow_image)

                # Panel lateral derecho: máscara de pesos
                if self.mask_label is not None:
                    self.mask_image = self._encode_side_panel(mask_bgr)
                    self.mask_label.configure(image=self.mask_image)
            else:
                self.preview_label.configure(text="No se pudo leer la cámara.", image="")
        else:
            self.preview_label.configure(text="Selecciona una cámara disponible.", image="")

        self.preview_job = self.root.after(15, self.update_frame)

    def _prepare_preview(self, frame: np.ndarray) -> tk.PhotoImage:
        """Convierte un frame de OpenCV a PhotoImage sin dependencias externas."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_rgb = self._resize_with_aspect(frame_rgb, PREVIEW_SIZE)
        encoded = cv2.imencode(".png", cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))[1]
        return tk.PhotoImage(data=encoded.tobytes())

    @staticmethod
    def _encode_side_panel(frame_bgr: np.ndarray | None) -> tk.PhotoImage:
        """Escala un frame BGR del pipeline a SIDE_PANEL_SIZE y lo convierte a PhotoImage."""
        if frame_bgr is None:
            img = np.zeros((SIDE_PANEL_SIZE[1], SIDE_PANEL_SIZE[0], 3), dtype=np.uint8)
        else:
            img = cv2.resize(frame_bgr, SIDE_PANEL_SIZE, interpolation=cv2.INTER_NEAREST)
        encoded = cv2.imencode(".png", img)[1]
        return tk.PhotoImage(data=encoded.tobytes())

    @staticmethod
    def _resize_with_aspect(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Redimensiona manteniendo proporción y añade fondo negro si hace falta."""
        target_w, target_h = target_size
        frame_h, frame_w = frame.shape[:2]

        scale = min(target_w / frame_w, target_h / frame_h)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    def _release_camera(self) -> None:
        """Libera el recurso de cámara actual si existe."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self) -> None:
        """Cierra la aplicación de forma segura."""
        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        self._release_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SequenceCaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()