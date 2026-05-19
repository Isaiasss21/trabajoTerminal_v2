"""
inference_engine.py
--------------------
Motor de inferencia para clasificación de microexpresiones.

Arquitectura: FlowClassifier (train_densenet_flow.py)
  - Backbone: resnet18 / resnet34 / resnet50 / densenet121 (detectado automáticamente)
  - Temporal pooling: TemporalAttentionPool | LSTMTemporalPool |
                      TemporalGRU | TemporalTCN | weighted mean-pool
  - Checkpoint: state_dict-only  (torch.save(model.state_dict(), path))

Preprocesamiento (idéntico a FlowSequenceDataset.__getitem__ del entrenamiento):
  (N, H, W, 3) float32 [dx, dy, mag]
    → sample T frames uniformly
    → f0=clip((dx+1)/2,0,1), f1=clip((dy+1)/2,0,1), f2=clip(mag,0,1)
    → uint8 pseudo-RGB
    → Resize(224,224) + ToTensor + Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    → (1, T, 3, 224, 224) tensor

Label map por defecto (sorted alphabetically = build_label_map del entrenamiento):
  asco→0, enojo→1, felicidad→2, miedo→3, neutral→4, sorpresa→5, tristeza→6

NO contiene: captura de cámara, GUI, almacenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ── Constantes ────────────────────────────────────────────────────────────────

# Label-map por defecto (sorted alphabetically = build_label_map del entrenamiento).
DEFAULT_LABEL_MAP: dict[str, int] = {
    "asco":      0,
    "enojo":     1,
    "felicidad": 2,
    "miedo":     3,
    "neutral":   4,
    "sorpresa":  5,
    "tristeza":  6,
}

# Nombres legibles en español para la UI (RB05)
EMOTION_LABELS_ES: dict[str, str] = {
    "asco":      "Asco",
    "enojo":     "Enojo",
    "felicidad": "Alegría",
    "miedo":     "Miedo",
    "neutral":   "Neutral",
    "sorpresa":  "Sorpresa",
    "tristeza":  "Tristeza",
}

# Colores hex por emoción para la UI
EMOTION_COLORS: dict[str, str] = {
    "Alegría":  "#FFD700",
    "Asco":     "#66BB6A",
    "Enojo":    "#EF5350",
    "Miedo":    "#AB47BC",
    "Neutral":  "#90A4AE",
    "Sorpresa": "#FF9800",
    "Tristeza": "#4FC3F7",
}

# RB02: umbrales de confianza
CONFIDENCE_UNCERTAIN = 0.50   # zona incierta
CONFIDENCE_VALID     = 0.70   # umbral de aceptación

NUM_FRAMES_DEFAULT = 16       # frames muestreados por secuencia


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULOS DE ARQUITECTURA
# Copiados de train_densenet_flow.py — deben coincidir exactamente con los
# pesos guardados.
# ══════════════════════════════════════════════════════════════════════════════

class _GeM(nn.Module):
    """Generalized Mean pooling (path DenseNet)."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class _TemporalAttentionPool(nn.Module):
    """Multi-head self-attention + mean-pool (B,T,D) → (B,D)."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x; x = self.norm1(x); x, _ = self.attn(x, x, x); x = x + r
        r = x; x = self.norm2(x); x = self.ffn(x); x = x + r
        return x.mean(dim=1)


class _LSTMTemporalPool(nn.Module):
    """BiLSTM (B,T,D) → (B, hidden*2)."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float = 0.6) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop    = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.drop(torch.cat([h_n[-2], h_n[-1]], dim=-1))


class _TemporalGRU(nn.Module):
    """BiGRU con LayerNorm (B,T,D) → (B, hidden*2)."""
    def __init__(self, dim: int, hidden: int = 256, layers: int = 1,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru  = nn.GRU(input_size=dim, hidden_size=hidden, num_layers=layers,
                           batch_first=True, bidirectional=True,
                           dropout=0.0 if layers == 1 else dropout)
        self.drop    = nn.Dropout(dropout)
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        self.gru.flatten_parameters()
        _, h_n = self.gru(x)
        return self.drop(torch.cat([h_n[-2], h_n[-1]], dim=-1))


class _TemporalTCN(nn.Module):
    """TCN con bloques dilatados (B,T,D) → (B, channels)."""
    def __init__(self, input_dim: int, channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm       = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1), nn.GELU()
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=1),
            )
            for dilation in [1, 2, 4]
        ])
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.drop    = nn.Dropout(dropout)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.drop(self.pool(x).squeeze(-1))


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class _GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, lambda_)


class _FlowClassifier(nn.Module):
    """
    FlowClassifier: backbone (ResNet/DenseNet) + temporal pooling + clasificador.
    Reconstruido desde train_densenet_flow.py para coincidir con los pesos guardados.
    Solo se instancia internamente desde InferenceEngine._load().
    """

    _FEATURE_DIMS: dict[str, int] = {
        "resnet18":    512,
        "resnet34":    512,
        "resnet50":    2048,
        "densenet121": 1024,
    }

    def __init__(
        self,
        num_classes:       int,
        backbone:          str  = "resnet18",
        use_temporal_attn: bool = False,
        use_lstm:          bool = False,
        lstm_hidden:       int  = 128,
        lstm_layers:       int  = 1,
        use_gru:           bool = False,
        gru_hidden:        int  = 256,
        gru_layers:        int  = 1,
        use_tcn:           bool = False,
        tcn_channels:      int  = 256,
        use_dann:          bool = False,
    ) -> None:
        super().__init__()
        backbone = backbone.lower()

        if backbone == "resnet18":
            net = models.resnet18(weights=None)
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone == "resnet34":
            net = models.resnet34(weights=None)
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone == "resnet50":
            net = models.resnet50(weights=None)
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone == "densenet121":
            net = models.densenet121(weights=None)
            self._densenet_features = net.features
            self._densenet_pool     = _GeM()
            self.backbone = None
        else:
            raise ValueError(f"Backbone desconocido: '{backbone}'")

        self._backbone_name = backbone
        self.feature_dim    = self._FEATURE_DIMS[backbone]

        self.use_temporal_attn = use_temporal_attn
        self.use_lstm          = use_lstm
        self.use_gru           = use_gru
        self.use_tcn           = use_tcn

        self.temporal_pool: Optional[nn.Module] = None
        self.lstm_pool:     Optional[nn.Module] = None
        self.gru_pool:      Optional[nn.Module] = None
        self.tcn_pool:      Optional[nn.Module] = None

        if use_temporal_attn:
            self.temporal_pool = _TemporalAttentionPool(
                dim=self.feature_dim, num_heads=4, dropout=0.1
            )
        if use_lstm:
            self.lstm_pool = _LSTMTemporalPool(
                input_dim=self.feature_dim, hidden_dim=lstm_hidden,
                num_layers=lstm_layers, dropout=0.6,
            )
        elif use_tcn:
            self.tcn_pool = _TemporalTCN(
                input_dim=self.feature_dim, channels=tcn_channels, dropout=0.2,
            )

        if use_lstm:
            head_dim = lstm_hidden * 2
        elif use_gru:
            head_dim = gru_hidden * 2
        elif use_tcn:
            head_dim = tcn_channels
        else:
            head_dim = self.feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(head_dim, num_classes),
        )

        self.use_dann = use_dann
        if use_dann:
            self._grl = _GradientReversalLayer()
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 2),
            )
        else:
            self._grl = None
            self.domain_classifier = None

    def forward(self, x: torch.Tensor, dann_lambda: float = 1.0) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        if self.backbone is not None:
            feat = self.backbone(x_flat)
        else:
            feat = self._densenet_features(x_flat)
            feat = F.relu(feat, inplace=True)
            feat = self._densenet_pool(feat).flatten(1)

        feat = feat.view(B, T, -1)

        if self.use_temporal_attn and self.temporal_pool is not None:
            feat = self.temporal_pool(feat)
        elif self.use_lstm and self.lstm_pool is not None:
            feat = self.lstm_pool(feat)
        elif self.use_gru and self.gru_pool is not None:
            feat = self.gru_pool(feat)
        elif self.use_tcn and self.tcn_pool is not None:
            feat = self.tcn_pool(feat)
        else:
            w = torch.linspace(0.5, 1.5, steps=T, device=feat.device, dtype=feat.dtype)
            w = w / w.sum()
            feat = (feat * w.view(1, T, 1)).sum(dim=1)

        return self.classifier(feat)


# ── Resultado de inferencia ────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Resultado de una predicción de emoción."""
    emotion:      str              # Nombre legible en español (e.g. "Alegría")
    raw_label:    str              # Label del modelo (e.g. "felicidad")
    confidence:   float            # Confianza de la clase predicha [0, 1]
    is_valid:     bool             # confidence >= CONFIDENCE_VALID (RB02)
    is_uncertain: bool             # CONFIDENCE_UNCERTAIN <= conf < CONFIDENCE_VALID
    probs:        dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Carga un checkpoint FlowClassifier y realiza predicciones sobre secuencias
    de optical flow capturadas en tiempo real.

    El checkpoint es un state_dict puro:
        torch.save(model.state_dict(), "best_resnet18_flow_fold1.pth")

    La arquitectura (backbone, temporal pooling, nº de clases) se detecta
    automáticamente desde las claves del state_dict.

    Uso:
        engine = InferenceEngine("microexpression-system/models/best_resnet18_flow.pth")
        result = engine.predict(flow_sequence)   # (N, 64, 64, 3) float32
        print(result.emotion, result.confidence)
    """

    def __init__(
        self,
        model_path: str | Path,
        device:     str = "auto",
        num_frames: int = NUM_FRAMES_DEFAULT,
    ) -> None:
        self._model_path = Path(model_path)
        self._num_frames = num_frames

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model:     Optional[_FlowClassifier] = None
        self._label_map: dict[str, int]            = {}
        self._idx_map:   dict[int, str]            = {}

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        if self._model_path.exists():
            self._load()

    # ── API pública ────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def label_map(self) -> dict[str, int]:
        return dict(self._label_map)

    def predict(self, flow_sequence: np.ndarray) -> InferenceResult:
        """
        Infiere la emoción de una secuencia de optical flow.

        Args:
            flow_sequence: np.ndarray (N, H, W, 3) float32 con canales [dx, dy, mag].
                           dx,dy ∈ [-1, 1],  mag ∈ [0, 1].
        """
        if not self.is_ready:
            raise RuntimeError("El modelo no está cargado. Verifica la ruta del checkpoint.")
        tensor = self._preprocess(flow_sequence)
        return self._run_inference(tensor)

    def predict_tta(self, flow_sequence: np.ndarray, passes: int = 4) -> InferenceResult:
        """
        Inferencia con Test-Time Augmentation (hasta 4 vistas).
        Idéntico a evaluate_with_tta() del entrenamiento.
        """
        if not self.is_ready:
            raise RuntimeError("El modelo no está cargado.")
        tensor  = self._preprocess(flow_sequence)
        n_views = max(1, min(passes, 4))
        views   = self._tta_views(tensor, n_views)

        self._model.eval()
        avg_probs = torch.zeros(1, len(self._label_map), device=self._device)
        with torch.no_grad():
            for view in views:
                avg_probs += torch.softmax(self._model(view), dim=1)
        avg_probs /= len(views)
        return self._probs_to_result(avg_probs[0].cpu().numpy())

    # ── Carga y detección automática de arquitectura ───────────────────────

    def _load(self) -> None:
        """Carga el checkpoint e instancia FlowClassifier con los parámetros correctos."""
        state_dict = torch.load(self._model_path, map_location="cpu", weights_only=True)

        # Checkpoint podría ser state_dict directo o dict con clave "state_dict"
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        keys = set(state_dict.keys())

        # ── Backbone ────────────────────────────────────────────────────────
        if any(k.startswith("_densenet_features") for k in keys):
            backbone = "densenet121"
        elif any("backbone.layer4.0.conv3" in k for k in keys):
            backbone = "resnet50"
        else:
            backbone = "resnet18"   # resnet18 o resnet34, mismo feature_dim

        # ── Temporal pooling ─────────────────────────────────────────────────
        use_temporal_attn = any(k.startswith("temporal_pool") for k in keys)
        use_lstm          = any(k.startswith("lstm_pool")      for k in keys)
        use_gru           = any(k.startswith("gru_pool")       for k in keys)
        use_tcn           = any(k.startswith("tcn_pool")       for k in keys)
        use_dann          = any(k.startswith("domain_class")   for k in keys)

        # ── Número de clases y head_dim desde el clasificador ────────────────
        head_weight = state_dict["classifier.1.weight"]
        num_classes = int(head_weight.shape[0])
        head_dim    = int(head_weight.shape[1])

        # Parámetros de pools inferidos desde head_dim
        lstm_hidden  = head_dim // 2 if use_lstm else 128
        lstm_layers  = _detect_rnn_layers(state_dict, "lstm_pool.lstm")
        gru_hidden   = head_dim // 2 if use_gru  else 256
        gru_layers   = _detect_rnn_layers(state_dict, "gru_pool.gru")
        tcn_channels = head_dim if use_tcn else 256

        model = _FlowClassifier(
            num_classes=num_classes, backbone=backbone,
            use_temporal_attn=use_temporal_attn,
            use_lstm=use_lstm, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            use_gru=use_gru, gru_hidden=gru_hidden, gru_layers=gru_layers,
            use_tcn=use_tcn, tcn_channels=tcn_channels,
            use_dann=use_dann,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._model = model.to(self._device)

        # ── Label map ────────────────────────────────────────────────────────
        # 1) Companion JSON file  (<model_name>.json  next to the .pth)
        json_path = self._model_path.with_suffix(".json")
        loaded_map: dict[str, int] | None = None
        if json_path.exists():
            import json
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                candidate = raw.get("label_map", raw) if isinstance(raw, dict) else None
                if isinstance(candidate, dict) and len(candidate) == num_classes:
                    loaded_map = {str(k): int(v) for k, v in candidate.items()}
            except Exception:
                pass

        if loaded_map is not None:
            self._label_map = loaded_map
        elif num_classes == len(DEFAULT_LABEL_MAP):
            self._label_map = DEFAULT_LABEL_MAP.copy()
        else:
            # Generic fallback for unknown num_classes
            self._label_map = {f"clase_{i}": i for i in range(num_classes)}
        self._idx_map = {v: k for k, v in self._label_map.items()}

        print(f"[InferenceEngine] Cargado: {self._model_path.name} | "
              f"backbone={backbone} | clases={num_classes} | device={self._device}")

    # ── Preprocesamiento ───────────────────────────────────────────────────

    def _preprocess(self, flow_seq: np.ndarray) -> torch.Tensor:
        """
        (N, H, W, 3) float32 → (1, T, 3, 224, 224).
        Idéntico a FlowSequenceDataset.__getitem__() del entrenamiento.
        """
        frames = _sample_frames(flow_seq, self._num_frames)  # (T, H, W, 3)

        # Normalizar canales al rango [0,1] → uint8 pseudo-RGB
        f0  = np.clip((frames[..., 0] + 1.0) / 2.0, 0.0, 1.0)
        f1  = np.clip((frames[..., 1] + 1.0) / 2.0, 0.0, 1.0)
        f2  = np.clip(frames[..., 2], 0.0, 1.0)
        rgb = (np.stack([f0, f1, f2], axis=-1) * 255).clip(0, 255).astype(np.uint8)

        tensor_frames = torch.stack([
            self._transform(Image.fromarray(frame, mode="RGB"))
            for frame in rgb
        ])  # (T, 3, 224, 224)

        return tensor_frames.unsqueeze(0).to(self._device)  # (1, T, 3, 224, 224)

    # ── Inferencia ─────────────────────────────────────────────────────────

    def _run_inference(self, tensor: torch.Tensor) -> InferenceResult:
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0].cpu().numpy()
        return self._probs_to_result(probs)

    def _probs_to_result(self, probs: np.ndarray) -> InferenceResult:
        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        raw_label  = self._idx_map.get(idx, "neutral")
        emotion    = EMOTION_LABELS_ES.get(raw_label, raw_label.capitalize())

        probs_dict = {
            EMOTION_LABELS_ES.get(self._idx_map.get(i, ""), str(i)): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return InferenceResult(
            emotion      = emotion,
            raw_label    = raw_label,
            confidence   = round(confidence, 4),
            is_valid     = confidence >= CONFIDENCE_VALID,
            is_uncertain = CONFIDENCE_UNCERTAIN <= confidence < CONFIDENCE_VALID,
            probs        = probs_dict,
        )

    # ── TTA helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _tta_views(tensor: torch.Tensor, n_views: int) -> list[torch.Tensor]:
        """Genera hasta 4 vistas deterministas (= evaluate_with_tta del entrenamiento)."""
        views = [tensor]
        if n_views >= 2:                      # flip horizontal → negar dx
            hf = tensor.flip(-1).clone()
            hf[:, :, 0] = -hf[:, :, 0]
            views.append(hf)
        if n_views >= 3:                      # flip vertical → negar dy
            vf = tensor.flip(-2).clone()
            vf[:, :, 1] = -vf[:, :, 1]
            views.append(vf)
        if n_views >= 4:                      # ambos flips
            hvf = tensor.flip(-1).flip(-2).clone()
            hvf[:, :, 0] = -hvf[:, :, 0]
            hvf[:, :, 1] = -hvf[:, :, 1]
            views.append(hvf)
        return views


# ── Utilidades internas ────────────────────────────────────────────────────────

def _sample_frames(sequence: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Muestreo uniforme de `num_frames` desde `sequence` (N, ...).
    Si N < num_frames repite (tile). Idéntico a sample_frames() del entrenamiento.
    """
    n = sequence.shape[0]
    if n == 0:
        return np.zeros((num_frames,) + sequence.shape[1:], dtype=sequence.dtype)
    if n >= num_frames:
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
    else:
        base = np.arange(n)
        indices = np.tile(base, (num_frames // n) + 1)[:num_frames]
    return sequence[indices]


def _detect_rnn_layers(state_dict: dict, prefix: str) -> int:
    """Detecta número de capas RNN contando sufijos '_lN_' en las claves."""
    max_layer = 0
    for k in state_dict:
        if not k.startswith(prefix):
            continue
        for part in k.split("."):
            # weight_ih_l0, weight_ih_l1_reverse, etc.
            if part.startswith("weight_ih_l") or part.startswith("weight_hh_l"):
                try:
                    layer_num = int("".join(c for c in part if c.isdigit()))
                    max_layer = max(max_layer, layer_num + 1)
                except ValueError:
                    pass
    return max(1, max_layer)

