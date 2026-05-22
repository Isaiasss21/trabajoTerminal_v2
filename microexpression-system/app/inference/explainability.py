"""
explainability.py
-----------------
Grad-CAM para modelos FlowClassifier (ResNet18/34/50 backbone).

Genera un mapa de calor que indica qué regiones del flujo óptico
contribuyeron más a la predicción de una clase determinada.

Referencia:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Uso:
    extractor = GradCAMExtractor(model)
    heatmap   = extractor.compute(tensor, class_idx, out_size=(64, 64))
    # heatmap → np.ndarray (H, W, 3) BGR uint8
    extractor.remove()   # liberar hooks al terminar
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAMExtractor:
    """
    Grad-CAM sobre la última capa convolucional del backbone ResNet.

    Registra hooks en `model.backbone.layer4[-1]` para capturar:
      - activaciones (forward hook)  → (B*T, C, h, w)
      - gradientes   (backward hook) → (B*T, C, h, w)

    B=1 siempre (batch size), T=16 (frames por secuencia), h×w=7×7
    para entradas 224×224.

    Solo compatible con backbones que expongan `backbone.layer4`
    (resnet18, resnet34, resnet50).  DenseNet no está soportado.
    """

    def __init__(self, model: nn.Module) -> None:
        if not (hasattr(model, "backbone") and model.backbone is not None
                and hasattr(model.backbone, "layer4")):
            raise ValueError(
                "GradCAM requiere un backbone ResNet con atributo 'layer4'."
            )

        self._model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._handles: list = []

        # Última capa del bloque layer4
        target_layer = list(model.backbone.layer4.children())[-1]
        self._handles.append(
            target_layer.register_forward_hook(self._fwd_hook)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._bwd_hook)
        )

    # ── Hooks ─────────────────────────────────────────────────────────────

    def _fwd_hook(self, module, inp, out) -> None:          # noqa: ANN001
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out) -> None:  # noqa: ANN001
        if grad_out and grad_out[0] is not None:
            self._gradients = grad_out[0].detach()

    # ── API pública ────────────────────────────────────────────────────────

    def compute(
        self,
        tensor:    torch.Tensor,
        class_idx: int,
        out_size:  tuple[int, int] = (64, 64),
    ) -> np.ndarray:
        """
        Calcula el mapa de calor Grad-CAM para la clase indicada.

        Args:
            tensor:    (1, T, 3, 224, 224) — preprocesado por InferenceEngine
            class_idx: índice de la clase predicha
            out_size:  (ancho, alto) del heatmap de salida, por defecto (64, 64)

        Returns:
            heatmap BGR uint8 de forma (out_size[1], out_size[0], 3)
        """
        self._activations = None
        self._gradients   = None

        self._model.eval()

        # Detach + requiere grad para el backward
        x = tensor.detach().requires_grad_(True)

        # Forward pass — hooks capturan activaciones
        logits = self._model(x)           # (1, num_classes)
        self._model.zero_grad()

        # Backward sobre el score de la clase objetivo
        logits[0, class_idx].backward()

        if self._activations is None or self._gradients is None:
            return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)

        # acts/grads: (B*T, C, h, w)
        acts  = self._activations   # e.g. (16, 512, 7, 7)
        grads = self._gradients     # e.g. (16, 512, 7, 7)

        # Peso por canal = promedio espacial global del gradiente (Global-Avg-Pool)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (B*T, C, 1, 1)

        # Combinación lineal ponderada + ReLU
        cam = torch.relu((weights * acts).sum(dim=1))    # (B*T, h, w)

        # Normalizar cada frame y redimensionar al tamaño de salida
        cam_np = cam.cpu().numpy()   # (B*T, h, w)
        frames_cam = []
        for t in range(cam_np.shape[0]):
            c = cam_np[t]
            vmin, vmax = float(c.min()), float(c.max())
            if vmax > vmin:
                c = (c - vmin) / (vmax - vmin)
            frames_cam.append(cv2.resize(c, out_size))

        # Promedio temporal → mapa final
        avg_cam = np.mean(frames_cam, axis=0)            # (H, W)

        # Pseudo-colorear con COLORMAP_JET
        u8 = (avg_cam * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_JET)  # (H, W, 3) BGR

    def remove(self) -> None:
        """Elimina los hooks. Llamar al cerrar o cambiar de modelo."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
