from __future__ import annotations

import torch
import torch.nn as nn

from .model import EmotionLSTM


class TransformerSequenceModel(nn.Module):
    """Lightweight sequence classifier for the existing padded-feature contract."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward_packed(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_seq_len = x.size(1)
        mask = torch.arange(max_seq_len, device=x.device)[None, :] >= lengths[:, None]
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        valid_mask = (~mask).unsqueeze(-1).type_as(x)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        pooled = self.norm(pooled)
        return self.classifier(pooled)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        return self.forward_packed(x, lengths)


def get_sequence_model(arch: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    arch_l = arch.lower()
    if arch_l == "lstm":
        return EmotionLSTM(input_dim=input_dim, num_classes=num_classes, **kwargs)
    if arch_l == "transformer":
        return TransformerSequenceModel(input_dim=input_dim, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown sequence architecture: {arch}")