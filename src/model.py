import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

class TemporalAttention(nn.Module):
    """
    Mecanismo de Atención Temporal (Bahdanau/Luong style adaptado).
    Aprende a darle más 'peso' a los frames donde ocurre el ápice de la microexpresión.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        # Capa para proyectar los outputs de la LSTM
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        # Vector de contexto que aprenderá a identificar lo "importante"
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs, mask=None):
        # rnn_outputs shape: (batch_size, seq_len, hidden_dim)
        
        # 1. Proyección no lineal
        u = torch.tanh(self.attention(rnn_outputs))
        
        # 2. Calcular scores de atención sin normalizar
        scores = self.context_vector(u).squeeze(-1) # shape: (batch_size, seq_len)

        # 3. Aplicar máscara (Crucial para ignorar el padding de secuencias cortas)
        if mask is not None:
            # Reemplazamos los scores de los frames falsos (padding) por -infinito
            # Así, al pasar por el softmax, su peso será exactamente 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Normalizar scores a probabilidades (pesos que suman 1)
        weights = F.softmax(scores, dim=1) # shape: (batch_size, seq_len)

        # 5. Multiplicar cada frame por su peso de atención y sumarlos
        # bmm = Batch Matrix Multiplication
        context = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1) 
        # context shape: (batch_size, hidden_dim)

        return context, weights


class EmotionLSTM(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=64,
        num_layers=1,
        num_classes=4,
        dropout=0.6,
        bidirectional=True,
        baseline_subtract=False,
        baseline_frames=3,
    ):
        super(EmotionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.baseline_subtract = bool(baseline_subtract)
        self.baseline_frames = int(baseline_frames)
        
        # Si es bidireccional, dividimos a la mitad para que la salida final siga siendo 'hidden_dim'
        self.lstm_hidden_dim = hidden_dim // self.num_directions if bidirectional else hidden_dim

        # PyTorch arroja warning si num_layers=1 y dropout>0 en la LSTM
        lstm_dropout = dropout if num_layers > 1 else 0.0

        # Capa 1: La LSTM Profunda
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )

        # Capa 2: Mecanismo de Atención
        self.attention = TemporalAttention(hidden_dim)

        # Capa 3: Clasificador con Regularización Severa (Dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def _apply_baseline_subtraction(self, x, lengths):
        # Baseline per sequence using first K valid frames.
        if self.baseline_frames <= 0:
            return x

        x_norm = x.clone()
        for i in range(x.size(0)):
            seq_len = int(lengths[i].item())
            if seq_len <= 0:
                continue
            k = min(self.baseline_frames, seq_len)
            baseline = x[i, :k, :].mean(dim=0, keepdim=True)
            x_norm[i, :seq_len, :] = x[i, :seq_len, :] - baseline
        return x_norm

    def forward_packed(self, x, lengths):
        # x shape: (batch, max_seq_len, input_dim)
        _, max_seq_len, _ = x.size()
        device = x.device

        if self.baseline_subtract:
            x = self._apply_baseline_subtraction(x, lengths)
        
        # Crear máscara booleana: 1 para frames reales, 0 para padding
        mask = torch.arange(max_seq_len, device=device)[None, :] < lengths[:, None]

        # Empacar la secuencia para eficiencia
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pasar por LSTM
        packed_out, _ = self.lstm(packed_x)

        # Desempacar la salida de la LSTM
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_seq_len)
        # out shape: (batch_size, max_seq_len, hidden_dim)

        # Pasar por el mecanismo de Atención
        context, _ = self.attention(out, mask)

        # Clasificación Final sobre el Vector de Contexto Ponderado
        out = F.relu(self.fc1(context))
        out = self.dropout_layer(out)
        logits = self.fc2(out)

        return logits

    def forward(self, x, lengths=None):
        # Fallback path for callers that do not pass explicit lengths.
        if lengths is None:
            lengths = torch.full(
                (x.size(0),),
                x.size(1),
                dtype=torch.long,
                device=x.device,
            )
        return self.forward_packed(x, lengths)


class ResidualMaskingGate(nn.Module):
    """Compuerta ligera inspirada en ResMaskNet: x * (1 + sigmoid(mask(x)))."""

    def __init__(self, channels: int, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        m = self.conv1(x)
        m = self.bn1(m)
        m = F.relu(m, inplace=True)
        m = torch.sigmoid(self.conv2(m))
        return x * (1.0 + m)


class FrameEncoderResNet18(nn.Module):
    """Encoder de frame con ResNet-18 y máscara residual ligera para microexpresiones."""

    def __init__(
        self,
        embedding_dim: int = 256,
        imagenet_pretrained: bool = True,
        affectnet_weights_path: str | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        backbone = models.resnet18(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Similar al espíritu de ResMaskNet: modulación multiplicativa por máscara.
        self.mask_gate = ResidualMaskingGate(channels=512, hidden_channels=128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        if affectnet_weights_path:
            info = self.load_affectnet_weights(affectnet_weights_path)
            print(
                f"[INFO] AffectNet load: matched={info['matched']} "
                f"missing={info['missing']} unexpected={info['unexpected']}"
            )

    @staticmethod
    def _extract_state_dict(ckpt_obj):
        if isinstance(ckpt_obj, dict):
            for key in ["state_dict", "model_state", "net", "model"]:
                maybe = ckpt_obj.get(key)
                if isinstance(maybe, dict):
                    return maybe
            # Algunos checkpoints ya son state_dict puro.
            tensor_values = [v for v in ckpt_obj.values() if torch.is_tensor(v)]
            if tensor_values:
                return ckpt_obj
        raise ValueError("No se encontró un state_dict válido en el checkpoint")

    @staticmethod
    def _strip_prefixes(name: str) -> str:
        prefixes = [
            "module.",
            "model.",
            "state_dict.",
            "encoder.",
            "backbone.",
            "feature_extractor.",
            "resnet.",
        ]
        out = name
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if out.startswith(p):
                    out = out[len(p) :]
                    changed = True
        return out

    def load_affectnet_weights(self, checkpoint_path: str):
        """Carga parcial robusta de pesos AffectNet/FER preentrenados."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = self._extract_state_dict(ckpt)

        cleaned = {}
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            nk = self._strip_prefixes(k)
            cleaned[nk] = v

        target = self.state_dict()
        matched = {}
        for k, v in cleaned.items():
            if k in target and target[k].shape == v.shape:
                matched[k] = v

        missing, unexpected = self.load_state_dict(matched, strict=False)
        return {
            "matched": len(matched),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }

    def extract_gap_features(self, x):
        """Devuelve features del backbone tras Global Average Pooling (pre-clasificación/proyección)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.mask_gate(x)
        gap = self.pool(x).flatten(1)
        return gap

    def forward(self, x):
        # x: [N, 3, H, W]
        x = self.extract_gap_features(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class EmotionFrameLSTM(nn.Module):
    """Modelo visual-temporal: FrameEncoder (ResNet18) + BiLSTM + atención temporal."""

    def __init__(
        self,
        frame_embedding_dim=256,
        hidden_dim=128,
        num_layers=1,
        num_classes=7,
        dropout=0.5,
        bidirectional=True,
        baseline_subtract=True,
        baseline_frames=3,
        imagenet_pretrained=True,
        affectnet_weights_path: str | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.baseline_subtract = bool(baseline_subtract)
        self.baseline_frames = int(baseline_frames)

        self.frame_encoder = FrameEncoderResNet18(
            embedding_dim=frame_embedding_dim,
            imagenet_pretrained=imagenet_pretrained,
            affectnet_weights_path=affectnet_weights_path,
            dropout=dropout * 0.5,
        )

        self.lstm_hidden_dim = hidden_dim // self.num_directions if bidirectional else hidden_dim
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=frame_embedding_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.attention = TemporalAttention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def load_affectnet_encoder_weights(self, checkpoint_path: str):
        return self.frame_encoder.load_affectnet_weights(checkpoint_path)

    def encode_frames(self, frames):
        # frames: [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        emb = self.frame_encoder(flat)
        return emb.reshape(b, t, -1)

    def _apply_baseline_subtraction(self, x, lengths):
        if self.baseline_frames <= 0:
            return x

        x_norm = x.clone()
        for i in range(x.size(0)):
            seq_len = int(lengths[i].item())
            if seq_len <= 0:
                continue
            k = min(self.baseline_frames, seq_len)
            baseline = x[i, :k, :].mean(dim=0, keepdim=True)
            x_norm[i, :seq_len, :] = x[i, :seq_len, :] - baseline
        return x_norm

    def forward_packed(self, frames, lengths):
        x = self.encode_frames(frames)
        _, max_seq_len, _ = x.size()
        device = x.device

        if self.baseline_subtract:
            x = self._apply_baseline_subtraction(x, lengths)

        mask = torch.arange(max_seq_len, device=device)[None, :] < lengths[:, None]
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_seq_len)

        context, _ = self.attention(out, mask)
        out = F.relu(self.fc1(context))
        out = self.dropout_layer(out)
        logits = self.fc2(out)
        return logits

    def forward(self, frames, lengths=None):
        if lengths is None:
            lengths = torch.full(
                (frames.size(0),),
                frames.size(1),
                dtype=torch.long,
                device=frames.device,
            )
        return self.forward_packed(frames, lengths)