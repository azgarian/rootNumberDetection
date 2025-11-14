import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Builds a model with an attention-augmented classification head to improve decision boundaries.
    The head:
      - Projects the penultimate feature to a token sequence
      - Applies a tiny Transformer encoder (MultiheadAttention + FFN with LayerNorm & residuals)
      - Pools tokens, then passes through an MLP (BN/GELU/Dropout)
      - Ends with a weight-normalized classifier
    """
    name = name.lower()

    if name == "alexnet":

        class TinyTransformerBlock(nn.Module):
            def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, p: float = 0.2):
                super().__init__()
                self.ln1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
                self.ln2 = nn.LayerNorm(dim)
                hidden = int(dim * mlp_ratio)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.Dropout(p),
                    nn.Linear(hidden, dim),
                    nn.Dropout(p),
                )

            def forward(self, x):
                # x: (B, T, C)
                x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
                x = x + self.ffn(self.ln2(x))
                return x

        class AttnHead(nn.Module):
            def __init__(self, n_in: int, n_classes: int, n_tokens: int = 16, attn_dim: int | None = None):
                super().__init__()
                # Ensure we can form tokens evenly; project if needed.
                attn_dim = attn_dim or max(64, (n_in // n_tokens // 16) * 16)
                self.proj_in = nn.Linear(n_in, n_tokens * attn_dim, bias=False)
                self.transformer = TinyTransformerBlock(attn_dim, num_heads=max(1, attn_dim // 64), mlp_ratio=2.0, p=0.1)
                # Lightweight squeeze-excitation style channel gating on the pooled token
                self.se = nn.Sequential(
                    nn.Linear(attn_dim, max(32, attn_dim // 4), bias=True),
                    nn.GELU(),
                    nn.Linear(max(32, attn_dim // 4), attn_dim, bias=True),
                    nn.Sigmoid(),
                )
                # MLP head after attention pooling
                self.mlp = nn.Sequential(
                    nn.Linear(attn_dim, 512, bias=False),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256, bias=False),
                    nn.BatchNorm1d(256),
                    nn.GELU(),
                    nn.Dropout(0.2),
                )
                self.classifier = torch.nn.utils.parametrizations.weight_norm(nn.Linear(256, n_classes), name="weight")

            def forward(self, x):
                # x: (B, n_in) penultimate feature vector
                B = x.size(0)
                x = self.proj_in(x)                      # (B, T*C)
                # form tokens
                T = self.transformer.attn.embed_dim  # attn_dim
                x = x.view(B, -1, T)                   # (B, n_tokens, attn_dim)
                x = self.transformer(x)                # (B, n_tokens, attn_dim)
                x = x.mean(dim=1)                      # token mean pool -> (B, attn_dim)
                # SE gating
                gate = self.se(x)
                x = x * gate
                # MLP + classifier
                x = self.mlp(x)
                return self.classifier(x)

        def make_head(n_in: int, n_classes: int) -> nn.Module:
            return AttnHead(n_in, n_classes, n_tokens=16)

        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.alexnet(weights=weights)
        # Replace only the last linear stage; earlier classifier layers remain (Dropout/FC/ReLU/...).
        n_in = model.classifier[-1].in_features  # 4096
        model.classifier[-1] = make_head(n_in, num_classes)
        return model

    if name == "efficientnet_b0":
        n_tokens = 16
        dropout = None
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[-1].in_features
        # Remove final classifier; we'll take pooled features and feed the attention head
        backbone.classifier[-1] = nn.Identity()

        class TinyTransformerBlock(nn.Module):
            def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, p: float = 0.1):
                super().__init__()
                self.ln1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
                self.ln2 = nn.LayerNorm(dim)
                hidden = int(dim * mlp_ratio)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.Dropout(p),
                    nn.Linear(hidden, dim),
                    nn.Dropout(p),
                )
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
                x = x + self.ffn(self.ln2(x))
                return x

        class AttnHead(nn.Module):
            def __init__(self, n_in: int, n_classes: int, n_tokens: int = 16, attn_dim: int | None = None):
                super().__init__()
                attn_dim = attn_dim or max(64, (n_in // n_tokens // 16) * 16)
                self.proj_in = nn.Linear(n_in, n_tokens * attn_dim, bias=False)
                self.transformer = TinyTransformerBlock(attn_dim, num_heads=max(1, attn_dim // 64), mlp_ratio=2.0, p=0.1)
                self.se = nn.Sequential(
                    nn.Linear(attn_dim, max(32, attn_dim // 4), bias=True),
                    nn.GELU(),
                    nn.Linear(max(32, attn_dim // 4), attn_dim, bias=True),
                    nn.Sigmoid(),
                )
                mlp = [nn.Linear(attn_dim, 512, bias=False), nn.BatchNorm1d(512), nn.GELU()]
                if dropout is not None:
                    mlp.append(nn.Dropout(float(dropout)))
                else:
                    mlp.append(nn.Dropout(0.3))
                mlp += [nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2)]
                self.mlp = nn.Sequential(*mlp)
                self.classifier = torch.nn.utils.parametrizations.weight_norm(nn.Linear(256, n_classes), name="weight")
                self.n_tokens = int(n_tokens)
                self.attn_dim = int(attn_dim)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b = x.size(0)
                x = self.proj_in(x)
                x = x.view(b, self.n_tokens, self.attn_dim)
                x = self.transformer(x)
                x = x.mean(dim=1)
                x = x * self.se(x)
                x = self.mlp(x)
                return self.classifier(x)

        class EfficientNetWithHead(nn.Module):
            def __init__(self, base: nn.Module, head: nn.Module):
                super().__init__()
                self.base = base
                self.head = head
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Follow torchvision forward to pooled features
                features = self.base.features(x)
                x = self.base.avgpool(features)
                x = torch.flatten(x, 1)
                return self.head(x)

        head = AttnHead(in_features, num_classes, n_tokens=int(n_tokens))
        return EfficientNetWithHead(backbone, head)


    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Identity()

        class TinyTransformerBlock(nn.Module):
            def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, p: float = 0.1):
                super().__init__()
                self.ln1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
                self.ln2 = nn.LayerNorm(dim)
                hidden = int(dim * mlp_ratio)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.Dropout(p),
                    nn.Linear(hidden, dim),
                    nn.Dropout(p),
                )
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
                x = x + self.ffn(self.ln2(x))
                return x

        class AttnHead(nn.Module):
            def __init__(self, n_in: int, n_classes: int, n_tokens: int = 16, attn_dim: int | None = None):
                super().__init__()
                attn_dim = attn_dim or max(64, (n_in // n_tokens // 16) * 16)
                self.proj_in = nn.Linear(n_in, n_tokens * attn_dim, bias=False)
                self.transformer = TinyTransformerBlock(attn_dim, num_heads=max(1, attn_dim // 64), mlp_ratio=2.0, p=0.1)
                self.se = nn.Sequential(
                    nn.Linear(attn_dim, max(32, attn_dim // 4), bias=True),
                    nn.GELU(),
                    nn.Linear(max(32, attn_dim // 4), attn_dim, bias=True),
                    nn.Sigmoid(),
                )
                self.mlp = nn.Sequential(
                    nn.Linear(attn_dim, 512, bias=False),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256, bias=False),
                    nn.BatchNorm1d(256),
                    nn.GELU(),
                    nn.Dropout(0.2),
                )
                self.classifier = torch.nn.utils.parametrizations.weight_norm(nn.Linear(256, n_classes), name="weight")
                self.n_tokens = int(n_tokens)
                self.attn_dim = int(attn_dim)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b = x.size(0)
                x = self.proj_in(x)
                x = x.view(b, self.n_tokens, self.attn_dim)
                x = self.transformer(x)
                x = x.mean(dim=1)
                x = x * self.se(x)
                x = self.mlp(x)
                return self.classifier(x)

        class DenseNetWithHead(nn.Module):
            def __init__(self, base: nn.Module, head: nn.Module):
                super().__init__()
                self.base = base
                self.head = head
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Mimic torchvision densenet forward to pooled features
                features = self.base.features(x)
                out = F.relu(features, inplace=False)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
                return self.head(out)

        head = AttnHead(in_features, num_classes)
        return DenseNetWithHead(backbone, head)
