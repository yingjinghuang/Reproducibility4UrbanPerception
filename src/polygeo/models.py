"""Architecture factories for A1 / A2 / A3.

All return a model with a 6-dim regression head (one output per perceptual dimension).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm

N_DIM = 6


def build_resnet50(pretrained: bool = True) -> nn.Module:
    """A1: ResNet-50 full fine-tune, ImageNet-1k init."""
    m = timm.create_model("resnet50", pretrained=pretrained, num_classes=N_DIM)
    return m


def build_vit_b16(pretrained: bool = True, init: str = "imagenet21k") -> nn.Module:
    """A2: ViT-B/16 full fine-tune.

    init ∈ {'imagenet21k', 'dinov2'}
      - imagenet21k → vit_base_patch16_224.augreg_in21k_ft_in1k (timm)
      - dinov2      → vit_base_patch14_dinov2.lvd142m  (note: patch 14, 518 input)
    """
    if init == "imagenet21k":
        m = timm.create_model(
            "vit_base_patch16_224.augreg_in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=N_DIM,
        )
    elif init == "dinov2":
        m = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=N_DIM,
            img_size=224,
        )
    else:
        raise ValueError(f"unknown ViT init {init!r}")
    return m


def build_dinov2_frozen(pretrained: bool = True, hidden: int = 256) -> nn.Module:
    """A3: DINOv2 ViT-B/14 frozen + linear probe + small MLP head.

    Only the head is trainable. Backbone is frozen at eval mode.
    """

    class FrozenDinoHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                num_classes=0,  # global feature
                img_size=224,
            )
            for p in self.backbone.parameters():
                p.requires_grad = False
            d = self.backbone.num_features
            self.head = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, N_DIM),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats = self.backbone(x)
            return self.head(feats)

        def train(self, mode: bool = True):
            # keep backbone in eval mode regardless of train(); only head toggles
            super().train(mode)
            self.backbone.eval()
            return self

    return FrozenDinoHead()


def build(arch: str, **kwargs) -> nn.Module:
    if arch == "resnet50":
        return build_resnet50(**kwargs)
    if arch == "vit_b16":
        return build_vit_b16(**kwargs)
    if arch == "dinov2_frozen":
        return build_dinov2_frozen(**kwargs)
    raise ValueError(f"unknown arch {arch!r}")
