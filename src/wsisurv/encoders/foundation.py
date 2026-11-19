"""Pathology foundation-model tile encoders (adapted from the clonality_detection repo).

Input contract: RGB tiles as float 0–255, NCHW, 224×224 — exactly what wsi_patching's
`TorchStreamWriter` yields. Each model's own normalisation is applied here; feeding raw
0–255 pixels to a ViT is a silent, common bug.

| name          | dim  | tiles at | access                                              |
|---------------|------|----------|-----------------------------------------------------|
| UNI2          | 1536 | 0.5 mpp  | gated: accept terms on hf.co/MahmoodLab/UNI2-h       |
| Virchow2      | 2560 | 0.5 mpp  | gated: hf.co/paige-ai/Virchow2 (CLS + mean patch)    |
| H-optimus-1   | 1536 | 0.5 mpp  | gated: hf.co/bioptimus/H-optimus-1                   |
| resnet50      | 2048 | any      | open ImageNet weights — a *non-pathology* baseline / smoke test |

Gated models: `huggingface-cli login` once, after accepting the model's terms. On an HPC
without internet: download once, then set HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HOPTIMUS = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))


class FoundationEncoder(nn.Module):
    def __init__(self, model_name: str = "UNI2", weights_path: str | None = None):
        super().__init__()
        name = model_name.lower()
        self.model_name = model_name
        self.cls_plus_mean = False

        if name.startswith("uni2"):
            self.encoder = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14, depth=24,
                num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU,
                reg_tokens=8, dynamic_img_size=True)
            stats = IMAGENET
        elif name.startswith("virchow"):
            self.encoder = timm.create_model(
                "hf-hub:paige-ai/Virchow2", pretrained=True, num_classes=0,
                mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU)
            cfg = timm.data.resolve_data_config({}, model=self.encoder)
            stats = (tuple(cfg["mean"]), tuple(cfg["std"]))
            self.cls_plus_mean = True   # model card: concat CLS token and mean of patch tokens
        elif "optimus" in name:
            kw = dict(pretrained=True, num_classes=0, init_values=1e-5, dynamic_img_size=False, img_size=224)
            if weights_path:
                self.encoder = timm.create_model("vit_giant_patch14_reg4_dinov2",
                                                 pretrained_cfg_overlay=dict(file=str(weights_path)), **kw)
            else:
                self.encoder = timm.create_model("hf-hub:bioptimus/H-optimus-1", **kw)
            stats = HOPTIMUS   # pinned: a local checkpoint would otherwise carry ImageNet stats
        elif name == "resnet50":
            self.encoder = timm.create_model("resnet50", pretrained=True, num_classes=0)
            stats = IMAGENET
        else:
            raise ValueError(f"unknown model_name {model_name!r}")

        self.register_buffer("mean", torch.tensor(stats[0]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(stats[1]).view(1, 3, 1, 1))
        log.info("Encoder %s, normalisation mean=%s std=%s", model_name, stats[0], stats[1])

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float 0–255 (B, 3, 224, 224) -> (B, dim)"""
        x = (x.float() / 255.0 - self.mean) / self.std
        if self.cls_plus_mean:
            tokens = self.encoder.forward_features(x)
            return torch.cat([tokens[:, 0], tokens[:, self.encoder.num_prefix_tokens:].mean(1)], dim=-1)
        return self.encoder(x)
