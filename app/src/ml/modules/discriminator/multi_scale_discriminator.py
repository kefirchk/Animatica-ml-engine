import torch
from src.ml.modules.discriminator.pix2pix_discriminator import Discriminator
from torch import nn


class MultiScaleDiscriminator(nn.Module):
    """Discriminator that processes inputs at multiple spatial scales."""

    def __init__(self, scales=(), **kwargs) -> None:
        super().__init__()
        self.scales = scales

        discs = {}
        for scale in scales:
            discs[str(scale).replace(".", "-")] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x: dict, kp=None) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        out_dict = {}

        for scale, disc in self.discs.items():
            scale = str(scale).replace("-", ".")
            key = f"prediction_{scale}"

            feature_maps, prediction_map = disc(x[key], kp)

            out_dict[f"feature_maps_{scale}"] = feature_maps
            out_dict[f"prediction_map_{scale}"] = prediction_map

        return out_dict
