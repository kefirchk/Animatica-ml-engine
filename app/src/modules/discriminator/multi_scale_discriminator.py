from src.modules.discriminator.pix2pix_discriminator import Discriminator
from torch import nn


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale (scale) discriminator."""

    def __init__(self, scales=(), **kwargs) -> None:
        super().__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace(".", "-")] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace("-", ".")
            key = "prediction_" + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict["feature_maps_" + scale] = feature_maps
            out_dict["prediction_map_" + scale] = prediction_map
        return out_dict
