import torch
from src.modules.blocks import AntiAliasInterpolation2d
from torch import nn


class ImagePyramide(torch.nn.Module):
    """Create image pyramide for computing pyramide perceptual loss. See Sec 3.3."""

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace(".", "-")] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict["prediction_" + str(scale).replace("-", ".")] = down_module(x)
        return out_dict
