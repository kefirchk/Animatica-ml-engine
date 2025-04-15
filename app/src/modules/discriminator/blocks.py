import torch.nn.functional as F
from torch import nn


class DownBlock2d(nn.Module):
    """Simple block for processing video (encoder)."""

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)
        self.conv = nn.utils.spectral_norm(self.conv) if sn else self.conv

        self.norm = nn.InstanceNorm2d(out_features, affine=True) if norm else None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out
