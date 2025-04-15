import torch
import torch.nn.functional as F
from torch import nn


class DownBlock2d(nn.Module):
    """Simple block for processing video (encoder).
    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
        norm (bool): Whether to use InstanceNorm2d.
        kernel_size (int): Size of the convolutional kernel.
        pool (bool): Whether to apply average pooling.
        sn (bool): Whether to apply spectral normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: bool = False,
        kernel_size: int = 4,
        pool: bool = False,
        sn: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)
        self.conv = nn.utils.spectral_norm(self.conv) if sn else self.conv
        self.norm: nn.Module = nn.InstanceNorm2d(out_features, affine=True) if norm else nn.Identity()
        self.pool: nn.Module = nn.AvgPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.leaky_relu(x, 0.2)
        x = self.pool(x)
        return x
