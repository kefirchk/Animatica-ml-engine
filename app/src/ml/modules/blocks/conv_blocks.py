import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d


class ResBlock2d(nn.Module):
    """Res block, preserve spatial resolution."""

    def __init__(self, in_features: int, kernel_size: int | tuple[int, int], padding: int | tuple[int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = F.relu(self.norm1(x))
        out = F.relu(self.norm2(self.conv1(out)))
        return self.conv2(out) + x


class UpBlock2d(nn.Module):
    """Upsampling block for use in decoder."""

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        return F.relu(self.norm(self.conv(out)))


class DownBlock2d(nn.Module):
    """Downsampling block for use in encoder."""

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        return self.pool(F.relu(self.norm(self.conv(x))))


class SameBlock2d(nn.Module):
    """Simple block, preserve spatial resolution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int = 1,
        kernel_size: int | tuple[int, int] = 3,
        padding: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))
