import torch
from src.modules.discriminator.blocks import DownBlock2d
from src.modules.utils.utils import kp2gaussian
from torch import nn


class Discriminator(nn.Module):
    """Discriminator similar to Pix2Pix."""

    def __init__(
        self,
        num_channels: int = 3,
        block_expansion: int = 64,
        num_blocks: int = 4,
        max_features: int = 512,
        sn: bool = False,
        use_kp: bool = False,
        num_kp: int = 10,
        kp_variance: float = 0.01,
        **_
    ) -> None:
        super().__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2**i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    norm=(i != 0),
                    kernel_size=4,
                    pool=(i != num_blocks - 1),
                    sn=sn,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        self.conv = nn.utils.spectral_norm(self.conv) if sn else self.conv
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = torch.cat([out, heatmap], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map
