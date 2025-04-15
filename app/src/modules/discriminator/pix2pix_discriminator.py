import torch
from src.modules.discriminator.blocks import DownBlock2d
from src.modules.utils import kp2gaussian
from torch import nn


class Discriminator(nn.Module):
    """Patch-based discriminator similar to Pix2Pix, with optional keypoint heatmap input."""

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
        self.use_kp = use_kp
        self.kp_variance = kp_variance

        self.down_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.down_blocks.append(
                DownBlock2d(
                    in_features=(
                        num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2**i))
                    ),
                    out_features=min(max_features, block_expansion * (2 ** (i + 1))),
                    norm=(i != 0),
                    kernel_size=4,
                    pool=(i != num_blocks - 1),
                    sn=sn,
                )
            )

        conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        self.conv = nn.utils.spectral_norm(conv) if sn else conv

    def forward(self, x: torch.Tensor, kp: dict | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        Returns:
            - List of intermediate feature maps (for multiscale loss).
            - Final prediction map.
        """
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            x = torch.cat([x, heatmap], dim=1)

        feature_maps = []
        for down_block in self.down_blocks:
            x = down_block(x)
            feature_maps.append(x)

        prediction_map = self.conv(x)

        return feature_maps, prediction_map
