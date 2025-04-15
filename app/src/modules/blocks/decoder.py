import torch
from src.modules.blocks.conv_blocks import UpBlock2d
from torch import nn


class Decoder(nn.Module):
    """Hourglass Decoder."""

    def __init__(self, block_expansion: int, in_features: int, num_blocks: int = 3, max_features: int = 256) -> None:
        super().__init__()

        def num_channels(scale):
            return min(max_features, block_expansion * (2**scale))

        self.up_blocks = nn.ModuleList(
            [
                UpBlock2d(
                    in_features=(1 if i == num_blocks - 1 else 2) * num_channels(i + 1), out_features=num_channels(i)
                )
                for i in reversed(range(num_blocks))
            ]
        )

        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for block in self.up_blocks:
            out = block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out
