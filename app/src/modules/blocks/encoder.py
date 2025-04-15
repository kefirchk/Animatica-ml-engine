from src.modules.blocks.conv_blocks import DownBlock2d
from torch import nn


class Encoder(nn.Module):
    """Hourglass Encoder."""

    def __init__(self, block_expansion: int, in_features: int, num_blocks: int = 3, max_features: int = 256):
        super(Encoder, self).__init__()

        def num_channels(scale):
            return min(max_features, block_expansion * (2**scale))

        self.down_blocks = nn.ModuleList(
            [
                DownBlock2d(in_features=in_features if i == 0 else num_channels(i), out_features=num_channels(i + 1))
                for i in range(num_blocks)
            ]
        )

    def forward(self, x):
        outs = [x]
        for block in self.down_blocks:
            outs.append(block(outs[-1]))
        return outs
