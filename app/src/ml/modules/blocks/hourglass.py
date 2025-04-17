from src.ml.modules.blocks.decoder import Decoder
from src.ml.modules.blocks.encoder import Encoder
from torch import nn


class Hourglass(nn.Module):
    """Hourglass architecture."""

    def __init__(self, block_expansion: int, in_features: int, num_blocks: int = 3, max_features: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))
