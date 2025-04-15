import torch
from src.modules.blocks import AntiAliasInterpolation2d
from torch import nn


class ImagePyramide(torch.nn.Module):
    """Creates an image pyramid for computing pyramidal perceptual loss.
    Args:
        scales: List of scale factors for the pyramid (e.g., [1.0, 0.5, 0.25])
        num_channels: Number of channels in input images
    """

    def __init__(self, scales: list[float], num_channels: int) -> None:
        super().__init__()

        # Validate input scales
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("Scales must be positive values")

        # Create downsampling modules
        self.downs = nn.ModuleDict(
            {f"scale_{scale}": AntiAliasInterpolation2d(num_channels, scale) for scale in scales}
        )

        # Cache scale names for faster forward pass
        self.scale_names = list(self.downs.keys())
        self.output_keys = [f'prediction_{scale.split("_")[1]}' for scale in self.scale_names]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Process input image through the pyramid.
        Returns:
            Dictionary with keys like 'prediction_1.0', 'prediction_0.5' etc.
        """
        return {key: down_module(x) for key, down_module in zip(self.output_keys, self.downs.values())}
