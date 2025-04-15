import torch
import torch.nn.functional as F


class AntiAliasInterpolation2d(torch.nn.Module):
    """Band-limited downsampling with antialiasing for better signal preservation.
    Args:
        channels: Number of input channels
        scale: Scaling factor (must be <= 1.0 for downsampling)
    """

    def __init__(self, channels: int, scale: float) -> None:
        super().__init__()
        if scale > 1.0:
            raise ValueError("Scale factor must be <= 1.0 for anti-aliased downsampling")

        self.scale = scale
        self.groups = channels

        if scale == 1.0:
            return  # Identity operation

        # Calculate optimal Gaussian kernel
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = kernel_size - self.ka - 1  # More efficient padding calculation

        # Vectorized kernel creation
        grid = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize

        # Register as buffer for proper device handling
        self.register_buffer(
            "weight", kernel.view(1, 1, kernel_size, kernel_size).expand(channels, -1, -1, -1).contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1.0:
            return x

        x = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        x = F.conv2d(x, weight=self.weight, groups=self.groups)
        x = F.interpolate(x, scale_factor=(self.scale, self.scale))

        return x
