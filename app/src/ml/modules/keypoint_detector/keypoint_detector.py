import torch
import torch.nn.functional as F
from src.ml.modules.blocks import AntiAliasInterpolation2d, Hourglass
from src.ml.modules.utils.utils import gaussian2kp
from torch import Tensor, nn


class KPDetector(nn.Module):
    """Detects keypoints and (optionally) estimates a Jacobian matrix near each keypoint."""

    def __init__(
        self,
        block_expansion: int,
        num_kp: int,
        num_channels: int,
        max_features: int,
        num_blocks: int,
        temperature: float,
        estimate_jacobian: bool = False,
        scale_factor: float = 1.0,
        single_jacobian_map: bool = False,
        pad: str | int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.estimate_jacobian = estimate_jacobian
        self.num_kp = num_kp

        # Backbone feature extractor
        self.predictor = Hourglass(
            block_expansion, in_features=num_channels, max_features=max_features, num_blocks=num_blocks
        )

        # Keypoint heatmap prediction
        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=7, padding=pad)

        # Optional Jacobian estimation
        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(
                in_channels=self.predictor.out_filters,
                out_channels=4 * self.num_jacobian_maps,
                kernel_size=7,
                padding=pad,
            )
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.down = AntiAliasInterpolation2d(num_channels, scale_factor) if scale_factor != 1 else None

    def forward(self, x) -> dict[str, Tensor]:
        if self.down:
            x = self.down(x)

        feature_map = self.predictor(x)
        heatmap = self.kp(feature_map)

        B, K, H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, K, -1)
        heatmap = F.softmax(heatmap_flat / self.temperature, dim=2).view(B, K, H, W)

        out = gaussian2kp(heatmap)

        if self.jacobian:
            jacobian_map = self.jacobian(feature_map)  # shape: [B, 4*num_maps, H, W]
            jacobian_map = jacobian_map.view(B, self.num_jacobian_maps, 4, H, W)

            weighted = (heatmap.unsqueeze(2) * jacobian_map).view(B, K, 4, -1)
            jacobian = weighted.sum(dim=-1).view(B, K, 2, 2)
            out["jacobian"] = jacobian

        return out
