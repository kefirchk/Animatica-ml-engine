import torch
import torch.nn.functional as F
from src.modules.blocks import AntiAliasInterpolation2d, Hourglass
from src.modules.utils.utils import kp2gaussian, make_coordinate_grid
from torch import nn


class DenseMotionNetwork(nn.Module):
    """Predicts dense motion from sparse keypoint representations using hourglass architecture."""

    def __init__(
        self,
        block_expansion: int,
        num_blocks: int,
        max_features: int,
        num_kp: int,
        num_channels: int,
        estimate_occlusion_map: bool = False,
        scale_factor: int = 1,
        kp_variance: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        # Input feature calculation (K+1)*(C+1) for heatmaps and deformed features
        in_features = (num_kp + 1) * (num_channels + 1)

        # Core components
        self.hourglass = Hourglass(block_expansion, in_features, num_blocks, max_features)

        # Output layers
        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)
        self.occlusion = (
            nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=7, padding=3) if estimate_occlusion_map else None
        )

        # Preprocessing
        self.down = AntiAliasInterpolation2d(num_channels, scale_factor) if scale_factor != 1 else None

    def create_heatmap_representations(
        self, source_image: torch.Tensor, kp_driving: dict, kp_source: dict
    ) -> torch.Tensor:
        """Generates heatmap representations for keypoint differences."""
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)

        # Add background channel
        heatmaps = torch.cat(
            [torch.zeros_like(gaussian_driving[:, :1]), gaussian_driving - gaussian_source], dim=1  # Background
        )

        return heatmaps.unsqueeze(2)  # Add dimension for concatenation

    def create_sparse_motions(self, source_image: torch.Tensor, kp_driving: dict, kp_source: dict) -> torch.Tensor:
        """Computes sparse motion transformations between keypoints."""
        bs, _, h, w = source_image.shape
        device = source_image.device

        # Create base coordinate grid
        identity_grid = make_coordinate_grid((h, w), kp_source["value"].type()).to(device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)

        # Calculate driving to source transformation
        coordinate_grid = identity_grid - kp_driving["value"].view(bs, self.num_kp, 1, 1, 2)

        # Apply Jacobian transformation if available
        if "jacobian" in kp_driving:
            jacobian = torch.matmul(kp_source["jacobian"], torch.inverse(kp_driving["jacobian"]))
            jacobian = jacobian.view(bs, self.num_kp, 1, 1, 2, 2)
            coordinate_grid = torch.einsum("bkhwij,bkhwj->bkhwi", jacobian, coordinate_grid)

        driving_to_source = coordinate_grid + kp_source["value"].view(bs, self.num_kp, 1, 1, 2)

        # Combine with identity grid for background
        return torch.cat([identity_grid.expand(bs, -1, h, w, 2), driving_to_source], dim=1)

    def _compute_deformation(self, sparse_motions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes final deformation field using predicted masks."""
        return torch.einsum("bkhw,bkhwc->bhwc", mask, sparse_motions.permute(0, 1, 4, 2, 3))

    def create_deformed_source_image(self, source_image: torch.Tensor, sparse_motions: torch.Tensor) -> torch.Tensor:
        """Warps source image using computed sparse motions."""
        bs, _, h, w = source_image.shape

        # Prepare source image for warping
        source_expanded = source_image.unsqueeze(1).expand(-1, self.num_kp + 1, -1, -1, -1)
        source_expanded = source_expanded.reshape(bs * (self.num_kp + 1), -1, h, w)

        # Perform grid sampling
        motions_flat = sparse_motions.view(bs * (self.num_kp + 1), h, w, 2)
        deformed = F.grid_sample(source_expanded, motions_flat, align_corners=True)

        return deformed.view(bs, self.num_kp + 1, -1, h, w)

    def forward(self, source_image: torch.Tensor, kp_driving: dict, kp_source: dict) -> dict[str, torch.Tensor]:
        # Preprocess input
        if self.down:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape
        out_dict = dict()

        # Compute motion components
        heatmaps = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motions = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motions)
        out_dict["sparse_deformed"] = deformed_source

        hourglass_input = torch.cat([heatmaps, deformed_source], dim=2)
        hourglass_input = hourglass_input.view(bs, -1, h, w)
        prediction = self.hourglass(hourglass_input)
        mask = F.softmax(self.mask(prediction), dim=1)
        out_dict["mask"] = mask

        mask = mask.unsqueeze(2)
        sparse_motions = sparse_motions.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motions * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        out_dict["deformation"] = deformation

        # Add occlusion map if needed
        if self.occlusion:
            out_dict["occlusion_map"] = torch.sigmoid(self.occlusion(prediction))

        return out_dict
