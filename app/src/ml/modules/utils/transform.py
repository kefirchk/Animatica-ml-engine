import torch
import torch.nn.functional as F
from src.ml.modules.utils.utils import make_coordinate_grid
from torch.autograd import grad


class Transform:
    """Thin Plate Spline (TPS) transformation for equivariance constraints with both affine and non-linear components.
    Args:
        bs: Batch size
        sigma_affine: Standard deviation for affine transformation parameters
        sigma_tps: Standard deviation for TPS control point parameters (optional)
        points_tps: Number of control points along each axis (optional)
    """

    def __init__(self, bs: int, **kwargs) -> None:
        self.bs = bs

        # Initialize random affine transformation parameters
        noise = torch.normal(mean=0, std=kwargs["sigma_affine"] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)

        # Initialize TPS transformation if parameters provided
        self.tps = ("sigma_tps" in kwargs) and ("points_tps" in kwargs)
        if self.tps:
            self.control_points = make_coordinate_grid((kwargs["points_tps"], kwargs["points_tps"]), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(
                mean=0, std=kwargs["sigma_tps"] * torch.ones([bs, 1, kwargs["points_tps"] ** 2])
            )

    def transform_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply transformation to input frame.
        Args:
            frame: Input tensor of shape [bs, C, H, W]
        Returns:
            Transformed frame of same shape
        """
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        warped_grid = self.warp_coordinates(grid).view(self.bs, *frame.shape[2:], 2)
        return F.grid_sample(frame, warped_grid, padding_mode="reflection", align_corners=True)

    def warp_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Apply both affine and TPS transformations to coordinates.
        Args:
            coordinates: Tensor of shape [bs, N, 2]
        Returns:
            Transformed coordinates of same shape
        """
        # Apply affine transformation
        transformed = self._apply_affine_transform(coordinates)

        # Apply TPS transformation if enabled
        if self.tps:
            transformed = self._apply_tps_transform(transformed, coordinates)

        return transformed

    def _apply_affine_transform(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation to coordinates."""
        # Efficient affine transform using einsum
        return torch.einsum(
            "bni,bij->bnj",
            torch.cat([coordinates, torch.ones_like(coordinates[..., :1])], dim=-1),
            self.theta.type_as(coordinates),
        )

    def _apply_tps_transform(self, transformed: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Apply Thin Plate Spline non-linear transformation."""
        # Calculate distances between coordinates and control points
        diff = coordinates.unsqueeze(2) - self.control_points.type_as(coordinates).view(1, 1, -1, 2)
        distances = torch.norm(diff, p=1, dim=-1)  # L1 distance

        # Compute TPS radial basis function
        result = distances**2 * torch.log(distances + 1e-6)
        result = (result * self.control_params.type_as(coordinates)).sum(dim=2)

        return transformed + result.unsqueeze(-1)

    def jacobian(self, coordinates: torch.Tensor) -> torch.Tensor:
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian
