import torch


def kp2gaussian(kp: dict, spatial_size: tuple[int, int], kp_variance: float) -> torch.Tensor:
    """Converts keypoints to Gaussian heatmaps with optimized tensor operations."""
    mean = kp["value"]
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()).view(1, 1, *spatial_size, 2)

    # Expand dimensions for broadcasting
    mean = mean.view(*mean.shape[:2], 1, 1, 2)
    diff = coordinate_grid - mean
    return torch.exp(-0.5 * (diff.pow(2).sum(-1) / kp_variance))


def gaussian2kp(heatmap):
    """Extract the mean and from a heatmap."""
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
    value = (heatmap * grid).sum(dim=(2, 3))
    kp = {"value": value}
    return kp


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


def make_coordinate_grid(spatial_size: tuple[int, int], type) -> torch.Tensor:
    """Creates normalized grid coordinates with modern PyTorch features."""
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed
