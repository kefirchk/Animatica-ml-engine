import torch


def kp2gaussian(kp, spatial_size, kp_variance):
    """Transform a keypoint into gaussian like representation."""
    mean = kp["value"]

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = coordinate_grid - mean

    out = torch.exp(-0.5 * (mean_sub**2).sum(-1) / kp_variance)

    return out


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


def make_coordinate_grid(spatial_size, type):
    """Create a meshgrid [-1,1] x [-1,1] of given spatial_size."""
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed
