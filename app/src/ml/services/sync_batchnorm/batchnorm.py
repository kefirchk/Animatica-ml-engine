import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class _SynchronizedBatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or not dist.is_initialized():
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )

        # Flatten input to (N, C, *) and compute stats
        N, C = input.size(0), self.num_features
        input_reshaped = input.contiguous().view(N, C, -1)
        mean = input_reshaped.mean(dim=[0, 2])
        var = input_reshaped.var(dim=[0, 2], unbiased=False)

        # All-reduce across GPUs
        world_size = dist.get_world_size()
        mean_sum = mean.clone()
        var_sum = var.clone()
        dist.all_reduce(mean_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(var_sum, op=dist.ReduceOp.SUM)

        mean = mean_sum / world_size
        var = var_sum / world_size

        # Update running stats
        self.running_mean = self.running_mean.to(mean.device) * (1 - self.momentum) + mean.detach() * self.momentum
        self.running_var = self.running_var.to(var.device) * (1 - self.momentum) + var.detach() * self.momentum

        inv_std = torch.rsqrt(var + self.eps)
        x_hat = (input - mean[None, :, *([None] * (input.dim() - 2))]) * inv_std[None, :, *([None] * (input.dim() - 2))]

        if self.affine:
            x_hat = (
                x_hat * self.weight[None, :, *([None] * (input.dim() - 2))]
                + self.bias[None, :, *([None] * (input.dim() - 2))]
            )
        return x_hat


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    """Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a mini-batch."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D input (got {input.dim()}D input)")
        return super().forward(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch of 3d inputs."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input (got {input.dim()}D)")
        return super().forward(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 5d input that is seen as a mini-batch of 4d inputs."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 5:
            raise ValueError(f"Expected 5D input (got {input.dim()}D)")
        return super().forward(input)
