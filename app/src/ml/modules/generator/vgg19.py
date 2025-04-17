import numpy as np
import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    """VGG19 network for perceptual loss as described in Sec 3.3."""

    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        vgg_features = models.vgg19(pretrained=True).features

        # Define VGG slices corresponding to relu1_2, relu2_2, relu3_2, relu4_2, relu5_2
        self.slices = torch.nn.ModuleList(
            [
                vgg_features[:2],  # relu1_2
                vgg_features[2:7],  # relu2_2
                vgg_features[7:12],  # relu3_2
                vgg_features[12:21],  # relu4_2
                vgg_features[21:30],  # relu5_2
            ]
        )

        # Normalization buffers (not learnable, no grad)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features
