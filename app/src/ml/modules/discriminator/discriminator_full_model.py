import torch
from src.ml.modules.utils import ImagePyramide
from src.ml.modules.utils.utils import detach_kp
from torch import nn


class DiscriminatorFullModel(torch.nn.Module):
    """Wraps the discriminator forward pass and GAN loss calculation for improved multi-GPU usage."""

    def __init__(self, kp_extractor, generator: nn.Module, discriminator: nn.Module, train_params: dict) -> None:
        super().__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

        self.scales = self.discriminator.scales
        self.loss_weights = train_params["loss_weights"]

        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

    def forward(self, x: dict, generated: dict) -> dict[str, float]:
        # Extract pyramids for real and generated frames
        pyramide_real = self.pyramid(x["driving"])
        pyramide_fake = self.pyramid(generated["prediction"].detach())

        # Detach keypoints to stop gradient to generator
        kp_driving_detached = detach_kp(generated["kp_driving"])

        # Run discriminator
        disc_fake = self.discriminator(pyramide_fake, kp=kp_driving_detached)
        disc_real = self.discriminator(pyramide_real, kp=kp_driving_detached)

        # Compute GAN loss (LSGAN formulation)
        disc_loss_total = 0.0
        for scale in self.scales:
            key = f"prediction_map_{scale}"
            real_map = disc_real[key]
            fake_map = disc_fake[key]
            scale_loss = (1 - real_map) ** 2 + fake_map**2
            disc_loss_total += self.loss_weights["discriminator_gan"] * scale_loss.mean()

        return {"disc_gan": disc_loss_total}
