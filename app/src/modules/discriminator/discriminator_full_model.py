import torch
from src.modules.utils.image_pyramide import ImagePyramide
from src.modules.utils.utils import detach_kp


class DiscriminatorFullModel(torch.nn.Module):
    """Merge all discriminator related updates into single model for better multi-gpu usage."""

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params["loss_weights"]

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x["driving"])
        pyramide_generated = self.pyramid(generated["prediction"].detach())

        kp_driving = generated["kp_driving"]
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = "prediction_map_%s" % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights["discriminator_gan"] * value.mean()
        loss_values["disc_gan"] = value_total

        return loss_values
