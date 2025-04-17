import torch
from src.ml.modules.generator.vgg19 import Vgg19
from src.ml.modules.utils import ImagePyramide, Transform
from src.ml.modules.utils.utils import detach_kp
from torch import nn


class GeneratorFullModel(nn.Module):
    """Merges generator-related updates into a single model to better support multi-GPU training."""

    def __init__(self, kp_extractor, generator: nn.Module, discriminator: nn.Module, train_params: dict) -> None:
        super().__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

        self.scales = train_params["scales"]
        self.disc_scales = self.discriminator.scales
        self.loss_weights = train_params["loss_weights"]

        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        if any(self.loss_weights["perceptual"]):
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x: dict) -> tuple[dict, dict]:
        # Keypoint extraction
        kp_source = self.kp_extractor(x["source"])
        kp_driving = self.kp_extractor(x["driving"])

        generated = self.generator(x["source"], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({"kp_source": kp_source, "kp_driving": kp_driving})

        loss_values = {}

        # Pyramids
        pyramide_real = self.pyramid(x["driving"])
        pyramide_fake = self.pyramid(generated["prediction"])

        # Perceptual loss
        if any(self.loss_weights["perceptual"]):
            perceptual_loss = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_fake[f"prediction_{scale}"])
                y_vgg = self.vgg(pyramide_real[f"prediction_{scale}"])
                for i, weight in enumerate(self.loss_weights["perceptual"]):
                    perceptual_loss += weight * torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
            loss_values["perceptual"] = perceptual_loss

        # --- GAN + Feature Matching ---
        if self.loss_weights["generator_gan"] != 0:
            detached_kp_driving = detach_kp(kp_driving)
            disc_maps_fake = self.discriminator(pyramide_fake, kp=detached_kp_driving)
            disc_maps_real = self.discriminator(pyramide_real, kp=detached_kp_driving)

            value_total = 0
            for scale in self.disc_scales:
                key = f"prediction_map_{scale}"
                value = ((1 - disc_maps_fake[key]) ** 2).mean()
                value_total += self.loss_weights["generator_gan"] * value
            loss_values["gen_gan"] = value_total

            if any(self.loss_weights["feature_matching"]):
                fm_loss = 0
                for scale in self.disc_scales:
                    key = f"feature_maps_{scale}"
                    for i, (real_f, gen_f) in enumerate(zip(disc_maps_real[key], disc_maps_fake[key])):
                        weight = self.loss_weights["feature_matching"][i]
                        if weight != 0:
                            fm_loss += weight * torch.abs(real_f - gen_f).mean()
                loss_values["feature_matching"] = value_total

        # --- Equivariance losses ---
        eq_val_w = self.loss_weights["equivariance_value"]
        eq_jac_w = self.loss_weights["equivariance_jacobian"]

        if eq_val_w + eq_jac_w != 0:
            transform = Transform(x["driving"].shape[0], **self.train_params["transform_params"])
            transformed_frame = transform.transform_frame(x["driving"])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated.update({"transformed_frame": transformed_frame, "transformed_kp": transformed_kp})

            # Value loss part
            if eq_val_w != 0:
                warped = transform.warp_coordinates(transformed_kp["value"])
                val_loss = torch.abs(kp_driving["value"] - warped).mean()
                loss_values["equivariance_value"] = eq_val_w * val_loss

            # jacobian loss part
            if eq_jac_w != 0:
                jacobian_transformed = torch.matmul(
                    transform.jacobian(transformed_kp["value"]), transformed_kp["jacobian"]
                )
                normed = torch.matmul(torch.inverse(kp_driving["jacobian"]), jacobian_transformed)
                identity = torch.eye(2).view(1, 1, 2, 2).type(normed.type())
                jac_loss = torch.abs(identity - normed).mean()
                loss_values["equivariance_jacobian"] = eq_jac_w * jac_loss

        return loss_values, generated
