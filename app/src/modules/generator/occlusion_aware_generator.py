import torch
import torch.nn.functional as F
from src.modules.blocks import DownBlock2d, ResBlock2d, SameBlock2d, UpBlock2d
from src.modules.dense_motion import DenseMotionNetwork
from torch import nn


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that transforms source image according to movement trajectories induced by keypoints.
    Follows the Johnson architecture.
    """

    def __init__(
        self,
        num_channels: int,
        num_kp: int,
        block_expansion: int,
        max_features: int,
        num_down_blocks: int,
        num_bottleneck_blocks: int,
        estimate_occlusion_map: bool = False,
        dense_motion_params: dict = None,
        estimate_jacobian: bool = False,
    ) -> None:
        super().__init__()

        # Dense Motion Network setup
        self.dense_motion_network = (
            DenseMotionNetwork(
                num_kp=num_kp,
                num_channels=num_channels,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
            )
            if dense_motion_params
            else None
        )

        # First block: SameBlock2d
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        # Downsampling blocks
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2d(
                    min(max_features, block_expansion * (2**i)), min(max_features, block_expansion * (2 ** (i + 1)))
                )
                for i in range(num_down_blocks)
            ]
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList(
            [
                UpBlock2d(
                    min(max_features, block_expansion * (2 ** (num_down_blocks - i))),
                    min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1))),
                )
                for i in range(num_down_blocks)
            ]
        )

        # Bottleneck ResBlocks
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2**num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module("r" + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        # Final output layer
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

        # Store options for occlusion map and channels
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    @staticmethod
    def deform_input(inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode="bilinear")
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        output_dict = {}

        # Transforming feature representation according to deformation and occlusion
        if self.dense_motion_network:
            dense_motion = self.dense_motion_network(
                source_image=source_image, kp_driving=kp_driving, kp_source=kp_source
            )
            output_dict.update(
                {
                    "mask": dense_motion["mask"],
                    "sparse_deformed": dense_motion["sparse_deformed"],
                }
            )

            if "occlusion_map" in dense_motion:
                occlusion_map = dense_motion["occlusion_map"]
                output_dict["occlusion_map"] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion["deformation"]
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2:] != occlusion_map.shape[2:]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode="bilinear")
                out = out * occlusion_map

            output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out = self.bottleneck(out)
        for up_block in self.up_blocks:
            out = up_block(out)

        # Final prediction and sigmoid activation
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
