import os

import numpy as np
import torch
from scipy.spatial import ConvexHull
from src.datasets.frames_dataset import PairedDataset
from src.services.logging import LoggingService
from src.services.sync_batchnorm import DataParallelWithCallback
from src.services.visualization import VisualizationService
from torch.utils.data import DataLoader
from tqdm import tqdm


class AnimationService:
    @staticmethod
    def make_animation(
        source_image,
        driving_video,
        generator,
        kp_detector,
        relative: bool = True,
        adapt_movement_scale: bool = True,
        cpu: bool = False,
    ):
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = AnimationService.normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative,
                    use_relative_jacobian=relative,
                    adapt_movement_scale=adapt_movement_scale,
                )
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

    @staticmethod
    def animate(config, generator, kp_detector, checkpoint, log_dir, dataset, imageio=None):
        log_dir = os.path.join(log_dir, "services")
        png_dir = os.path.join(log_dir, "png")
        animate_params = config["animate_params"]

        dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params["num_pairs"])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        if not checkpoint:
            raise AttributeError("Checkpoint should be specified for mode='animate'.")

        LoggingService.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        if torch.cuda.is_available():
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        for it, x in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                predictions = []
                visualizations = []

                driving_video = x["driving_video"]
                source_frame = x["source_video"][:, :, 0, :, :]

                kp_source = kp_detector(source_frame)
                kp_driving_initial = kp_detector(driving_video[:, :, 0])

                for frame_idx in range(driving_video.shape[2]):
                    driving_frame = driving_video[:, :, frame_idx]
                    kp_driving = kp_detector(driving_frame)
                    kp_norm = AnimationService.normalize_kp(
                        kp_source=kp_source,
                        kp_driving=kp_driving,
                        kp_driving_initial=kp_driving_initial,
                        **animate_params["normalization_params"]
                    )
                    out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                    out["kp_driving"] = kp_driving
                    out["kp_source"] = kp_source
                    out["kp_norm"] = kp_norm

                    del out["sparse_deformed"]

                    predictions.append(np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0])

                    visualization = VisualizationService(**config["visualizer_params"]).visualize(
                        source=source_frame, driving=driving_frame, out=out
                    )
                    visualization = visualization
                    visualizations.append(visualization)

                predictions = np.concatenate(predictions, axis=1)
                result_name = "-".join([x["driving_name"][0], x["source_name"][0]])
                imageio.imsave(os.path.join(png_dir, result_name + ".png"), (255 * predictions).astype(np.uint8))

                image_name = result_name + animate_params["format"]
                imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    @staticmethod
    def normalize_kp(
        kp_source,
        kp_driving,
        kp_driving_initial,
        adapt_movement_scale: bool = False,
        use_relative_movement: bool = False,
        use_relative_jacobian: bool = False,
    ):
        if adapt_movement_scale:
            source_area = ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
            driving_area = ConvexHull(kp_driving_initial["value"][0].data.cpu().numpy()).volume
            adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
        else:
            adapt_movement_scale = 1

        kp_new = {k: v for k, v in kp_driving.items()}

        if use_relative_movement:
            kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
            kp_value_diff *= adapt_movement_scale
            kp_new["value"] = kp_value_diff + kp_source["value"]

            if use_relative_jacobian:
                jacobian_diff = torch.matmul(kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"]))
                kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

        return kp_new
