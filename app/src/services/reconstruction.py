import os

import imageio
import numpy as np
import torch
from src.services.logging import LoggingService
from src.services.sync_batchnorm import DataParallelWithCallback
from src.services.visualization import VisualizationService
from torch.utils.data import DataLoader
from tqdm import tqdm

log = LoggingService.setup_logger(__name__)


class ReconstructionService:
    @staticmethod
    def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
        png_dir = os.path.join(log_dir, "reconstruction/png")
        log_dir = os.path.join(log_dir, "reconstruction")

        if not checkpoint:
            raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
        LoggingService.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        loss_list = []
        if torch.cuda.is_available():
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        for it, x in tqdm(enumerate(dataloader)):
            if config["reconstruction_params"]["num_videos"] is not None:
                if it > config["reconstruction_params"]["num_videos"]:
                    break
            with torch.no_grad():
                predictions = []
                visualizations = []
                if torch.cuda.is_available():
                    x["video"] = x["video"].cuda()

                kp_source = kp_detector(x["video"][:, :, 0])
                for frame_idx in range(x["video"].shape[2]):
                    source = x["video"][:, :, 0]
                    driving = x["video"][:, :, frame_idx]
                    kp_driving = kp_detector(driving)
                    out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                    out["kp_source"] = kp_source
                    out["kp_driving"] = kp_driving
                    del out["sparse_deformed"]
                    predictions.append(np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0])

                    visualization = VisualizationService(**config["visualizer_params"]).visualize(
                        source=source, driving=driving, out=out
                    )
                    visualizations.append(visualization)

                    loss_list.append(torch.abs(out["prediction"] - driving).mean().cpu().numpy())

                predictions = np.concatenate(predictions, axis=1)
                imageio.imsave(os.path.join(png_dir, x["name"][0] + ".png"), (255 * predictions).astype(np.uint8))

                image_name = x["name"][0] + config["reconstruction_params"]["format"]
                imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

        log.info("Reconstruction loss:", np.mean(loss_list))
