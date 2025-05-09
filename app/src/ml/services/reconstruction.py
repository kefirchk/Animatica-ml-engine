import os

import cv2
import numpy as np
import torch
from src.ml.datasets.replicate import DataParallelWithCallback
from src.ml.services.logging import LoggingService
from src.ml.services.visualization import VisualizationService
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

                # Save concatenated predictions as PNG using OpenCV
                image = (255 * predictions).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(png_dir, x["name"][0] + ".png"), image)

                image_name = x["name"][0] + config["reconstruction_params"]["format"]

                # Save visualizations as video using OpenCV
                if visualizations:
                    fps = config["reconstruction_params"].get("fps", 30)
                    frame_height, frame_width = visualizations[0].shape[:2]

                    # Determine FourCC based on file extension
                    file_ext = config["reconstruction_params"]["format"].lower()
                    fourcc = {".mp4": "mp4v", ".avi": "XVID", ".mov": "MJPG"}.get(file_ext, "mp4v")
                    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                    video_path = os.path.join(log_dir, image_name)
                    video_writer = cv2.VideoWriter(video_path, fourcc_code, fps, (frame_width, frame_height))
                    for frame in visualizations:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                    video_writer.release()

        log.info("Reconstruction loss:", np.mean(loss_list))
