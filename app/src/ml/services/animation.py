import os

import numpy as np
import torch
from scipy.spatial import ConvexHull
from src.ml.datasets import PairedDataset
from src.ml.services.logging import LoggingService
from src.ml.services.sync_batchnorm import DataParallelWithCallback
from src.ml.services.visualization import VisualizationService
from torch.utils.data import DataLoader
from tqdm import tqdm


class AnimationService:
    """Service for generating animations using keypoint detection and image generation."""

    @classmethod
    def make_animation(
        cls,
        source_image: np.ndarray,
        driving_video: list[np.ndarray],
        generator: torch.nn.Module,
        kp_detector: torch.nn.Module,
        relative: bool = True,
        adapt_movement_scale: bool = True,
        cpu: bool = False,
    ) -> list[np.ndarray]:
        """
        Generate fomm from source image and driving video.
        Args:
            source_image: Source image (H, W, C)
            driving_video: Driving video frames (T, H, W, C)
            generator: Image generator model
            kp_detector: Keypoint detector model
            relative: Use relative movement
            adapt_movement_scale: Adapt movement scale
            cpu: Force CPU execution
        Returns:
            List of predicted frames
        """
        device = torch.device("cpu") if cpu else torch.device("cuda")

        with torch.no_grad():
            # Convert inputs to tensors
            source = cls._image_to_tensor(source_image).to(device)
            driving = cls._video_to_tensor(driving_video).to(device)

            # Get keypoints
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            # Process each frame
            predictions = []
            for frame_idx in tqdm(range(driving.shape[2]), desc="Generating fomm"):
                driving_frame = driving[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)

                kp_norm = cls.normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative,
                    use_relative_jacobian=relative,
                    adapt_movement_scale=adapt_movement_scale,
                )

                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                predictions.append(cls._tensor_to_image(out["prediction"]))

        return predictions

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        return torch.from_numpy(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image."""
        return np.transpose(tensor.data.cpu().numpy(), [0, 2, 3, 1])[0]

    @staticmethod
    def _video_to_tensor(video: list[np.ndarray]) -> torch.Tensor:
        """Convert numpy video to tensor."""
        return torch.from_numpy(np.array(video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

    @staticmethod
    def animate(
        config: dict,
        generator: torch.nn.Module,
        kp_detector: torch.nn.Module,
        checkpoint: str,
        log_dir: str,
        dataset: PairedDataset,
        imageio=None,
    ) -> None:
        """
        Main fomm pipeline with visualization and saving.
        Args:
            config: Configuration dictionary
            generator: Image generator model
            kp_detector: Keypoint detector model
            checkpoint: Path to model checkpoint
            log_dir: Directory to save outputs
            dataset: Dataset for fomm
            imageio: ImageIO instance for saving
        """
        # Setup directories
        log_dir = os.path.join(log_dir, "services")
        png_dir = os.path.join(log_dir, "png")
        os.makedirs(png_dir, exist_ok=True)

        # Load checkpoint
        if not checkpoint:
            raise ValueError("Checkpoint must be specified for fomm mode")
        LoggingService.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)

        # Prepare models
        if torch.cuda.is_available():
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        # Create dataset and dataloader
        animate_params = config["animate_params"]
        dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params["num_pairs"])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        # Initialize visualizer
        visualizer = VisualizationService(**config["visualizer_params"])

        # Process each pair
        for x in tqdm(dataloader, desc="Animating dataset"):
            with torch.no_grad():
                predictions, visualizations = [], []
                driving_video = x["driving_video"]
                source_frame = x["source_video"][:, :, 0, :, :]

                # Get keypoints
                kp_source = kp_detector(source_frame)
                kp_driving_initial = kp_detector(driving_video[:, :, 0])

                # Process each frame
                for frame_idx in range(driving_video.shape[2]):
                    driving_frame = driving_video[:, :, frame_idx]
                    kp_driving = kp_detector(driving_frame)

                    kp_norm = AnimationService.normalize_kp(
                        kp_source=kp_source,
                        kp_driving=kp_driving,
                        kp_driving_initial=kp_driving_initial,
                        **animate_params["normalization_params"],
                    )

                    out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                    # Prepare outputs
                    out.update({"kp_driving": kp_driving, "kp_source": kp_source, "kp_norm": kp_norm})
                    del out["sparse_deformed"]

                    # Save results
                    predictions.append(AnimationService._tensor_to_image(out["prediction"]))
                    visualizations.append(visualizer.visualize(source=source_frame, driving=driving_frame, out=out))

                # Save outputs
                result_name = f"{x['driving_name'][0]}-{x['source_name'][0]}"
                predictions = np.concatenate(predictions, axis=1)

                imageio.imsave(
                    os.path.join(png_dir, f"{result_name}.png"),
                    (255 * predictions).astype(np.uint8),
                )

                imageio.mimsave(
                    os.path.join(log_dir, f"{result_name}{animate_params['format']}"),
                    visualizations,
                )

    @classmethod
    def normalize_kp(
        cls,
        kp_source: dict[str, torch.Tensor],
        kp_driving: dict[str, torch.Tensor],
        kp_driving_initial: dict[str, torch.Tensor],
        adapt_movement_scale: bool = False,
        use_relative_movement: bool = False,
        use_relative_jacobian: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Normalize keypoints between source and driving frames.
        Args:
            kp_source: Source keypoints
            kp_driving: Driving keypoints
            kp_driving_initial: Initial driving keypoints
            adapt_movement_scale: Adjust for scale differences
            use_relative_movement: Use relative movement
            use_relative_jacobian: Use relative jacobian
        Returns:
            Normalized keypoints
        """
        kp_new = kp_driving.copy()

        if adapt_movement_scale:
            scale = cls._calculate_movement_scale(kp_source, kp_driving_initial)
        else:
            scale = 1.0

        if use_relative_movement:
            kp_value_diff = (kp_driving["value"] - kp_driving_initial["value"]) * scale
            kp_new["value"] = kp_value_diff + kp_source["value"]

            if use_relative_jacobian:
                jacobian_diff = torch.matmul(kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"]))
                kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

        return kp_new

    @staticmethod
    def _calculate_movement_scale(
        kp_source: dict[str, torch.Tensor], kp_driving_initial: dict[str, torch.Tensor]
    ) -> float:
        """Calculate movement scale based on convex hull areas."""
        source_area = ConvexHull(kp_source["value"][0].cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial["value"][0].cpu().numpy()).volume
        return np.sqrt(source_area) / np.sqrt(driving_area)
