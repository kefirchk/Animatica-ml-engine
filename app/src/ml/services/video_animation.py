from pathlib import Path

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from src.ml.services.animation import AnimationService
from src.ml.services.logging import LoggingService
from src.ml.services.model import ModelService
from src.ml.services.utils import find_best_frame, load_video, preprocess_image


class VideoAnimationService:
    """Service for generating animated videos from source images and driving videos."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        source_image_path: str,
        driving_video_path: str,
        result_video_path: str,
        relative: bool = False,
        adapt_scale: bool = False,
        find_best: bool = False,
        best_frame: int | None = None,
        cpu: bool = False,
    ) -> None:
        """Initialize video fomm service.
        Args:
            config_path: Path to model configuration
            checkpoint_path: Path to model checkpoint
            source_image_path: Path to source image
            driving_video_path: Path to driving video
            result_video_path: Output video path
            relative: Use relative keypoint movement
            adapt_scale: Adapt movement scale
            find_best: Find best frame automatically
            best_frame: Specify best frame index
            cpu: Force CPU execution
        """
        self.log = LoggingService.setup_logger(__name__)
        self._validate_paths(source_image_path, driving_video_path)

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.source_image_path = source_image_path
        self.driving_video_path = driving_video_path
        self.result_video_path = result_video_path
        self.relative = relative
        self.adapt_scale = adapt_scale
        self.find_best = find_best
        self.best_frame = best_frame
        self.cpu = cpu

        # Initialize models
        model_service = ModelService(config_path, checkpoint_path, cpu=cpu)
        self.generator, self.kp_detector = model_service.load_eval_models()
        self._log_init()

    @staticmethod
    def _validate_paths(*paths: str) -> None:
        """Validate input file paths exist."""
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Path does not exist: {path}")

    def _log_init(self) -> None:
        """Log initialization parameters."""
        self.log.info(f"Initialized VideoAnimationService with:")
        self.log.info(f"Source: {self.source_image_path}")
        self.log.info(f"Driving: {self.driving_video_path}")
        self.log.info(f"Output: {self.result_video_path}")
        self.log.info(f"Settings: relative={self.relative}, adapt_scale={self.adapt_scale}")

    def run(self) -> None:
        """Execute the full fomm pipeline."""
        try:
            # Load and preprocess media
            source_image = preprocess_image(self.source_image_path)
            driving_video, fps = load_video(self.driving_video_path)

            # Generate animations
            predictions = self._generate_animations(source_image, driving_video)

            # Save results
            self._save_video(predictions, fps)
        except Exception as e:
            self.log.error(f"Animation failed: {str(e)}")
            raise

    def _generate_animations(self, source_image: np.ndarray, driving_video: list[np.ndarray]) -> list[np.ndarray]:
        """Generate fomm frames with optional best frame processing."""
        if not self.find_best and self.best_frame is None:
            return AnimationService.make_animation(
                source_image,
                driving_video,
                self.generator,
                self.kp_detector,
                relative=self.relative,
                adapt_movement_scale=self.adapt_scale,
                cpu=self.cpu,
            )

        # Handle best frame processing
        i = (
            self.best_frame
            if self.best_frame is not None
            else find_best_frame(source_image, driving_video, cpu=self.cpu)
        )
        self.log.info(f"Using frame {i} as best match")

        # Split video at best frame
        driving_forward = driving_video[i:]
        driving_backward = driving_video[: i + 1][::-1]

        # Process both segments
        predictions_forward = AnimationService.make_animation(
            source_image,
            driving_forward,
            self.generator,
            self.kp_detector,
            relative=self.relative,
            adapt_movement_scale=self.adapt_scale,
            cpu=self.cpu,
        )

        predictions_backward = AnimationService.make_animation(
            source_image,
            driving_backward,
            self.generator,
            self.kp_detector,
            relative=self.relative,
            adapt_movement_scale=self.adapt_scale,
            cpu=self.cpu,
        )

        # Combine results (excluding duplicate middle frame)
        return predictions_backward[::-1] + predictions_forward[1:]

    def _save_video(self, frames: list[np.ndarray], fps: float) -> None:
        """Save frames as images, then encode with ffmpeg directly to mp4 (H.264)"""
        import subprocess
        import tempfile

        import imageio.v2 as imageio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save all frames
            for i, frame in enumerate(frames):
                imageio.imwrite(f"{tmpdir}/frame_{i:04d}.png", img_as_ubyte(frame))

            # Encode with ffmpeg (H.264, web-compatible)
            cmd = [
                "ffmpeg",
                "-y",
                "-r",
                str(fps),
                "-i",
                f"{tmpdir}/frame_%04d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                self.result_video_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
