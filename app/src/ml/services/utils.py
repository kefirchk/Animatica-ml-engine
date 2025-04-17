import warnings

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from skimage.transform import resize


def find_best_frame(source_image: np.ndarray, driving_video: list[np.ndarray], cpu: bool = False) -> int:
    """
    Find the driving frame with most similar facial landmarks to source image.
    Args:
        source_image: Source image (H, W, C)
        driving_video: List of driving frames
        cpu: Force CPU execution
    Returns:
        Index of best matching frame
    """
    import face_alignment  # Lazy import to reduce dependencies

    def normalize_kp(kp: np.ndarray) -> np.ndarray:
        """Normalize keypoints by centering and scaling by face area."""
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = np.sqrt(ConvexHull(kp[:, :2]).volume)
        return kp / area

    # Initialize face alignment
    device = "cpu" if cpu else "cuda"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device=device)

    # Process source landmarks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress skimage resize warnings
        kp_source = fa.get_landmarks((255 * source_image).astype(np.uint8))

    if not kp_source:
        raise ValueError("No faces detected in source image")
    kp_source = normalize_kp(kp_source[0])

    # Find best matching frame
    min_norm = float("inf")
    best_frame = 0

    for i, frame in enumerate(driving_video):
        kp_driving = fa.get_landmarks((255 * frame).astype(np.uint8))
        if not kp_driving:
            continue

        kp_driving = normalize_kp(kp_driving[0])
        current_norm = np.sum((kp_source - kp_driving) ** 2)

        if current_norm < min_norm:
            min_norm = current_norm
            best_frame = i

    return best_frame


def preprocess_image(image_path: str, target_size: tuple[int, int] = (256, 256), normalize: bool = True) -> np.ndarray:
    """
    Load and preprocess an image.
    Args:
        image_path: Path to image file
        target_size: Target dimensions (H, W)
        normalize: Scale to [0,1] range
    Returns:
        Processed image array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Convert color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize with antialiasing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = resize(image, target_size, preserve_range=False, anti_aliasing=True)

    # Ensure 3 channels and normalize
    image = image[..., :3]
    if normalize:
        image = np.clip(image, 0, 1).astype(np.float32)

    return image


def load_video(
    video_path: str, target_size: tuple[int, int] = (256, 256), max_frames: int | None = None
) -> tuple[list[np.ndarray], float]:
    """
    Load and preprocess video frames.
    Args:
        video_path: Path to video file
        target_size: Target dimensions (H, W)
        max_frames: Maximum number of frames to load
    Returns:
        Tuple of (frames, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break

        # Convert and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = resize(frame, target_size, preserve_range=False, anti_aliasing=True)
        frames.append(frame[..., :3])

    cap.release()
    return frames, fps
