import random

from skimage.transform import rotate
from src.ml.datasets.augmentation.utils import is_numpy_clip, is_pil_clip


class RandomRotation:
    """Rotate entire clip randomly by a random angle within given bounds.
    Args:
        degrees (sequence or int): Range of degrees to select from
                                   If degrees is a number instead of sequence like (min, max),
                                   the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees) -> None:
        if isinstance(degrees, int):
            if degrees < 0:
                raise ValueError("If degrees is a single number," "must be positive")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence," "it must be of len 2.")

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
            clip (PIL.Image or numpy.ndarray): List of videos to be cropped in format (h, w, c) in numpy.ndarray
        Returns:
            PIL.Image or numpy.ndarray: Cropped list of videos
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if is_numpy_clip(clip):
            rotated = [rotate(image=img, angle=angle, preserve_range=True) for img in clip]
        elif is_pil_clip(clip):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))

        return rotated
