import random

from src.datasets.augmentation.utils import (
    crop_clip,
    is_numpy_clip,
    is_pil_clip,
    pad_clip,
)


class RandomCrop:
    """Extract random crop at the same location for a list of videos.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w)
    """

    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, clip):
        """
        Args:
            img (PIL.Image or numpy.ndarray): List of videos to be cropped in format (h, w, c) in numpy.ndarray
        Returns:
            PIL.Image or numpy.ndarray: Cropped list of videos
        """
        h, w = self.size
        if is_numpy_clip(clip):
            im_h, im_w, im_c = clip[0].shape
        elif is_pil_clip(clip):
            im_w, im_h = clip[0].size
        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))

        clip = pad_clip(clip, h, w)
        im_h, im_w = clip.shape[1:3]
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped
