import random

import numpy as np
import PIL
from src.datasets.augmentation.utils import resize_clip


class RandomResize:
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
        interpolation (str): Can be one of 'nearest', 'bilinear' defaults to nearest
        ratio (tuple): (widht, height)
    """

    def __init__(self, ratio: tuple[float] = (3.0 / 4.0, 4.0 / 3.0), interpolation: str = "nearest"):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(clip, new_size, interpolation=self.interpolation)

        return resized
