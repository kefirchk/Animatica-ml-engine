import random

import numpy as np
import torchvision
from skimage.util import img_as_float, img_as_ubyte
from src.datasets.augmentation.utils import is_numpy_clip, is_pil_clip


class ColorJitter:
    """Randomly change the brightness, contrast and saturation and hue of the clip.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
                            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
                          is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
                            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
                    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness: float, contrast: float, saturation: float, hue: float):
        return (
            random.uniform(max(0.0, 1 - brightness), 1 + brightness) if brightness > 0 else None,
            random.uniform(max(0.0, 1 - contrast), 1 + contrast) if contrast > 0 else None,
            random.uniform(max(0.0, 1 - saturation), 1 + saturation) if saturation > 0 else None,
            random.uniform(-hue, hue) if hue > 0 else None,
        )

    def __call__(self, clip):
        """
        Args:
            clip (list): list of PIL.Image
        Returns:
            list PIL.Image : list of transformed PIL.Image
        """
        return self.apply_color_jitter(clip, self.brightness, self.contrast, self.saturation, self.hue)

    @staticmethod
    def apply_pipeline(img, transforms):
        for t in transforms:
            img = t(img)
        return img

    @classmethod
    def apply_color_jitter(cls, clip, brightness, contrast, saturation, hue):
        brightness, contrast, saturation, hue = ColorJitter.get_params(brightness, contrast, saturation, hue)
        transforms = []
        if brightness:
            transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation:
            transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue:
            transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast:
            transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(transforms)

        if is_numpy_clip(clip):
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + transforms + [np.array, img_as_float]
            return [cls.apply_pipeline(img, img_transforms) for img in clip]
        elif is_pil_clip(clip):
            return [cls.apply_pipeline(img, transforms) for img in clip]
        else:
            raise TypeError(f"Unsupported clip type: {type(clip[0])}")
