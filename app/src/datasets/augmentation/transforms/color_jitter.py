import random
import warnings

import numpy as np
import PIL
import torchvision
from skimage.util import img_as_float, img_as_ubyte


class ColorJitter:
    """Randomly change the brightness, contrast and saturation and hue of the clip
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

    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness: float, contrast: float, saturation: float, hue: float):
        brightness_factor = random.uniform(max(0.0, 1 - brightness), 1 + brightness) if brightness > 0 else None
        contrast_factor = random.uniform(max(0.0, 1 - contrast), 1 + contrast) if contrast > 0 else None
        saturation_factor = random.uniform(max(0.0, 1 - saturation), 1 + saturation) if saturation > 0 else None
        hue_factor = random.uniform(-hue, hue) if hue > 0 else None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            # Create img transform function sequence
            img_transforms = []
            if brightness:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = (
                [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array, img_as_float]
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img.astype("float32"))
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            # Create img transform function sequence
            img_transforms = []
            if brightness:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all videos
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))
        return jittered_clip
