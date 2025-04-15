from src.datasets.augmentation.transforms import (
    ColorJitter,
    RandomCrop,
    RandomFlip,
    RandomResize,
    RandomRotation,
)


class AllAugmentationTransform:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None, jitter_param=None):
        self.transforms = []

        params = {
            RandomFlip: flip_param,
            RandomRotation: rotation_param,
            RandomResize: resize_param,
            RandomCrop: crop_param,
            ColorJitter: jitter_param,
        }
        for Factory, param in params.items():
            if param:
                self.transforms.append(Factory(**param))

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip
