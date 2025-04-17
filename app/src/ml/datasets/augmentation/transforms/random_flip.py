import random

import numpy as np


class RandomFlip:
    def __init__(self, time_flip: bool = False, horizontal_flip: bool = False) -> None:
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            clip = clip[::-1]
        if random.random() < 0.5 and self.horizontal_flip:
            clip = [np.fliplr(img) for img in clip]
        return clip
