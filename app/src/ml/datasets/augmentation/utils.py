import numpy as np
import PIL
from skimage.transform import resize


def is_numpy_clip(clip) -> bool:
    return isinstance(clip[0], np.ndarray)


def is_pil_clip(clip) -> bool:
    return isinstance(clip[0], PIL.Image.Image)


def get_clip_shape(clip):
    if is_numpy_clip(clip):
        return clip[0].shape[:2]
    elif is_pil_clip(clip):
        return clip[0].size[1], clip[0].size[0]
    raise TypeError(f"Unsupported clip type: {type(clip[0])}")


def crop_clip(clip, min_h: int, min_w: int, h: int, w: int):
    if is_numpy_clip(clip):
        return [img[min_h : min_h + h, min_w : min_w + w, :] for img in clip]
    elif is_pil_clip(clip):
        return [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]
    else:
        raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))


def pad_clip(clip, h: int, w: int):
    im_h, im_w = clip[0].shape[:2]
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)
    return np.pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode="edge")


def get_resize_sizes(im_h: int, im_w: int, size: int):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def resize_clip(clip, size: int | tuple[int, int], interpolation: str = "bilinear"):
    if isinstance(size, int):
        im_h, im_w = get_clip_shape(clip)
        if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
            return clip
        new_h, new_w = get_resize_sizes(im_h, im_w, size)
        size = (new_w, new_h)
    else:
        size = size[1], size[0]

    if is_numpy_clip(clip):
        order = 1 if interpolation == "bilinear" else 0
        return [
            resize(img, size, order=order, preserve_range=True, mode="constant", anti_aliasing=True) for img in clip
        ]
    elif is_pil_clip(clip):
        pil_inter = PIL.Image.BILINEAR if interpolation == "bilinear" else PIL.Image.NEAREST
        return [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError(f"Unsupported clip type: {type(clip[0])}")
