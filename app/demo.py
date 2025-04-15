import sys
from argparse import ArgumentParser

import imageio
import matplotlib
import numpy as np
from scipy.spatial import ConvexHull
from skimage.transform import resize
from skimage.util import img_as_ubyte
from src.services.animation import AnimationService
from src.services.logging import LoggingService
from src.services.model import ModelService
from tqdm import tqdm

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

matplotlib.use("Agg")


# def load_checkpoints(config_path, checkpoint_path, cpu=False):
#     with open(config_path) as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#
#     generator = OcclusionAwareGenerator(
#         **config["model_params"]["generator_params"], **config["model_params"]["common_params"]
#     )
#     if not cpu:
#         generator.cuda()
#
#     kp_detector = KPDetector(**config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"])
#     if not cpu:
#         kp_detector.cuda()
#
#     if cpu:
#         checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
#     else:
#         checkpoint = torch.load(checkpoint_path)
#
#     generator.load_state_dict(checkpoint["generator"])
#     kp_detector.load_state_dict(checkpoint["kp_detector"])
#
#     if not cpu:
#         generator = DataParallelWithCallback(generator)
#         kp_detector = DataParallelWithCallback(kp_detector)
#
#     generator.eval()
#     kp_detector.eval()
#
#     return generator, kp_detector


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=True, device="cpu" if cpu else "cuda"
    )
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float("inf")
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


if __name__ == "__main__":
    log = LoggingService.setup_logger(__name__)
    parser = ArgumentParser()
    parser.add_argument("--configs", required=True, help="path to configs")
    parser.add_argument("--checkpoint", default="vox-cpk.pth.tar", help="path to checkpoint to restore")
    parser.add_argument("--source_image", default="sup-mat/source.png", help="path to source image")
    parser.add_argument("--driving_video", default="sup-mat/source.png", help="path to driving video")
    parser.add_argument("--result_video", default="result.mp4", help="path to output")
    parser.add_argument(
        "--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates"
    )
    parser.add_argument(
        "--adapt_scale",
        dest="adapt_scale",
        action="store_true",
        help="adapt movement scale based on convex hull of keypoints",
    )
    parser.add_argument(
        "--find_best_frame",
        dest="find_best_frame",
        action="store_true",
        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)",
    )
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()["fps"]
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError as e:
        log.error(e)
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    model_service = ModelService(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    generator, kp_detector = model_service.load_eval_models()

    if opt.find_best_frame or opt.best_frame:
        i = opt.best_frame if opt.best_frame else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[: (i + 1)][::-1]
        predictions_forward = AnimationService.make_animation(
            source_image,
            driving_forward,
            generator,
            kp_detector,
            relative=opt.relative,
            adapt_movement_scale=opt.adapt_scale,
            cpu=opt.cpu,
        )
        predictions_backward = AnimationService.make_animation(
            source_image,
            driving_backward,
            generator,
            kp_detector,
            relative=opt.relative,
            adapt_movement_scale=opt.adapt_scale,
            cpu=opt.cpu,
        )
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = AnimationService.make_animation(
            source_image,
            driving_video,
            generator,
            kp_detector,
            relative=opt.relative,
            adapt_movement_scale=opt.adapt_scale,
            cpu=opt.cpu,
        )
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
