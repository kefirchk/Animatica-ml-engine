import os
from shutil import copy
from time import gmtime, strftime

import torch
import yaml
from src.modules.discriminator import MultiScaleDiscriminator
from src.modules.generator import OcclusionAwareGenerator
from src.modules.keypoint_detector import KPDetector
from src.services.sync_batchnorm import DataParallelWithCallback


class ModelService:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str = None,
        log_dir: str = "log",
        cpu: bool = False,
        verbose: bool = False,
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.log_dir = log_dir
        self.cpu = cpu
        self.verbose = verbose
        # self.config = self._load_config()
        # self.log_dir = self._prepare_log_dir()
        self.device = torch.device("cpu" if cpu else "cuda:0")

    def load_checkpoints(self, config_path, checkpoint_path, cpu=False):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        generator = OcclusionAwareGenerator(
            **config["model_params"]["generator_params"], **config["model_params"]["common_params"]
        )
        if not cpu:
            generator.cuda()

        kp_detector = KPDetector(
            **config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"]
        )
        if not cpu:
            kp_detector.cuda()

        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(checkpoint_path)

        generator.load_state_dict(checkpoint["generator"])
        kp_detector.load_state_dict(checkpoint["kp_detector"])

        if not cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector
