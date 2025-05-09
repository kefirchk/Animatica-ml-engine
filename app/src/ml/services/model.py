import os
from shutil import copy
from time import gmtime, strftime

import torch
import yaml
from src.ml.datasets.replicate import DataParallelWithCallback
from src.ml.modules.discriminator import MultiScaleDiscriminator
from src.ml.modules.generator import OcclusionAwareGenerator
from src.ml.modules.keypoint_detector import KPDetector


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
        self.config = self._load_config()
        self.log_dir = self._prepare_log_dir()
        self.device = torch.device("cpu" if cpu else "cuda:0")

    def _load_config(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _prepare_log_dir(self):
        if self.checkpoint_path:
            return os.path.join(*os.path.split(self.checkpoint_path)[:-1])
        else:
            log_dir = os.path.join(self.log_dir, os.path.basename(self.config_path).split(".")[0])
            log_dir += " " + strftime("%d_%m_%y_%H.%M.%S", gmtime())
            os.makedirs(log_dir, exist_ok=True)

            config_copy_path = os.path.join(log_dir, os.path.basename(self.config_path))
            if not os.path.exists(config_copy_path):
                copy(self.config_path, config_copy_path)

            return log_dir

    def init_training_models(self, device_ids: list) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        generator = OcclusionAwareGenerator(
            **self.config["model_params"]["generator_params"], **self.config["model_params"]["common_params"]
        )
        discriminator = MultiScaleDiscriminator(
            **self.config["model_params"]["discriminator_params"], **self.config["model_params"]["common_params"]
        )
        kp_detector = KPDetector(
            **self.config["model_params"]["kp_detector_params"], **self.config["model_params"]["common_params"]
        )

        if torch.cuda.is_available() and not self.cpu:
            generator.to(device_ids[0])
            discriminator.to(device_ids[0])
            kp_detector.to(device_ids[0])

        if self.verbose:
            print(generator)
            print(discriminator)
            print(kp_detector)

        return generator, discriminator, kp_detector

    def load_eval_models(self):
        generator = OcclusionAwareGenerator(
            **self.config["model_params"]["generator_params"],
            **self.config["model_params"]["common_params"],
        )

        kp_detector = KPDetector(
            **self.config["model_params"]["kp_detector_params"],
            **self.config["model_params"]["common_params"],
        )

        if not self.cpu:
            generator.cuda()
            kp_detector.cuda()

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        generator.load_state_dict(checkpoint["generator"])
        kp_detector.load_state_dict(checkpoint["kp_detector"])

        if not self.cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector
