import collections
import logging
import os
import sys

import imageio
import numpy as np
import torch
from services.visualization import VisualizationService


class LoggingService:
    def __init__(
        self,
        log_dir: str,
        log_file_name: str = "log.txt",
        checkpoint_freq: int = 100,
        visualizer_params=None,
        zfill_num: int = 8,
    ) -> None:
        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, "train-vis")
        os.makedirs(self.visualizations_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, log_file_name), "a")
        self.zfill_num = zfill_num
        self.visualizer = VisualizationService(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float("inf")
        self.names = None
        self.models = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp["driving"], inp["source"], out)
        imageio.imsave(
            os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image
        )

    def save_cpk(self, emergent: bool = False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk["epoch"] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, "%s-checkpoint.pth.tar" % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(
        checkpoint_path: str,
        generator=None,
        discriminator=None,
        kp_detector=None,
        optimizer_generator=None,
        optimizer_discriminator=None,
        optimizer_kp_detector=None,
    ):
        log = LoggingService.setup_logger(__name__)
        checkpoint = torch.load(checkpoint_path)
        if generator:
            generator.load_state_dict(checkpoint["generator"])
        if kp_detector:
            kp_detector.load_state_dict(checkpoint["kp_detector"])
        if discriminator:
            try:
                discriminator.load_state_dict(checkpoint["discriminator"])
            except:
                log.error("No discriminator in the state-dict. Dicriminator will be randomly initialized")
        if optimizer_generator:
            optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
        if optimizer_discriminator:
            try:
                optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])
            except RuntimeError:
                log.error("No discriminator optimizer in the state-dict. Optimizer will be not initialized")
        if optimizer_kp_detector:
            optimizer_kp_detector.load_state_dict(checkpoint["optimizer_kp_detector"])

        return checkpoint["epoch"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if "models" in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)

    @staticmethod
    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
