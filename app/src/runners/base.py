from abc import ABC, abstractmethod
from argparse import Namespace
from logging import Logger

from src.services.model import ModelService


class BaseRunner(ABC):
    def __init__(self, args: Namespace, log: Logger):
        self.args = args
        self.log = log
        self.device_ids = args.device_ids
        self.verbose = args.verbose

        model_service = ModelService(args.configs, args.checkpoint, args.log_dir, cpu=args.cpu)

        self.config = model_service.config
        self.log_dir = model_service.log_dir

        self.generator, self.discriminator, self.kp_detector = model_service.init_training_models(self.device_ids)

    @abstractmethod
    def run(self):
        pass
