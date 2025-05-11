from src.ml.datasets import FramesDataset
from src.ml.runners.base import BaseRunner
from src.ml.services.animation import AnimationService


class AnimationRunner(BaseRunner):
    def __init__(self, args, log):
        super().__init__(args, log)
        self.animation_service = AnimationService()

    def run(self):
        dataset = FramesDataset(is_train=False, **self.config["dataset_params"])
        self.log.info("Animation started...")
        self.animation_service.animate(
            self.config, self.generator, self.kp_detector, self.args.checkpoint, self.log_dir, dataset
        )
