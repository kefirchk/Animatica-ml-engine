from src.ml.datasets import FramesDataset
from src.ml.runners.base import BaseRunner
from src.ml.services.training import TrainingService


class TrainingRunner(BaseRunner):
    def run(self):
        dataset = FramesDataset(is_train=True, **self.config["dataset_params"])
        self.log.info("Training started...")
        TrainingService.train(
            self.config,
            self.generator,
            self.discriminator,
            self.kp_detector,
            self.args.checkpoint,
            self.log_dir,
            dataset,
            self.device_ids,
        )
