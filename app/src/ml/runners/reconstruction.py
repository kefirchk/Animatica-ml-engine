from src.ml.datasets import FramesDataset
from src.ml.runners.base import BaseRunner
from src.ml.services.reconstruction import ReconstructionService


class ReconstructionRunner(BaseRunner):
    def run(self):
        dataset = FramesDataset(is_train=False, **self.config["dataset_params"])
        self.log.info("Reconstruction started...")
        ReconstructionService.reconstruction(
            self.config, self.generator, self.kp_detector, self.args.checkpoint, self.log_dir, dataset
        )
