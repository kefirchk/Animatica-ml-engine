from src.datasets import FramesDataset
from src.runners.base import BaseRunner
from src.services.reconstruction import ReconstructionService


class ReconstructionRunner(BaseRunner):
    def run(self):
        dataset = FramesDataset(is_train=False, **self.config["dataset_params"])
        self.log.info("Reconstruction started...")
        ReconstructionService.reconstruction(
            self.config, self.generator, self.kp_detector, self.args.checkpoint, self.log_dir, dataset
        )
