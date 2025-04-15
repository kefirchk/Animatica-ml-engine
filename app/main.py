import sys
from argparse import ArgumentParser, Namespace

import torch
from src.runners import AnimationRunner, ReconstructionRunner, TrainingRunner
from src.services.logging import LoggingService


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--mode", choices=["train", "reconstruction", "animate"], required=True)
    parser.add_argument("--log_dir", default="log")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(","))))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception("Use Python 3 or higher.")

    torch.multiprocessing.set_start_method("spawn", force=True)  # Safety for CUDA

    args = parse_args()
    log = LoggingService.setup_logger(__name__)

    mode_to_runner = {"train": TrainingRunner, "reconstruction": ReconstructionRunner, "animate": AnimationRunner}

    runner_class = mode_to_runner[args.mode]
    runner = runner_class(args, log)
    runner.run()
