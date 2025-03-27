# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import importlib
import random
import sys
from typing import Optional
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.config import YoloxConfig
from yolox.core import launch
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("yolox train")
    parser.add_argument("-c", "--config", type=str, help="A builtin config such as yolox_s, or a custom Python class given as {module}:{classname} such as yolox.config:YoloxS")
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name; defaults to the model name specified in config")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
                Implemented loggers include `tensorboard`, `mlflow` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def train(config: YoloxConfig, args):
    if config.seed is not None:
        random.seed(config.seed)
#        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = config.get_trainer(args)
    trainer.train()


def resolve_config(config_str: str) -> YoloxConfig:
    config_class = YoloxConfig.get_named_config(config_str)
    if config_class is None:
        classpath = config_str.split(":")
        if len(classpath) == 2:
            try:
                module = importlib.import_module(classpath[0])
                config_class = getattr(module, classpath[1], None)
            except ImportError:
                pass
    if config_class is None:
        raise ValueError(f"Unknown config class: {config_str}")
    if not issubclass(config_class, YoloxConfig):
        raise ValueError(f"Invalid config class (does not extend `YoloxConfig`): {config_str}")

    try:
        return config_class()
    except Exception as e:
        raise ValueError(f"Error loading model config: {config_str}") from e


def main(argv: list[str]) -> None:
    configure_module()
    args = make_parser().parse_args(argv)
    if args.config is None:
        raise AttributeError("Please specify a config file.")
    config = resolve_config(args.config)
    config.update(args.opts)
    config.validate()

    if not args.name:
        args.name = config.name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        config.dataset = config.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        train,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(config, args),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
