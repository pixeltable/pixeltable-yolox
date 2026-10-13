# Copyright (c) Megvii, Inc. and its affiliates.

import argparse

import torch
from loguru import logger

from yolox.cli.utils import resolve_config
from yolox.models import Yolox


def make_parser():
    parser = argparse.ArgumentParser("yolox torchscript-deploy")
    parser.add_argument(
        "-n", "--name", type=str, required=True,
        help="A builtin model name such as yolox_s, or a checkpoint path (requires --config)"
    )
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="A builtin config such as yolox_s, or a custom class given as {module}:{classname}"
    )
    parser.add_argument(
        "--output-name", type=str, default="yolox.torchscript.pt", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--decode-in-inference",
        action="store_true",
        help="decode in inference or not"
    )
    return parser


def main():
    args = make_parser().parse_args()
    config = resolve_config(args.config) if args.config else None

    model = Yolox.from_pretrained(args.name, config)
    module = model.module
    module.eval()
    module.head.decode_in_inference = args.decode_in_inference

    test_size = model.processor.config.test_size
    dummy_input = torch.randn(args.batch_size, 3, test_size[0], test_size[1])

    mod = torch.jit.trace(module, dummy_input)
    mod.save(args.output_name)
    logger.info("generated torchscript model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
