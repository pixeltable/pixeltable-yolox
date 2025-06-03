import logging
import pytest
import subprocess

logger = logging.getLogger(__name__)


def test_training_sagemaker_structure():
    rs = subprocess.run(
        ["python", "yolox/cli/train.py", "--name", "yolox-sagemaker", "--config", "yolox-s", "--batch-size", "4",
         "--data-dir", "/path/to/datasets/sm_scooter_training", "--train-data-suffix", "train",
         "--val-data-suffix", "val", "--images-suffix", "images", "--train-anno",
         "train/labels/annotations.json", "--val-anno", "val/labels/annotations.json", "--output-dir",
         "/path/to/datasets/sm_scooter_training/output", "--fp16"])
    if rs.returncode != 0:
        pytest.fail("yolox/cli/train.py failed. See the log for details!")


def test_training_coco_structure():
    rs = subprocess.run(
        ["python", "yolox/cli/train.py", "--name", "yolox-coco", "--config", "yolox-s", "--fp16"])
    if rs.returncode != 0:
        pytest.fail("yolox/cli/train.py failed. See the log for details!")
