# Copyright (c) Megvii, Inc. and its affiliates.

import json
from pathlib import Path

import pytest

from yolox.data import CocoDataset


def test_missing_annotation_file_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "COCO"
    (data_dir / "train2017").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Annotation file not found"):
        CocoDataset(data_dir=str(data_dir))


def test_missing_image_dir_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "COCO"
    (data_dir / "annotations").mkdir(parents=True)
    ann = {"images": [{"id": 1, "file_name": "img1.jpg"}], "annotations": [], "categories": []}
    (data_dir / "annotations" / "instances_train2017.json").write_text(json.dumps(ann))
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        CocoDataset(data_dir=str(data_dir))


def test_empty_annotation_file_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "COCO"
    (data_dir / "annotations").mkdir(parents=True)
    (data_dir / "train2017").mkdir()
    ann = {"images": [], "annotations": [], "categories": []}
    (data_dir / "annotations" / "instances_train2017.json").write_text(json.dumps(ann))
    with pytest.raises(RuntimeError, match="No images declared"):
        CocoDataset(data_dir=str(data_dir))
