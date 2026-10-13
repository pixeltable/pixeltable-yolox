# Copyright (c) Megvii, Inc. and its affiliates.

import json
from pathlib import Path

import cv2
import numpy as np
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


def test_absolute_annotation_path_roboflow_layout(tmp_path: Path) -> None:
    # Roboflow COCO exports keep the annotation file next to the images of each split
    split_dir = tmp_path / "roboflow" / "train"
    split_dir.mkdir(parents=True)
    cv2.imwrite(str(split_dir / "img1.jpg"), np.zeros((8, 12, 3), dtype=np.uint8))
    ann = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 12, "height": 8}],
        "annotations": [],
        "categories": [{"id": 1, "name": "thing"}],
    }
    ann_path = split_dir / "_annotations.coco.json"
    ann_path.write_text(json.dumps(ann))
    dataset = CocoDataset(data_dir=str(tmp_path / "roboflow"), json_file=str(ann_path), name="train")
    assert len(dataset) == 1
    assert dataset.load_image(0).shape == (8, 12, 3)


def test_yolox_datadir_hint_only_when_data_dir_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLOX_DATADIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="YOLOX_DATADIR") as exc_info:
        CocoDataset()
    assert str(tmp_path / "COCO") in str(exc_info.value)
    with pytest.raises(FileNotFoundError) as exc_info:
        CocoDataset(data_dir=str(tmp_path / "COCO"))
    assert "YOLOX_DATADIR" not in str(exc_info.value)
