import json
from pathlib import Path

import pytest
from PIL import Image

from yolox.data import CocoDataset


@pytest.fixture()
def coco_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "COCO"
    (data_dir / "annotations").mkdir(parents=True)
    (data_dir / "train2017").mkdir()
    for i in (1, 2):
        Image.new("RGB", (64, 64)).save(data_dir / "train2017" / f"img{i}.jpg")
    ann = {
        "info": {"description": "mini COCO fixture"},
        "images": [
            {"id": i, "file_name": f"img{i}.jpg", "width": 64, "height": 64} for i in (1, 2)
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 2, 10, 20],
                "area": 200,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    (data_dir / "annotations" / "instances_train2017.json").write_text(json.dumps(ann))
    return data_dir


def test_coco_dataset_preserves_info(coco_dir: Path) -> None:
    # 'info' must survive dataset construction: pycocotools loadRes() reads it during eval
    dataset = CocoDataset(data_dir=str(coco_dir))
    assert "info" in dataset.coco.dataset
    assert dataset.num_imgs == 2
