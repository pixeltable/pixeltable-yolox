<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/assets/logo.png"
     alt="YOLOX" width="350">
</div>

`pixeltable-yolox` is an Apache-licensed, pip-installable YOLOX object detection package maintained by
[Pixeltable](https://github.com/pixeltable/pixeltable). It is the YOLOX Python package for developers who
want pretrained inference and training, and for Pixeltable users who want a computed column for YOLOX
object detection on an image table or YOLOX video object detection on frames from a video table. The
library is a fork of [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) by Zheng Ge et al. and
runs on Python 3.10+. Install it with `pip install pixeltable-yolox`. `pip install yolox` installs the older
Megvii distribution published on PyPI. License: [Apache 2.0](LICENSE).

## Quick start

Python 3.10 or newer.

```bash
pip install pixeltable-yolox
```

`Yolox.from_pretrained` accepts `yolox_nano`, `yolox_tiny`, `yolox_s`, `yolox_m`, `yolox_l`, or `yolox_x`.
The first call downloads `{model_id}.pth` from the Megvii YOLOX `0.1.1rc0` release into
`$YOLOX_HOME/weights`, or `~/.cache/yolox/weights` when `YOLOX_HOME` is unset. The default device is CPU.

Inputs are PIL images or file paths. Each detection dict has `bboxes` as `(x1, y1, x2, y2)` in the original
image, `scores`, and `labels` as COCO category indices (`yolox.data.datasets.COCO_CLASSES`). The default
score threshold is `0.5`.

```python
import io
import urllib.request

from PIL import Image
from yolox.data.datasets import COCO_CLASSES
from yolox.models import Yolox

url = (
    'https://raw.githubusercontent.com/pixeltable/pixeltable-yolox'
    '/main/tests/data/000000000001.jpg'
)
with urllib.request.urlopen(url) as response:
    image = Image.open(io.BytesIO(response.read()))
    image.load()

model = Yolox.from_pretrained('yolox_s')
detections = model([image], threshold=0.65)
labels = detections[0]['labels']
print(labels)
print([COCO_CLASSES[label] for label in labels])
```

A file path works the same way: `model(['tests/data/000000000001.jpg'], threshold=0.65)`.

The detection test in this repo records labels `[7, 2, 12]` for that image at `threshold=0.65` with
`yolox_s` (COCO classes `truck`, `car`, and `parking meter`) on torch 2.2.2 and torchvision 0.17.2.
Other torch builds can shift boxes and scores. The test allows small numeric differences in boxes and
scores, but requires labels and their order to match exactly.

## YOLOX in Pixeltable

Pixeltable calls this package through
[`pixeltable.functions.yolox`](https://docs.pixeltable.com/sdk/latest/yolox). A computed column runs YOLOX
object detection for every image. YOLOX video object detection is the same function on frames from a video
table. The UDF on Pixeltable main loads the model with `device='cpu'`.

The Pixeltable integration requires Python 3.11 or newer.

```bash
pip install pixeltable pixeltable-yolox
```

Notebooks and the REPL use `create_table` and `add_computed_column`. A Pixeltable application assigns the
same `yolox(...)` call on a `TableModel` and creates the table with `pxt schema update`
([quick start](https://docs.pixeltable.com/overview/quick-start)).

`model_id` is one of `yolox_nano`, `yolox_tiny`, `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x`. The default
`threshold` is `0.5`. The stored value has `bboxes`, `scores`, and `labels`.

```python
import pixeltable as pxt
from pixeltable.functions.yolox import yolox, yolo_to_coco

images = pxt.create_table('images', {'image': pxt.Image})
images.add_computed_column(
    detections=yolox(images.image, model_id='yolox_s', threshold=0.5)
)
images.add_computed_column(detections_coco=yolo_to_coco(images.detections))
images.insert([{
    'image': (
        'https://raw.githubusercontent.com/pixeltable/pixeltable-yolox'
        '/main/tests/data/000000000001.jpg'
    ),
}])
print(images.select(images.detections).collect())
```

`yolo_to_coco` returns a list of `{bbox, category}` dicts. `bbox` is COCO `[x, y, width, height]` with each
value rounded to an integer, and `category` is the YOLOX label.

For video, extract frames with `frame_iterator`, then add the same computed column. At most one of `fps`,
`num_frames`, and `keyframes_only` may be set. This is the pattern in
[Object detection in videos](https://docs.pixeltable.com/howto/use-cases/object-detection-in-videos).

```python
import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.yolox import yolox

videos = pxt.create_table('videos', {'video': pxt.Video})
frames = pxt.create_view(
    'frames',
    videos,
    iterator=frame_iterator(videos.video, fps=1.0),
)
frames.add_computed_column(
    detections=yolox(frames.frame, model_id='yolox_s', threshold=0.5)
)
videos.insert([{
    'video': (
        'https://raw.githubusercontent.com/pixeltable/pixeltable'
        '/release/docs/resources/bangkok.mp4'
    ),
}])
print(frames.select(frames.frame_attrs, frames.detections).limit(3).collect())
```

Inserting a video fills the frame view and runs YOLOX object detection on those frames. Frame metadata
is stored in `frame_attrs`; `frame_attrs['time']` is the frame timestamp in seconds.

## Models

Sizes below are the named configs in `yolox/config.py`. Input size is the default `input_size` and
`test_size` `(height, width)`. Every named config has `num_classes` 80. Pretrained weights are the
matching `{model_id}.pth` checkpoint from Megvii release `0.1.1rc0`.

| Model id | Depth | Width | Input size | Depthwise |
| --- | ---: | ---: | --- | --- |
| `yolox_nano` | 0.33 | 0.25 | 416×416 | yes |
| `yolox_tiny` | 0.33 | 0.375 | 416×416 | no |
| `yolox_s` | 0.33 | 0.50 | 640×640 | no |
| `yolox_m` | 0.67 | 0.75 | 640×640 | no |
| `yolox_l` | 1.0 | 1.0 | 640×640 | no |
| `yolox_x` | 1.33 | 1.25 | 640×640 | no |

## Training

`yolox train` trains a named config on CUDA. `-d` is the GPU count, `-b` is the batch size, and `--fp16`
enables mixed precision. The run name defaults to the config name, and `output_dir` defaults to `./out`,
so this command writes `./out/yolox_s/latest_ckpt.pth` after each epoch. Training runs for 300 epochs by
default; add `-D max_epoch=1` for a single epoch.

```bash
yolox train -c yolox_s -d 1 -b 8 --fp16
yolox train -h
```

Training reads a COCO dataset. With `YOLOX_DATADIR` unset, the root is `datasets/COCO` under the parent of
the `yolox` package directory (the repository root in a source checkout). Otherwise it is
`$YOLOX_DATADIR/COCO`.

```text
COCO/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  train2017/
  val2017/
```

Training images are loaded from `<data_dir>/train2017/`, the default `train_img_dir`. Evaluation during
training reads `<data_dir>/val2017/` (`val_img_dir`) and `annotations/instances_val2017.json`. Override
config fields with repeated `-D name=value` flags, for example `-D num_classes=71` or `-D max_epoch=50`.
Annotation names (`train_ann`, `val_ann`, `test_ann`) resolve under `<data_dir>/annotations/` unless the
value is an absolute path. `train_img_dir`, `val_img_dir`, and `test_img_dir` are the image subdirectory
names under `data_dir`.

A Roboflow COCO export keeps `_annotations.coco.json` next to the images of each split. Point the
annotation options at those files and the image options at the split directories:

```bash
yolox train -c yolox_s -d 1 -b 16 --fp16 \
  -D data_dir=/data/roboflow -D num_classes=3 \
  -D train_ann=/data/roboflow/train/_annotations.coco.json -D train_img_dir=train \
  -D val_ann=/data/roboflow/valid/_annotations.coco.json -D val_img_dir=valid
```

`-e` / `--start_epoch` takes effect only together with `--resume`. It is the epoch to resume from, not
the total epoch count. `latest_ckpt.pth` is written after every epoch. `best_ckpt.pth` is written only
when evaluation AP rises above the previous best, which starts at 0, so a run that stays at AP 0 has no
`best_ckpt.pth`.

Pass `--ckpt` to fine-tune from a checkpoint. The file must contain a `model` state dict, which the
Megvii release checkpoints do. The loader skips tensors whose shapes do not match, including the
classification head when `num_classes` is not 80.

```bash
yolox train -c yolox_s -d 1 -b 8 --fp16 \
  --ckpt /path/to/yolox_s.pth \
  -D num_classes=71 \
  -D data_dir=/path/to/dataset
```

Evaluate a checkpoint with the same dataset overrides used for training. `yolox eval` reads `-n` / `--name`
for the run name.

```bash
yolox eval -c yolox_s -b 8 --ckpt out/yolox_s/latest_ckpt.pth \
  -D data_dir=/path/to/dataset -D num_classes=71
```

Load that checkpoint for inference with a config whose `num_classes` matches training:

```python
from yolox.config import YoloxS
from yolox.models import Yolox

config = YoloxS()
config.num_classes = 3
model = Yolox.from_pretrained('out/yolox_s/latest_ckpt.pth', config=config)
```

`Yolox.__call__` keeps detections at or above `threshold`, which defaults to 0.5. Training evaluation
uses `test_conf=0.01`, so a weak checkpoint can return no boxes at the default threshold. Use
`best_ckpt.pth` in place of `latest_ckpt.pth` when that file exists. Other scripts under `yolox/cli/`
are not subcommands of `yolox`.

## Documentation

- Pixeltable documentation: <https://docs.pixeltable.com/>
- `pixeltable.functions.yolox`: <https://docs.pixeltable.com/sdk/latest/yolox>
- Object detection in videos: <https://docs.pixeltable.com/howto/use-cases/object-detection-in-videos>
- Pixeltable repository: <https://github.com/pixeltable/pixeltable>
- Questions: [contact@pixeltable.com](mailto:contact@pixeltable.com)
- Contributions: [CONTRIBUTING.md](CONTRIBUTING.md)

## In memory of Dr. Jian Sun

Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would
not have been released and open sourced to the community. The passing away of Dr. Sun is a huge loss to the
Computer Vision field. We add this section here to express our remembrance and condolences to Dr. Sun. It
is hoped that every AI practitioner in the world will stick to the belief of "continuous innovation to
expand cognitive boundaries, and extraordinary technology to achieve product value" and move forward all
the way.

<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/assets/sunjian.png"
     alt="Dr. Jian Sun" width="200">
</div>
