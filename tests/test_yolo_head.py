import pytest
import torch

from yolox.models import YoloxHead

IMG_SIZE = 64
WIDTH = 0.25
IN_CHANNELS = [256, 512, 1024]
STRIDES = [8, 16, 32]


def _train_forward(boxes: list[tuple[float, float, float, float, float]]) -> tuple:
    torch.manual_seed(0)
    head = YoloxHead(num_classes=2, width=WIDTH, in_channels=IN_CHANNELS)
    head.train()
    feats = [torch.randn(1, int(c * WIDTH), IMG_SIZE // s, IMG_SIZE // s) for c, s in zip(IN_CHANNELS, STRIDES)]
    labels = torch.zeros(1, 4, 5)  # (cls, cx, cy, w, h) in pixels, zero rows are padding
    labels[0, : len(boxes)] = torch.tensor(boxes)
    with torch.no_grad():
        return head(feats, labels=labels, imgs=torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))


@pytest.mark.parametrize(
    'boxes',
    [
        [(1, 1000, 32, 10, 10)],  # center far outside the image
        [(1, 1000, 1000, 0, 0)],  # degenerate box, also outside
    ],
)
def test_losses_when_no_anchor_matches_any_gt(boxes: list[tuple[float, float, float, float, float]]) -> None:
    # Previously raised "RuntimeError: selected index k out of range" in simota_matching.
    loss, iou_loss, obj_loss, cls_loss, _, _ = _train_forward(boxes)
    assert torch.isfinite(loss)
    assert float(iou_loss) == pytest.approx(0.0)  # no foreground anchors
    assert float(cls_loss) == pytest.approx(0.0)
    assert float(obj_loss) > 0.0  # every anchor is still trained as background


def test_losses_with_matchable_gt() -> None:
    loss, iou_loss, _, cls_loss, _, _ = _train_forward([(1, 32, 32, 20, 20)])
    assert torch.isfinite(loss)
    assert float(iou_loss) > 0.0
    assert float(cls_loss) > 0.0
