import torch

from yolox.config import YoloxConfig, YoloxS
from yolox.models import YoloxModule


def test_named_config_returns_new_instance() -> None:
    first = YoloxConfig.get_named_config('yolox_s')
    first.num_classes = 3
    second = YoloxConfig.get_named_config('yolox-s')
    assert isinstance(second, YoloxS)
    assert second is not first
    assert second.num_classes == 80
    assert YoloxConfig.get_named_config('yolox_unknown') is None


def test_from_pretrained_models_are_independent() -> None:
    first = YoloxModule.from_pretrained('yolox_nano')
    bias = first.head.cls_preds[0].bias.detach().clone()
    second = YoloxModule.from_pretrained('yolox_nano')
    assert second is not first
    # get_model() re-initializes biases and switches to train mode; it must not touch a loaded model
    YoloxConfig.get_named_config('yolox_nano').get_model()
    assert torch.equal(first.head.cls_preds[0].bias.detach(), bias)
    assert not first.training
