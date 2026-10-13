import os
from pathlib import Path

import numpy as np
import pytest
import torch

from yolox.config import YoloxConfig
from yolox.models import YoloxModule
from yolox.utils import load_checkpoint

SAFE_GLOBALS_SUPPORTED = tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (2, 6)


def _save(path: Path, best_ap: object, curr_ap: object) -> None:
    # Same keys as Trainer.save_ckpt()
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    state = {
        'start_epoch': 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_ap': best_ap,
        'curr_ap': curr_ap,
    }
    torch.save(state, path)


class _Payload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple:
        return (os.system, (f'echo PWNED > {self.marker}',))


class TestLoadCheckpoint:
    def test_builtin_floats(self, tmp_path: Path) -> None:
        _save(tmp_path / 'new.pth', 0.25, 0.25)
        assert load_checkpoint(tmp_path / 'new.pth')['best_ap'] == pytest.approx(0.25)

    @pytest.mark.skipif(not SAFE_GLOBALS_SUPPORTED, reason='needs torch >= 2.6')
    def test_numpy_scalars(self, tmp_path: Path) -> None:
        _save(tmp_path / 'old.pth', np.float64(0.25), np.float64(0.25))
        ckpt = load_checkpoint(tmp_path / 'old.pth')
        assert (ckpt['best_ap'], ckpt['curr_ap']) == pytest.approx((0.25, 0.25))

    @pytest.mark.parametrize('with_numpy', [False, True])
    def test_rejects_arbitrary_code(self, tmp_path: Path, with_numpy: bool) -> None:
        marker = tmp_path / 'pwned.txt'
        best_ap = np.float64(0.1) if with_numpy else 0.1
        _save(tmp_path / 'evil.pth', best_ap, _Payload(marker))
        with pytest.raises(RuntimeError, match='Refusing to load'):
            load_checkpoint(tmp_path / 'evil.pth')
        assert not marker.exists()

    def test_from_pretrained_file(self, tmp_path: Path) -> None:
        config = YoloxConfig.get_named_config('yolox_nano')
        model = config.get_model()
        torch.save({'model': model.state_dict(), 'best_ap': 0.0, 'curr_ap': None}, tmp_path / 'nano.pth')
        loaded = YoloxModule.from_pretrained(str(tmp_path / 'nano.pth'), config=config)
        assert loaded.state_dict().keys() == model.state_dict().keys()
