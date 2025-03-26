from typing import Optional

from .yolox_config import YoloxConfig, YoloxL, YoloxM, YoloxNano, YoloxS, YoloxTiny, YoloxX

_NAMED_CONFIG = {
    'yolox_s': YoloxS(),
    'yolox_m': YoloxM(),
    'yolox_l': YoloxL(),
    'yolox_x': YoloxX(),
    'yolox_tiny': YoloxTiny(),
    'yolox_nano': YoloxNano(),
}


def get_named_config(name: str) -> Optional[YoloxConfig]:
    return _NAMED_CONFIG.get(name)
