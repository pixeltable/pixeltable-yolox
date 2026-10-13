# Copyright (c) Megvii Inc. All rights reserved.
import contextlib
import os
import pickle
import shutil
from typing import Union

import numpy as np
import torch
from loguru import logger


def _numpy_scalar_globals() -> list:
    # Older trainers stored best_ap/curr_ap as numpy.float64 (from cocoEval.stats).
    # Unpickling one needs exactly these three globals.
    try:
        from numpy._core.multiarray import scalar
    except ImportError:
        from numpy.core.multiarray import scalar
    allowed = [np.dtype, type(np.dtype(np.float64))]
    if tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (2, 6):
        # Files name the module of the numpy that wrote them: numpy.core (1.x) or numpy._core (2.x).
        allowed += [(scalar, 'numpy.core.multiarray.scalar'), (scalar, 'numpy._core.multiarray.scalar')]
    else:
        allowed.append(scalar)
    return allowed


def load_checkpoint(path: Union[str, os.PathLike], map_location: Union[str, torch.device] = 'cpu') -> dict:
    """Load a checkpoint with torch.load(weights_only=True). Never falls back to full unpickling."""
    # torch.serialization.safe_globals() needs torch >= 2.5; on older torch, numpy scalars are rejected.
    safe_globals = getattr(torch.serialization, 'safe_globals', None)
    scope = safe_globals(_numpy_scalar_globals()) if safe_globals else contextlib.nullcontext()
    try:
        with scope:
            return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f'Refusing to load {path}: it contains objects that torch.load(weights_only=True) rejects. '
            'Checkpoints from older yolox trainers store numpy scalars, which load safely on torch >= 2.6. '
            'If you trust the file, convert it once: '
            "ckpt = torch.load(path, weights_only=False); ckpt['best_ap'] = float(ckpt['best_ap']); "
            "ckpt['curr_ap'] = None; torch.save(ckpt, path)"
        ) from e


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
