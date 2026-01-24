"""Monkey patch to fix ManiSkill3 flatten_state_dict failing on numpy.ndarray inputs."""
from typing import Optional

import numpy as np
import torch

from mani_skill.utils import common
from mani_skill.utils.structs.types import Device

_old_flatten = common.flatten_state_dict

def patched_flatten_state_dict(state_dict, use_torch=False, device: Optional[Device] = None):
    use_torch = isinstance(state_dict, np.ndarray)
    flat = _old_flatten(state_dict, use_torch=use_torch, device=device)
    if use_torch and isinstance(flat, torch.Tensor):
        flat = flat.cpu().detach().numpy()
    return flat

common.flatten_state_dict = patched_flatten_state_dict
