"""Monkey patch to fix ManiSkill3 flatten_state_dict failing on numpy.ndarray inputs."""
import numpy as np
import torch
from mani_skill.utils import common

_old_flatten = common.flatten_state_dict

def patched_flatten_state_dict(state_dict):
    return _old_flatten(
        state_dict, 
        use_torch=isinstance(state_dict, np.ndarray),
    )

common.flatten_state_dict = patched_flatten_state_dict
