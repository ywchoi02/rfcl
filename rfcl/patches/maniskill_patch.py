"""Monkey patch to fix ManiSkill3 flatten_state_dict failing on numpy.ndarray inputs."""
from typing import Optional

import numpy as np
import torch

from mani_skill.utils import common
from mani_skill.utils.structs.types import Array, Device

to_tensor = common.to_tensor


def _is_empty(x):
    if isinstance(x, torch.Tensor):
        return x.numel() == 0
    elif isinstance(x, np.ndarray):
        return x.size == 0
    else:
        return False
        
def flatten_state_dict(
    state_dict: dict, use_torch=False, device: Optional[Device] = None
) -> Array:
    """Flatten a dictionary containing states recursively. Expects all data to be either torch or numpy

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.
        use_torch (bool): Whether to convert the data to torch tensors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray | torch.Tensor: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. dict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []

    for key, value in state_dict.items():
        if isinstance(value, dict):
            state = flatten_state_dict(value, use_torch=use_torch, device=device)
            # if state.nelement() == 0:
            #     state = None
            if _is_empty(state):
                state = None
            elif use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
            if use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
            if use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
            if use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value if value.size > 0 else None
            if use_torch:
                state = to_tensor(state, device=device)

        elif isinstance(value, torch.Tensor):
            state = value
            if len(state.shape) == 1:
                state = state[:, None]
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)

    if use_torch:
        if len(states) == 0:
            return torch.empty(0, device=device)
        else:
            return torch.hstack(states)
    else:
        if len(states) == 0:
            return np.empty(0)
        else:
            return np.hstack(states)


common.flatten_state_dict = flatten_state_dict
