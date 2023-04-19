import torch
import numpy as np

def apply_along_axis(function, x, axis: int = 0):
    npax = x.numpy()
    res = np.apply_along_axis(function, axis, npax)
    return torch.from_numpy(res)