import numpy as np


RNG = np.random.RandomState(2021)
"""
The defaults random value generator.
"""
eps = np.finfo(float).eps
"""
The smallest non-zero positive.
"""


def set_rng(seed):
    """
    Sets the default random state.

    Parameters
    ----------
    seed: int
    """
    global RNG
    RNG = np.random.RandomState(seed)
