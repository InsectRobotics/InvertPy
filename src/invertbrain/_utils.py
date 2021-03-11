import numpy as np

RNG = np.random.RandomState(2021)
eps = np.finfo(float).eps


def set_rng(seed):
    global RNG
    RNG = np.random.RandomState(seed)
