import numpy as np

RNG = np.random.RandomState(2021)
eps = np.finfo(float).eps


def leaky_relu(x, leak=.08, cmin=-np.inf, cmax=np.inf):
    return np.clip(np.maximum(x, leak * x), cmin, cmax)


def linear(x, cmin=-np.inf, cmax=np.inf):
    return leaky_relu(x, leak=1., cmin=cmin, cmax=cmax)


def relu(x, cmin=-np.inf, cmax=np.inf):
    return leaky_relu(x, leak=0., cmin=cmin, cmax=cmax)


def set_rng(seed):
    global RNG
    RNG = np.random.RandomState(seed)
