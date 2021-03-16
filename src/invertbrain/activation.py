from ._helpers import RNG

from scipy.special import expit

import numpy as np


def leaky_relu(x, leak=.08, cmin=-np.inf, cmax=np.inf, noise=0., rng=RNG):
    lr = np.maximum(x, leak * x) + rng.normal(scale=noise, size=len(x))
    return np.clip(lr, cmin, cmax)


def linear(x, cmin=-np.inf, cmax=np.inf, noise=0., rng=RNG):
    return leaky_relu(x, leak=1., cmin=cmin, cmax=cmax, noise=noise, rng=rng)


def relu(x, cmin=-np.inf, cmax=np.inf, noise=0., rng=RNG):
    return leaky_relu(x, leak=0., cmin=cmin, cmax=cmax, noise=noise, rng=rng)


def sigmoid(x, cmin=0, cmax=1, noise=0., rng=RNG):
    """
    Takes a vector v as input, puts through sigmoid and adds Gaussian noise. Results are clipped to return rate
    between 0 and 1.

    Parameters
    ----------
    x
    cmin
    cmax
    noise
    rng

    Returns
    -------

    """
    sig = expit(x) + rng.normal(scale=noise, size=len(x))
    return np.clip(sig, cmin, cmax)
