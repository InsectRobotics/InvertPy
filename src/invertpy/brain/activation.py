"""
Some commonly used activation functions.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from ._helpers import RNG

from scipy.special import expit

import numpy as np


def leaky_relu(x, leak=.08, cmin=-np.inf, cmax=np.inf, noise=0., rng=RNG):
    """
    The noisy leaky ReLU activation function reduces the negative values of the input based to the leak, adds Gaussian
    noise and clips the output to its boundaries.

        y = max(x, leak * x) + rng.normal(scale=noise, size=len(x))

        y' = clip(y, cmin, cmax)

    Parameters
    ----------
    x: np.ndarray | float
        the input values.
    leak: float, optional
        the leaking constant
    cmin: float, optional
        the minimum constant
    cmax: float, optional
        the maximum constant
    noise: float, optional
        the variance of the noise
    rng
        the random generator

    Returns
    -------
    y': np.ndarray
        the output of the activation function
    """
    lr = np.maximum(x, leak * x) + _get_noise(noise, rng=rng, size=_get_size(x))
    return np.clip(lr, cmin, cmax)


def linear(x, cmin=-np.inf, cmax=np.inf, noise=0., rng=RNG):
    """
    The noisy linear activation function just adds Gaussian noise to the input and clips it to its boundaries.

        y = x + rng.normal(scale=noise, size=len(x))

        y' = clip(y, cmin, cmax)

    Parameters
    ----------
    x: np.ndarray | float
        the input values.
    cmin: float, optional
        the minimum constant
    cmax: float, optional
        the maximum constant
    noise: float, optional
        the variance of the noise
    rng
        the random generator

    Returns
    -------
    y': np.ndarray
        the output of the activation function
    """
    return leaky_relu(x, leak=1., cmin=cmin, cmax=cmax, noise=noise, rng=rng)


def relu(x, cmin=0., cmax=np.inf, noise=0., rng=RNG):
    """
    The noisy ReLU activation function ignores the negative values of the input, adds Gaussian noise and
    clips the output to its boundaries.

        y = max(x, 0) + rng.normal(scale=noise, size=len(x))

        y' = clip(y, cmin, cmax)

    Parameters
    ----------
    x: np.ndarray | float
        the input values.
    cmin: float, optional
        the minimum constant
    cmax: float, optional
        the maximum constant
    noise: float, optional
        the variance of the noise
    rng
        the random generator

    Returns
    -------
    y': np.ndarray
        the output of the activation function
    """
    return leaky_relu(x, leak=0., cmin=cmin, cmax=cmax, noise=noise, rng=rng)


def sigmoid(x, cmin=0, cmax=1, noise=0., rng=RNG):
    """
    Takes a vector v as input, puts through sigmoid and adds Gaussian noise. Results are clipped to return rate
    between 0 and 1.

        y = 1 / (1 + exp(-x)) + rng.normal(scale=noise, size=len(x))

        y' = clip(y, cmin, cmax)

    Parameters
    ----------
    x: np.ndarray | float
        the input values.
    cmin: float, optional
        the minimum constant
    cmax: float, optional
        the maximum constant
    noise: float, optional
        the variance of the noise
    rng
        the random generator

    Returns
    -------
    y': np.ndarray
        the output of the activation function
    """
    sig = expit(x) + _get_noise(noise, rng=rng, size=_get_size(x))
    return np.clip(sig, cmin, cmax)


def softmax(x, tau=1., cmin=0., cmax=1, noise=0., rng=RNG, axis=None):
    """
    The Softmax function can be used to convert values to probabilities.

        y = exp(x / tau) / sum(exp(x / tau)) + rng.normal(scale=noise, size=len(x))

        y' = clip(y, cmin, cmax)

    The tau is called a temperature parameter (in allusion to statistical mechanics).
    For high temperatures (tau -> inf), all probabilities are the same (uniform distribution) and for
    low temperatures (tau -> 0) the value affects the probability more (probability of the highest value tends to 1).

    Parameters
    ----------
    x: np.ndarray, float
        The input values.
    tau: float, optional
        The temperature parameter.
    cmin: float, optional
        the minimum constant
    cmax: float, optional
        the maximum constant
    noise: float, optional
        the variance of the noise
    rng
        the random generator
    axis: int | tuple
        the axis to perform the normalisation on.
    Returns
    -------
    y': np.ndarray
        The probability of each each of the input values
    """
    y = np.exp(x / tau)
    y = np.clip(y / np.sum(y, axis=axis), 0., 1e+16) + _get_noise(noise, rng=rng, size=_get_size(x))
    return np.clip(y, cmin, cmax)


def winner_takes_all(x, tau=None, percentage=.05, normalise=False, cmin=0., cmax=1., noise=0., rng=RNG):
    """
    The Winner Takes All (WTA) algorithm can be used to force sparse coding.

    This can be done by either specifying the percentage of active neurons that we want or a fixed threshold (tau).

        y = x >= np.sort(x)[::-1][percentage * len(x)] + rng.normal(scale=noise)

        or

        y = x >= tau + rng.normal(scale=noise)

        y' = clip(y, cmin, cmax)


    Parameters
    ----------
    x: np.ndarray, float
        the input values.
    tau: float, optional
        anything higher than this threshold will be active and anything lower will be inactive. If None, then the
        percentage approach is applied.
    percentage: float, optional
        the percentage of the active neurons that we want to keep.
    normalise: bool, optional
        if True, then the output will sum to one.
    cmin: float, optional
        the minimum constant
    cmax: float, optional
        the maximum constant
    noise: float, optional
        the variance of the noise
    rng
        the random generator
    Returns
    -------
    y': np.ndarray
        The output of the activation function.
    """
    y = x + _get_noise(noise, rng=rng, size=_get_size(x))
    if tau is None:
        y = np.asarray(np.greater(y.T, np.quantile(y, 1 - percentage, axis=-1)).T, dtype=x.dtype)
    else:
        y = np.asarray(np.greater_equal(y, tau), dtype=x.dtype)

    if normalise:
        y /= (y.sum() + np.finfo(float).eps)

    return np.clip(y, cmin, cmax)


def hardmax(x, cmin=0., cmax=1., noise=0., rng=RNG, axis=None):
    y = x + _get_noise(noise, rng=rng, size=_get_size(x))
    y = np.eye(y.shape[-1], dtype=x.dtype)[np.argmax(y, axis=axis)]

    return np.clip(y, cmin, cmax)


def _get_noise(eta, size=None, rng=RNG):
    return rng.uniform(low=-eta, high=eta, size=size)


def _get_size(x):
    if hasattr(x, "shape"):
        return x.shape
    elif hasattr(x, "len"):
        return len(x)
    else:
        return None
