__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

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


def softmax(z, tau=1., axis=None):
    """
    The Softmax function can be used to convert values to probabilities.
    P(z) = exp(z / tau) / sum(exp(z / tau))
    The tau is called a temperature parameter (in allusion to statistical mechanics).
    For high temperatures (tau -> inf), all probabilities are the same (uniform distribution) and for
    low temperatures (tau -> 0) the value affects the probability more (probability of the highest value tends to 1).
    Parameters
    ----------
    z: np.ndarray, float
        The input values.
    tau: float
        The temperature parameter.
    axis:
        The axis to perform the normalisation on.
    Returns
    -------
        The probability of each each of the input values
    """
    s = np.exp(z / tau)
    return np.clip(s.T / np.sum(s, axis=axis), 0., 1e+16).T
