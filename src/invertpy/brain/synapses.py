"""
Package that contains helpers for initialising the synaptic weights between groups of neurons.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

import sys

import scipy.signal

from ._helpers import RNG, pca, whitening, eps
from .activation import softmax

from scipy.special import comb
from scipy.spatial.transform import Rotation as R

import numpy as np


def random_synapses(nb_in, nb_out, w_min=-1, w_max=1, dtype='float32', bias=None, rng=np.random.RandomState(2021)):
    """
    Creates random synapses.

    Parameters
    ----------
    nb_in : int
        the number of the input units.
    nb_out: int
        the number of the output units.
    w_min : float, optional
        the minimum synaptic weight
    w_max : float, optional
        the maximum synaptic weight
    dtype : np.dtype | str
        the type of the values for the synaptic weights.
    bias : float | bool
        the value of all the biases. If bool, the biases are also random. If None, no bias is returned.
    rng : np.random.RandomState
        the random number generator

    Returns
    -------
    np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    w = np.asarray(rng.uniform(low=w_min, high=w_max, size=(nb_in, nb_out)), dtype=dtype)
    if bias is None:
        return w
    elif isinstance(bias, bool) and bias:
        return w, np.asarray(rng.uniform(low=w_min, high=w_max, size=(nb_out,)), dtype=dtype)
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def uniform_synapses(nb_in, nb_out, fill_value=0, dtype='float32', bias=None):
    """
    Creates uniform synapses.

    Examples
    --------
    >>> uniform_synapses(3, 2, fill_value=1)
    >>> np.array([[1, 1],
    >>>           [1, 1],
    >>>           [1, 1]], dtype='float32')

    Parameters
    ----------
    nb_in: int
        the number of the input units.
    nb_out: int
        the number of the output units.
    fill_value: float, optional
        the value of all the synaptic weights.
    dtype: np.dtype | str
        the type of the values for the synaptic weights.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    w = np.full((nb_in, nb_out), fill_value=fill_value, dtype=dtype)
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def diagonal_synapses(nb_in, nb_out, fill_value=1, tile=False, dtype='float32', bias=None):
    """
    Creates diagonal synapses.

    Examples
    --------
    >>> diagonal_synapses(3, 4, fill_value=2)
    >>> np.array([[2, 0, 0, 0],
    >>>           [0, 2, 0, 0],
    >>>           [0, 0, 2, 0]], dtype='float32')

    >>> diagonal_synapses(3, 6, tile=True, fill_value=1)
    >>> np.array([[1, 0, 0, 1, 0, 0],
    >>>           [0, 1, 0, 0, 1, 0],
    >>>           [0, 0, 1, 0, 0, 1]], dtype='float32')

    Parameters
    ----------
    nb_in: int
        the number of the input units.
    nb_out: int
        the number of the output units.
    fill_value: float
        the value of the non-zero synaptic weights.
    tile: bool, optional
        if True and nb_in != nb_out, then it wraps the diagonal starting from the beginning.
    dtype: np.dtype | str
        the type of the values for the synaptic weights.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    w = None
    if tile:
        if nb_in // nb_out > 1:
            w = fill_value * np.tile(np.eye(nb_out, dtype=dtype), (nb_in//nb_out, 1))
        elif nb_out // nb_in > 1:
            w = fill_value * np.tile(np.eye(nb_in, dtype=dtype), (1, nb_out//nb_in))
        else:
            tile = False
    if not tile:
        w = fill_value * np.eye(nb_in, nb_out, dtype=dtype)
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


# def sparse_synapses(nb_in, nb_out, nb_in_min=None, nb_in_max=None, max_corr=0.8, max_repeat=20, normalise=True,
#                     dtype='float32', rng=RNG, bias=None):
#     """
#     Creates sparse synapses.
#
#     Parameters
#     ----------
#     nb_in: int
#         the number of the input units.
#     nb_out: int
#         the number of the output units.
#     nb_in_min: int, optional
#         the minimum number of input neurons connected to each output neuron.
#     nb_in_max: int, optional
#         the maximum number of input neurons connected to each output neuron.
#     max_corr : float, optional
#         the maximum allowed correlation between connection patterns of the sparse code.
#     normalise: bool, optional
#         if the synaptic weights for each output neuron should sum to 1.
#     dtype: np.dtype | str
#         the type of the values for the synaptic weights.
#     rng
#         the random value generator.
#     bias: float, optional
#         the value of all the biases. If None, no bias is returned.
#
#     Returns
#     -------
#     params: np.ndarray | tuple
#         the generated synaptic weights and the bias (if requested)
#     """
#     w = uniform_synapses(nb_in, nb_out * max_repeat, fill_value=0., dtype=dtype)
#
#     if nb_in_min is None:  # default: 6
#         nb_in_min = max(int(nb_in * 6. / 1000.), 1)
#     if nb_in_max is None:  # default: 14
#         nb_in_max = max(int(nb_in * 14. / 1000.), 1)
#
#     max_corr = np.clip(max_corr, 0, 1)  # make sure that it is in the correct range
#
#     # number of input connections for each of of the output (sparse) neurons
#     nb_out_in = np.asarray(rng.rand(nb_out * max_repeat) * (nb_in_max - nb_in_min) + nb_in_min, dtype='int32')
#
#     c_out_in = np.zeros(nb_out * max_repeat, dtype=int)  # accumulated output connections from input neurons
#     duplicates = np.arange(nb_out * max_repeat)  # the indices of the duplicated PN-KC entries
#
#     while len(duplicates) > 0:
#         i = rng.randint(0, nb_in - 1)
#         w[:, duplicates] = 0.
#         c_out_in[duplicates] = 0
#         while c_out_in.sum() < nb_out_in.sum():
#             ii = i
#             j_s = duplicates[np.argsort(nb_out_in[rng.permutation(duplicates)])[::-1]]
#             # print(np.sort(nb_out_in[duplicates]))
#             for j in j_s:  # iterate over the different output neurons
#
#                 # if the number of connections for the output neuron has reached its limit
#                 # continue to the next output neuron
#                 if c_out_in[j] >= nb_out_in[j] or w[i, j] > 0:
#                     continue
#                 w[i, j] = 1. / nb_out_in[j] if normalise else 1.
#                 i = (i + 1) % nb_in
#                 c_out_in[j] += 1
#             if i == ii:
#                 i = (i + 1) % nb_in
#
#         c = np.dot(w.T, w) / np.outer(np.linalg.norm(w, axis=0), np.linalg.norm(w, axis=0))
#         duplicates = np.arange(nb_out)[np.greater_equal(c - np.diag(np.diagonal(c)), max_corr).max(axis=0)]
#         print(f"Number of duplicates: {duplicates.shape[0]}")
#
#     w = rng.permutation(w)
#
#     if bias is None:
#         return w
#     else:
#         return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def sparse_synapses(nb_in, nb_out, nb_in_min=None, nb_in_max=None, max_samples=80000, normalise=True,
                    dtype='float32', rng=RNG, bias=None):
    """
    Creates sparse synapses.

    Parameters
    ----------
    nb_in: int
        the number of the input units.
    nb_out: int
        the number of the output units.
    nb_in_min: int, optional
        the minimum number of input neurons connected to each output neuron.
    nb_in_max: int, optional
        the maximum number of input neurons connected to each output neuron.
    max_samples : int, optional
        the number of times that the algorithm will generate different weights in order to choose the ones with the
        least correlation. Default is 5
    normalise: bool, optional
        if the synaptic weights for each output neuron should sum to 1.
    dtype: np.dtype | str
        the type of the values for the synaptic weights.
    rng
        the random value generator.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    w = uniform_synapses(nb_in, max_samples, fill_value=0., dtype=dtype)

    if nb_in_min is None:  # default: 6
        # nb_in_min = max(int(nb_in * 6. / 1000.), 1)
        nb_in_min = int(max(nb_in / nb_out, 2))
    if nb_in_max is None:  # default: 14
        # nb_in_max = max(int(nb_in * 14. / 1000.), 1)
        nb_in_max = nb_in_min
        while comb(nb_in, nb_in_max - 1) < comb(nb_in, nb_in_max) < np.power(nb_in * nb_out, 4):
            nb_in_max += 1

    # print(f"in: {nb_in}, out: {nb_out}, max(in): {nb_in_max}, repeats: {max_samples}")

    # number of input connections for each of of the output (sparse) neurons
    nb_out_in = np.asarray(rng.rand(max_samples) * (nb_in_max - nb_in_min + 1) + nb_in_min, dtype='int32')
    c_out_in = np.zeros(max_samples, dtype=int)  # accumulated output connections from input neurons

    z = nb_out_in if normalise else np.ones_like(nb_out_in)

    i_rep = 0
    nb_unchanged = 0
    last_completed = 0
    while np.any(np.greater(nb_out_in, c_out_in)):
        # explore only the output neurons that have not reached their connections limit yet
        j_c = np.arange(max_samples)[np.greater(nb_out_in, c_out_in)]

        nb_syn_unique = np.sort(np.setdiff1d(np.unique(nb_out_in[j_c]), {0}))

        for nb_syn in nb_syn_unique:
            # find the output neurons that have the target number of synapses
            j_s = rng.permutation(j_c[nb_syn == nb_out_in[j_c]])

            # maximum number of unique combinations of N (nb_in) choose k (nb_syn) elements
            max_nb_comb = int(comb(nb_in, nb_syn))

            # if the number of input patterns requested is not realistic given the number of unique inputs
            # increase the number of unique inputs
            if max_nb_comb < len(j_s) and nb_syn < nb_in:
                sub_set = max(max_nb_comb // 2, nb_in)
                nb_out_in[j_s[sub_set:]] += 1
                j_s = j_s[:sub_set]

            for syn in range(nb_syn):
                for j in j_s:
                    i = np.argmin(w.sum(axis=1))
                    w[i, j] = 1. / z[j]

                    c_out_in[j] += 1

            w = w / (w.sum(axis=0) + eps)

        i_rep += 1
        if i_rep < nb_in:
            # find the unique PN patterns
            u_w, ui = np.unique(w, return_index=True, axis=1)

            # delete the non-unique (redundant) PN patterns
            oi = np.setdiff1d(np.arange(max_samples), ui)
            print(f"{i_rep+1:02}/{nb_in:02}: completed: {ui.shape[0]:05}/{w.shape[1]}")

            w[:, oi] = 0.
            c_out_in[oi] = 0

            if ui.shape[0] == last_completed:
                nb_unchanged += 1
            else:
                nb_unchanged = 0
            last_completed = ui.shape[0]
            if nb_unchanged > 1:
                w = u_w
                break
        else:
            break

    # normalise the synapses so that each group to inputs sum to 1
    w = w / (w.sum(axis=0) + eps)

    # find the unique input patterns
    u_w = np.unique(w, axis=1)

    # check if the unique patterns are enough for the weights matrix
    if u_w.shape[1] < nb_out:

        # if they are not enough create copies
        w = uniform_synapses(nb_in, nb_out, fill_value=0., dtype=dtype)
        for i in range(w.shape[1] // u_w.shape[1] + 1):
            i_start = i * u_w.shape[1]
            i_end = min((i + 1) * u_w.shape[1], w.shape[1])
            u_end = (i_end - 1) % u_w.shape[1] + 1

            w[:, i_start:i_end] = u_w[:, :u_end]
    else:
        # if they are enough keep only the unique input patterns
        w = u_w

    # create the correlation matrix among the generated input patterns
    c = np.dot(w.T, w) / (np.outer(np.linalg.norm(w, axis=0), np.linalg.norm(w, axis=0)) + eps)

    # sort the indices of the correlation matrix and keep only the ones representing the nb_out least correlations
    c_sorted_indices = np.argsort(c, axis=1)[:, :nb_out]

    # count the number of times that the index has been observed in the top nb_out (low) correlations
    indices, counts = np.unique(c_sorted_indices, return_counts=True)

    # sort the indices so that most frequent ones come to the front
    i = indices[np.argsort(counts)[::-1]]

    # select the best indices for the synapses
    w = rng.permutation(w[:, i[:nb_out]])

    # calculate correlation (for visualisation only)
    c = np.dot(w.T, w) / (np.outer(np.linalg.norm(w, axis=0), np.linalg.norm(w, axis=0)) + eps)
    cc = c - np.diag(np.diag(c) * np.nan)
    print(f"\nCorrelation: max={np.nanmax(cc):.2}, mean={np.nanmean(cc):.2}")

    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def opposing_synapses(nb_in, nb_out, fill_value=1., dtype='float32', bias=None):
    """
    Creates opposing synapses which is similar to some shifted diagonal synapses.

    Examples
    --------
    >>> diagonal_synapses(4, 4, fill_value=2)
    >>> np.array([[0, 0, 2, 0],
    >>>           [0, 0, 0, 2],
    >>>           [2, 0, 0, 0],
    >>>           [0, 2, 0, 0]], dtype='float32')

    >>> diagonal_synapses(2, 6, fill_value=1)
    >>> np.array([[0, 0, 0, 1, 1, 1],
    >>>           [1, 1, 1, 0, 0, 0]], dtype='float32')

    Parameters
    ----------
    nb_in: int
        the number of the input units.
    nb_out: int
        the number of the output units.
    fill_value: float, optional
        the value of the non-zero synaptic weights.
    dtype: np.dtype | str
        the type of the values for the synaptic weights.
    rng
        the random value generator.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    w = np.kron(fill_value * np.array([[0, 1], [1, 0]], dtype=dtype), np.eye(nb_in//2, nb_out//2, dtype=dtype))
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def sinusoidal_synapses(nb_in, nb_out, fill_value=1., dtype='float32', bias=None):
    """
    Creates a diagonal of sunusoidal synapses.

    Parameters
    ----------
    nb_in: int
        the number of the input units.
    nb_out: int
        the number of the output units.
    fill_value: float, optional
        the value of all the synaptic weights.
    dtype: np.dtype | str
        the type of the values for the synaptic weights.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    w = np.zeros((nb_in, nb_out), dtype=dtype)
    pref_in = np.linspace(0, 2*np.pi, nb_in, endpoint=False)
    for i in range(nb_in):
        w[i, :] = fill_value * (-np.cos(np.linspace(0, 2 * np.pi, nb_out, endpoint=False) + pref_in[i]) + 1) / 2
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def chessboard_synapses(nb_in, nb_out, fill_value=1., nb_rows=2, nb_cols=2, dtype='float32', bias=None):
    """
    Creates chessboard-like synapses.

    Parameters
    ----------
    nb_in: int
        the number of the input units.
    nb_out: int
        the number of the output units.
    fill_value: float, optional
        the value of all the synaptic weights.
    nb_rows: int, optional
        the number of chessboard rows
    nb_cols: int, optional
        the number of chessboard columns
    dtype: np.dtype | str
        the type of the values for the synaptic weights.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    pattern = np.array([[(i % 2 == 0) == (j % 2 == 0) for j in range(nb_cols)] for i in range(nb_rows)], dtype=dtype)
    if nb_out // nb_in > 1:
        patch = np.full((1, nb_out // nb_in), fill_value=fill_value, dtype=dtype)
    elif nb_in // nb_out > 1:
        patch = np.full((nb_in // nb_out, 1), fill_value=fill_value, dtype=dtype)
    else:
        patch = np.full((1, 1), fill_value=fill_value, dtype=dtype)
    return pattern_synapses(pattern, patch, dtype=dtype, bias=bias)


def dct_synapses(nb_in, dtype='float32'):
    """
    Creates Discrete Cosine Transform (DCT) synapses.

    nb_in: int
        the number of input neurons is the same as the number of output neurons.
    dtype: np.dtype, optional
        the type of the values for the synaptic weights.

    Returns
    -------
    params: np.ndarray
        the generated synaptic weights
    """
    n = np.arange(nb_in)
    m = np.arange(nb_in)
    c = (1 / np.sqrt(1 + np.asarray(np.isclose(m, 0), dtype=dtype)))[..., np.newaxis]
    d = np.cos(np.pi * m[..., np.newaxis] * (2 * n + 1) / (2 * nb_in))
    A = np.sqrt(2 / nb_in) * c * d

    return A


def dct_omm_synapses(omm_ori, dtype='float32'):
    """
    Creates Discrete Cosine Transform (DCT) synapses based on the ommatidia orientations.

    Parameters
    ----------
    omm_ori: R
        the ommatidia orientations.
    dtype: np.dtype, optional
        the type of the values for the synaptic weights.

    Returns
    -------
    params: np.ndarray
        the generated synaptic weights
    """
    nb_in = float(np.shape(omm_ori)[0])

    phi, theta, _ = omm_ori.as_euler('ZYX', degrees=False).T
    phi = phi % (2 * np.pi)
    theta = (np.pi/2 + theta) % np.pi

    m = np.argsort(phi)
    n = np.argsort(theta)

    c = (1 / np.sqrt(1 + np.asarray(np.isclose(m, 0), dtype=dtype)))[..., np.newaxis]
    d = np.cos(np.pi * m[..., np.newaxis] * (2 * n + 1) / (2 * nb_in))
    A = np.sqrt(2 / nb_in) * c * d

    return A.T


def whitening_synapses(samples, nb_out=None, samples_mean=None, w_func=pca, dtype='float32', bias=None):
    """
    Whitening synapses based on the samples and function.

    Parameters
    ----------
    samples: np.ndarray
        the samples from which the whitening synaptic weights will be created.
    samples_mean: np.ndarray, optional
        the mean value of the samples. If None, it will be calculated automatically.
    w_func: callable, optional
        the whitening function.
    dtype: np.dtype, optional
        the type of the values for the synaptic weights.
    bias: bool, optional
        whether to return the mean value of the samples as a bias or not.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the mean of the samples (if requested).
    """
    if samples_mean is None:
        samples_mean = samples.mean(axis=0)

    w = w_func(samples, nb_out=nb_out, m=samples_mean, dtype=dtype)

    if bias:
        return w, samples_mean
    else:
        return w


def pattern_synapses(pattern, patch, dtype='float32', bias=None):
    """
    Created synapses by repeating a patch over a pattern.

    Parameters
    ----------
    pattern: np.ndarray
        a matrix where each value will be multiplied with the patch creating a pattern.
    patch: np.ndarray
        a matrix that will be repeated based on the pattern
    dtype: np.dtype, optional
        the type of teh values for the synaptic weights.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the mean of the samples (if requested).
    """

    w = np.kron(pattern, patch)
    if bias is None:
        return w
    else:
        return w, np.full(w.shape[1], fill_value=bias, dtype=dtype)


def roll_synapses(w, left=None, right=None, up=None, down=None):
    """
    Rolls the synapses for a number of position and towards a given direction.

    Parameters
    ----------
    w: np.ndarray
        the input synaptic wegiths.
    left: int, optional
        the number of positions to shift towards the left.
    right: int, optional
        the number of positions to shift towards the right.
    up: int, optional
        the number of positions to shift upwards.
    down: int, optional
        the number of positions to shift downwards.

    Returns
    -------
    w_out: np.ndarray
        the result synaptic weights.
    """

    if left is not None:
        w = np.hstack([w[:, int(left):], w[:, :int(left)]])
    elif right is not None:
        w = np.hstack([w[:, -int(right):], w[:, :-int(right)]])

    if up is not None:
        w = np.vstack([w[int(left):, :], w[:int(left), :]])
    elif down is not None:
        w = np.vstack([w[-int(right):, :], w[:-int(right), :]])

    return w


def mental_rotation_synapses(omm_ori, nb_out, phi_out=None, sigma=.02, dtype='float32'):
    """
    Builds a matrix (nb_om x nb_om x nb_out) that performs mental rotation of the visual input.

    In practice, it builds a maps for each of the uniformly distributed nb_out view directions,
    that allow internal rotation of the visual input for different orientations of interest (preference angles).

    Parameters
    ----------
    omm_ori: R
        orientations of the ommatidia
    nb_out: int
        number of the different tuning points (preference angles)
    phi_out: np.ndarray, optional
        list of the preference angles for the mental rotations. Default is angles uniformly distributed in a circle.
    sigma: float, optional
        mental radius of each ommatidium
    dtype: np.dtype, optional
        the type of the data in the array of weights

    Returns
    -------
    np.ndarray[float]
        A matrix that maps the input space of the eye to nb_out uniformly distributed
    """

    nb_omm = np.shape(omm_ori)[0]
    w = np.zeros((nb_omm, nb_omm, nb_out), dtype=dtype)
    if phi_out is None:
        phi_out = np.linspace(0, 2 * np.pi, nb_out, endpoint=False)

    assert len(phi_out) == nb_out, (
        "The list of preference angles should be of the same size as the 'nb_out'."
    )

    for i in range(nb_out):
        i_ori = R.from_euler('Z', -phi_out[i], degrees=False) * omm_ori
        for j in range(nb_omm):
            j_ori = omm_ori[j]
            d = np.linalg.norm(j_ori.apply([1, 0, 0]) - i_ori.apply([1, 0, 0]), axis=1) / 2
            w[j, :, i] = softmax(1. - d, tau=sigma)

    return w
