from ._helpers import RNG, pca, whitening

import numpy as np


def init_synapses(nb_in, nb_out, fill_value=0, dtype='float32', bias=None):
    w = np.full((nb_in, nb_out), fill_value=fill_value, dtype=dtype)
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def diagonal_synapses(nb_in, nb_out, fill_value=1, tile=False, dtype='float32', bias=None):
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


def sparse_synapses(nb_in, nb_out, nb_in_min=None, nb_in_max=None, normalise=True, dtype='float32', rng=RNG, bias=None):
    w = init_synapses(nb_in, nb_out, dtype=dtype)

    if nb_in_min is None:  # default: 6
        nb_in_min = max(int(nb_in * 6. / 1000.), 1)
    if nb_in_max is None:  # default: 14
        nb_in_max = max(int(nb_in * 14. / 1000.), 1)

    # number of input connections for each of of the output (sparse) neurons
    nb_out_in = np.asarray(rng.rand(nb_out) * (nb_in_max - nb_in_min) + nb_in_min, dtype='int32')

    c_out_in = np.zeros(nb_out, dtype=int)  # accumulated output connections from input neurons

    i = 0
    while c_out_in.sum() < nb_out_in.sum():
        for j in range(nb_out):
            if c_out_in[j] >= nb_out_in[j] or w[i, j] > 0:
                continue
            w[i, j] = 1
            i = (i + 1) % nb_in
            c_out_in[j] += 1
    w = rng.permutation(w)
    if normalise:
        w = w / w.sum(axis=0)

    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def opposing_synapses(nb_in, nb_out, fill_value=1., dtype='float32', bias=None):
    w = np.kron(fill_value * np.array([[0, 1], [1, 0]], dtype=dtype), np.eye(nb_in//2, nb_out//2, dtype=dtype))
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def sinusoidal_synapses(nb_in, nb_out, fill_value=1., dtype='float32', bias=None):
    w = np.zeros((nb_in, nb_out), dtype=dtype)
    sinusoid = fill_value * (-np.cos(np.linspace(0, 2 * np.pi, nb_out, endpoint=False)) + 1) / 2
    for i in range(nb_in):
        w[i, :] = np.roll(sinusoid, i)
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def chessboard_synapses(nb_in, nb_out, fill_value=1., nb_rows=2, nb_cols=2, dtype='float32', bias=None):
    pattern = np.array([[(i % 2 == 0) == (j % 2 == 0) for j in range(nb_cols)] for i in range(nb_rows)], dtype=dtype)
    if nb_out // nb_in > 1:
        patch = np.full((1, nb_out // nb_in), fill_value=fill_value, dtype=dtype)
    elif nb_in // nb_out > 1:
        patch = np.full((nb_in // nb_out, 1), fill_value=fill_value, dtype=dtype)
    else:
        patch = np.full((1, 1), fill_value=fill_value, dtype=dtype)
    return pattern_synapses(pattern, patch, dtype=dtype, bias=bias)


def dct_synapses(nb_in, dtype='float32'):
    n = np.arange(nb_in)
    m = np.arange(nb_in)
    c = (1 / np.sqrt(1 + np.asarray(np.isclose(m, 0), dtype=dtype)))[..., np.newaxis]
    d = np.cos(np.pi * m[..., np.newaxis] * (2 * n + 1) / (2 * nb_in))
    A = np.sqrt(2 / nb_in) * c * d

    return A


def dct_omm_synapses(omm_ori, dtype='float32'):
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


def whitening_synapses(samples, samples_mean=None, w_func=pca, dtype='float32', bias=None):
    if samples_mean is None:
        samples_mean = samples.mean(axis=0)

    w = w_func(samples, m=samples_mean, dtype=dtype)

    if bias:
        return w, samples_mean
    else:
        return w


def pattern_synapses(pattern, patch, dtype='float32', bias=None):
    w = np.kron(pattern, patch)
    if bias is None:
        return w
    else:
        return w, np.full(w.shape[1], fill_value=bias, dtype=dtype)


def roll_synapses(w, left=None, right=None, up=None, down=None):

    if left is not None:
        w = np.hstack([w[:, int(left):], w[:, :int(left)]])
    elif right is not None:
        w = np.hstack([w[:, -int(right):], w[:, :-int(right)]])

    if up is not None:
        w = np.vstack([w[int(left):, :], w[:int(left), :]])
    elif down is not None:
        w = np.vstack([w[-int(right):, :], w[:-int(right), :]])

    return w
