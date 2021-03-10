from ._utils import RNG

import numpy as np


def init_synapses(nb_in, nb_out, fill_value=0, dtype='float32', bias=None):
    w = np.full((nb_in, nb_out), fill_value=fill_value, dtype=dtype)
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)


def diagonal_synapses(nb_in, nb_out, fill_value=1, dtype='float32', bias=None):
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
