import numpy as np


def dopaminergic(w, r_in, r_out, dop_fact, learning_rate=1., w_rest=1.):
    # When DAN > 0 and KC > W - W_rest increase the weight (if DAN < 0 it is reversed)
    # When DAN > 0 and KC < W - W_rest decrease the weight (if DAN < 0 it is reversed)
    # When DAN = 0 no learning happens
    dop_fact = dop_fact[np.newaxis, ...]
    r_in = r_in[..., np.newaxis]
    return w + learning_rate * dop_fact * (r_in + w - w_rest)


def prediction_error(w, r_in, r_out, rein, learning_rate=1., w_rest=1.):
    # When KC > 0 and DAN > W - W_rest increase the weight (if KC < 0 it is reversed)
    # When KC > 0 and DAN < W - W_rest decrease the weight (if KC < 0 it is reversed)
    # When KC = 0 no learning happens
    rein = rein[np.newaxis, ...]
    r_in = r_in[..., np.newaxis]
    r_out = r_out[np.newaxis, ...]
    return w + learning_rate * r_in * (rein - r_out + w_rest)


def hebbian(w, r_in, r_out, rein, learning_rate=1., w_rest=1.):
    # When DAN > 0 and MBON > 0 increase the weight
    # When DAN <= 0 no learning happens
    rein = rein[np.newaxis, ...]
    r_in = r_in[..., np.newaxis]
    r_out = r_out[np.newaxis, ...]
    return w + learning_rate * (rein * np.outer(r_in, r_out) + w_rest)


def anti_hebbian(w, r_in, r_out, rein, learning_rate=1., w_rest=1.):
    # When DAN > 0 and KC > 0 decrease the weight
    # When DAN <= 0 no learning happens
    rein = np.maximum(rein[np.newaxis, ...], 0)
    r_in = r_in[..., np.newaxis]
    return w + learning_rate * (-rein * (r_in * w) + w_rest)
