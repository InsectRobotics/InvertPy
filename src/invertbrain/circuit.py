from .plasticity import dopaminergic
from ._helpers import RNG

import numpy as np
import warnings


class Component(object):

    def __init__(self, nb_input: int, nb_output: int, repeats=4, repeat_learning_rate=None, learning_rule=dopaminergic,
                 eligibility_trace=0., noise=0., rng: np.random.RandomState = RNG, dtype: np.dtype = np.float32):
        self.dtype = dtype
        self.rng = rng

        self.params = []

        self._lambda = eligibility_trace
        self._learning_rule = learning_rule
        self.__update = False
        self._nb_input = nb_input
        self._nb_output = nb_output
        self._repeats = repeats
        self._noise = noise

        if repeat_learning_rate is None:
            # self.__eta = np.power(1. / float(repeats), 1. / float(repeats))
            self.__eta = 1. / float(repeats)
        else:
            self.__eta = repeat_learning_rate

    def reset(self):
        raise NotImplementedError()

    def _fprop(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        callback = kwargs.pop('callback', None)
        out = self._fprop(*args, **kwargs)

        if callback is not None:
            callback(self)

        return out

    def __repr__(self):
        return "Component(in=%d, out=%d, lr='%s')" % (self._nb_input, self._nb_output, self.learning_rule)

    def update_values(self, v, v_pre=None, eta=None):
        if eta is None:
            eta = self.__eta
        if v_pre is None:
            v_pre = 0.
            eta = 1.
        epsilon = self.rng.uniform(-self._noise, self._noise, v.shape)
        return eta * v + (1. - eta) * v_pre + epsilon

    def update_weights(self, w_pre, r_in, r_out, rein, w_rest=1., eta=None):
        if not callable(self._learning_rule):
            warnings.warn("Variable learning_rule is not callable! Update is skipped.", RuntimeWarning)
            return w_pre

        if eta is None:
            eta = self.__eta

        return self._learning_rule(w_pre, r_in, r_out, rein, learning_rate=eta, w_rest=w_rest)

    @property
    def learning_rule(self):
        return self._learning_rule.__name__

    @property
    def update(self):
        return self.__update

    @update.setter
    def update(self, value):
        self.__update = value

    @property
    def repeats(self):
        return self._repeats

