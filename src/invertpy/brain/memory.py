"""
Package that holds implementations of the Mushroom Body component of the insect brain.

References:
    .. [1] Wessnitzer, J., Young, J. M., Armstrong, J. D. & Webb, B. A model of non-elemental olfactory learning in
       Drosophila. J Comput Neurosci 32, 197–212 (2012).
    .. [2] Ardin, P., Peng, F., Mangan, M., Lagogiannis, K. & Webb, B. Using an Insect Mushroom Body Circuit to Encode
       Route Memory in Complex Natural Environments. Plos Comput Biol 12, e1004683 (2016).
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

import numbers

from .component import Component
from .mushroombody import VectorMemoryMB
from .synapses import uniform_synapses, sparse_synapses, random_synapses
from .activation import relu, sigmoid, winner_takes_all

from sklearn.metrics import mean_squared_error
from copy import copy
from abc import ABC

import numpy as np


class MemoryComponent(Component, ABC):
    def __init__(self, *args, nb_hidden=0, novelty_mode=0, **kwargs):
        """
        Abstract class of a memory component in the insect brain. Memory components are use to store information related to
        the visual navigation and other tasks. They have been used by numerous works usually as models of the mushroom
        bodies [1]_, [2]_. Here we keep them more abstracted allowing them to be used as mushroom bodies or not.

        Parameters
        ----------
        nb_hidden : int
            the number of the hidden units. Default is 0 (no hidden units)

        Notes
        -----
        .. [1] Wessnitzer, J., Young, J. M., Armstrong, J. D. & Webb, B. A model of non-elemental olfactory learning in
           Drosophila. J Comput Neurosci 32, 197–212 (2012).
        .. [2] Ardin, P., Peng, F., Mangan, M., Lagogiannis, K. & Webb, B. Using an Insect Mushroom Body Circuit to Encode
           Route Memory in Complex Natural Environments. Plos Comput Biol 12, e1004683 (2016).
        """
        super().__init__(*args, **kwargs)

        self._nb_hidden = nb_hidden

        self._inp = np.zeros((self.neuron_dims, self.nb_input), dtype=self.dtype)
        self._out = np.zeros((self.neuron_dims, self.nb_output), dtype=self.dtype)
        self._hid = np.zeros((self.neuron_dims, self.nb_hidden), dtype=self.dtype)

        self._novelty_name = ["", "pca", "zca", "zernike-zca", "zernike"]
        if isinstance(novelty_mode, str):
            novelty_mode = self._novelty_name.index(novelty_mode)
        self._novelty_mode = np.clip(novelty_mode, 0, len(self._novelty_name) - 1)

    def reset(self):
        """
        By default a memory component is open for updates.
        """
        self._inp *= 0.
        self._hid *= 0.
        self._out *= 0.

        self.update = True

    def __repr__(self):
        return "MemoryComponent(in=%d, out=%d, plasticity='%s', free-space=%.2f)" % (
            self.nb_input, self.nb_output, self.learning_rule, self.free_space
        )

    @property
    def r_inp(self):
        """
        The responses of the input layer.

        Returns
        -------
        np.ndarray[float]
        """
        return self._inp

    @property
    def r_out(self):
        """
        The responses of the output layer.

        Returns
        -------
        np.ndarray[float]
        """
        return self._out

    @property
    def r_hid(self):
        """
        The responses of the hidden layer.

        Returns
        -------
        np.ndarray[float]
        """
        return self._hid

    @property
    def nb_input(self):
        """
        The number of units in the input layer.

        Returns
        -------
        int
        """
        return self._nb_input

    @property
    def nb_output(self):
        """
        The number of units in the output layer.

        Returns
        -------
        int
        """
        return self._nb_output

    @property
    def nb_hidden(self):
        """
        The number of units in the hidden layer.

        Returns
        -------
        int
        """
        return self._nb_hidden

    @property
    def free_space(self):
        """
        Percentile of the  available space in the memory.

        Returns
        -------
        float
        """
        raise NotImplementedError()

    @property
    def novelty(self):
        """
        The novelty of the last presented input extracted from the memory.

        Returns
        -------
        np.ndarray[float]
        """
        return np.clip(1 - self._out, 0, 1)

    @property
    def familiarity(self):
        """
        The familiarity to the last presented input extracted from the memory.
        Typically: 1 - novelty

        Returns
        -------
        np.ndarray[float]
        """
        return 1 - self.novelty

    @property
    def novelty_mode(self):
        """
        The novelty mode the preprocessing layers that the memory will create the novelty of the input signal given the output of the memory.

        Returns
        -------
        str
        """
        return self._novelty_name[self._novelty_mode]

    @novelty_mode.setter
    def novelty_mode(self, v):
        if isinstance(v, str):
            self._novelty_mode = self._novelty_name.index(v)
        elif isinstance(v, int):
            self._novelty_mode = v


class IncentiveCircuitMemory(MemoryComponent):

    def __init__(self, nb_input, nb_output=1, nb_sparse=None, learning_rule='dopaminergic', eligibility_trace=.1,
                 sparseness=.03, *args, **kwargs):
        """
        The Whillshaw Network is a simplified Mushroom Body circuit that is used for associative memory tasks. It
        contains the input, sparse and output layers. In the sparse layer, we create a sparse representation of the
        input layer, and its synaptic weights are fixed. The sparse-to-output layer synapses are plastic.

        Examples
        --------
        >>> wn = IncentiveCircuitMemory(nb_input=360, nb_kc=1000)
        >>> wn.nb_input
        360
        >>> wn.nb_sparse
        1000

        Parameters
        ----------
        nb_input : int
            the number of input units
        nb_output : int, optional
            the number of output units. Default is 1
        nb_sparse : int, optional
            the number of sparse units. Default is 40 times the number of input units
        learning_rule : callable, str
            the name of a learning rule or a function representing it. The function could have as input:
                w - the synaptic weights to be updated,
                r_pre - the pre-synaptic responses,
                r_post - the post synaptic responses,
                rein - the reinforcement signal or the dopaminergic factor,
                learning_rate - the learning rate,
                w_rest - the resting values for the synaptic weights.
            Default is the 'anti_hebbian' learning rule.
        eligibility_trace : float, optional
            the lambda parameter for the eligibility traces. The higher the lambda, the more the new responses will rely
            on the previous ones.
        sparseness : float, optional
            the percentage of the number of KCs that needs to be active. Default is 3%.
        """
        if nb_sparse is not None:
            kwargs['nb_hidden'] = nb_sparse
        else:
            kwargs.setdefault('nb_hidden', nb_input * 40)

        super().__init__(nb_input=nb_input, nb_output=nb_output, learning_rule=learning_rule,
                         eligibility_trace=eligibility_trace, *args, **kwargs)

        self._ic = VectorMemoryMB(nb_cs=nb_input, nb_us=2, nb_kc=nb_sparse, learning_rule=learning_rule,
                                  eligibility_trace=eligibility_trace, sparseness=sparseness,
                                  cs_magnitude=1, us_magnitude=2, ltm_charging_speed=0.0005)

        self.params.extend(self._ic.params)

        # PN=.0: C=72.99 : 100% on (random), m_fam=<1%
        # PN=.1: C=73.37 : 90% on, m_fam=<1%
        # PN=.2: C=74.30 : 80% on, m_fam=<1%
        # PN=.3: C=75.49 : 70% on, m_fam=1%
        # PN=.4: C=77.31 : 60% on, m_fam=1.5%
        # PN=.5: C=79.94 : 50% on, m_fam=1.5%
        # PN=.6: C=82.50 : 40% on, m_fam=3%
        # PN=.7: C=85.78 : 30% on, m_fam=10%
        # PN=.8: C=90.23 : 20% on, m_fam=15%
        # PN=.9: C=95.55 : 10% on, m_fam=80%
        # PN=1.: C=72.97 : 0% on (random), m_fam=<1%
        self._ic.f_cs = lambda x: np.asarray(
            (x.T - x.min(axis=-1)) / (x.max(axis=-1) - x.min(axis=-1)), dtype=self.dtype).T
        self._ic.f_kc = lambda x: np.asarray(
            winner_takes_all(x, percentage=self.sparseness, noise=.01, normalise=False), dtype=self.dtype)
        f_mbon = copy(self._ic.f_mbon)
        self._ic.f_mbon = lambda x: np.asarray(f_mbon(x), dtype=self.dtype)

        self.__s = True
        self.__r = True
        self.__m = True

        self.__pos, self.__neg = [], []
        if self.__s:
            self.__pos.append([0])
            self.__neg.append([1])
        if self.__r:
            self.__pos.append([2])
            self.__neg.append([3])
        if self.__m:
            self.__pos.append([4])
            self.__neg.append([5])

    def reset(self):
        """
        Resets the synaptic weights and internal responses.
        """
        self._ic.reset()

        super().reset()

    def _fprop(self, inp=None, reinforcement=None):
        """
        Running the forward propagation.

        Parameters
        ----------
        inp: np.ndarray[float]
            The current input.
        reinforcement: np.ndarray[float]
            The current reinforcement.

        Returns
        -------
        np.ndarray[float]
            the novelty of the input element before the update
        """
        if inp is None:
            inp = np.zeros_like(self._inp, dtype=self.dtype)
        if reinforcement is None:
            reinforcement = [0, 0]
        elif isinstance(reinforcement, numbers.Number):
            reinforcement = [reinforcement, 0]
        elif len(reinforcement) < 2:
            reinforcement = [reinforcement[0], 0]

        inp = np.array(inp, dtype=self.dtype)
        reinforcement = np.array(reinforcement, dtype=self.dtype)
        if inp.ndim < 2:
            inp = inp[np.newaxis, ...]

        if self.update:
            nb_repeats = 1
        else:
            nb_repeats = 1

        for i in range(nb_repeats):
            self._ic(cs=inp, us=reinforcement)

        # print(self._ic.r_mbon[0], self._ic.r_dan[0])

        self._inp = self._ic.r_cs[:1]
        self._hid = self._ic.r_kc[:1]

        self._out = np.clip(np.mean(self._ic.r_mbon[:1, self.__pos] - self._ic.r_mbon[:1, self.__neg], axis=1), -1, 1)
        # self._out = np.absolute(self._out)

        return self._out

    def __repr__(self):
        return "IncentiveCircuitMemory(in=%d, sparse=%d, out=%d, eligibility_trace=%.2f, plasticity='%s')" % (
            self.nb_input, self.nb_sparse, self.nb_output, self._lambda, self.learning_rule
        )

    @property
    def sparseness(self):
        """
        The sparseness of the KCs: the percentage of the KCs that are active in every time-step.
        """
        return self._ic.sparseness

    @property
    def nb_sparse(self):
        """
        The number of units in the sparse layer.
        """
        return self._ic.nb_kc

    @property
    def r_spr(self):
        """
        The responses of the sparse layer.

        Returns
        -------
        np.ndarray[float]
        """
        return self.r_hid

    @property
    def w_i2s(self):
        """
        The input-to-sparse synaptc weights.
        """
        return self._ic.w_c2k

    @w_i2s.setter
    def w_i2s(self, v):
        self._ic.w_c2k = v

    @property
    def w_s2o(self):
        """
        The sparse-to-output synaptic weights.
        """
        return self._ic.w_k2m

    @w_s2o.setter
    def w_s2o(self, v):
        self._ic.w_k2m = v

    @property
    def w_rest(self):
        return self._ic.w_rest

    @property
    def free_space(self):
        """
        Percentile of the  available space in the memory.

        Returns
        -------
        float
        """

        ids = self.__pos + self.__neg
        return np.clip(1 - np.absolute(self.w_s2o[:, ids] - self.w_rest[:, ids]), 0, 1).mean()

    @property
    def novelty(self):
        return 1 - np.clip(self._out, 0, 1)

    @property
    def update(self):
        return self._ic.update

    @update.setter
    def update(self, v):
        self._ic.update = v


class WillshawNetwork(MemoryComponent):

    def __init__(self, nb_input, nb_output=1, nb_sparse=None, learning_rule='anti_hebbian', eligibility_trace=.1,
                 sparseness=.03, *args, **kwargs):
        """
        The Whillshaw Network is a simplified Mushroom Body circuit that is used for associative memory tasks. It
        contains the input, sparse and output layers. In the sparse layer, we create a sparse representation of the
        input layer, and its synaptic weights are fixed. The sparse-to-output layer synapses are plastic.
        This model is a modified version of the one presented in [1]_.

        Examples
        --------
        >>> wn = WillshawNetwork(nb_input=360, nb_kc=1000)
        >>> wn.nb_input
        360
        >>> wn.nb_sparse
        1000

        Parameters
        ----------
        nb_input : int
            the number of input units
        nb_output : int, optional
            the number of output units. Default is 1
        nb_sparse : int, optional
            the number of sparse units. Default is 40 times the number of input units
        learning_rule : callable, str
            the name of a learning rule or a function representing it. The function could have as input:
                w - the synaptic weights to be updated,
                r_pre - the pre-synaptic responses,
                r_post - the post synaptic responses,
                rein - the reinforcement signal or the dopaminergic factor,
                learning_rate - the learning rate,
                w_rest - the resting values for the synaptic weights.
            Default is the 'anti_hebbian' learning rule.
        eligibility_trace : float, optional
            the lambda parameter for the eligibility traces. The higher the lambda, the more the new responses will rely
            on the previous ones.
        sparseness : float, optional
            the percentage of the number of KCs that needs to be active. Default is 3%.

        Notes
        -----
        .. [1] Ardin, P., Peng, F., Mangan, M., Lagogiannis, K. & Webb, B. Using an Insect Mushroom Body Circuit to
           Encode Route Memory in Complex Natural Environments. Plos Comput Biol 12, e1004683 (2016).
        """
        if nb_sparse is not None:
            kwargs['nb_hidden'] = nb_sparse
        else:
            kwargs.setdefault('nb_hidden', nb_input * 40)

        super().__init__(nb_input=nb_input, nb_output=nb_output, learning_rule=learning_rule,
                         eligibility_trace=eligibility_trace, *args, **kwargs)

        # nb_in_max = nb_sparse // nb_input + 6
        max_samples = max(nb_sparse, 10000)
        # max_repeat = int(np.round(nb_sparse / nb_input / 20))
        self._w_i2s, self._b_s = sparse_synapses(nb_input, nb_sparse, max_samples=max_samples, dtype=self.dtype, bias=0.)
        self._w_s2o, self._b_o = uniform_synapses(nb_sparse, nb_output, dtype=self.dtype, bias=0.)
        self._w_rest = 1.

        self.params.extend([self._w_i2s, self._w_s2o, self._w_rest, self._b_s, self._b_o])
        if sparseness * self.nb_hidden < 1:
            sparseness = 1 / self.nb_hidden
        self._sparseness = sparseness

        # PN=.0: C=72.99 : 100% on (random), m_fam=<1%
        # PN=.1: C=73.37 : 90% on, m_fam=<1%
        # PN=.2: C=74.30 : 80% on, m_fam=<1%
        # PN=.3: C=75.49 : 70% on, m_fam=1%
        # PN=.4: C=77.31 : 60% on, m_fam=1.5%
        # PN=.5: C=79.94 : 50% on, m_fam=1.5%
        # PN=.6: C=82.50 : 40% on, m_fam=3%
        # PN=.7: C=85.78 : 30% on, m_fam=10%
        # PN=.8: C=90.23 : 20% on, m_fam=15%
        # PN=.9: C=95.55 : 10% on, m_fam=80%
        # PN=1.: C=72.97 : 0% on (random), m_fam=<1%
        self.f_input = lambda x: np.asarray(
            (x.T - x.min(axis=-1)) / (x.max(axis=-1) - x.min(axis=-1)), dtype=self.dtype).T
        # self.f_input = lambda x: np.asarray(np.greater(x, np.quantile(x, .7)), dtype=self.dtype)  # 30% is on
        self.f_sparse = lambda x: np.asarray(
            winner_takes_all(x, percentage=self.sparseness, noise=.01), dtype=self.dtype)
        self.f_output = lambda x: np.asarray(relu(x), dtype=self.dtype)

    def reset(self):
        """
        Resets the synaptic weights and internal responses.
        """
        self.w_s2o = uniform_synapses(self.nb_sparse, self.nb_output, fill_value=1, dtype=self.dtype)

        super().reset()

    def _fprop(self, inp=None, reinforcement=None):
        """
        Running the forward propagation.

        Parameters
        ----------
        inp: np.ndarray[float]
            The current input.
        reinforcement: np.ndarray[float]
            The current reinforcement.

        Returns
        -------
        np.ndarray[float]
            the novelty of the input element before the update
        """
        if inp is None:
            inp = np.zeros_like(self._inp)
        if reinforcement is None:
            reinforcement = 0.

        inp = np.array(inp, dtype=self.dtype)
        reinforcement = np.array(reinforcement, dtype=self.dtype)
        if inp.ndim < 2:
            inp = inp[np.newaxis, ...]

        a_inp = self.f_input(inp)

        spr = np.dot(a_inp, self.w_i2s) + self._b_s
        a_spr = self.f_sparse(self.update_values(spr, v_pre=self.r_spr, eta=1. - self._lambda))

        out = np.dot(a_spr, self.w_s2o) + self._b_o
        a_out = self.f_output(self.update_values(out, v_pre=self.r_out, eta=1. - self._lambda))

        if self.update:
            self.w_s2o = np.clip(
                self.update_weights(self._w_s2o, a_spr, a_out, reinforcement, w_rest=self._w_rest), 0, 1)

        self._inp = a_inp
        self._hid = a_spr
        self._out = a_out

        return self._out

    def __repr__(self):
        return "WillshawNetwork(in=%d, sparse=%d, out=%d, eligibility_trace=%.2f, plasticity='%s')" % (
            self.nb_input, self.nb_sparse, self.nb_output, self._lambda, self.learning_rule
        )

    @property
    def sparseness(self):
        """
        The sparseness of the KCs: the percentage of the KCs that are active in every time-step.
        """
        return self._sparseness

    @property
    def nb_sparse(self):
        """
        The number of units in the sparse layer.
        """
        return self._nb_hidden

    @property
    def r_spr(self):
        """
        The responses of the sparse layer.

        Returns
        -------
        np.ndarray[float]
        """
        return self.r_hid

    @property
    def w_i2s(self):
        """
        The input-to-sparse synaptc weights.
        """
        return self._w_i2s

    @w_i2s.setter
    def w_i2s(self, v):
        self._w_i2s[:] = v

    @property
    def w_s2o(self):
        """
        The sparse-to-output synaptic weights.
        """
        return self._w_s2o

    @w_s2o.setter
    def w_s2o(self, v):
        self._w_s2o[:] = v

    @property
    def free_space(self):
        """
        Percentile of the  available space in the memory.

        Returns
        -------
        float
        """
        return np.clip(1 - np.absolute(self.w_s2o - self._w_rest), 0, 1).mean()

    @property
    def novelty(self):
        z = np.maximum(np.sum(self._hid > 0, axis=1), 1)
        r_out = (self._out.T / z).T
        if self.novelty_mode == "zernike-zca" and False:
            return sigmoid(30 * (r_out - .06))
        elif self.novelty_mode == "zernike" and False:
            return sigmoid(25 * (r_out - .085))
        elif self.novelty_mode == "pca" and False:
            return sigmoid(5 * (np.power(r_out, .2) - .05))
        else:
            return r_out


class Infomax(MemoryComponent):

    def __init__(self, nb_input, learning_rule="infomax", learning_rate=1.1, *args, **kwargs):
        kwargs.setdefault("nb_hidden", nb_input)
        # learning_rate *= nb_input
        super().__init__(nb_input=nb_input, nb_output=1, learning_rule=learning_rule,
                         repeat_rate=learning_rate, *args, **kwargs)

        self._w_i2h = random_synapses(nb_input, self.nb_hidden, w_min=-.5, w_max=.5, dtype=self.dtype, rng=self.rng)
        self._w_h2o = uniform_synapses(self.nb_hidden, self.nb_output, fill_value=1. / self.nb_hidden, dtype=self.dtype)

        self.params.extend([self._w_i2h, self._w_h2o])

        self.f_inp = lambda x: np.asarray(x, dtype=self.dtype)
        self.f_hid = lambda x: np.asarray(np.tanh(x), dtype=self.dtype)
        self.f_out = lambda x: np.asarray(x / 10, dtype=self.dtype)

    def _fprop(self, inp=None, reinforcement=None):
        a_inp = self.f_inp(inp)
        hid = np.dot(a_inp, self._w_i2h)
        a_hid = self.f_hid(hid)
        a_out = self.f_out(np.dot(np.absolute(hid), self._w_h2o))

        if self.update:
            self.w_i2h = self.update_weights(self._w_i2h, a_inp, hid, reinforcement, w_rest=0)

        self._inp = a_inp
        self._hid = a_hid
        self._out = a_out

        return self._out

    def __repr__(self):
        return f"Infomax(in={self.nb_input}, out={self.nb_output}, free_space={self.free_space * 100:.2f}%)"

    @property
    def free_space(self):
        return np.clip(1 - np.absolute(self.w_i2h), 0, 1).mean()

    @property
    def novelty(self):
        if self.novelty_mode == "zernike-zca":
            nov = np.power(1 - self.r_out, 32)
            return sigmoid(50 * (nov - .34))
        elif self.novelty_mode == "zernike":
            # raise NotImplementedError()
            return self.r_out
        elif self.novelty_mode == "pca":
            nov = np.power(self.r_out, 8)
            return sigmoid(30 * (nov - .35))
        else:
            return self.r_out


class PerfectMemory(MemoryComponent):

    def __init__(self, nb_input, nb_output=1, maximum_capacity=1000, error_metric=mean_squared_error, *args, **kwargs):
        """
        The Perfect Memory is a simplified memory component and it does not contain any neural connections.
        This model stores all the input received in a database and searches for the best match every time that receives
        a new input and reports the minimum difference. This was used for comparison by many papers including [1]_.

        Parameters
        ----------
        nb_input : int
            the number of input units
        nb_output : int, optional
            the number of output units. Default is 1
        maximum_capacity : int
            the maximum number of elements that can be stored. Default is 1000
        error_metric: callable
            the metric that measures the error between the observation and the database. Default is mean square error
            (MSE)

        Notes
        -----
        .. [1] Ardin, P., Peng, F., Mangan, M., Lagogiannis, K. & Webb, B. Using an Insect Mushroom Body Circuit to
           Encode Route Memory in Complex Natural Environments. Plos Comput Biol 12, e1004683 (2016).
        """

        kwargs.setdefault('nb_repeats', 1)
        super().__init__(nb_input=nb_input, nb_output=nb_output, eligibility_trace=0., *args, **kwargs)

        self.f_inp = lambda x: x

        self._error_metric = error_metric
        self._database = None
        self._max_capacity = maximum_capacity
        self._write = 0

        self.reset()

    def reset(self):
        """
        Resets the database.
        """

        # erase the database
        self._database = np.zeros((self._max_capacity, self.nb_input), dtype=self.dtype)
        self._write = 0

        super().reset()

    def _fprop(self, inp=None, reinforcement=None):
        """
        Calculates the novelty of the input with respect to the stored elements and updates the memory.

        Parameters
        ----------
        inp : np.ndarray[float]
            the input element
        reinforcement : np.ndarrya[float]
            the reinforcement

        Returns
        -------
            the novelty of the input element before the update
        """
        if inp is None:
            inp = np.zeros_like(self._database[0])

        a_inp = self.f_inp(inp)

        if self._write > 0:
            y_true = self.database[:self._write].T
            y_pred = np.vstack([a_inp] * self._write).T
            a_out = self._error_metric(y_true, y_pred, multioutput='raw_values', squared=False).min()
        else:
            a_out = np.zeros(self.nb_output, dtype=self.dtype)

        if np.ndim(a_out) < 1:
            a_out = np.array([a_out])
        if np.ndim(a_out) < 2:
            a_out = a_out[np.newaxis, ...]

        self._inp = a_inp
        self._out = a_out

        if self.update:
            self._database[self._write % self._max_capacity] = a_inp
            self._write += 1

        return self._out

    def __repr__(self):
        return "PerfectMemory(in=%d, out=%d, error=%s)" % (self.nb_input, self.nb_output, self.error_metric)

    @property
    def database(self):
        """
        The database of elements.

        Returns
        -------
        np.ndarray[float]
        """
        return np.array(self._database)

    @property
    def error_metric(self):
        """
        The name function that calculates the error.

        Returns
        -------
        str
        """
        return self._error_metric.__name__

    @property
    def free_space(self):
        """
        Percentile of the  available space in the memory.

        Returns
        -------
        float
        """
        return 1. - self._write / self._max_capacity

    @property
    def novelty(self):
        nov = 1 - np.power(1 - self.r_out, 4096)
        if self.novelty_mode == "zernike-zca":
            return sigmoid(15 * (self.r_out - 0.87))
        elif self.novelty_mode == "zernike":
            return sigmoid(300 * (self.r_out - 0.01))
        elif self.novelty_mode == "pca":
            return sigmoid(15 * (nov - .2))
        else:
            return self.r_out
