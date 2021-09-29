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

from .component import Component
from .synapses import uniform_synapses, sparse_synapses
from .activation import relu, winner_takes_all

from sklearn.metrics import mean_squared_error
from abc import ABC

import numpy as np


class MemoryComponent(Component, ABC):
    """
    Abstract class of a memory component in the insect brain. Memory components are use to store information related to
    the visual navigation and other tasks. They have been used by numerous works usually as models of the mushroom
    bodies [1]_, [2]_. Here we keep them more abstracted allowing them to be used as mushroom bodies or not.

    Notes
    -----
    .. [1] Wessnitzer, J., Young, J. M., Armstrong, J. D. & Webb, B. A model of non-elemental olfactory learning in
       Drosophila. J Comput Neurosci 32, 197–212 (2012).
    .. [2] Ardin, P., Peng, F., Mangan, M., Lagogiannis, K. & Webb, B. Using an Insect Mushroom Body Circuit to Encode
       Route Memory in Complex Natural Environments. Plos Comput Biol 12, e1004683 (2016).
    """

    def reset(self):
        """
        By default a memory component is open for updates.
        """
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
        raise NotImplementedError()

    @property
    def r_out(self):
        """
        The responses of the output layer.

        Returns
        -------
        np.ndarray[float]
        """
        raise NotImplementedError()

    @property
    def r_hid(self):
        """
        The responses of the hidden layer.

        Returns
        -------
        np.ndarray[float]
        """
        raise NotImplementedError()

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
    def free_space(self):
        """
        Percentile of the  available space in the memory.

        Returns
        -------
        float
        """
        raise NotImplementedError()


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

        super().__init__(nb_input=nb_input, nb_output=nb_output, learning_rule=learning_rule,
                         eligibility_trace=eligibility_trace, *args, **kwargs)

        self._w_i2s, self._b_s = uniform_synapses(nb_input, nb_sparse, dtype=self.dtype, bias=0.)
        self._w_s2o, self._b_o = uniform_synapses(nb_sparse, nb_output, dtype=self.dtype, bias=0.)
        self._w_rest = 1.

        self.params.extend([self._w_i2s, self._w_s2o, self._w_rest, self._b_s, self._b_o])

        # reserve space for the responses
        self._inp = np.zeros(nb_input, dtype=self.dtype)
        self._spr = np.zeros(nb_sparse, dtype=self.dtype)
        self._out = np.zeros(nb_output, dtype=self.dtype)

        self._nb_sparse = nb_input * 40 if nb_sparse is None else nb_sparse

        self.f_input = lambda x: np.asarray(x > np.sort(x)[int(self.nb_input * .7)], dtype=self.dtype)
        self.f_sparse = lambda x: np.asarray(winner_takes_all(x, percentage=self.sparseness), dtype=self.dtype)
        self.f_output = lambda x: relu(x)

        self._sparseness = sparseness

    def reset(self):
        """
        Resets the synaptic weights and internal responses.
        """
        self.w_i2s = sparse_synapses(self.nb_input, self.nb_sparse, dtype=self.dtype)
        self.w_i2s *= self.nb_input / self._w_i2s.sum(axis=1)[:, np.newaxis]
        self.w_s2o = uniform_synapses(self.nb_sparse, self.nb_output, fill_value=1, dtype=self.dtype)

        self._inp *= 0.
        self._spr *= 0.
        self._out *= 0.

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

        a_inp = self.f_input(inp)

        spr = a_inp @ self.w_i2s + self._b_s
        a_spr = self.f_sparse(self.update_values(spr, v_pre=self.r_spr, eta=1. - self._lambda))

        out = a_spr @ self.w_s2o + self._b_o
        a_out = self.f_output(self.update_values(out, v_pre=self.r_out, eta=1. - self._lambda))

        if self.update:
            self.w_s2o = np.clip(
                self.update_weights(self._w_s2o, a_spr, a_out, reinforcement, w_rest=self._w_rest), 0, 1)

        self._inp = a_inp
        self._spr = a_spr
        self._out = a_out

        return a_out

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
        return self._nb_sparse

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
    def r_spr(self):
        """
        The responses of the sparse layer.

        Returns
        -------
        np.ndarray[float]
        """
        return self._spr

    @property
    def r_hid(self):
        """
        The responses of the hidden layer is the same as the ones from the sparse layer.

        Returns
        -------
        np.ndarray[float]
        """
        return self._spr

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

        self._inp = None
        self._hid = None
        self._out = None

        self.reset()

    def reset(self):
        """
        Resets the database.
        """

        # erase the database
        self._database = np.zeros((self._max_capacity, self.nb_input), dtype=self.dtype)
        self._write = 0
        self._inp = np.zeros(self.nb_input, dtype=self.dtype)
        self._hid = np.zeros(0, dtype=self.dtype)
        self._out = np.zeros(self.nb_output, dtype=self.dtype)

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
            y_pred = np.array([a_inp] * self._write).T
            a_out = self._error_metric(y_true, y_pred, multioutput='raw_values', squared=False).min()
        else:
            a_out = np.zeros(self.nb_output, dtype=self.dtype)

        self._inp = a_inp
        self._out = a_out

        if self.update:
            self._database[self._write % self._max_capacity] = a_inp
            self._write += 1

        return a_out

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
