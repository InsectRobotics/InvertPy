"""
Abstract class for brain components. Implements functions like the value and weights update based on the given learning
rule and contains basic attributes of the brain component, such as the type of the values, the random generator, the
eligibility traces parameter, the learning rule and noise.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from .plasticity import dopaminergic, get_learning_rule
from ._helpers import RNG

from copy import copy

import numpy as np
import warnings


class Component(object):

    def __init__(self, nb_input, nb_output, nb_repeats=1, repeat_rate=None, learning_rule=dopaminergic, ndim=1,
                 eligibility_trace=0., noise=0., rng=RNG, dtype=np.float32):
        """
        Abstract class of a brain component that implements its basic functions and sets the abstract methods that need
        to be implemented by every component (i.e. the 'reset' and 'forward propagation').

        Parameters
        ----------
        nb_input: int
            The number of input units of the component.
        nb_output: int
            The number of output units of the component.
        nb_repeats: int, optional
            The number of times we want to repeat the forward propagation. This is particularly helpful when there are
            feedback connections in the component, as it allows smooth update of all of the layers. The more repeats,
            the smoother the transition, but also more computationally expensive.
        repeat_rate: float, optional
            The rate of update in each repeat. By default this is set to: 1 / nb_repeats.
        learning_rule: callable, str
            The name of a learning rule or a function representing it. The function could have as input:
                w - the synaptic weights to be updated,
                r_pre - the pre-synaptic responses,
                r_post - the post synaptic responses,
                rein - the reinforcement signal or the dopaminergic factor,
                learning_rate - the learning rate,
                w_rest - the resting values for the synaptic weights.
        ndim: int, optional
            The number of dimensions per neuron. Default is 1.
        eligibility_trace: float, optional
            The lambda parameter for the eligibility traces. The higher the lambda, the more the new responses will rely
            on the previous ones.
        noise: float, optional
            The noise introduced in the responses of the component.
        rng: np.random.RandomState, int
            An instance of the numpy.random.RandomState that will be used in order to generate random patterns.
        dtype: np.dtype, optional
            The type of the values used in this component.
        """
        self.dtype = dtype
        if isinstance(rng, int):
            rng = np.random.RandomState(rng)
        self.rng = rng

        self.params = []

        self._lambda = eligibility_trace
        self._learning_rule = learning_rule if callable(learning_rule) else get_learning_rule(learning_rule)
        self.__update = False
        self._nb_input = nb_input
        self._ndim = ndim
        self._nb_output = nb_output
        self._repeats = nb_repeats
        self._noise = noise

        if repeat_rate is None:
            # self.__eta = np.power(1. / float(repeats), 1. / float(repeats))
            self.__eta = 1. / float(nb_repeats)
        else:
            self.__eta = repeat_rate

    def reset(self):
        """
        This method is called whenever we want to re-initialise the component. It should implement the synaptic weights
        and internal values initialisation.
        """
        raise NotImplementedError()

    def _fprop(self, *args, **kwargs):
        """
        The forward propagation function should process the input that the component receives and calculate its output.
        It automatically runs when the component is called.

        Returns
        -------
        r_out: np.ndarray
            The output of the component given the input.
        """
        raise NotImplementedError()

    def __call__(self, *args, callback=None, **kwargs):
        """
        When the component is called, the forward propagation is executed and the output is calculated. Then the
        callback function is called (if provided), which gets as input the instance of the component itself. Finally,
        the output is returned.

        Parameters
        ----------
        callback: callable, optional
            Customised processing of the component every time that the component is called. It gets as input the
            component itself.

        Returns
        -------
        r_out: np.ndarray
            The output of the component given the input.
        """
        out = self._fprop(*args, **kwargs)

        if callback is not None:
            callback(self)

        return out

    def copy(self):
        """
        Creates a clone of the instance.

        Returns
        -------
        copy: Component
            another instance of exactly the same class and parameters.
        """
        return copy(self)

    def __copy__(self):
        component = self.__class__(nb_input=self._nb_input, nb_output=self._nb_output)
        for att in self.__dict__:
            component.__dict__[att] = copy(self.__dict__[att])

        return component

    def __repr__(self):
        return f"Component(in={self._nb_input}, out={self._nb_output}, lr='{self.learning_rule}')"

    def update_values(self, v, v_pre=None, eta=None):
        """
        Updates the new value (v) by adding the contribution from the previous value (v_pre) based on the blending
        parameter (eta) and noise. Formally the output is:
                                v_post = eta * v + (1 - eta) * v_pre + noise

        Parameters
        ----------
        v: np.ndarray, float
            The value to be used as the new value.
        v_pre: np.ndarray, float
            The value to be used as the previous value. Default is 0.
        eta: float, optional
            The blending parameter defines how much to rely on the previous value. For eta=0, it will rely only on the
            old value, while for eta=1, it will rely only on the new value. Default is eta=repeat_rate.

        Returns
        -------
        v_post: np.ndarray
            The updated value using the input parameters.
        """
        if eta is None:
            eta = self.__eta
        if v_pre is None:
            v_pre = 0.
            eta = 1.
        epsilon = self.rng.uniform(-self._noise, self._noise, v.shape)
        return v_pre + eta * (v - v_pre) + epsilon

    def update_weights(self, w_pre, r_pre, r_post, rein, w_rest=1., eta=None):
        """
        Updates the given synaptic weights (w_pre) by using the pre-synaptic (r_pre) and post-synaptic (r_post)
        responses, the reinforcement (rein), the resting weights (w_rest) and the learning rate (eta). These are passed
        as parameters to the learning rule of the component as:
            w_post = learning_rule(w_pre, r_pre, r_post, rein, eta, w_rest)
        
        Parameters
        ----------
        w_pre: np.ndarray, float
            The current synaptic weights.
        r_pre: np.ndarray, float
            The pre-synaptic neural responses.
        r_post: np.ndarray, float
            The post-synaptic neural responses.
        rein: np.ndarray, float
            The responses of the reinforcement neurons or the respective dopaminergic factors.
        w_rest: np.ndarray, float
            The resting synaptic weights.
        eta: float, optional
            The learning rate.

        Returns
        -------
        w_post: np.ndarray
            The updated weights.
        """
        if not callable(self._learning_rule):
            warnings.warn("Variable learning_rule is not callable! Update is skipped.", RuntimeWarning)
            return w_pre

        if eta is None:
            eta = self.__eta

        return self._learning_rule(w_pre, r_pre, r_post, rein, learning_rate=eta, w_rest=w_rest)

    @property
    def learning_rule(self):
        """
        The learning rule as a string.

        Returns
        -------
        str
        """
        if self._learning_rule is None:
            return 'None'
        else:
            return self._learning_rule.__name__

    @property
    def update(self):
        """
        The update status. If True, then the component updates its synaptic weights during the forward propagation. If
        False, then the update is skipped.

        Returns
        -------
        bool
        """
        return self.__update

    @update.setter
    def update(self, to):
        """
        The update status. If True, then the component updates its synaptic weights during the forward propagation. If
        False, then the update is skipped.

        Parameters
        ----------
        to: bool
            The new value for the 'update' attribute.
        """
        self.__update = to

    @property
    def repeats(self):
        """
        The number of repeats for every update step.

        Returns
        -------
        int
        """
        return self._repeats

    @property
    def ndim(self):
        """
        The number of dimension of the neural values.

        Returns
        -------
        int
        """
        return self._ndim
