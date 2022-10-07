"""
Package that holds implementations of the Mushroom Body component of the insect brain.

References:
    .. [1] Wessnitzer, J., Young, J. M., Armstrong, J. D. & Webb, B. A model of non-elemental olfactory learning in
       Drosophila. J Comput Neurosci 32, 197–212 (2012).
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from ._helpers import eps
from .memory import MemoryComponent
from .synapses import uniform_synapses, diagonal_synapses, sparse_synapses, opposing_synapses, roll_synapses,\
    pattern_synapses
from .activation import linear, relu, winner_takes_all

from copy import copy

import numpy as np


class MushroomBody(MemoryComponent):
    def __init__(self, nb_cs, nb_us, nb_kc, nb_dan, nb_mbon, nb_apl=1, learning_rule='dopaminergic', sparseness=0.03,
                 *args, **kwargs):
        """
        The Mushroom Body component of the insect brain is responsible for creating associations between the input (CS)
        and output (MBON) based on the reinforcement (US) and by modulating the KC-MBON connections. The KCs receive CS
        input and their output activates the MBONs. The APL works as a dynamic threshold for the KCs receiving input
        from them and inhibiting them via global and local inhibition. DANs get US input and their output modulate the
        KC-MBON connections. There are also KC-KC and MBON-DAN connections supported.

        Examples
        --------
        >>> mb = MushroomBody(nb_cs=2, nb_us=2, nb_kc=10, nb_dan=3, nb_mbon=3, nb_apl=1, learning_rule='dopaminergic', sparseness=0.03)
        >>> print(mb.nb_cs, mb.nb_us, mb.nb_kc, mb.nb_dan, mb.nb_mbon, mb.nb_apl)
        2 2 10 3 3 1
        >>> print(mb.learning_rule)
        dopaminergic
        >>> print(mb.sparseness)
        0.03

        Parameters
        ----------
        nb_cs: int
            The number of dimensions for the Conditional Stimulus (CS), equivalent to the number of Projection Neurons
            (PNs).
        nb_us: int
            The number of dimensions for the Unconditional Stimulus (US), equivalent to the reinforcement signal.
        nb_kc: int
            The number of Kenyon cells (KCs); intrinsic neurons of the mushroom body.
        nb_dan: int
            The number of Dopaminergic Neurons (DANs); reinforcement encoding extrinsic neurons of the mushroom body.
        nb_mbon: int
            The number of Mushroom Body Output Neurons (MBONs). This is equivalent to the nb_output.
        nb_apl: int, optional
            The number of Anterior Pair Lateral (APL) neurons. Default is 1.
        learning_rule: callable, str
            The name of a learning rule or a function representing it. The function could have as input:
                w - the synaptic weights to be updated,
                r_pre - the pre-synaptic responses,
                r_post - the post synaptic responses,
                rein - the reinforcement signal or the dopaminergic factor,
                learning_rate - the learning rate,
                w_rest - the resting values for the synaptic weights.
            Default is the 'dopaminergic' learning rule.
        sparseness: float, optional
            The percentage of the number of KCs that needs to be active. Default is 3%.
        """
        if nb_mbon is None and 'nb_output' in kwargs.keys():
            nb_mbon = kwargs.pop('nb_output')

        kwargs.setdefault("nb_input", nb_cs + nb_us)
        kwargs.setdefault("nb_output", nb_mbon)
        kwargs.setdefault("nb_hidden", nb_kc)
        super().__init__(learning_rule=learning_rule, *args, **kwargs)

        self._nb_cs = nb_cs
        self._nb_us = nb_us
        self._nb_dan = nb_dan
        self._nb_apl = nb_apl

        # set the parameters (synapses)
        max_samples = np.maximum(1000, int(1.5 * nb_kc))
        self._w_c2k = sparse_synapses(self.nb_cs, self.nb_kc, dtype=self.dtype,
                                      nb_in_min=1, nb_in_max=4, max_samples=max_samples)

        self._w_k2k = None
        self._w_a2k, self._b_k = uniform_synapses(nb_apl, nb_kc, dtype=self.dtype, bias=0)
        self._w_k2m = uniform_synapses(nb_kc, nb_mbon, dtype=self.dtype)
        self._w_m2m, self._b_m = uniform_synapses(nb_mbon, nb_mbon, dtype=self.dtype, bias=0)
        self._w_u2d = uniform_synapses(nb_us, nb_dan, dtype=self.dtype)
        self._w_m2d, self._b_d = uniform_synapses(nb_mbon, nb_dan, dtype=self.dtype, bias=0)
        self._w_d2d = uniform_synapses(nb_dan, nb_dan, dtype=self.dtype)
        self._w_k2a, self._b_a = uniform_synapses(nb_kc, nb_apl, dtype=self.dtype, bias=0)

        self._w_d2m = uniform_synapses(nb_dan, nb_mbon, dtype=self.dtype)
        self._w_rest = uniform_synapses(nb_kc, nb_mbon, dtype=self.dtype)

        self.params.extend([self.w_c2k, self.w_a2k, self.w_k2m, self.w_m2m, self.w_u2d, self.w_m2d, self._w_d2d,
                            self.w_k2a, self.w_d2m, self.w_rest, self.b_k, self.b_m, self.b_d, self.b_a])

        # reserve space for the responses
        self._inp = np.zeros((self._repeats, self.ndim, nb_cs + nb_us), dtype=self.dtype)
        self._dan = np.zeros((self._repeats, self.ndim, nb_dan), dtype=self.dtype)
        self._hid = np.zeros((self._repeats, self.ndim, nb_kc), dtype=self.dtype)
        self._apl = np.zeros((self._repeats, self.ndim, nb_apl), dtype=self.dtype)
        self._out = np.zeros((self._repeats, self.ndim, nb_mbon), dtype=self.dtype)

        self.f_cs = lambda x: np.asarray(linear(x, noise=self._noise, rng=self.rng))
        self.f_us = lambda x: np.asarray(linear(x, noise=self._noise, rng=self.rng))
        self.f_dan = lambda x: np.asarray(relu(x, cmax=2, noise=self._noise, rng=self.rng))
        self.f_kc = lambda x: np.asarray(relu(x, cmax=2, noise=self._noise, rng=self.rng))
        self.f_apl = lambda x: np.asarray(relu(x, cmax=2, noise=self._noise, rng=self.rng))
        self.f_mbon = lambda x: np.asarray(relu(x, cmax=2, noise=self._noise, rng=self.rng))

        if sparseness * nb_kc < 1:
            sparseness = 1 / nb_kc
        self._sparseness = sparseness
        self._maximum_weight = 50

        self.cs_names = ["c_{%d}" % i for i in range(nb_cs)]
        self.us_names = ["u_{%d}" % i for i in range(nb_us)]
        self.dan_names = ["d_{%d}" % i for i in range(nb_dan)]
        self.kc_names = ["k_{%d}" % i for i in range(nb_kc)]
        self.apl_names = ["a_{%d}" % i for i in range(nb_apl)]
        self.mbon_names = ["m_{%d}" % i for i in range(nb_mbon)]

        self.reset()

    def reset(self):
        """
        Examples
        --------
        >>> mb = MushroomBody(2, 2, 10, 3, 3, 1)
        >>> mb.update = False
        >>> print(mb.update)
        False
        >>> mb.reset()
        >>> print(mb.update)
        True
        """
        # reset synapses
        # by default KC2KC connections are not supported so we save space by not allocating the memory
        self._w_k2k = None
        self.w_a2k, self.b_k = uniform_synapses(self.nb_apl, self.nb_kc, fill_value=-1, dtype=self.dtype, bias=0)
        self.w_k2m = uniform_synapses(self.nb_kc, self.nb_mbon, fill_value=1, dtype=self.dtype)
        self.w_m2m, self.b_m = uniform_synapses(self.nb_mbon, self.nb_mbon, fill_value=0, dtype=self.dtype, bias=0)
        self.w_u2d = diagonal_synapses(self.nb_us, self.nb_dan, fill_value=2, dtype=self.dtype)
        self.w_m2d, self.b_d = uniform_synapses(self.nb_mbon, self.nb_dan, fill_value=0, dtype=self.dtype, bias=0)
        self.w_k2a, self.b_a = uniform_synapses(self.nb_kc, self.nb_apl, dtype=self.dtype, bias=0,
                                                fill_value=2. * (1. - self._sparseness) / float(self.nb_kc + eps))

        self.w_d2m = diagonal_synapses(self.nb_dan, self.nb_mbon, fill_value=-1, dtype=self.dtype)
        self.w_rest = uniform_synapses(self.nb_kc, self.nb_mbon, fill_value=1, dtype=self.dtype)

        self.reset_responses()

        self.update = True

    def reset_responses(self):

        # reset responses
        self._inp = np.zeros((self._repeats, self.ndim, self.nb_input), dtype=self.dtype)
        self._dan = np.zeros((self._repeats, self.ndim, self.nb_dan), dtype=self.dtype)
        self._hid = np.zeros((self._repeats, self.ndim, self.nb_hidden), dtype=self.dtype)
        self._apl = np.zeros((self._repeats, self.ndim, self.nb_apl), dtype=self.dtype)
        self._out = np.zeros((self._repeats, self.ndim, self.nb_output), dtype=self.dtype)

    def get_response(self, neuron_name, all_repeats=False):
        """
        Identifies a neuron by its name and returns its response or all the updates of its response during the repeats.

        Parameters
        ----------
        neuron_name: str
            The name of the neuron.
        all_repeats: bool, optional
            Where or not to return the responses during all the repeats. By default it returns only the final response.

        Returns
        -------
        r: np.ndarray
            the response(s) of the specified neuron. If the name of the neuron is not found, it returns None.

        """
        if neuron_name in self.cs_names:
            if all_repeats:
                return self.r_cs[..., self.cs_names.index(neuron_name)]
            else:
                return self.r_cs[0, ..., self.cs_names.index(neuron_name)]

        elif neuron_name in self.us_names:
            if all_repeats:
                return self.r_us[..., self.us_names.index(neuron_name)]
            else:
                return self.r_us[0, ..., self.us_names.index(neuron_name)]
        elif neuron_name in self.dan_names:
            if all_repeats:
                return self.r_dan[..., self.dan_names.index(neuron_name)]
            else:
                return self.r_dan[0, ..., self.dan_names.index(neuron_name)]
        elif neuron_name in self.kc_names:
            if all_repeats:
                return self.r_kc[..., self.kc_names.index(neuron_name)]
            else:
                return self.r_kc[0, ..., self.kc_names.index(neuron_name)]
        elif neuron_name in self.apl_names:
            if all_repeats:
                return self.r_apl[..., self.apl_names.index(neuron_name)]
            else:
                return self.r_apl[0, ..., self.apl_names.index(neuron_name)]
        elif neuron_name in self.mbon_names:
            if all_repeats:
                return self.r_mbon[..., self.mbon_names.index(neuron_name)]
            else:
                return self.r_mbon[0, ..., self.mbon_names.index(neuron_name)]
        else:
            return None

    def set_maximum_weight(self, new_max):
        """
        Sets the maximum value for the KC-MBON synaptic weights.

        Parameters
        ----------
        new_max: float
            the new maximum weight
        """
        self._maximum_weight = new_max

    def _fprop(self, cs=None, us=None):
        """
        It propagates the CS and US signal forwards through the connections of the model for nb_repeats times. It
        updates the internal responses of the neurons and returns the final output of the MBONs.

        Parameters
        ----------
        cs: np.ndarray[float], optional
            the CS input. Default is 0.
        us: np.ndarray[float], optional
            the US reinforcement. Default is 0.

        Returns
        -------
        r_mbon: np.ndarray
            the MBONs' responses

        """
        if cs is None:
            cs = np.zeros_like(self._cs[0])
        if us is None:
            us = np.zeros_like(self._us[0])
        elif isinstance(us, float):
            _us = us
            us = np.zeros_like(self._us[0])
            us[..., 0] = np.maximum(_us, 0)
            us[..., 1:] = np.maximum(-_us, 0)
        elif len(us) < 2:
            _us = us
            us = np.zeros_like(self._us[0])
            us[..., 0] = np.maximum(_us[0], 0)
            us[..., 1:] = np.maximum(-_us[0], 0)
        cs = np.array(cs, dtype=self.dtype)
        us = np.array(us, dtype=self.dtype)
        if cs.ndim < 2:
            cs = cs[np.newaxis, ...]

        if us.ndim == 1:
            if us.shape[0] == self.nb_us:
                us = np.vstack([us] * self.ndim)
            elif us.shape[0] == self.ndim:
                us = np.vstack([us] + [np.zeros_like(us)] * (self.nb_us - 1)).T
        elif us.shape[1] < self.nb_us:
            _us = us
            us = np.zeros(self.ndim, self.nb_us)
            us[:, :len(_us)] = _us
        elif us.shape[1] > self.nb_us:
            us = us[:, :self.nb_us]
        us = np.asarray(us, dtype=self.dtype)

        for r in range(self.repeats):
            self._kc[-r-1], self._apl[-r-1], self._dan[-r-1], self._mbon[-r-1] = self._rprop(
                cs, us, self.r_kc[-r], self.r_apl[-r], self.r_dan[-r], self.r_mbon[-r], v_update=r > 0)
            self._cs[-r-1] = self.f_cs(cs)
            self._us[-r-1] = self.f_us(us)

        return self._mbon[0]

    def _rprop(self, cs, us, kc_pre, apl_pre, dan_pre, mbon_pre, v_update=True):
        """
        Running the forward propagation for a single repeat.

        Parameters
        ----------
        cs: np.ndarray[float]
            The current CS input.
        us: np.ndarray[float]
            The current US reinforcement.
        kc_pre: np.ndarray[float]
            The old KC responses.
        apl_pre: np.ndarray[float]
            The old APL responses.
        dan_pre: np.ndarray[float]
            The old DAN responses.
        mbon_pre: np.ndarray[float]
            The old MBON responses.
        v_update: bool, optional
            Whether or not to update the value based on the old one or not. If not, then it is updated based on the
            eligibility trace.

        Returns
        -------
        kc_post: np.ndarray[float]
            the new KC responses.
        apl_post: np.ndarray[float]
            the new APL responses.
        dan_post: np.ndarray[float]
            the new DAN responses.
        mbon_post: np.ndarray[float]
            the new MBON responses.
        """
        a_cs = self.f_cs(cs)

        _kc = kc_pre.dot(self.w_k2k) if self.w_k2k is not None else 0.
        _kc += a_cs.dot(self.w_c2k) + apl_pre.dot(self.w_a2k) + self.b_k
        a_kc = self.f_kc(self.update_values(_kc, v_pre=kc_pre, eta=None if v_update else (1. - self._lambda)))

        us = np.array(us, dtype=self.dtype)
        a_us = self.f_us(us)

        _dan = a_us.dot(self.w_u2d) + mbon_pre.dot(self.w_m2d) + dan_pre.dot(self.w_d2d) + self.b_d
        a_dan = self.f_dan(self.update_values(_dan, v_pre=dan_pre, eta=None if v_update else (1. - self._lambda)))

        _apl = kc_pre.dot(self.w_k2a) + self.b_a
        a_apl = self.f_apl(self.update_values(_apl, v_pre=apl_pre, eta=None if v_update else (1. - self._lambda)))

        _mbon = kc_pre.dot(self.w_k2m) / (np.sum(kc_pre) + eps) + mbon_pre.dot(self.w_m2m) + self.b_m
        a_mbon = self.f_mbon(self.update_values(_mbon, v_pre=mbon_pre, eta=None if v_update else (1. - self._lambda)))

        if self.update:
            if self.learning_rule == "dopaminergic":
                D = np.maximum(a_dan, 0).dot(self.w_d2m)
            else:
                D = a_dan
            self.w_k2m = np.clip(self.update_weights(self.w_k2m, a_kc, a_mbon, D, w_rest=self.w_rest),
                                 0, self._maximum_weight)

        # print(a_kc)
        # print(a_dan)
        # print(a_mbon)

        return a_kc, a_apl, a_dan, a_mbon

    def __repr__(self):
        return "MushroomBody(CS=%d, US=%d, KC=%d, APL=%d, DAN=%d, MBON=%d, sparseness=%0.3f, plasticity='%s')" % (
            self.nb_cs, self.nb_us, self.nb_kc, self.nb_apl, self.nb_dan, self.nb_mbon,
            self.sparseness, self.learning_rule
        )

    @property
    def free_space(self):
        """
        Percentile of the  available space in the memory.

        Returns
        -------
        float
        """
        return np.clip(1 - np.absolute(self.w_k2m - self.w_rest), 0, 1).mean()

    @property
    def novelty(self):
        fam_0 = np.maximum((np.roll(self.r_mbon, axis=-1, shift=-1) - self.r_mbon) / 2, 0)
        fam_1 = np.maximum((np.roll(self.r_mbon, axis=-1, shift=1) - self.r_mbon) / 2, 0)
        fam = fam_0 + fam_1
        return np.clip(1 - fam, 0, 1)

    @property
    def w_c2k(self):
        """
        The CS-KC synaptic weights.
        """
        return self._w_c2k

    @w_c2k.setter
    def w_c2k(self, v):
        self._w_c2k[:] = v[:]

    @property
    def w_k2k(self):
        """
        The KC-KC synaptic weights.
        """
        return self._w_k2k

    @w_k2k.setter
    def w_k2k(self, v):
        self._w_k2k[:] = v[:]

    @property
    def w_a2k(self):
        """
        The APL-KC synaptic weights.
        """
        return self._w_a2k

    @w_a2k.setter
    def w_a2k(self, v):
        self._w_a2k[:] = v[:]

    @property
    def w_k2m(self):
        """
        The KC-MBON synaptic weights.
        """
        return self._w_k2m

    @w_k2m.setter
    def w_k2m(self, v):
        self._w_k2m[:] = v[:]

    @property
    def w_m2m(self):
        """
        The MBON-MBON synaptic weights.
        """
        return self._w_m2m

    @w_m2m.setter
    def w_m2m(self, v):
        self._w_m2m[:] = v[:]

    @property
    def w_u2d(self):
        """
        The US-DAN synaptic weights.
        """
        return self._w_u2d

    @w_u2d.setter
    def w_u2d(self, v):
        self._w_u2d[:] = v[:]

    @property
    def w_m2d(self):
        """
        The MBON-DAN synaptic weights.
        """
        return self._w_m2d

    @w_m2d.setter
    def w_m2d(self, v):
        self._w_m2d[:] = v[:]

    @property
    def w_d2d(self):
        """
        The DAN-DAN synaptic weights.
        """
        return self._w_m2d

    @w_d2d.setter
    def w_d2d(self, v):
        self._w_d2d[:] = v

    @property
    def w_k2a(self):
        """
        The KC-APL synaptic weights.
        """
        return self._w_k2a

    @w_k2a.setter
    def w_k2a(self, v):
        self._w_k2a[:] = v[:]

    @property
    def w_d2m(self):
        """
        The dopaminergic strength of each DAN that transforms them into the dopaminergic factor.
        """
        return self._w_d2m

    @w_d2m.setter
    def w_d2m(self, v):
        self._w_d2m[:] = v[:]

    @property
    def b_k(self):
        """
        The KC bias.
        """
        return self._b_k

    @b_k.setter
    def b_k(self, v):
        self._b_k[:] = v[:]

    @property
    def b_m(self):
        """
        The MBON bias.
        """
        return self._b_m

    @b_m.setter
    def b_m(self, v):
        self._b_m[:] = v[:]

    @property
    def b_d(self):
        """
        The DAN bias.
        """
        return self._b_d

    @b_d.setter
    def b_d(self, v):
        self._b_d[:] = v[:]

    @property
    def b_a(self):
        """
        The APL bias.
        """
        return self._b_a

    @b_a.setter
    def b_a(self, v):
        self._b_a[:] = v[:]

    @property
    def w_rest(self):
        """
        The resting synaptic weights for the w_k2m.
        """
        return self._w_rest

    @w_rest.setter
    def w_rest(self, v):
        self._w_rest[:] = v[:]

    @property
    def nb_cs(self):
        """
        The CS number of dimensions.
        """
        return self._nb_cs

    @property
    def nb_us(self):
        """
        The US number of dimensions.
        """
        return self._nb_us

    @property
    def nb_dan(self):
        """
        The number of DANs.
        """
        return self._nb_dan

    @property
    def nb_kc(self):
        """
        The number of KCs.
        """
        return self._nb_hidden

    @property
    def nb_apl(self):
        """
        The number of APLs.
        """
        return self._nb_apl

    @property
    def nb_mbon(self):
        """
        The number of MBONs.
        """
        return self._nb_output

    @property
    def _cs(self):
        return self._inp[..., :self.nb_cs]

    @_cs.setter
    def _cs(self, v):
        self._inp[..., :self.nb_cs] = v

    @property
    def _us(self):
        return self._inp[..., self.nb_cs:]

    @_us.setter
    def _us(self, v):
        self._inp[..., self.nb_cs:] = v

    @property
    def _kc(self):
        return self._hid

    @_kc.setter
    def _kc(self, v):
        self._hid[:] = v

    @property
    def _mbon(self):
        return self._out

    @_mbon.setter
    def _mbon(self, v):
        self._out[:] = v

    @property
    def r_cs(self):
        """
        The CS responses.
        """
        return self._cs

    @property
    def r_us(self):
        """
        The US responses.
        """
        return self._us

    @property
    def r_dan(self):
        """
        The DAN responses.
        """
        return self._dan

    @property
    def r_kc(self):
        """
        The KC responses.
        """
        return self._kc

    @property
    def r_apl(self):
        """
        The APL responses.
        """
        return self._apl

    @property
    def r_mbon(self):
        """
        The MBON responses.
        """
        return self._mbon

    @property
    def sparseness(self):
        """
        The sparseness of the KCs: the percentage of the KCs that are active in every time-step.
        """
        return self._sparseness


class IncentiveCircuit(MushroomBody):
    def __init__(self, nb_cs=2, nb_us=2, nb_kc=10, nb_apl=0, nb_dan=6, nb_mbon=6, learning_rule='dopaminergic',
                 cs_magnitude=2., us_magnitude=2., ltm_charging_speed=.05, *args, **kwargs):
        """
        The Incentive Circuit is a representative compartment of the Mushroom Body that encodes the memory dynamics of
        model in small scale and it contains MBON-DAN and MBON-MBON feedback connections. This model was first
        presented in Gkanias et al. (2022).

        Parameters
        ----------
        nb_cs: int, optional
            the number of neurons representing the conditional stimulus (CS) or the projection neurons (PN).
        nb_us: int, optional
            the number of neurons representing the unconditional stimulus (US).
        nb_kc: int, optional
            the number of Kenyon cells (KCs).
        nb_apl: int, optional
            the number of Anterior pair lateral (APL) neurons
        nb_dan: int, optional
            the number of Dopaminergic neurons (DANs).
        nb_mbon: int, optional
            the number of MB output neurons (MBONs).
        learning_rule: callable | str
            the learning rule for the updates of the KC-MBON synaptic weights.
        cs_magnitude: float, optional
            a constant that the CS will be multiplied with before feeding to the KCs.
        us_magnitude: float, optional
            a constant that the US will be multiplied with before feeding to the DANs.
        ltm_charging_speed: float, optional
            the charging (and discharging) speed of the long-term memory MBONs.
        """
        kwargs.setdefault('nb_repeats', 4)
        kwargs.setdefault('repeat_rate', 1)

        self._cs_magnitude = cs_magnitude
        self._us_magnitude = us_magnitude
        self._memory_charging_speed = ltm_charging_speed

        if not hasattr(self, "_pds") or not hasattr(self, "_pde"):
            self._pds, self._pde = 0, 2  # d-DANs
        if not hasattr(self, "_pcs") or not hasattr(self, "_pce"):
            self._pcs, self._pce = 2, 4  # c-DANs
        if not hasattr(self, "_pfs") or not hasattr(self, "_pfe"):
            self._pfs, self._pfe = 4, 6  # m-DANs
        if not hasattr(self, "_pss") or not hasattr(self, "_pse"):
            self._pss, self._pse = 0, 2  # s-MBONs
        if not hasattr(self, "_prs") or not hasattr(self, "_pre"):
            self._prs, self._pre = 2, 4  # r-MBONs
        if not hasattr(self, "_pms") or not hasattr(self, "_pme"):
            self._pms, self._pme = 4, 6  # m-MBONs

        super().__init__(nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_apl=nb_apl, nb_dan=nb_dan, nb_mbon=nb_mbon,
                         learning_rule=learning_rule, *args, **kwargs)

        self.w_c2k *= self._cs_magnitude

        self.us_names = ["punishment", "reward"]
        self.cs_names = ["A", "B"]
        self.mbon_names = ["s_{av}", "s_{at}", "r_{av}", "r_{at}", "m_{av}", "m_{at}"]
        self.dan_names = ["d_{av}", "d_{at}", "c_{av}", "c_{at}", "f_{av}", "f_{at}"]

    def reset(self, has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True):
        super().reset()

        pds, pde = self._pds, self._pde
        pcs, pce = self._pcs, self._pce
        pfs, pfe = self._pfs, self._pfe
        pss, pse = self._pss, self._pse
        prs, pre = self._prs, self._pre
        pms, pme = self._pms, self._pme

        self.w_d2m, self.b_m = uniform_synapses(self.nb_dan, self.nb_mbon, dtype=self.dtype, bias=0.)
        self.w_m2d, self.b_d = uniform_synapses(self.nb_mbon, self.nb_dan, dtype=self.dtype, bias=0.)
        self.w_m2m = uniform_synapses(self.nb_mbon, self.nb_mbon, dtype=self.dtype)

        # self.b_d[pds:pde] = -.5
        # self.b_d[pcs:pce] = -.15
        # self.b_d[pfs:pfe] = -.15
        # self.b_m[pss:pse] = -2.
        # self.b_m[prs:pre] = -2.
        # self.b_m[pms:pme] = -2.
        self.b_d[pds:pde] = 0.
        self.b_d[pcs:pce] = 0.
        self.b_d[pfs:pfe] = 0.
        self.b_m[pss:pse] = -2.
        self.b_m[prs:pre] = 0.
        self.b_m[pms:pme] = -1.

        self._dan[0] = self.b_d.copy()
        self._mbon[0] = self.b_m.copy()

        # susceptible memory (SM) sub-circuit
        if has_sm:
            # Susceptible memories depress their opposite DANs
            self.w_m2d[pss:pse, pds:pde] += opposing_synapses(pse-pss, pde-pds, fill_value=-1, dtype=self.dtype)
            # Discharging DANs depress their opposite susceptible MBONs
            self.w_d2m[pds:pde, pss:pse] += opposing_synapses(pde-pds, pse-pss, fill_value=-1, dtype=self.dtype)
            # Susceptible MBONs depress the other susceptible MBONs
            # self.w_m2m[pss:pse, pss:pse] = opposing_synapses(pse-pss, pse-pss, fill_value=-1, dtype=self.dtype)

        # restrained memory (RM) sub-circuit
        if has_rm:
            # Susceptible memories depress their opposite restrained MBONs
            self.w_m2m[pss:pse, prs:pre] += opposing_synapses(pse-pss, pre-prs, fill_value=-1, dtype=self.dtype)

        if has_ltm:
            # Long-term memory (LTM) sub-circuit
            self.w_m2d[pms:pme, pcs:pce] += opposing_synapses(pme-pms, pce-pcs, fill_value=self.memory_charging_speed,
                                                             dtype=self.dtype)
            self.w_m2d[pms:pme, pcs:pce] += roll_synapses(self.w_m2d[pms:pme, pcs:pce], left=1)
            # Charging DANs enhance their respective memory MBONs
            self.w_d2m[pcs:pce, pms:pme] += opposing_synapses(pce-pcs, pme-pms, fill_value=self.memory_charging_speed,
                                                             dtype=self.dtype)
            self.w_d2m[pcs:pce, pms:pme] += roll_synapses(self.w_d2m[pcs:pce, pms:pme], right=1)

        # reciprocal restrained memories (RRM) sub-circuit
        if has_rrm:
            # Restrained memories enhance their respective DANs
            self.w_m2d[prs:pre, pcs:pce] += diagonal_synapses(pre-prs, pce-pcs, fill_value=1., dtype=self.dtype)

            # Charging DANs depress their opposite restrained MBONs
            self.w_d2m[pcs:pce, prs:pre] += opposing_synapses(pce-pcs, pre-prs, fill_value=-1, dtype=self.dtype)

        # reciprocal forgetting memories (RFM) sub-circuit
        if has_rfm:
            # Relative states enhance their respective DANs
            self.w_m2d[pms:pme, pfs:pfe] += diagonal_synapses(pme-pms, pfe-pfs, fill_value=1., dtype=self.dtype)

            # Forgetting DANs depress their opposite long-term memory MBONs
            self.w_d2m[pfs:pfe, pms:pme] += opposing_synapses(pfe-pfs, pme-pms, fill_value=-1, dtype=self.dtype)

        # Memory assimilation mechanism (MAM)
        if has_mam:
            self.w_d2m[pfs:pfe, prs:pre] += diagonal_synapses(pfe-pfs, pre-prs, fill_value=-self.memory_charging_speed,
                                                              dtype=self.dtype)

        self.w_u2d = uniform_synapses(self.nb_us, self.nb_dan, dtype=self.dtype)
        self.w_u2d[:, pds:pde] = diagonal_synapses(pde - pds, pde - pds,
                                                   fill_value=self._us_magnitude, dtype=self.dtype)
        self.w_u2d[:, pcs:pce] = diagonal_synapses(pce - pcs, pce - pcs,
                                                   fill_value=self._us_magnitude, dtype=self.dtype)

    @property
    def memory_charging_speed(self):
        """
        the charging (and discharging) speed of the long-term memory MBONs.
        """
        return self._memory_charging_speed

    @property
    def novelty(self):
        r_s = self.r_mbon[0, :, self._pss:self._pse]
        r_r = self.r_mbon[0, :, self._prs:self._pre]
        r_m = self.r_mbon[0, :, self._pms:self._pme]

        fam_s = r_s[:, 0] - np.mean(r_s[:, 1:], axis=1)
        fam_r = r_r[:, 0] - np.mean(r_r[:, 1:], axis=1)
        fam_m = r_m[:, 0] - np.mean(r_m[:, 1:], axis=1)
        return np.clip(1 - (fam_s + fam_r + fam_m) / 3, 0, 1)

    def __repr__(self):
        return "IncentiveCircuit(CS=%d, US=%d, KC=%d, DAN=%d, MBON=%d, LTM_charging_speed=%.2f, plasticity='%s')" % (
            self.nb_cs, self.nb_us, self.nb_kc, self.nb_dan, self.nb_mbon,
            self.memory_charging_speed, self.learning_rule
        )


class IncentiveWheel(IncentiveCircuit):
    def __init__(self, nb_cs=8, nb_us=8, nb_kc=None, nb_dan=None, nb_mbon=None, nb_apl=0, learning_rule='dopaminergic',
                 *args, **kwargs):
        """
        The Incentive Wheel is an extension of the Incentive Circuit and more complete model of the Mushroom Body that
        encodes the memory dynamics of model related with the susceptible, restrained and lont-term memory MBONs. It
        contains MBON-DAN and MBON-MBON feedback connections similarly to the Incentive Circuit, but it also connects
        different incentive circuits that share MBONs with different roles. This model was first presented in
        Gkanias et al (2021).

        Parameters
        ----------
        nb_cs: int, optional
        nb_us: int, optional
        nb_kc: int, optional
        nb_apl: int, optional
        nb_dan: int, optional
        nb_mbon: int, optional
        learning_rule
        """

        if nb_cs is None:
            nb_cs = 8
        if nb_us is None and nb_dan is not None:
            nb_us = nb_dan // 2
            nb_dan = 2 * nb_us
        elif nb_us is None and nb_mbon is not None:
            nb_us = nb_mbon // 2
            nb_mbon = 2 * nb_us
        elif nb_us is None:
            nb_us = 8
        if nb_kc is None:
            nb_kc = 5 * nb_cs
        if nb_dan is None:
            nb_dan = 2 * nb_us
        if nb_mbon is None:
            nb_mbon = nb_dan

        self._pds, self._pde = 0, nb_dan // 2  # d-DANs
        self._pcs, self._pce = nb_dan // 2, nb_dan  # c-DANs
        self._pfs, self._pfe = nb_dan // 2, nb_dan  # m-DANs
        self._pss, self._pse = 0, nb_mbon // 2  # s-MBONs
        self._prs, self._pre = nb_mbon // 2, nb_mbon  # r-MBONs
        self._pms, self._pme = nb_mbon // 2, nb_mbon  # m-MBONs

        super(IncentiveWheel, self).__init__(nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_apl=nb_apl, nb_dan=nb_dan,
                                             nb_mbon=nb_mbon, learning_rule=learning_rule, *args, **kwargs)

        self.us_names = ["friendly", "predator", "unexpected", "failure",
                         "abominable", "enemy", "new territory", "posses"]
        self.mbon_names = ["s_%d" % i for i in range(self.nb_mbon//2)] + ["m_%d" % i for i in range(self.nb_mbon//2)]
        self.dan_names = ["d_%d" % i for i in range(self.nb_dan//2)] + ["f_%d" % i for i in range(self.nb_dan//2)]

    def reset(self, **kwargs):
        kwargs.setdefault("has_rfm", False)
        super().reset(**kwargs)

    def __repr__(self):
        return super().__repr__().replace("IncentiveCircuit", "IncentiveWheel")


class CrossIncentive(IncentiveCircuit):
    def __init__(self, nb_cs=10, nb_us=4, nb_kc=None, nb_dan=None, nb_mbon=None, nb_apl=0, learning_rule='dopaminergic',
                 *args, **kwargs):
        """
        The Incentive Wheel is an extension of the Incentive Circuit and more complete model of the Mushroom Body that
        encodes the memory dynamics of model related with the susceptible, restrained and lont-term memory MBONs. It
        contains MBON-DAN and MBON-MBON feedback connections similarly to the Incentive Circuit, but it also connects
        different incentive circuits that share MBONs with different roles. This model was first presented in
        Gkanias et al (2021).

        Parameters
        ----------
        nb_cs: int, optional
        nb_us: int, optional
        nb_kc: int, optional
        nb_apl: int, optional
        nb_dan: int, optional
        nb_mbon: int, optional
        learning_rule
        """

        if nb_cs is None:
            nb_cs = 8
        if nb_us is None and nb_dan is not None:
            nb_us = nb_dan // 2
            nb_dan = 2 * nb_us
        elif nb_us is None and nb_mbon is not None:
            nb_us = nb_mbon // 2
            nb_mbon = 2 * nb_us
        elif nb_us is None:
            nb_us = 8
        if nb_kc is None:
            nb_kc = 5 * nb_cs
        if nb_dan is None:
            nb_dan = 2 * nb_us
        if nb_mbon is None:
            nb_mbon = nb_dan

        self._pds, self._pde = 0, nb_dan // 2  # d-DANs
        self._pcs, self._pce = nb_dan // 2, nb_dan  # c-DANs
        self._pfs, self._pfe = nb_dan // 2, nb_dan  # m-DANs
        self._pss, self._pse = 0, nb_mbon // 2  # s-MBONs
        self._prs, self._pre = nb_mbon // 2, nb_mbon  # r-MBONs
        self._pms, self._pme = nb_mbon // 2, nb_mbon  # m-MBONs

        super(CrossIncentive, self).__init__(nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_apl=nb_apl, nb_dan=nb_dan,
                                             nb_mbon=nb_mbon, learning_rule=learning_rule, *args, **kwargs)

        self.us_names = ["left turn", "right turn", "not left turn", "not right turn"]
        self.mbon_names = ["s_{L}", "s_{R}", "s_{nL}", "s_{nR}", "m_{L}", "m_{R}", "m_{nL}", "m_{nR}"]
        self.dan_names = ["d_{L}", "d_{R}", "d_{nL}", "d_{nR}", "c_{L}", "c_{R}", "c_{nL}", "c_{nR}"]

    def reset(self, **kwargs):
        kwargs.setdefault("has_ltm", False)
        kwargs.setdefault("has_rfm", False)
        kwargs.setdefault("has_mam", False)
        super().reset(**kwargs)

        pds, pde = self._pds, self._pde
        pcs, pce = self._pcs, self._pce
        pfs, pfe = self._pfs, self._pfe
        pss, pse = self._pss, self._pse
        prs, pre = self._prs, self._pre
        pms, pme = self._pms, self._pme

        self.b_d[pds:pde] = 0.
        self.b_d[pcs:pce] = 0.
        self.b_d[pfs:pfe] = 0.
        self.b_m[pss:pse] = 0.
        self.b_m[prs:pre] = 0.
        self.b_m[pms:pme] = 0.

        self._dan[0, :, ...] = self.b_d.copy()
        self._mbon[0, :, ...] = self.b_m.copy()

        # susceptible memory (SM) sub-circuit
        # Susceptible MBONs depress the other susceptible MBONs
        self.w_m2m[pss:pse, pss:pse] = np.array([
            [0., 1., 1., 0.],
            [1., 0., 0., 1.],
            [1., 0., 0., 1.],
            [0., 1., 1., 0.]
        ]) * (-.5)

        # Long-term memory (LTM) sub-circuit
        # Charging DANs enhance their respective memory MBONs
        self.w_d2m[pcs:pce, pms:pme] += np.array([
            [0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]
        ]) * self.memory_charging_speed

        # Memory assimilation mechanism (MAM)
        self.w_d2m[pfs:pfe, prs:pre] += np.array([
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]
        ]) * (-self.memory_charging_speed)

    def __repr__(self):
        return super().__repr__().replace("IncentiveCircuit", "CrossIncentive")


class VisualIncentiveCircuit(IncentiveCircuit):
    def __init__(self, nb_cs=None, nb_us=None, nb_kc=None, nb_dan=None, nb_mbon=None, *args, **kwargs):
        """
        The Incentive Wheel is an extension of the Incentive Circuit and more complete model of the Mushroom Body that
        encodes the memory dynamics of model related with the susceptible, restrained and lont-term memory MBONs. It
        contains MBON-DAN and MBON-MBON feedback connections similarly to the Incentive Circuit, but it also connects
        different incentive circuits that share MBONs with different roles. This model was first presented in
        Gkanias et al (2021).

        Parameters
        ----------
        nb_cs: int, optional
        nb_us: int, optional
        nb_kc: int, optional
        nb_apl: int, optional
        nb_dan: int, optional
        nb_mbon: int, optional
        learning_rule
        """
        kwargs.setdefault('nb_repeats', 4)
        nb_cs = kwargs.pop('nb_input', nb_cs)
        nb_kc = kwargs.pop('nb_sparse', kwargs.pop('nb_hidden', nb_kc))

        assert nb_cs is not None, "__init__() missing 1 required positional argument: 'nb_cs'"

        if nb_us is None and nb_dan is not None:
            nb_us = nb_dan // 3
        elif nb_us is None and nb_mbon is not None:
            nb_us = nb_mbon // 3
        elif nb_us is None:
            nb_us = 2
        if nb_us % 2 != 0:  # make sure that the number of US is even
            nb_us += 1
        if nb_kc is None:
            # nb_kc = 40 * nb_cs
            nb_kc = 4000
        if nb_dan is None:
            nb_dan = 3 * nb_us
        if nb_mbon is None:
            nb_mbon = nb_dan

        self._pds, self._pde = 0, nb_dan // 3  # d-DANs
        self._pcs, self._pce = nb_dan // 3, 2 * nb_dan // 3  # c-DANs
        self._pfs, self._pfe = 2 * nb_dan // 3, 3 * nb_dan // 3  # m-DANs
        self._pss, self._pse = 0, nb_mbon // 3  # s-MBONs
        self._prs, self._pre = nb_mbon // 3, 2 * nb_mbon // 3  # r-MBONs
        self._pms, self._pme = 2 * nb_mbon // 3,  3 * nb_mbon // 3  # m-MBONs

        kwargs.setdefault("cs_magnitude", 1)
        kwargs.setdefault("us_magnitude", 2)
        kwargs.setdefault("sparseness", 10. / nb_kc)
        kwargs.setdefault("eligibility_trace", 0.)
        kwargs.setdefault("ltm_charging_speed", 5e-4)
        super().__init__(nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_dan=nb_dan, nb_mbon=nb_mbon, *args, **kwargs)

        self.f_cs = lambda x: np.asarray(
            (x.T - x.min(axis=-1)) / (x.max(axis=-1) - x.min(axis=-1)), dtype=self.dtype).T
        self.f_kc = lambda x: np.asarray(
            winner_takes_all(x, percentage=self.sparseness, noise=.01, normalise=False), dtype=self.dtype)

        self.us_names = (["attractive", "repulsive"] +
                         [f"{chr(ord('A') + s)}_{mot} " for s in range(nb_us - 2) for mot in ["a", "r"]])
        self.mbon_names = (["s_{a}", "s_{r}"] + [f"s_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                 for mot in ["a", "r"]] +
                           ["r_{a}", "r_{r}"] + [f"r_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                 for mot in ["a", "r"]] +
                           ["m_{a}", "m_{r}"] + [f"m_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                 for mot in ["a", "r"]])
        self.dan_names = (["d_{a}", "d_{r}"] + [f"d_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                for mot in ["a", "r"]] +
                          ["c_{a}", "c_{r}"] + [f"c_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                for mot in ["a", "r"]] +
                          ["f_{a}", "f_{r}"] + [f"f_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                for mot in ["a", "r"]])

        if self.__class__ == VisualIncentiveCircuit:
            self.reset()

    def reset(self, **kwargs):
        super().reset(**kwargs)

        pds, pde = self._pds, self._pde
        pcs, pce = self._pcs, self._pce
        pfs, pfe = self._pfs, self._pfe
        pss, pse = self._pss, self._pse
        prs, pre = self._prs, self._pre
        pms, pme = self._pms, self._pme

        self.b_d[pds:pde] = 0.
        self.b_d[pcs:pce] = 0.
        self.b_d[pfs:pfe] = 0.
        self.b_m[pss:pse] = -0.
        self.b_m[prs:pre] = -0.
        self.b_m[pms:pme] = -0.

        self._dan[0, :, ...] = self.b_d.copy()
        self._mbon[0, :, ...] = self.b_m.copy()

        self.w_d2m *= 0.
        self.w_m2m *= 0.
        self.w_m2d *= 0.
        self.w_d2d *= 0.

        # SUSCEPTIBLE MEMORY (SM) microcircuit

        # Susceptible MBONs inhibit their opposite discharging DANs
        # self.w_m2d[pss:pse, pds:pde] += np.array(
        #     [[0.] + [1.] * (self.nb_dan // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_dan // 3 - 1)] * (self.nb_mbon // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pse - pss - 1)
        v = 1
        self.w_m2d[pss:pse, pds:pde] += pattern_synapses(diagonal_synapses((pse-pss) // 2, (pde-pds) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # Discharging DANs depress their opposite susceptible MBONs
        # self.w_d2m[pds:pde, pss:pse] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        v = 1
        self.w_d2m[pds:pde, pss:pse] += pattern_synapses(diagonal_synapses((pde-pds) // 2, (pse-pss) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # RESTRAINED MEMORY (RM) microcircuit

        # Susceptible MBONs depress their opposite restrained MBONs
        # self.w_m2m[pss:pse, prs:pre] = np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_mbon // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pse - pss - 1)
        v = 1
        self.w_m2m[pss:pse, prs:pre] += pattern_synapses(diagonal_synapses((pse-pss) // 2, (pre-prs) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # RESTRAINED MEMORY (RM) microcircuit

        # Susceptible MBONs depress their opposite restrained MBONs
        # self.w_m2m[pss:pse, pss:pse] = np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_mbon // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # self.w_m2m[pss:pse, pss:pse] = diagonal_synapses(pse-pss, pse-pss, fill_value=1., dtype=self.dtype) - 1.

        # RECIPROCAL SHORT-TERM MEMORIES (RSM) microcircuit

        # Restrained MBONs excite their respective charging DANs
        v = 1
        self.w_m2d[prs:pre, pcs:pce] += diagonal_synapses(pre-prs, pce-pcs, fill_value=v, dtype=self.dtype)

        # Restrained MBONs inhibit the opposite charging DANs
        # self.w_m2d[prs:pre, pcs:pce] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pre - prs - 1)
        # v = 1
        # self.w_m2d[prs:pre, pcs:pce] += pattern_synapses(diagonal_synapses((pre-prs) // 2, (pce-pcs) // 2),
        #                                                  opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # # Charging DANs inhibit other charging DANs
        # self.w_d2d[pcs:pce, pcs:pce] += diagonal_synapses(pce-pcs, pce-pcs, fill_value=1., dtype=self.dtype) - 1

        # Charging DANs depress their opposite restrained MBONs
        # self.w_d2m[pcs:pce, prs:pre] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        v = 1
        self.w_d2m[pcs:pce, prs:pre] += pattern_synapses(diagonal_synapses((pce-pcs) // 2, (pre-prs) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # LONG-TERM MEMORY (LTM) microcircuit

        # LTM MBONs excite their respective charging DANs
        v = self.memory_charging_speed
        self.w_m2d[pms:pme, pcs:pce] += diagonal_synapses(pme-pms, pce-pcs, fill_value=v, dtype=self.dtype)

        # Forgetting DANs inhibit other forgetting DANs
        # self.w_d2d[pfs:pfe, pfs:pfe] += diagonal_synapses(pce-pcs, pce-pcs, fill_value=1., dtype=self.dtype) - 1

        # Charging DANs potentiate their respective LTM MBONs
        v = self.memory_charging_speed
        self.w_d2m[pcs:pce, pms:pme] += diagonal_synapses(pce-pcs, pme-pms, fill_value=v, dtype=self.dtype)

        # RECIPROCAL LONG-TERM MEMORIES (RLM) microcircuit

        # LTM MBONs excite their respective forgetting DANs
        v = 1
        self.w_m2d[pms:pme, pfs:pfe] += diagonal_synapses(pme-pms, pfe-pfs, fill_value=v, dtype=self.dtype)

        # LTM MBONs inhibit their opposite forgetting DANs
        # self.w_m2d[pms:pme, pfs:pfe] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pme - pfs - 1)
        # v = 1
        # self.w_m2d[pms:pme, pfs:pfe] += pattern_synapses(diagonal_synapses((pme-pms) // 2, (pfe-pfs) // 2),
        #                                                  opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # Forgetting DANs depress their opposite long-term memory MBONs
        # self.w_d2m[pfs:pfe, pms:pme] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        v = 1
        self.w_d2m[pfs:pfe, pms:pme] += pattern_synapses(diagonal_synapses((pfe-pfs) // 2, (pme-pms) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # MEMORY ASSIMILATION MECHANISM (MAM) microcircuit

        # Forgetting DANs depress their respective restrained MBONs
        v = self.memory_charging_speed
        self.w_d2m[pfs:pfe, prs:pre] += diagonal_synapses(pfe-pfs, pre-prs, fill_value=-v, dtype=self.dtype)

    def __repr__(self):
        return super().__repr__().replace("IncentiveCircuit", "VisualIncentiveCircuit")


class VectorMemoryMB(VisualIncentiveCircuit):
    def __init__(self, *args, **kwargs):
        """
        The Incentive Wheel is an extension of the Incentive Circuit and more complete model of the Mushroom Body that
        encodes the memory dynamics of model related with the susceptible, restrained and lont-term memory MBONs. It
        contains MBON-DAN and MBON-MBON feedback connections similarly to the Incentive Circuit, but it also connects
        different incentive circuits that share MBONs with different roles. This model was first presented in
        Gkanias et al (2021).
        """
        super().__init__(*args, **kwargs)

        self.us_names = (["approach home", "avoid home"] +
                         [f"{mot} {chr(ord('A') + s)}" for s in range(self.nb_us - 2) for mot in ["approach", "avoid"]])
        self.mbon_names = (["s_{ap}", "s_{av}"] +
                           [f"s_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(self.nb_us // 2 - 1)
                            for mot in ["ap", "av"]] +
                           ["r_{ap}", "r_{av}"] +
                           [f"r_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(self.nb_us // 2 - 1)
                            for mot in ["ap", "av"]] +
                           ["m_{ap}", "m_{av}"] +
                           [f"m_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(self.nb_us // 2 - 1)
                            for mot in ["ap", "av"]])
        self.dan_names = (["d_{ap}", "d_{av}"] +
                          [f"d_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(self.nb_us // 2 - 1)
                           for mot in ["ap", "av"]] +
                          ["c_{ap}", "c_{av}"] +
                          [f"c_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(self.nb_us // 2 - 1)
                           for mot in ["ap", "av"]] +
                          ["f_{ap}", "f_{av}"] +
                          [f"f_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(self.nb_us // 2 - 1)
                           for mot in ["ap", "av"]])

    def reset(self, **kwargs):
        super().reset(**kwargs)

        pds, pde = self._pds, self._pde
        pcs, pce = self._pcs, self._pce
        pfs, pfe = self._pfs, self._pfe
        pss, pse = self._pss, self._pse
        prs, pre = self._prs, self._pre
        pms, pme = self._pms, self._pme

        self.b_d[pds:pde] = -0.5
        self.b_d[pcs:pce] = 0.
        self.b_d[pfs:pfe] = 0.
        self.b_m[pss:pse] = -0.
        self.b_m[prs:pre] = -0.
        self.b_m[pms:pme] = -0.

        self._dan[0, :, ...] = self.b_d.copy()
        self._mbon[0, :, ...] = self.b_m.copy()

        self.w_d2m *= 0.
        self.w_m2m *= 0.
        self.w_m2d *= 0.
        self.w_d2d *= 0.

        # SUSCEPTIBLE MEMORY (SM) microcircuit

        # Susceptible MBONs inhibit their opposite discharging DANs
        # self.w_m2d[pss:pse, pds:pde] += np.array(
        #     [[0.] + [1.] * (self.nb_dan // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_dan // 3 - 1)] * (self.nb_mbon // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pse - pss - 1)
        v = 1
        self.w_m2d[pss:pse, pds:pde] += pattern_synapses(diagonal_synapses((pse-pss) // 2, (pde-pds) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # Discharging DANs depress their opposite susceptible MBONs
        # self.w_d2m[pds:pde, pss:pse] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        v = 1
        self.w_d2m[pds:pde, pss:pse] += pattern_synapses(diagonal_synapses((pde-pds) // 2, (pse-pss) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # RESTRAINED MEMORY (RM) microcircuit

        # Susceptible MBONs depress their opposite restrained MBONs
        # self.w_m2m[pss:pse, prs:pre] = np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_mbon // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pse - pss - 1)
        v = 1
        self.w_m2m[pss:pse, prs:pre] += pattern_synapses(diagonal_synapses((pse-pss) // 2, (pre-prs) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # RESTRAINED MEMORY (RM) microcircuit

        # Susceptible MBONs depress their opposite restrained MBONs
        # self.w_m2m[pss:pse, pss:pse] = np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_mbon // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # self.w_m2m[pss:pse, pss:pse] = diagonal_synapses(pse-pss, pse-pss, fill_value=1., dtype=self.dtype) - 1.

        # RECIPROCAL SHORT-TERM MEMORIES (RSM) microcircuit

        # Restrained MBONs excite their respective charging DANs
        v = 1
        self.w_m2d[prs:pre, pcs:pce] += diagonal_synapses(pre-prs, pce-pcs, fill_value=v, dtype=self.dtype)

        # Restrained MBONs inhibit the opposite charging DANs
        # self.w_m2d[prs:pre, pcs:pce] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pre - prs - 1)
        v = 1
        self.w_m2d[prs:pre, pcs:pce] += pattern_synapses(diagonal_synapses((pre-prs) // 2, (pce-pcs) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # # Charging DANs inhibit other charging DANs
        # self.w_d2d[pcs:pce, pcs:pce] += diagonal_synapses(pce-pcs, pce-pcs, fill_value=1., dtype=self.dtype) - 1

        # Charging DANs depress their opposite restrained MBONs
        # self.w_d2m[pcs:pce, prs:pre] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        v = 1
        self.w_d2m[pcs:pce, prs:pre] += pattern_synapses(diagonal_synapses((pce-pcs) // 2, (pre-prs) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # LONG-TERM MEMORY (LTM) microcircuit

        # LTM MBONs excite their respective charging DANs
        v = self.memory_charging_speed
        self.w_m2d[pms:pme, pcs:pce] += diagonal_synapses(pme-pms, pce-pcs, fill_value=v, dtype=self.dtype)

        # Forgetting DANs inhibit other forgetting DANs
        # self.w_d2d[pfs:pfe, pfs:pfe] += diagonal_synapses(pce-pcs, pce-pcs, fill_value=1., dtype=self.dtype) - 1

        # Charging DANs potentiate their respective LTM MBONs
        v = self.memory_charging_speed
        self.w_d2m[pcs:pce, pms:pme] += diagonal_synapses(pce-pcs, pme-pms, fill_value=v, dtype=self.dtype)

        # RECIPROCAL LONG-TERM MEMORIES (RLM) microcircuit

        # LTM MBONs excite their respective forgetting DANs
        v = 1
        self.w_m2d[pms:pme, pfs:pfe] += diagonal_synapses(pme-pms, pfe-pfs, fill_value=v, dtype=self.dtype)

        # LTM MBONs inhibit their opposite forgetting DANs
        # self.w_m2d[pms:pme, pfs:pfe] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        # v = 1 / (pme - pfs - 1)
        v = 1
        self.w_m2d[pms:pme, pfs:pfe] += pattern_synapses(diagonal_synapses((pme-pms) // 2, (pfe-pfs) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # Forgetting DANs depress their opposite long-term memory MBONs
        # self.w_d2m[pfs:pfe, pms:pme] += np.array(
        #     [[0.] + [1.] * (self.nb_mbon // 3 - 1)] +
        #     [[1.] + [0.] * (self.nb_mbon // 3 - 1)] * (self.nb_dan // 3 - 1),
        #     dtype=self.dtype) * (-1.)
        v = 1
        self.w_d2m[pfs:pfe, pms:pme] += pattern_synapses(diagonal_synapses((pfe-pfs) // 2, (pme-pms) // 2),
                                                         opposing_synapses(2, 2, fill_value=-v), dtype=self.dtype)

        # MEMORY ASSIMILATION MECHANISM (MAM) microcircuit

        # Forgetting DANs depress their respective restrained MBONs
        v = self.memory_charging_speed
        self.w_d2m[pfs:pfe, prs:pre] += diagonal_synapses(pfe-pfs, pre-prs, fill_value=-v, dtype=self.dtype)

    def __repr__(self):
        return super().__repr__().replace("IncentiveCircuit", "VectorMemoryMB")


class IncentiveCircuitMemory(MemoryComponent):

    def __init__(self, nb_input, nb_output=1, nb_sparse=None, learning_rule='dopaminergic', eligibility_trace=.0,
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
                                  eligibility_trace=eligibility_trace, sparseness=sparseness, ndim=self.ndim,
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

    def _fprop(self, cs=None, us=None):
        """
        Running the forward propagation.

        Parameters
        ----------
        cs: np.ndarray[float]
            The current input.
        us: np.ndarray[float]
            The current reinforcement.

        Returns
        -------
        np.ndarray[float]
            the novelty of the input element before the update
        """
        if cs is None:
            cs = np.zeros_like(self._inp, dtype=self.dtype)
        if us is None:
            us = [0, 0]
        elif isinstance(us, float):
            us = [np.maximum(us, 0), np.maximum(-us, 0)]
        elif len(us) < 2:
            us = [np.maximum(us[0], 0), np.maximum(-us[0], 0)]
        elif isinstance(us, np.ndarray):
            if us.shape[0] == self.ndim and us.ndim == 1:
                us = [[np.maximum(r, 0), np.maximum(-r, 0)] for r in us]

        cs = np.array(cs, dtype=self.dtype)
        us = np.array(us, dtype=self.dtype)
        if cs.ndim < 2:
            cs = cs[np.newaxis, ...]

        if self.update:
            nb_repeats = 1
        else:
            nb_repeats = 1

        for i in range(nb_repeats):
            self._ic(cs=cs, us=us)

        # print(self._ic.r_mbon[0], self._ic.r_dan[0])

        self._inp[:] = self._ic.r_cs[0]
        self._hid[:] = self._ic.r_kc[0]

        self._out[:] = np.clip(np.mean(
            self._ic.r_mbon[0][..., self.__pos] - self._ic.r_mbon[0][..., self.__neg], axis=1), -1, 1)
        # print(self._out)
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
