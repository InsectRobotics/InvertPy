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
from .component import Component
from .synapses import uniform_synapses, diagonal_synapses, sparse_synapses, opposing_synapses, roll_synapses,\
    pattern_synapses
from .activation import linear, relu, winner_takes_all

import numpy as np


class MushroomBody(Component):
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

        super().__init__(nb_cs + nb_us, nb_mbon, learning_rule=learning_rule, *args, **kwargs)

        self._nb_cs = nb_cs
        self._nb_us = nb_us
        self._nb_dan = nb_dan
        self._nb_kc = nb_kc
        self._nb_apl = nb_apl
        self._nb_mbon = nb_mbon

        # set the parameters (synapses)
        self._w_c2k = sparse_synapses(self.nb_cs, self.nb_kc,  nb_in_min=1, nb_in_max=4,
                                      max_samples=400, dtype=self.dtype)
        # self._w_c2k *= self.nb_cs / self.w_c2k.sum(axis=1)[:, np.newaxis]
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
        self._cs = np.zeros((self._repeats, nb_cs), dtype=self.dtype)
        self._us = np.zeros((self._repeats, nb_us), dtype=self.dtype)
        self._dan = np.zeros((self._repeats, nb_dan), dtype=self.dtype)
        self._kc = np.zeros((self._repeats, nb_kc), dtype=self.dtype)
        self._apl = np.zeros((self._repeats, nb_apl), dtype=self.dtype)
        self._mbon = np.zeros((self._repeats, nb_mbon), dtype=self.dtype)

        self.f_cs = lambda x: linear(x, noise=self._noise, rng=self.rng)
        self.f_us = lambda x: linear(x, noise=self._noise, rng=self.rng)
        self.f_dan = lambda x: relu(x, cmax=2, noise=self._noise, rng=self.rng)
        self.f_kc = lambda x: relu(x, cmax=2, noise=self._noise, rng=self.rng)
        self.f_apl = lambda x: relu(x, cmax=2, noise=self._noise, rng=self.rng)
        self.f_mbon = lambda x: relu(x, cmax=2, noise=self._noise, rng=self.rng)

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

        # reset responses
        self._cs = np.zeros((self._repeats, self.nb_cs), dtype=self.dtype)
        self._us = np.zeros((self._repeats, self.nb_us), dtype=self.dtype)
        self._dan = np.zeros((self._repeats, self.nb_dan), dtype=self.dtype)
        self._kc = np.zeros((self._repeats, self.nb_kc), dtype=self.dtype)
        self._apl = np.zeros((self._repeats, self.nb_apl), dtype=self.dtype)
        self._mbon = np.zeros((self._repeats, self.nb_mbon), dtype=self.dtype)

        self.update = True

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
                return self.r_cs[0, self.cs_names.index(neuron_name)]

        elif neuron_name in self.us_names:
            if all_repeats:
                return self.r_us[..., self.us_names.index(neuron_name)]
            else:
                return self.r_us[0, self.us_names.index(neuron_name)]
        elif neuron_name in self.dan_names:
            if all_repeats:
                return self.r_dan[..., self.dan_names.index(neuron_name)]
            else:
                return self.r_dan[0, self.dan_names.index(neuron_name)]
        elif neuron_name in self.kc_names:
            if all_repeats:
                return self.r_kc[..., self.kc_names.index(neuron_name)]
            else:
                return self.r_kc[0, self.kc_names.index(neuron_name)]
        elif neuron_name in self.apl_names:
            if all_repeats:
                return self.r_apl[..., self.apl_names.index(neuron_name)]
            else:
                return self.r_apl[0, self.apl_names.index(neuron_name)]
        elif neuron_name in self.mbon_names:
            if all_repeats:
                return self.r_mbon[..., self.mbon_names.index(neuron_name)]
            else:
                return self.r_mbon[0, self.mbon_names.index(neuron_name)]
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
        cs = np.array(cs, dtype=self.dtype)
        us = np.array(us, dtype=self.dtype)

        if len(us) < self.nb_us:
            _us = us
            us = np.zeros(self.nb_us)
            us[:len(_us)] = _us
        elif us.shape[0] > self.nb_us:
            us = us[:self.nb_us]
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

        _mbon = kc_pre.dot(self.w_k2m) + mbon_pre.dot(self.w_m2m) + self.b_m
        a_mbon = self.f_mbon(self.update_values(_mbon, v_pre=mbon_pre, eta=None if v_update else (1. - self._lambda)))

        if self.update:
            if self.learning_rule == "dopaminergic":
                D = np.maximum(a_dan, 0).dot(self.w_d2m)
            else:
                D = a_dan
            self.w_k2m = np.clip(self.update_weights(self.w_k2m, a_kc, a_mbon, D, w_rest=self.w_rest),
                                 0, self._maximum_weight)

        return a_kc, a_apl, a_dan, a_mbon

    def __repr__(self):
        return "MushroomBody(CS=%d, US=%d, KC=%d, APL=%d, DAN=%d, MBON=%d, sparseness=%0.3f, plasticity='%s')" % (
            self.nb_cs, self.nb_us, self.nb_kc, self.nb_apl, self.nb_dan, self.nb_mbon,
            self.sparseness, self.learning_rule
        )

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
        return self._nb_kc

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
        return self._nb_mbon

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
        presented in Gkanias et al (2021).

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


class VectorMemoryMB(IncentiveCircuit):
    def __init__(self, nb_cs, nb_us=None, nb_kc=None, nb_dan=None, nb_mbon=None, learning_rule='dopaminergic',
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
        kwargs.setdefault('nb_repeats', 4)

        if nb_us is None and nb_dan is not None:
            nb_us = nb_dan // 3
        elif nb_us is None and nb_mbon is not None:
            nb_us = nb_mbon // 3
        elif nb_us is None:
            nb_us = 4
        if nb_us % 2 != 0:  # make sure that the number of US is even
            nb_us += 1
        if nb_kc is None:
            nb_kc = 10 * nb_cs
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
        kwargs.setdefault("ltm_charging_speed", .01)
        super().__init__(nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_dan=nb_dan,
                         nb_mbon=nb_mbon, learning_rule=learning_rule, *args, **kwargs)

        # self.f_kc = lambda x: relu(x, cmax=2, noise=self._noise, rng=self.rng)
        self.f_kc = lambda x: np.asarray(
            winner_takes_all(x, percentage=1 / self.nb_kc, noise=.01), dtype=self.dtype)

        self.us_names = (["approach home", "avoid home"] +
                         [f"{mot} {chr(ord('A') + s)}" for s in range(nb_us - 2) for mot in ["approach", "avoid"]])
        self.mbon_names = (["s_{ap}", "s_{av}"] + [f"s_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                   for mot in ["ap", "av"]] +
                           ["r_{ap}", "r_{av}"] + [f"r_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                   for mot in ["ap", "av"]] +
                           ["m_{ap}", "m_{av}"] + [f"m_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                   for mot in ["ap", "av"]])
        self.dan_names = (["d_{ap}", "d_{av}"] + [f"d_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                  for mot in ["ap", "av"]] +
                          ["c_{ap}", "c_{av}"] + [f"c_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                  for mot in ["ap", "av"]] +
                          ["f_{ap}", "f_{av}"] + [f"f_{{{mot}}}^{{{chr(ord('A') + s)}}}" for s in range(nb_us // 2 - 1)
                                                  for mot in ["ap", "av"]])

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
        self.b_m[pms:pme] = -1.

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
        self.w_m2d[prs:pre, pcs:pce] += diagonal_synapses(pre-prs, pce-pcs, fill_value=1, dtype=self.dtype)

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
        return super().__repr__().replace("IncentiveCircuit", "FamiliarityCircuit")

    @property
    def w_d2d(self):
        return self._w_d2d

    @w_d2d.setter
    def w_d2d(self, v):
        self._w_d2d[:] = v
