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

from ._helpers import eps
from .component import Component
from .plasticity import dopaminergic, anti_hebbian
from .synapses import uniform_synapses, diagonal_synapses, sparse_synapses, opposing_synapses, roll_synapses
from .activation import relu, winner_takes_all

from sklearn.metrics import mean_squared_error

import numpy as np


class MushroomBody(Component):
    def __init__(self, nb_cs, nb_us, nb_kc, nb_dan, nb_mbon, nb_apl=1, learning_rule=dopaminergic, sparseness=0.03,
                 *args, **kwargs):
        """
        The Mushroom Body component of the insect brain is responsible for creating associations between the input (CS)
        and output (MBON) based on the reinforcement (US) and by modulating the KC-MBON connections. The KCs receive CS
        input and their output activates the MBONs. The APL works as a dynamic threshold for the KCs receiving input
        from them and inhibiting them via global and local inhibition. DANs get US input and their output modulate the
        KC-MBON connections. There are also KC-KC and MBON-DAN connections supported.

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

        # set the parameters (synapses)
        self._w_c2k = uniform_synapses(nb_cs, nb_kc, dtype=self.dtype)
        self._w_k2k = None
        self._w_a2k, self._b_k = uniform_synapses(nb_apl, nb_kc, dtype=self.dtype, bias=0)
        self._w_k2m = uniform_synapses(nb_kc, nb_mbon, dtype=self.dtype)
        self._w_m2m, self._b_m = uniform_synapses(nb_mbon, nb_mbon, dtype=self.dtype, bias=0)
        self._w_u2d = uniform_synapses(nb_us, nb_dan, dtype=self.dtype)
        self._w_m2d, self._b_d = uniform_synapses(nb_mbon, nb_dan, dtype=self.dtype, bias=0)
        self._w_k2a, self._b_a = uniform_synapses(nb_kc, nb_apl, dtype=self.dtype, bias=0)

        self._w_d2m = uniform_synapses(nb_dan, nb_mbon, dtype=self.dtype)
        self._w_rest = uniform_synapses(nb_kc, nb_mbon, dtype=self.dtype)

        self.params.extend([self.w_c2k, self.w_a2k, self.w_k2m, self.w_m2m, self.w_u2d, self.w_m2d, self.w_k2a,
                            self.w_d2m, self.w_rest, self.b_k, self.b_m, self.b_d, self.b_a])

        # reserve space for the responses
        self._cs = np.zeros((self._repeats, nb_cs), dtype=self.dtype)
        self._us = np.zeros((self._repeats, nb_us), dtype=self.dtype)
        self._dan = np.zeros((self._repeats, nb_dan), dtype=self.dtype)
        self._kc = np.zeros((self._repeats, nb_kc), dtype=self.dtype)
        self._apl = np.zeros((self._repeats, nb_apl), dtype=self.dtype)
        self._mbon = np.zeros((self._repeats, nb_mbon), dtype=self.dtype)

        self._nb_cs = nb_cs
        self._nb_us = nb_us
        self._nb_dan = nb_dan
        self._nb_kc = nb_kc
        self._nb_apl = nb_apl
        self._nb_mbon = nb_mbon

        self.f_cs = lambda x: x
        self.f_us = lambda x: x
        self.f_dan = lambda x: relu(x, cmax=2)
        self.f_kc = lambda x: relu(x, cmax=2)
        self.f_apl = lambda x: relu(x, cmax=2)
        self.f_mbon = lambda x: relu(x, cmax=2)

        self._sparseness = sparseness

        self.cs_names = ["c_{%d}" % i for i in range(nb_cs)]
        self.us_names = ["u_{%d}" % i for i in range(nb_us)]
        self.dan_names = ["d_{%d}" % i for i in range(nb_dan)]
        self.kc_names = ["k_{%d}" % i for i in range(nb_kc)]
        self.apl_names = ["a_{%d}" % i for i in range(nb_apl)]
        self.mbon_names = ["m_{%d}" % i for i in range(nb_mbon)]

        self.reset()

    def reset(self):
        # reset synapses
        self.w_c2k = sparse_synapses(self.nb_cs, self.nb_kc, dtype=self.dtype)
        self.w_c2k *= self.nb_cs / self.w_c2k.sum(axis=1)[:, np.newaxis]

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

    def _fprop(self, cs=None, us=None):
        """
        It propagates the CS and US signal forwards through the connections of the model for nb_repeats times. It
        updates the internal responses of the neurons and returns the final output of the MBONs.

        Parameters
        ----------
        cs: np.ndarray, optional
            the CS input. Default is 0.
        us: np.ndarray, optional
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
        cs: np.ndarray
            The current CS input.
        us: np.ndarray
            The current US reinforcement.
        kc_pre: np.ndarray
            The old KC responses.
        apl_pre: np.ndarray
            The old APL responses.
        dan_pre: np.ndarray
            The old DAN responses.
        mbon_pre: np.ndarray
            The old MBON responses.
        v_update: bool, optional
            Whether or not to update the value based on the old one or not. If not, then it is updated based on the
            eligibility trace.

        Returns
        -------
        r_post: tuple
            kc_post: np.ndarray
                the new KC responses.
            apl_post: np.ndarray
                the new APL responses.
            dan_post: np.ndarray
                the new DAN responses.
            mbon_post: np.ndarray
                the new MBON responses.
        """
        a_cs = self.f_cs(cs)
        _kc = kc_pre @ self.w_k2k if self.w_k2k is not None else 0.
        _kc += a_cs @ self.w_c2k + apl_pre @ self.w_a2k + self.b_k
        a_kc = self.f_kc(self.update_values(_kc, v_pre=kc_pre, eta=None if v_update else (1. - self._lambda)))

        a_us = self.f_us(us)
        _dan = a_us @ self.w_u2d + mbon_pre @ self.w_m2d + self.b_d
        a_dan = self.f_dan(self.update_values(_dan, v_pre=dan_pre, eta=None if v_update else (1. - self._lambda)))

        _apl = kc_pre @ self.w_k2a + self.b_a
        a_apl = self.f_apl(self.update_values(_apl, v_pre=apl_pre, eta=None if v_update else (1. - self._lambda)))

        _mbon = kc_pre @ self.w_k2m + mbon_pre @ self.w_m2m + self.b_m
        a_mbon = self.f_mbon(self.update_values(_mbon, v_pre=mbon_pre, eta=None if v_update else (1. - self._lambda)))

        if self.update:
            if self.learning_rule == "dopaminergic":
                D = np.maximum(a_dan, 0) @ self.w_d2m
            else:
                D = a_dan
            self.w_k2m = np.maximum(self.update_weights(self.w_k2m, a_kc, a_mbon, D, w_rest=self.w_rest), 0)

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


class WillshawNetwork(MushroomBody):

    def __init__(self, nb_cs=360, nb_us=1, nb_kc=200000, nb_apl=0, nb_dan=1, nb_mbon=1, learning_rule=anti_hebbian,
                 eligibility_trace=.1, *args, **kwargs):
        """
        The Whillshaw Network is a simplified Mushroom Body component and it does not contain MBON-to-DAN connections.
        This model is a modified version of the one presented in [1]_.

        Notes
        -----
        .. [1] Ardin, P., Peng, F., Mangan, M., Lagogiannis, K. & Webb, B. Using an Insect Mushroom Body Circuit to
           Encode Route Memory in Complex Natural Environments. Plos Comput Biol 12, e1004683 (2016).
        """

        super(WillshawNetwork, self).__init__(
            nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_apl=nb_apl, nb_dan=nb_dan, nb_mbon=nb_mbon,
            learning_rule=learning_rule, eligibility_trace=eligibility_trace, *args, **kwargs)

        self.f_cs = lambda x: np.asarray(x > np.sort(x)[int(self.nb_cs * .7)], dtype=self.dtype)
        self.f_dan = lambda x: relu(x, cmax=2)
        # self.f_kc = lambda x: np.asarray(x > 0, dtype=self.dtype)
        # self.f_kc = lambda x: np.asarray(
        #     x >= np.sort(x)[::-1][int(self.sparseness * self.nb_kc)], dtype=self.dtype)
        self.f_kc = lambda x: np.asarray(winner_takes_all(x, percentage=self.sparseness), dtype=self.dtype)
        self.f_mbon = lambda x: relu(x)

    def reset(self):
        super().reset()
        self.w_rest *= 0

    def __repr__(self):
        return "WillshawNetwork(PN=%d, KC=%d, EN=%d, eligibility_trace=%.2f, plasticity='%s')" % (
            self.nb_cs, self.nb_kc, self.nb_mbon, self._lambda, self.learning_rule
        )


class PerfectMemory(MushroomBody):

    def __init__(self, nb_cs=360, nb_mbon=1, error_metric=mean_squared_error, *args, **kwargs):
        """
        The Perfect Memory is a simplified Mushroom Body component and it does not contain any neural connections.
        This model stores all the input received in a list and searches for the best match every time that receives a
        new input are reports the minimum difference. This was used for comparison by many papers including [1]_.

        Parameters
        ----------
        error_metric: callable
            the metric that measures the error between the observation and the database.

        Notes
        -----
        .. [1] Ardin, P., Peng, F., Mangan, M., Lagogiannis, K. & Webb, B. Using an Insect Mushroom Body Circuit to
           Encode Route Memory in Complex Natural Environments. Plos Comput Biol 12, e1004683 (2016).
        """

        kwargs.setdefault('nb_repeats', 1)
        super().__init__(
            nb_cs=nb_cs, nb_us=0, nb_kc=0, nb_apl=0, nb_dan=0, nb_mbon=nb_mbon, eligibility_trace=0., *args, **kwargs)

        self.f_cs = lambda x: x

        self._error_metric = error_metric
        self._database = None
        self.reset()

    def reset(self):
        super().reset()

        # erase the database
        self._database = []

    def _fprop(self, cs=None, us=None):
        if cs is None:
            cs = np.zeros_like(self._cs[0])

        self._cs[0] = self.f_cs(cs)

        if len(self._database) > 0:
            y_true = self.database.T
            y_pred = np.array([self._cs[0]] * len(self._database)).T
            self._mbon[0] = self._error_metric(y_true, y_pred, multioutput='raw_values', squared=False).min()
        else:
            self._mbon[0] = 1.

        if self.update:
            self._database.append(self._cs[0].copy())

        return self._mbon[0]

    def __repr__(self):
        return "PerfectMemory(PN=%d, EN=%d, error=%s)" % (self.nb_cs, self.nb_mbon, self.error_metric)

    @property
    def database(self):
        return np.array(self._database)

    @property
    def error_metric(self):
        return self._error_metric.__name__


class IncentiveCircuit(MushroomBody):
    def __init__(self, nb_cs=2, nb_us=2, nb_kc=10, nb_apl=0, nb_dan=3, nb_mbon=3, learning_rule=dopaminergic,
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

        self._cs_magnitude = cs_magnitude
        self._us_magnitude = us_magnitude
        self._memory_charging_speed = ltm_charging_speed

        self._pds, self._pde = 0, 2  # d-DANs
        self._pcs, self._pce = 2, 4  # c-DANs
        self._pfs, self._pfe = 4, 6  # m-DANs
        self._pss, self._pse = 0, 2  # s-MBONs
        self._prs, self._pre = 2, 4  # r-MBONs
        self._pms, self._pme = 4, 6  # m-MBONs

        super().__init__(nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_apl=nb_apl, nb_dan=nb_dan, nb_mbon=nb_mbon,
                         learning_rule=learning_rule, *args, **kwargs)

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

        self.b_d[pds:pde] = -.5
        self.b_d[pcs:pce] = -.15
        self.b_d[pfs:pfe] = -.15
        self.b_m[pss:pse] = -2.
        self.b_m[prs:pre] = -2.
        self.b_m[pms:pme] = -2.

        self._dan[0] = self.b_d.copy()
        self._mbon[0] = self.b_m.copy()

        # susceptible memory (SM) sub-circuit
        if has_sm:
            # Susceptible memories depress their opposite DANs
            self.w_m2d[pss:pse, pds:pde] = opposing_synapses(pse-pss, pde-pds, fill_value=-.3, dtype=self.dtype)
            # Discharging DANs depress their opposite susceptible MBONs
            self.w_d2m[pds:pde, pss:pse] = opposing_synapses(pde-pds, pse-pss, fill_value=-1, dtype=self.dtype)

        # restrained memory (RM) sub-circuit
        if has_rm:
            # Susceptible memories depress their opposite restrained MBONs
            self.w_m2m[pss:pse, prs:pre] = opposing_synapses(pse-pss, pre-prs, fill_value=-1, dtype=self.dtype)

        if has_ltm:
            # Long-term memory (LTM) sub-circuit
            self.w_m2d[pms:pme, pcs:pce] = opposing_synapses(pme-pms, pce-pcs, fill_value=self.memory_charging_speed,
                                                             dtype=self.dtype)
            self.w_m2d[pms:pme, pcs:pce] = roll_synapses(self.w_m2d[pms:pme, pcs:pce], left=1)
            # Charging DANs enhance their respective memory MBONs
            self.w_d2m[pcs:pce, pms:pme] = opposing_synapses(pce-pcs, pme-pms, fill_value=self.memory_charging_speed,
                                                             dtype=self.dtype)
            self.w_d2m[pcs:pce, pms:pme] = roll_synapses(self.w_d2m[pcs:pce, pms:pme], right=1)

        # reciprocal restrained memories (RRM) sub-circuit
        if has_rrm:
            # Restrained memories enhance their respective DANs
            self.w_m2d[prs:pre, pcs:pce] = diagonal_synapses(pre-prs, pce-pcs, fill_value=.5, dtype=self.dtype)

            # Charging DANs depress their opposite restrained MBONs
            self.w_d2m[pcs:pce, prs:pre] = opposing_synapses(pce-pcs, pre-prs, fill_value=-1, dtype=self.dtype)

        # reciprocal forgetting memories (RFM) sub-circuit
        if has_rfm:
            # Relative states enhance their respective DANs
            self.w_m2d[pms:pme, pfs:pfe] = diagonal_synapses(pme-pms, pfe-pfs, fill_value=.5, dtype=self.dtype)

            # Forgetting DANs depress their opposite long-term memory MBONs
            self.w_d2m[pfs:pfe, pms:pme] = opposing_synapses(pfe-pfs, pme-pms, fill_value=-1, dtype=self.dtype)

        # Memory assimilation mechanism (MAM)
        if has_mam:
            self.w_d2m[pfs:pfe, prs:pre] = diagonal_synapses(pfe-pfs, pre-prs, fill_value=-self.memory_charging_speed,
                                                             dtype=self.dtype)

        self.w_c2k *= self._cs_magnitude

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
    def __init__(self, nb_cs=8, nb_us=8, nb_kc=40, nb_apl=0, nb_dan=16, nb_mbon=16, learning_rule=dopaminergic,
                 *args, **kwargs):
        """
        The Incentive Wheel is an extension of the Incentive Circuit and more complete model of the Mushroom Body that
        encodes the memory dynamics of model related with the susceptible, restrained and lont-term memory MBONs. It
        contains MBON-DAN and MBON-MBON feedback connections similarly to the Incentive Circuit, but it also connects
        different incentive circuits that share MBONs with different roles. This model was first presented in
        Gkanias et al (2021).

        Parameters
        ----------
        nb_cs
        nb_us
        nb_kc
        nb_apl
        nb_dan
        nb_mbon
        learning_rule
        args
        kwargs
        """

        self._pds, self._pde = 0, 8  # d-DANs
        self._pcs, self._pce = 8, 16  # c-DANs
        self._pfs, self._pfe = 8, 16  # m-DANs
        self._pss, self._pse = 0, 8  # s-MBONs
        self._prs, self._pre = 8, 16  # r-MBONs
        self._pms, self._pme = 8, 16  # m-MBONs

        super(IncentiveWheel, self).__init__(nb_cs=nb_cs, nb_us=nb_us, nb_kc=nb_kc, nb_apl=nb_apl, nb_dan=nb_dan,
                                             nb_mbon=nb_mbon, learning_rule=learning_rule, *args, **kwargs)

        self.us_names = ["friendly", "predator", "unexpected", "failure",
                         "abominable", "enemy", "new territory", "posses"]
        self.mbon_names = ["s_%d" % i for i in range(self.nb_mbon//2)] + ["m_%d" % i for i in range(self.nb_mbon//2)]
        self.dan_names = ["d_%d" % i for i in range(self.nb_dan//2)] + ["f_%d" % i for i in range(self.nb_dan//2)]

    def __repr__(self):
        return super().__repr__().replace("IncentiveCircuit", "IncentiveWheel")


class SusceptibleMemory(IncentiveCircuit):
    def __init__(self, *args, **kwargs):
        """
        The Susceptible Memory is a sub-circuit of the Incentive Circuit that has only susceptible MBONs and discharging
        DANs. This model was first presented in Gkanias et al (2021).
        """
        kwargs.setdefault('ltm_charging_speed', 0.5)
        super().__init__(*args, **kwargs)

        self.mbon_names = self.mbon_names[:2]
        self.dan_names = self.dan_names[:2]

    def reset(self, *args, **kwargs):
        super().reset(has_sm=True, has_rm=False, has_ltm=False, has_rrm=False, has_rfm=False, has_mam=False)

        self.b_m[self._prs:self._pre] = -2.
        self.b_m[self._pms:self._pme] = -4.

        self._mbon[0] = self.b_m.copy()

    def _fprop(self, *args, **kwargs):
        mbons = super()._fprop(*args, **kwargs)
        return mbons[:2]

    def __repr__(self):
        return "SusceptibleMemory(CS=%d, US=%d, KC=%d, DAN=2, MBON=2, plasticity='%s')" % (
            self.nb_cs, self.nb_cs, self.nb_cs, self.learning_rule
        )

    @property
    def r_mbon(self):
        return super().r_mbon[..., :2]

    @property
    def r_dan(self):
        return super().r_dan[..., :2]


class RestrainedMemory(IncentiveCircuit):
    def __init__(self, *args, **kwargs):
        """
        The Restrained Memory is a sub-circuit of the Incentive Circuit that has only susceptible and restrained MBONs
        and discharging DANs. This model was first presented in Gkanias et al (2021).
        """
        kwargs.setdefault('ltm_charging_speed', 0.5)
        super().__init__(*args, **kwargs)

        self.mbon_names = self.mbon_names[:4]
        self.dan_names = self.dan_names[:2]

    def reset(self, *args, **kwargs):
        super().reset(has_sm=True, has_rm=True, has_ltm=False, has_rrm=False, has_rfm=False, has_mam=False)

        self.b_m[self._prs:self._pre] = -2.
        self.b_m[self._pms:self._pme] = -4.

        self._mbon[0] = self.b_m.copy()

    def _fprop(self, *args, **kwargs):
        mbons = super()._fprop(*args, **kwargs)
        return mbons[:4]

    def __repr__(self):
        return "RestrainedMemory(CS=%d, US=%d, KC=%d, DAN=2, MBON=4, plasticity='%s')" % (
            self.nb_cs, self.nb_cs, self.nb_cs, self.learning_rule
        )

    @property
    def r_mbon(self):
        return super().r_mbon[..., :4]

    @property
    def r_dan(self):
        return super().r_dan[..., :2]


class LongTermMemory(IncentiveCircuit):
    def __init__(self, *args, **kwargs):
        """
        The Long-term Memory is a sub-circuit of the Incentive Circuit that has only long-term memory MBONs
        and charging DANs. This model was first presented in Gkanias et al (2021).
        """
        kwargs.setdefault('ltm_charging_speed', 0.5)
        super().__init__(*args, **kwargs)

        self.mbon_names = self.mbon_names[4:6]
        self.dan_names = self.dan_names[2:4]

    def reset(self, *args, **kwargs):
        super().reset(has_sm=True, has_rm=True, has_ltm=True, has_rrm=False, has_rfm=False, has_mam=False)

        self.b_m[self._prs:self._pre] = -2.
        self.b_m[self._pms:self._pme] = -4.

        self._mbon[0] = self.b_m.copy()

    def _fprop(self, *args, **kwargs):
        mbons = super()._fprop(*args, **kwargs)
        return mbons[4:6]

    def __repr__(self):
        return "LongTermMemory(CS=%d, US=%d, KC=%d, DAN=2, MBON=2, LTM_charging_speed=%.2f, plasticity='%s')" % (
            self.nb_cs, self.nb_cs, self.nb_cs, self._memory_charging_speed, self.learning_rule
        )

    @property
    def r_mbon(self):
        return super().r_mbon[..., 4:6]

    @property
    def r_dan(self):
        return super().r_dan[..., 2:4]


class ReciprocalRestrainedMemories(IncentiveCircuit):
    def __init__(self, *args, **kwargs):
        """
        The Reciprocal Restrained Memories is a sub-circuit of the Incentive Circuit that has only restrained MBONs and
        charging DANs. This model was first presented in Gkanias et al (2021).
        """
        kwargs.setdefault('ltm_charging_speed', 0.5)
        super().__init__(*args, **kwargs)

        self.mbon_names = self.mbon_names[2:4]
        self.dan_names = self.dan_names[2:4]

    def reset(self, *args, **kwargs):
        super().reset(has_sm=True, has_rm=True, has_ltm=False, has_rrm=True, has_rfm=False, has_mam=False)

        self.b_m[self._prs:self._pre] = -2.
        self.b_m[self._pms:self._pme] = -4.

        self._mbon[0] = self.b_m.copy()

    def _fprop(self, *args, **kwargs):
        mbons = super()._fprop(*args, **kwargs)
        return mbons[2:4]

    def __repr__(self):
        return "ReciprocalRestrainedMemories(CS=%d, US=%d, KC=%d, DAN=2, MBON=2, plasticity='%s')" % (
            self.nb_cs, self.nb_cs, self.nb_cs, self.learning_rule
        )

    @property
    def r_mbon(self):
        return super().r_mbon[..., 2:4]

    @property
    def r_dan(self):
        return super().r_dan[..., 2:4]


class ReciprocalForgettingMemories(IncentiveCircuit):
    def __init__(self, *args, **kwargs):
        """
        The Reciprocal Forgetting Memories is a sub-circuit of the Incentive Circuit that has only long-term memory
        MBONs and forgetting DANs. This model was first presented in Gkanias et al (2021).
        """
        kwargs.setdefault('ltm_charging_speed', 0.5)
        super().__init__(*args, **kwargs)

        self.mbon_names = self.mbon_names[4:6]
        self.dan_names = self.dan_names[4:6]

    def reset(self, *args, **kwargs):
        super().reset(has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=False)

        self.b_m[self._prs:self._pre] = -2.
        self.b_m[self._pms:self._pme] = -4.

        self._mbon[0] = self.b_m.copy()

    def _fprop(self, *args, **kwargs):
        mbons = super()._fprop(*args, **kwargs)
        return mbons[4:6]

    def __repr__(self):
        return ("ReciprocalForgettingMemories(CS=%d, US=%d, KC=%d, DAN=2, MBON=2, LTM_charging_speed=%.2f,"
                " plasticity='%s')") % (
            self.nb_cs, self.nb_cs, self.nb_cs, self._memory_charging_speed, self.learning_rule
        )

    @property
    def r_mbon(self):
        return super().r_mbon[..., 4:6]

    @property
    def r_dan(self):
        return super().r_dan[..., 4:6]


class MemoryAssimilationMechanism(IncentiveCircuit):
    def __init__(self, *args, **kwargs):
        """
        The Memory Assimilation Mechanism is a sub-circuit of the Incentive Circuit that has only restrained and
        long-term memory MBONs and charging and forgetting DANs. This model was first presented in Gkanias et al (2021).
        """
        kwargs.setdefault('ltm_charging_speed', 0.5)
        super().__init__(*args, **kwargs)

        self.mbon_names = self.mbon_names[2:6]
        self.dan_names = self.dan_names[2:6]

    def reset(self, *args, **kwargs):
        super().reset(has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True)

        self.b_m[self._prs:self._pre] = -2.
        self.b_m[self._pms:self._pme] = -4.

        self._mbon[0] = self.b_m.copy()

    def _fprop(self, *args, **kwargs):
        mbons = super()._fprop(*args, **kwargs)
        return mbons[2:6]

    def __repr__(self):
        return ("MemoryAssimilationMechanism(CS=%d, US=%d, KC=%d, DAN=4, MBON=4, LTM_charging_speed=%.2f,"
                " plasticity='%s')") % (
            self.nb_cs, self.nb_cs, self.nb_cs, self._memory_charging_speed, self.learning_rule
        )

    @property
    def r_mbon(self):
        return super().r_mbon[..., 2:6]

    @property
    def r_dan(self):
        return super().r_dan[..., 2:6]
