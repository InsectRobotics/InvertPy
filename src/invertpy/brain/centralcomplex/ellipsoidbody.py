
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

from abc import ABC

from invertpy.brain.synapses import *
from invertpy.brain.activation import sigmoid

from .centralcomplex import CentralComplexLayer


class EllipsoidBodyLayer(CentralComplexLayer):

    def __init__(self, nb_epg=16, nb_delta7=8, nb_cl1=None, nb_tb1=None, mix=.667, *args, **kwargs):
        if nb_cl1 is not None:
            nb_epg = nb_cl1
        if nb_tb1 is not None:
            nb_delta7 = nb_tb1
        kwargs.setdefault('nb_input', nb_epg)
        kwargs.setdefault('nb_output', nb_delta7)
        super().__init__(*args, **kwargs)

        self._nb_epg = nb_epg
        self._nb_delta7 = nb_delta7

        self._w_epg2delta7 = np.zeros((nb_epg, nb_delta7), dtype=self.dtype)
        self._w_delta72delta7 = np.zeros((nb_delta7, nb_delta7), dtype=self.dtype)

        self._epg_slope = 3.0
        self._delta7_slope = 5.0
        self._epg_b = -0.5
        self._delta7_b = 0

        self.params.extend([
            self._w_epg2delta7,
            self._w_delta72delta7,
            self._epg_slope,
            self._delta7_slope,
            self._epg_b,
            self._delta7_b
        ])

        self._epg = np.zeros(self.nb_epg, dtype=self.dtype)
        self._delta7 = np.zeros(self.nb_delta7, dtype=self.dtype)

        self.f_epg = lambda v: sigmoid(v * self._epg_slope - self._epg_b, noise=self._noise, rng=self.rng)
        self.f_delta7 = lambda v: sigmoid(v * self._delta7_slope - self._delta7_b, noise=self._noise, rng=self.rng)

        self._mix = mix

    def reset(self):
        self.w_epg2delta7 = diagonal_synapses(self.nb_cl1, self.nb_tb1, fill_value=1, tile=True, dtype=self.dtype)
        self.w_delta72delta7 = sinusoidal_synapses(self.nb_tb1, self.nb_tb1, fill_value=1, dtype=self.dtype) - 1

        self.r_epg = np.zeros(self.nb_epg, dtype=self.dtype)
        self.r_delta7 = np.zeros(self.nb_delta7, dtype=self.dtype)

        self.update = True

    def _fprop(self, epg=None, delta7=None, cl1=None, tb1=None):

        if tb1 is not None and delta7 is None:
            delta7 = tb1
        if delta7 is None:
            delta7 = self.r_delta7
        if cl1 is not None and epg is None:
            epg = cl1
        if epg is None:
            epg = self.r_epg

        self._epg = a_epg = self.f_epg(epg)

        delta7 = self._mix * a_epg.dot(self.w_epg2delta7) + (1 - self._mix) * delta7.dot(self.w_delta72delta7)
        self._delta7 = a_delta7 = self.f_delta7(delta7)

        return a_delta7

    @property
    def w_epg2delta7(self):
        return self._w_epg2delta7

    @w_epg2delta7.setter
    def w_epg2delta7(self, v):
        self._w_epg2delta7[:] = v

    @property
    def w_cl12tb1(self):
        return self.w_epg2delta7

    @property
    def w_delta72delta7(self):
        return self._w_delta72delta7

    @property
    def w_tb12tb1(self):
        return self.w_delta72delta7

    @w_delta72delta7.setter
    def w_delta72delta7(self, v):
        self._w_delta72delta7[:] = v

    @property
    def r_epg(self):
        return self._epg

    @r_epg.setter
    def r_epg(self, v):
        self._epg[:] = v

    @property
    def r_cl1(self):
        return self._epg

    @property
    def r_delta7(self):
        return self._delta7

    @r_delta7.setter
    def r_delta7(self, v):
        self._delta7[:] = v

    @property
    def r_tb1(self):
        return self._delta7

    @property
    def nb_epg(self):
        return self._nb_epg

    @property
    def nb_cl1(self):
        return self.nb_epg

    @property
    def nb_delta7(self):
        return self._nb_delta7

    @property
    def nb_tb1(self):
        return self.nb_delta7


class ProtocerebralBridgeLayer(CentralComplexLayer, ABC):

    def __init__(self, nb_pfl3=16, nb_cpu1=None, *args, **kwargs):
        if nb_cpu1 is not None:
            nb_pfl3 = nb_cpu1
        kwargs.setdefault('nb_output', nb_pfl3)
        super().__init__(*args, **kwargs)

        self._pfl3 = np.zeros(nb_pfl3, dtype=self.dtype)

        self._pfl3_slope = 5.0
        self._pfl3_b = 2.5

        self.f_pfl3 = lambda v: sigmoid(v * self._pfl3_slope - self._pfl3_b, noise=self._noise, rng=self.rng)

    def reset(self):
        self.r_pfl3 = np.zeros(self.nb_pfl3, dtype=self.dtype)

        self.update = True

    @property
    def r_pfl3(self):
        return self._pfl3

    @r_pfl3.setter
    def r_pfl3(self, v):
        self._pfl3[:] = v

    @property
    def r_cpu1(self):
        return self._pfl3

    @r_cpu1.setter
    def r_cpu1(self, v):
        self._pfl3[:] = v

    @property
    def nb_pfl3(self):
        return self._nb_output

    @property
    def nb_cpu1(self):
        return self._nb_output

    @property
    def f_cpu1(self):
        return self.f_pfl3


class SimpleCompass(EllipsoidBodyLayer):
    def __init__(self, nb_tl2=16, *args, **kwargs):
        kwargs.setdefault('nb_cl1', nb_tl2)
        super().__init__(*args, **kwargs)

        self._w_tl22cl1 = np.zeros((nb_tl2, self.nb_cl1), dtype=self.dtype)

        self._nb_tl2 = nb_tl2

        self._tl2_slope = 6.8
        self._tl2_b = 3.0

        self.params.extend([
            self._w_tl22cl1,
            self._tl2_slope,
            self._tl2_b
        ])

        self._tl2 = np.zeros(nb_tl2, dtype=self.dtype)

        self.tl2_prefs = np.tile(np.linspace(0, 2 * np.pi, self.nb_tb1, endpoint=False), 2)

        self.f_tl2 = lambda v: sigmoid(v * self._tl2_slope - self._tl2_b, noise=self._noise, rng=self.rng)

    def reset(self):
        super().reset()

        self.w_tl22cl1 = diagonal_synapses(self.nb_tl2, self.nb_cl1, fill_value=-1, dtype=self.dtype)
        self.r_tl2 = np.zeros(self.nb_tl2, dtype=self.dtype)

    def _fprop(self, phi=None, tl2=None, **kwargs):

        if isinstance(phi, np.ndarray) and phi.size == self.nb_tl2 // 2:
            if tl2 is None:
                tl2 = np.tile(phi, 2)
            if kwargs.get('cl1', None) is not None:
                cl1 = kwargs.pop('cl1', None)
            elif kwargs.get('epg', None) is not None:
                cl1 = kwargs.pop('epg', None)
            else:
                cl1 = np.tile(phi, 2)
            a_tl2 = self._tl2 = self.f_tl2(tl2[::-1])
            a_cl1 = self._epg = self.f_epg(cl1[::-1])
            a_tb1 = self._delta7 = self.f_delta7(5. * phi[::-1])
        else:
            if phi is not None:
                a_tl2 = self._tl2 = self.f_tl2(self.phi2tl2(phi))
            else:
                a_tl2 = self._tl2
            a_tb1 = super()._fprop(cl1=a_tl2.dot(self.w_tl22cl1))

        return a_tb1

    def phi2tl2(self, phi):
        """
        Transform the heading direction to the TL2 responses.

        Parameters
        ----------
        phi: float
            the feading direction in radiance.

        Returns
        -------
        r_tl2: np.ndarray
            the TL2 responses based on their preference angle
        """
        return np.cos(phi - self.tl2_prefs)

    @property
    def w_tl22cl1(self):
        return self._w_tl22cl1

    @w_tl22cl1.setter
    def w_tl22cl1(self, v):
        self._w_tl22cl1[:] = v

    @property
    def r_tl2(self):
        return self._tl2

    @r_tl2.setter
    def r_tl2(self, v):
        self._tl2[:] = v

    @property
    def nb_tl2(self):
        return self._nb_tl2


class SimpleSteering(ProtocerebralBridgeLayer):
    def __init__(self, nb_tb1=8, nb_cpu4=16, nb_delta7=None, nb_fbn=None, *args, **kwargs):
        if nb_delta7 is not None:
            nb_tb1 = nb_delta7
        if nb_fbn is not None:
            nb_cpu4 = nb_fbn
        kwargs.setdefault('nb_input', nb_tb1 + nb_cpu4)
        super().__init__(*args, **kwargs)

        self._w_delta72pfl3 = uniform_synapses(nb_tb1, self.nb_cpu1, fill_value=0, dtype=self.dtype)
        self._w_fbn2pfl3 = uniform_synapses(nb_cpu4, self.nb_cpu1, fill_value=0, dtype=self.dtype)
        self._w_cpu42cpu1 = uniform_synapses(nb_cpu4, self.nb_cpu1, fill_value=0, dtype=self.dtype)

        self._delta7 = np.zeros(nb_tb1, dtype=self.dtype)
        self._fbn = np.zeros(nb_cpu4, dtype=self.dtype)

        self._nb_delta7 = nb_tb1
        self._nb_fbn = nb_cpu4

    def reset(self):
        super().reset()

        w_tb12cpu1 = diagonal_synapses(self.nb_tb1, self.nb_cpu1, fill_value=-1, tile=True)
        self.w_tb12cpu1a = w_tb12cpu1[:, 1:-1]
        self.w_tb12cpu1b = np.hstack([w_tb12cpu1[:, -1:], w_tb12cpu1[:, :1]])

        w_cpu42cpu1 = opposing_synapses(self.nb_cpu4, self.nb_cpu1, fill_value=1, dtype=self.dtype)
        self.w_cpu42cpu1a = np.hstack([w_cpu42cpu1[:, :self.nb_cpu1a // 2], w_cpu42cpu1[:, -self.nb_cpu1a // 2:]])
        self.w_cpu42cpu1b = w_cpu42cpu1[:, [-self.nb_cpu1a // 2 - 1, self.nb_cpu1a // 2]]

        self.r_tb1 = np.zeros(self.nb_tb1, dtype=self.dtype)
        self.r_cpu4 = np.zeros(self.nb_cpu4, dtype=self.dtype)
        self.r_cpu1 = np.zeros(self.nb_cpu1, dtype=self.dtype)

        self.update = True

    def _fprop(self, cpu4=None, tb1=None, **kwargs):
        cpu4 = kwargs.pop('fbn', cpu4)
        tb1 = kwargs.pop('delta7', tb1)

        if cpu4 is None:
            cpu4 = self.r_cpu4
        if tb1 is None:
            tb1 = self.r_tb1

        self.r_tb1 = tb1
        self.r_cpu4 = cpu4
        a_cpu1 = self.r_cpu1 = self.f_cpu1(self._compute_cpu1(cpu4, tb1))

        return a_cpu1

    def _compute_cpu1(self, cpu4, tb1):
        cpu1a = cpu4.dot(self.w_cpu42cpu1a) + tb1.dot(self.w_tb12cpu1a)
        cpu1b = cpu4.dot(self.w_cpu42cpu1b) + tb1.dot(self.w_tb12cpu1b)

        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])

    @property
    def w_delta72pfl3(self):
        return self._w_delta72pfl3

    @property
    def w_tb12cpu1(self):
        return self._w_delta72pfl3

    @property
    def w_tb12cpu1a(self):
        """
        The TB1 to CPU1a synaptic weights.
        """
        return self._w_delta72pfl3[:, 1:-1]

    @w_tb12cpu1a.setter
    def w_tb12cpu1a(self, v):
        self._w_delta72pfl3[:, 1:-1] = v[:]

    @property
    def w_tb12cpu1b(self):
        """
        The TB1 to CPU1b synaptic weights.
        """
        return np.hstack([self._w_delta72pfl3[:, -1:], self._w_delta72pfl3[:, :1]])

    @w_tb12cpu1b.setter
    def w_tb12cpu1b(self, v):
        self._w_delta72pfl3[:, -1:] = v[:, :1]
        self._w_delta72pfl3[:, :1] = v[:, -1:]

    @property
    def w_fbn2pfl3(self):
        return self._w_fbn2pfl3

    @property
    def w_cpu42cpu1(self):
        return self._w_cpu42cpu1

    @w_cpu42cpu1.setter
    def w_cpu42cpu1(self, v):
        self._w_cpu42cpu1[:] = v

    @property
    def w_cpu42cpu1a(self):
        """
        The CPU4 to CPU1a synaptic weights.
        """
        return np.hstack([self.w_cpu42cpu1[:, :self.nb_cpu1a//2], self.w_cpu42cpu1[:, -self.nb_cpu1a//2:]])

    @w_cpu42cpu1a.setter
    def w_cpu42cpu1a(self, v):
        self.w_cpu42cpu1[:, :self.nb_cpu1a // 2] = v[:, :self.nb_cpu1a // 2]
        self.w_cpu42cpu1[:, -self.nb_cpu1a // 2:] = v[:, -self.nb_cpu1a // 2:]

    @property
    def w_cpu42cpu1b(self):
        """
        The CPU4 to CPU1b synaptic weights.
        """
        return self.w_cpu42cpu1[:, [-self.nb_cpu1a//2-1, self.nb_cpu1a//2]]

    @w_cpu42cpu1b.setter
    def w_cpu42cpu1b(self, v):
        self.w_cpu42cpu1[:, [-self.nb_cpu1a//2-1, self.nb_cpu1a//2]] = v[:]

    @property
    def r_delta7(self):
        return self._delta7

    @r_delta7.setter
    def r_delta7(self, v):
        self._delta7[:] = v

    @property
    def r_tb1(self):
        return self._delta7

    @r_tb1.setter
    def r_tb1(self, v):
        self._delta7[:] = v

    @property
    def r_fbn(self):
        return self._fbn

    @r_fbn.setter
    def r_fbn(self, v):
        self._fbn[:] = v

    @property
    def r_cpu4(self):
        return self._fbn

    @r_cpu4.setter
    def r_cpu4(self, v):
        self._fbn[:] = v

    @property
    def nb_delta7(self):
        return self._nb_delta7

    @property
    def nb_tb1(self):
        return self._nb_delta7

    @property
    def nb_fbn(self):
        return self._nb_fbn

    @property
    def nb_cpu4(self):
        return self._nb_fbn

    @property
    def nb_cpu1a(self):
        return self.nb_cpu1 - 2

    @property
    def nb_cpu1b(self):
        return 2


class PontineSteering(SimpleSteering):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._w_pou2cpu1a = uniform_synapses(self.nb_cpu1, self.nb_cpu1a, fill_value=0, dtype=self.dtype)
        self._w_pou2cpu1b = uniform_synapses(self.nb_cpu1, self.nb_cpu1b, fill_value=0, dtype=self.dtype)
        self._w_cpu42pou = uniform_synapses(self.nb_cpu4, self.nb_cpu1, fill_value=0, dtype=self.dtype)

        self._pfl3_slope = 7.5
        self._pfl3_b = -1.0
        self._pou_slope = 5.0
        self._pou_b = 2.5

        self.f_pou = lambda v: sigmoid(v * self._pou_slope - self._pou_b, noise=self._noise, rng=self.rng)

    def reset(self):
        super().reset()

        w_pontine2cpu1 = pattern_synapses(pattern=diagonal_synapses(4, 4, 1, dtype=self.dtype)[:, ::-1],
                                          patch=diagonal_synapses(self.nb_cpu1 // 4, self.nb_cpu1 // 4, 1,
                                                                  dtype=self.dtype),
                                          dtype=self.dtype) * (-1.)

        self.w_pou2cpu1a = np.hstack([w_pontine2cpu1[:, :self.nb_cpu1a // 2], w_pontine2cpu1[:, -self.nb_cpu1a // 2:]])
        self.w_pou2cpu1b = w_pontine2cpu1[:, [-self.nb_cpu1a // 2 - 1, self.nb_cpu1a // 2]]

        self.w_fbn2pou = diagonal_synapses(self.nb_cpu4, self.nb_cpu4, fill_value=1, dtype=self.dtype)

    def _compute_cpu1(self, cpu4, tb1):
        a_pontine = self.f_pou(cpu4.dot(self.w_cpu42pou))
        cpu1a = (.5 * cpu4.dot(self.w_cpu42cpu1a) +
                 .5 * a_pontine.dot(self.w_pou2cpu1a) +
                 tb1.dot(self.w_tb12cpu1a))
        cpu1b = (.5 * cpu4.dot(self.w_cpu42cpu1b) +
                 .5 * a_pontine.dot(self.w_pou2cpu1b) +
                 tb1.dot(self.w_tb12cpu1b))

        return np.hstack([cpu1b[..., 1:], cpu1a, cpu1b[..., :1]])

    @property
    def w_fbn2pou(self):
        return self._w_cpu42pou

    @w_fbn2pou.setter
    def w_fbn2pou(self, v):
        self._w_cpu42pou[:] = v

    @property
    def w_cpu42pou(self):
        return self._w_cpu42pou

    @w_cpu42pou.setter
    def w_cpu42pou(self, v):
        self._w_cpu42pou[:] = v

    @property
    def w_pou2cpu1a(self):
        """
        The Pontine to CPU1a synaptic weights.
        """
        return self._w_pou2cpu1a

    @w_pou2cpu1a.setter
    def w_pou2cpu1a(self, v):
        self._w_pou2cpu1a[:] = v[:]

    @property
    def w_pou2cpu1b(self):
        """
        The Pontine to CPU1b synaptic weights.
        """
        return self._w_pou2cpu1b

    @w_pou2cpu1b.setter
    def w_pou2cpu1b(self, v):
        self._w_pou2cpu1b[:] = v[:]

