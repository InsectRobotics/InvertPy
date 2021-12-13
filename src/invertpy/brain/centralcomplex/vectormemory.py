"""
The Central Complex (CX) model of the bee brain as introduced by _[1].

References:
    .. [1] Stone, T. et al. An Anatomically Constrained Model for Path Integration in the Bee Brain.
       Curr Biol 27, 3069-3085.e11 (2017).
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

from invertpy.brain.synapses import *
from invertpy.brain.activation import sigmoid, hardmax

from .stone import StoneCX
from ._helpers import tn_axes

import numpy as np
import os

# get path of the script
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

N_COLUMNS = 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class VectorMemoryCX(StoneCX):

    def __init__(self, nb_rings=4, nb_mbon=2, *args, **kwargs):
        """
        The Central Complex model of [1]_ as a component of the locust brain.

        Parameters
        ----------
        nb_tb1: int, optional
            the number of TB1 neurons. Default is 8
        nb_tn1: int, optional
            the number of TN1 neurons. Default is 2
        nb_tn2: int, optional
            the number of TN2 neurons. Default is 2
        nb_cl1: int, optional
            the number of CL1 neurons. Default is 16
        nb_tl2: int, optional
            the number of TL2 neurons. Default is 16
        nb_cpu4: int, optional
            the number of CPU4 neurons. Default is 16
        nb_cpu1a: int, optional
            the number of CPU1a neurons. Default is 14
        nb_cpu1b: int, optional
            the number of CPU1b neurons. Default is 2
        nb_rings: int, optional
            the maximum number of PI vectors to store. Default is 4
        nb_mbon: int, optional
            the number of motivations for which vector to use. Default is 2 (homing and foraging)
        tn_prefs: float, optional
            the angular offset of preference of the TN neurons from the front direction. Default is pi/4
        gain: float, optional
            the gain if used as charging speed for the memory. Default is 0.05
        pontin: bool, optional
            whether to include a pontin neuron in the circuit or not. Default is False.

        Notes
        -----
        .. [1] Stone, T. et al. An Anatomically Constrained Model for Path Integration in the Bee Brain.
           Curr Biol 27, 3069-3085.e11 (2017).
        """

        kwargs.setdefault('gain', 0.05)
        kwargs.setdefault('nb_cpu4', 16)
        kwargs.setdefault('noise', 0.)
        kwargs.setdefault('rng', RNG)
        kwargs.setdefault('dtype', np.float32)

        self._nb_rings = nb_rings
        self._nb_mbon = nb_mbon
        nb_cpu4 = kwargs['nb_cpu4']
        noise = kwargs['noise']
        rng = kwargs['rng']
        dtype = kwargs['dtype']

        # initialise the responses of the neurons
        self._vec = np.zeros(nb_rings)
        self._mbon = np.zeros(nb_mbon)
        self._vec_t = np.zeros_like(self._vec)

        # Weight matrices based on anatomy (These are not changeable!)

        # cpu4 memory
        self._w_mbon2vec = uniform_synapses(nb_mbon, nb_rings, fill_value=0, dtype=dtype)
        self._w_vec2cpu4 = uniform_synapses(nb_rings, nb_cpu4, fill_value=.5, dtype=dtype)
        self._w_ring2cpu4 = uniform_synapses(nb_cpu4 // 2, nb_cpu4, fill_value=0, dtype=dtype)

        self.f_vec = lambda v: hardmax(v, noise=noise + 0.01, rng=rng)
        self._multi = False
        self._c_rings = None  # the working rings
        self._c_vec = None  # the working vector
        self._v_change = False  # show if the vector is different from before

        super().__init__(*args, **kwargs)

        self.params.extend([
            self._w_mbon2vec,
            self._w_vec2cpu4,
            self._w_ring2cpu4
        ])

    def reset(self):
        super().reset()

        self.w_mbon2vec = np.zeros_like(self._w_mbon2vec)
        self.w_mbon2vec[0, 0] = 1
        self.w_mbon2vec[1:, 1:] = diagonal_synapses(self._nb_mbon - 1, self.nb_rings - 1, fill_value=1.,
                                                    tile=True, dtype=self.dtype)
        self.w_vec2cpu4 = uniform_synapses(self.nb_rings, self.nb_cpu4, fill_value=.5, dtype=self.dtype)
        self.w_ring2cpu4 = pattern_synapses(
            pattern=diagonal_synapses(self.nb_cpu4 // 2, self.nb_cpu4 // 2, fill_value=1, dtype=self.dtype),
            patch=np.array([1, 1], dtype=self.dtype), dtype=self.dtype)

        self._vec_t = np.zeros_like(self._vec)
        self._c_rings = None
        self._c_vec = None
        self._v_change = False

        self.update = True

    def _fprop(self, phi, flow, tl2=None, cl1=None, mbon=None, visual_rings=None):
        """
        Parameters
        ----------
        phi : float, np.ndarray[float]
            the global heading direction either as a ring or as a number (in rads)
        flow : np.ndarray[float]
            the [L, R] optic flow
        tl2 : np.ndarray[float]
            the TL2 responses
        cl1 : np.ndarray[float]
            the CL1 responses
        mbon: np.ndarray[float]
            the MBON activity comes from the mushroom body
        visual_rings: np.ndarray[float]
            rings based on teh visual cues

        Returns
        -------
        np.ndarray[float]
            the CPU1 responses that are used for steering
        """

        if mbon is None or np.all(np.isclose(mbon, 0)):  # default is homing PI
            mbon = np.zeros_like(self._mbon)
            mbon[0] = 1.

        # select vector based on motivation
        i_vec = np.argmax(mbon)
        vec_mot = self.w_mbon2vec[i_vec]
        # select most recent vector
        # vec_tim = np.exp(-self._vec_t)
        vec_tim = 1.
        # get the closest memory
        # vec_dis = self.get_vectors_distance()
        vec_dis = 1.
        # weight more the closest vector
        self._c_vec = a_vec = self.f_vec(vec_mot * vec_tim * vec_dis)
        # print(f"vec: {a_vec}, mot: {vec_mot}, time: {vec_tim}, dist: {vec_dis}, all: {vec_mot * vec_tim * vec_dis}")
        self._c_rings = visual_rings
        print(f"MBON: {np.array2string(mbon, precision=2)}, "
              f"VEC: {np.array2string(vec_mot, precision=1)}, "
              f"A_VEC: {np.array2string(a_vec, precision=1)}, "
              f"MEM2VEC: {np.absolute(self.mem2vector())}")

        a_cpu1 = super()._fprop(phi, flow, tl2=tl2, cl1=cl1)

        self._mbon = mbon
        self._vec = a_vec

        return a_cpu1

    def load_memory(self):
        cpu4_mem = super().load_memory()
        vec_mem = 0.
        visual = 0.
        if self._c_vec is not None:
            vec_mem = np.dot(self._c_vec, self.w_vec2cpu4 - 0.5)

            if self._c_rings is not None:
                # integrate the visual rings gated by the active vector
                visual = self._c_vec * np.dot(self._c_rings, self.w_ring2cpu4)

        return cpu4_mem + vec_mem + visual

    def update_memory(self, mem):
        super().update_memory(mem)

        if self._c_vec is not None:
            # check if the vector has changed
            self._v_change = np.argmax(self._vec) != np.argmax(self._c_vec)

            # reset the memory when the motivation changes
            if self._v_change:
                self.reset_current_memory()
                self._vec_t[np.argmax(self._vec)] += 1

    def reset_current_memory(self):
        max_v = np.argmax(self._vec)
        if self._multi:  # use multiple integrators
            print("RESET!")
            self.w_vec2cpu4[max_v] = 0.5
        elif max_v > 0:  # use single integrator
            print(f"STORE VECTOR AT VEC_{max_v+1}!")
            self.w_vec2cpu4[max_v] = 1. - super().load_memory()
            if max_v == 0:
                super().update_memory(np.full_like(super().load_memory(), 0.5))

    def get_vectors_distance(self):
        if self._multi:  # use multiple integrators
            return np.absolute(self.mem2vector())
        else:  # use single integrator
            w = self.w_vec2cpu4.copy() + self.w_vec2cpu4[0] - 0.5
            return np.absolute(mem2vector(w, self._gain))

    def mem2vector(self):
        """
        Transforms the internal vector memories to actual vectors in the Cartesian coordinate system.

        Returns
        -------
        np.ndarray[complex]
        """
        return mem2vector(self.w_vec2cpu4, self._gain)

    def __repr__(self):
        return f"VectorMemoryCX(TB1={self.nb_tb1:d}, TN1={self.nb_tn1:d}, TN2={self.nb_tn2:d}," \
               f" CL1={self.nb_cl1:d}, TL2={self.nb_tl2}, CPU4={self.nb_cpu4:d}," \
               f" MBON={self.nb_mbon:d}, vectors={self.nb_rings:d}, CPU1={self.nb_cpu1:d})"

    @property
    def v_change(self):
        return self._v_change

    @property
    def w_mbon2vec(self):
        """
        The motivation to vector selection synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_mbon2vec

    @w_mbon2vec.setter
    def w_mbon2vec(self, v):
        self._w_mbon2vec[:] = v[:]

    @property
    def w_vec2cpu4(self):
        """
        The vectors to CPU4 synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_vec2cpu4

    @w_vec2cpu4.setter
    def w_vec2cpu4(self, v):
        self._w_vec2cpu4[:] = v[:]

    @property
    def w_ring2cpu4(self):
        """
        The visual familiarity ring to CPU4 synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_ring2cpu4

    @w_ring2cpu4.setter
    def w_ring2cpu4(self, v):
        self._w_ring2cpu4[:] = v[:]

    @property
    def nb_rings(self):
        """
        The number of visual familiarity rings.

        Returns
        -------
        int
        """
        return self._nb_rings

    @property
    def nb_mbon(self):
        """
        The number of motivations supported.

        Returns
        -------
        int
        """
        return self._nb_mbon

    @property
    def r_vec(self):
        return self._vec


def mem2vector(mem, gain):
    """
    Transforms the given vector memories to actual vectors in the Cartesian coordinate system.

    Parameters
    ----------
    mem : np.ndarray[float]
        the different vectors stored in the memory
    gain : float
        the internal gain of the CX

    Returns
    -------
    np.ndarray[complex]
    """
    vec_reshaped = mem.reshape((mem.shape[0], 2, -1))
    vec_shifted = np.array([np.roll(vec_reshaped[:, 0], 1, axis=-1),
                            np.roll(vec_reshaped[:, 1], -1, axis=-1)])
    vec_signal = np.sum(vec_shifted, axis=0)
    vec = []
    for signal in vec_signal:
        fund_freq = np.fft.fft(signal)[1]
        angle = -np.angle(np.conj(fund_freq))
        distance = np.absolute(fund_freq) / gain
        vec.append(distance * np.exp(1j * angle))
    return np.array(vec)
