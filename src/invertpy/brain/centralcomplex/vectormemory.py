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

from .centralcomplex import CentralComplexBase
from ._helpers import tn_axes

import numpy as np
import os

# get path of the script
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

N_COLUMNS = 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class VectorMemoryCX(CentralComplexBase):

    def __init__(self, nb_tb1=8, nb_tn1=2, nb_tn2=2, nb_cl1=16, nb_tl2=16, nb_cpu4=16, nb_cpu1a=14, nb_cpu1b=2,
                 nb_rings=4, nb_mbon=2, tn_prefs=np.pi / 4, gain=0.05, pontin=False, *args, **kwargs):
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

        kwargs.setdefault('nb_input', nb_tb1 + nb_tn1 + nb_tn2)
        kwargs.setdefault('nb_compass', nb_tb1)
        kwargs.setdefault('nb_memory', nb_cpu4)
        kwargs.setdefault('nb_steering', nb_cpu1a + nb_cpu1b)
        kwargs.setdefault('learning_rule', None)
        super().__init__(*args, **kwargs)

        # set-up the learning speed
        if pontin:
            gain *= 5e-03
        self._gain = gain

        # set-up parameters
        self.tn_prefs = tn_prefs
        self.smoothed_flow = 0.
        self.pontin = pontin

        self._nb_tl2 = nb_tl2
        self._nb_cl1 = nb_cl1
        self._nb_tn1 = nb_tn1
        self._nb_tn2 = nb_tn2
        self._nb_cpu1a = nb_cpu1a
        self._nb_cpu1b = nb_cpu1b
        self._nb_rings = nb_rings
        self._nb_mbon = nb_mbon

        # initialise the responses of the neurons
        self._tl2 = np.zeros(self.nb_tl2)
        self._cl1 = np.zeros(self.nb_cl1)
        self._tn1 = np.zeros(self.nb_tn1)
        self._tn2 = np.zeros(self.nb_tn2)
        self._vec = np.zeros(self.nb_rings)
        self._mbon = np.zeros(self.nb_mbon)
        self._vec_t = np.zeros_like(self._vec)

        # Weight matrices based on anatomy (These are not changeable!)
        self._w_tl22cl1 = uniform_synapses(self.nb_tl2, self.nb_cl1, fill_value=0, dtype=self.dtype)
        self._w_cl12tb1 = uniform_synapses(self.nb_cl1, self.nb_tb1, fill_value=0, dtype=self.dtype)
        self._w_tn22cpu4 = uniform_synapses(self.nb_tn2, self.nb_cpu4, fill_value=0, dtype=self.dtype)

        self._w_pontin2cpu1a = uniform_synapses(self.nb_cpu1, self.nb_cpu1a, fill_value=0, dtype=self.dtype)
        self._w_pontin2cpu1b = uniform_synapses(self.nb_cpu1, self.nb_cpu1b, fill_value=0, dtype=self.dtype)
        self._w_cpu42pontin = uniform_synapses(self.nb_cpu4, self.nb_cpu4, fill_value=0, dtype=self.dtype)

        # cpu4 memory
        self._w_mbon2vec = uniform_synapses(self.nb_mbon, self.nb_rings, fill_value=0, dtype=self.dtype)
        self._w_vec2cpu4 = uniform_synapses(self.nb_rings, self.nb_cpu4, fill_value=.5, dtype=self.dtype)
        self._w_ring2cpu4 = uniform_synapses(self.nb_cpu4 // 2, self.nb_cpu4, fill_value=0, dtype=self.dtype)

        # The cell properties (for sigmoid function)
        self._tl2_slope = 6.8
        self._cl1_slope = 3.0
        self._tb1_slope = 5.0
        self._cpu4_slope = 5.0
        self._cpu1_slope = 5.0  # 7.5
        self._motor_slope = 1.0
        self._pontin_slope = 5.0

        self._b_tl2 = 3.0
        self._b_cl1 = -0.5
        self._b_pontin = 2.5

        self.params.extend([
            self._w_tl22cl1,
            self._w_cl12tb1,
            self._w_cpu42pontin,
            self._w_pontin2cpu1a,
            self._w_pontin2cpu1b,
            self._b_tl2,
            self._b_cl1,
            self._b_pontin
        ])

        self.tl2_prefs = np.tile(np.linspace(0, 2 * np.pi, self.nb_tb1, endpoint=False), 2)

        self.f_vec = lambda v: hardmax(v, noise=self._noise, rng=self.rng)
        self.f_tl2 = lambda v: sigmoid(v * self._tl2_slope - self.b_tl2, noise=self._noise, rng=self.rng)
        self.f_cl1 = lambda v: sigmoid(v * self._cl1_slope - self.b_cl1, noise=self._noise, rng=self.rng)
        self.f_tb1 = lambda v: sigmoid(v * self._tb1_slope - self.b_tb1, noise=self._noise, rng=self.rng)
        self.f_cpu4 = lambda v: sigmoid(v * self._cpu4_slope - self.b_cpu4, noise=self._noise, rng=self.rng)
        self.f_pontin = lambda v: sigmoid(v * self._pontin_slope - self.b_pontin, noise=self._noise, rng=self.rng)
        self.f_cpu1 = lambda v: sigmoid(v * self._cpu1_slope - self.b_cpu1, noise=self._noise, rng=self.rng)

        self._multi = False

        self.reset()

    def reset(self):
        # Weight matrices based on anatomy (These are not changeable!)
        self.w_tl22cl1 = diagonal_synapses(self.nb_tl2, self.nb_cl1, fill_value=-1, dtype=self.dtype)
        self.w_cl12tb1 = diagonal_synapses(self.nb_cl1, self.nb_tb1, fill_value=1, tile=True, dtype=self.dtype)
        self.w_tb12tb1 = sinusoidal_synapses(self.nb_tb1, self.nb_tb1, fill_value=-1, dtype=self.dtype)
        self.w_tb12cpu4 = diagonal_synapses(self.nb_tb1, self.nb_cpu4, fill_value=-1, tile=True, dtype=self.dtype)
        self.w_tn22cpu4 = chessboard_synapses(self.nb_tn2, self.nb_cpu4, nb_rows=2, nb_cols=2, fill_value=1,
                                              dtype=self.dtype)
        w_tb12cpu1 = diagonal_synapses(self.nb_tb1, self.nb_cpu1, fill_value=-1, tile=True)
        self.w_tb12cpu1a = w_tb12cpu1[:, 1:-1]
        self.w_tb12cpu1b = np.hstack([w_tb12cpu1[:, -1:], w_tb12cpu1[:, :1]])

        w_cpu42cpu1 = opposing_synapses(self.nb_cpu4, self.nb_cpu1, fill_value=1, dtype=self.dtype)
        self.w_cpu42cpu1a = np.hstack([w_cpu42cpu1[:, :self.nb_cpu1a//2], w_cpu42cpu1[:, -self.nb_cpu1a//2:]])
        self.w_cpu42cpu1b = w_cpu42cpu1[:, [-self.nb_cpu1a//2-1, self.nb_cpu1a//2]]

        self.w_cpu1a2motor = chessboard_synapses(self.nb_cpu1a, 2, nb_rows=2, nb_cols=2, fill_value=1, dtype=self.dtype)
        self.w_cpu1b2motor = 1 - chessboard_synapses(self.nb_cpu1b, 2, nb_rows=2, nb_cols=2, fill_value=1,
                                                     dtype=self.dtype)

        w_pontin2cpu1 = pattern_synapses(pattern=diagonal_synapses(4, 4, 1, dtype=self.dtype)[:, ::-1],
                                         patch=diagonal_synapses(self.nb_cpu1//4, self.nb_cpu1//4, 1, dtype=self.dtype),
                                         dtype=self.dtype)

        self.w_pontin2cpu1a = np.hstack([w_pontin2cpu1[:, :self.nb_cpu1a//2], w_pontin2cpu1[:, -self.nb_cpu1a//2:]])
        self.w_pontin2cpu1b = w_pontin2cpu1[:, [-self.nb_cpu1a // 2 - 1, self.nb_cpu1a // 2]]

        self.w_cpu42pontin = diagonal_synapses(self.nb_cpu4, self.nb_cpu4, fill_value=1, dtype=self.dtype)

        self.w_mbon2vec = np.zeros_like(self._w_mbon2vec)
        self.w_mbon2vec[0, 0] = 1
        self.w_mbon2vec[1:, 1:] = diagonal_synapses(self._nb_mbon - 1, self.nb_rings - 1, fill_value=1.,
                                                    tile=True, dtype=self.dtype)
        self.w_vec2cpu4 = uniform_synapses(self.nb_rings, self.nb_cpu4, fill_value=.5, dtype=self.dtype)
        self.w_ring2cpu4 = pattern_synapses(
            pattern=diagonal_synapses(self.nb_cpu4 // 2, self.nb_cpu4 // 2, fill_value=1, dtype=self.dtype),
            patch=np.array([1, 1], dtype=self.dtype), dtype=self.dtype)

        self.r_tl2 = np.zeros(self.nb_tl2)
        self.r_cl1 = np.zeros(self.nb_cl1)
        self.r_tb1 = np.zeros(self.nb_tb1)
        self.r_tn1 = np.zeros(self.nb_tn1)
        self.r_tn2 = np.zeros(self.nb_tn2)
        self.r_cpu4 = np.zeros(self.nb_cpu4)  # cpu4 output
        self.r_cpu1 = np.zeros(self.nb_cpu1)

        self._vec_t = np.zeros_like(self._vec)
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

        if mbon is None:  # default is homing PI
            mbon = np.zeros_like(self._mbon)
            mbon[0] = 1.

        # select vector based on motivation
        vec_mot = np.dot(mbon, self.w_mbon2vec)
        # select most recent vector
        vec_tim = np.exp(-self._vec_t)
        # get the closest memory
        vec_dis = 1 - self.get_vectors_distance()
        # weight more the closest vector
        a_vec = self.f_vec(vec_mot * vec_tim * vec_dis)
        # print(f"vec: {a_vec}, mot: {vec_mot}, time: {vec_tim}, dist: {vec_dis}, all: {vec_mot * vec_tim * vec_dis}")

        if isinstance(phi, np.ndarray) and phi.size == 8:
            if tl2 is None:
                tl2 = np.tile(phi, 2)
            if cl1 is None:
                cl1 = np.tile(phi, 2)
            self._tl2 = a_tl2 = self.f_tl2(tl2[::-1])
            self._cl1 = a_cl1 = self.f_cl1(cl1[::-1])
            self._com = a_tb1 = self.f_com(5. * phi[::-1])
        else:
            self._tl2 = a_tl2 = self.f_tl2(self.phi2tl2(phi))
            self._cl1 = a_cl1 = self.f_cl1(np.dot(a_tl2, self.w_tl22cl1))
            if self._com is None:
                self._com = a_tb1 = self.f_com(a_cl1)
            else:
                p = .667  # proportion of input from CL1 to TB1
                self._com = a_tb1 = self.f_com(p * np.dot(a_cl1, self.w_cl12tb1) + (1 - p) * np.dot(self._com, self.w_tb12tb1))
        self._tn1 = a_tn1 = self.flow2tn1(flow)
        self._tn2 = a_tn2 = self.flow2tn2(flow)

        if self.pontin:
            mem = .5 * self._gain * (np.clip(a_tn2 @ self.w_tn22cpu4 - a_tb1 @ self.w_tb12cpu4, 0, 1) - .25)
        else:
            # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
            # update = np.clip((.5 - tn1).dot(self.w_tn2cpu4), 0., 1.)  # normal
            mem_tn1 = np.dot(.5 - a_tn1, self.w_tn22cpu4)  # holonomic

            mem_tb1 = self._gain * np.dot(a_tb1 - 1., self.w_tb12cpu4)
            # update *= self.gain * (1. - tb1).dot(self.w_tb12cpu4)

            # Both CPU4 waves must have same average
            # If we don't normalise get drift and weird steering
            mem_tn2 = self._gain * .25 * a_tn2.dot(self.w_tn22cpu4)

            mem = mem_tn1 * mem_tb1 - mem_tn2

        if self.update:  # update the correct PI

            # # flush the vector memory to the last vector
            # if self._mot.argmax() != motivation.argmax():
            #     print("FLUSH!")
            #     self.w_vec2cpu4[1:] += np.outer(self._vec[1:], self.w_vec2cpu4[0])
            #
            # # always add the step to the home vector
            # self.w_vec2cpu4[0] += mem

            # add the vector to all the available vectors
            self.update_memory(mem)

            # reset the memory when the motivation changes
            if np.argmax(self._vec) != a_vec.argmax():
                self.reset_current_memory()
                self._vec_t[1:][np.argmax(self._vec[1:])] += 1

        # load the current memory
        mem_cpu4 = self.load_memory(a_vec)

        if visual_rings is None:
            a_ring = np.zeros_like(self._mem)
        else:
            # integrate the visual rings gated by the active motivation
            a_ring = a_vec * np.dot(visual_rings, self.w_ring2cpu4)

        # integrate the visual cues and PI by just adding them
        self._mem = a_cpu4 = self.f_mem(mem_cpu4 + a_ring)

        if self.pontin:
            a_pontin = self.f_pontin(a_cpu4 @ self.w_cpu42pontin)
            cpu1a = .5 * a_cpu4 @ self.w_cpu42cpu1a - .5 * a_pontin @ self.w_pontin2cpu1a - a_tb1 @ self.w_tb12cpu1a
            cpu1b = .5 * a_cpu4 @ self.w_cpu42cpu1b - .5 * a_pontin @ self.w_pontin2cpu1b - a_tb1 @ self.w_tb12cpu1b
        else:
            cpu1a = (a_cpu4 @ self.w_cpu42cpu1a) * ((a_tb1 - 1.) @ self.w_tb12cpu1a)
            cpu1b = (a_cpu4 @ self.w_cpu42cpu1b) * ((a_tb1 - 1.) @ self.w_tb12cpu1b)

        self._ste = a_cpu1 = self.f_cpu1(np.hstack([cpu1b[-1], cpu1a, cpu1b[0]]))
        self._mbon = mbon
        self._vec = a_vec

        return a_cpu1

    def load_memory(self, vec):
        if self._multi:
            return np.dot(vec, self.w_vec2cpu4)
        else:
            return self.w_vec2cpu4[0] + np.dot(vec[1:], self.w_vec2cpu4[1:] - .5)

    def update_memory(self, mem):
        if self._multi:  # use multiple integrators
            self.w_vec2cpu4 += mem
        else:  # use single integrator
            self.w_vec2cpu4[0] += mem

    def reset_current_memory(self):
        max_v = np.argmax(self._vec)
        if self._multi:  # use multiple integrators
            print("RESET!")
            self.w_vec2cpu4[max_v] = 0.5
        elif max_v != 0:  # use single integrator
            print("STORE VECTOR!")
            self.w_vec2cpu4[max_v] = 1 - self.w_vec2cpu4[0]

    def get_vectors_distance(self):
        bias = np.full_like(self.r_cpu4, fill_value=1 / self.nb_cpu4)
        if self._multi:  # use multiple integrators
            return np.dot(bias, np.maximum(self.w_vec2cpu4.T, 0.5))
        else:  # use single integrator
            w = self.w_vec2cpu4.copy()
            w[1:] += self.w_vec2cpu4[0] - 0.5
            # dist = np.dot(bias, np.maximum(self.w_vec2cpu4.T, 0))
            # dist[0] = np.maximum(dist[0], 0.5)
            # dist[1:] += 0.5
            return np.dot(bias, np.maximum(w.T, 0.5))

    def __repr__(self):
        return f"VectorMemoryCX(TB1={self.nb_tb1:d}, TN1={self.nb_tn1:d}, TN2={self.nb_tn2:d}," \
               f" CL1={self.nb_cl1:d}, TL2={self.nb_tl2}, CPU4={self.nb_cpu4:d}," \
               f" MBON={self.nb_mbon:d}, vectors={self.nb_rings:d}, CPU1={self.nb_cpu1:d})"

    def get_flow(self, heading, velocity, filter_steps=0):
        """
        Calculate optic flow depending on preference angles. [L, R]

        Parameters
        ----------
        heading: float
            the heading direction in radians.
        velocity: np.ndarray
            the 2D linear velocity.
        filter_steps: int, optional
            the number of steps as a smoothing parameter for the filter

        Returns
        -------
        np.ndarray[float]
            the estimated optic flow from both eyes [L, R]
        """
        A = tn_axes(heading, self.tn_prefs)
        flow = velocity.dot(A)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow

    def phi2tl2(self, phi):
        """
        Transform the heading direction to the TL2 responses.

        Parameters
        ----------
        phi: float
            the feading direction in radiance.

        Returns
        -------
        np.ndarray[float]
            the TL2 responses based on their preference angle
        """
        return np.cos(phi - self.tl2_prefs)

    def flow2tn1(self, flow):
        """
        Linearly inverse sensitive to forwards and backwards motion.

        Parameters
        ----------
        flow: np.ndarray
            the [L, R] optic flow

        Returns
        -------
        np.ndarray[float]
            the responses of the TN1 neurons

        """
        return np.clip((1. - flow) / 2. + self.rng.normal(scale=self._noise, size=flow.shape), 0, 1)

    def flow2tn2(self, flow):
        """
        Linearly sensitive to forwards motion only.

        Parameters
        ----------
        flow: np.ndarray
            the [L, R] optic flow

        Returns
        -------
        np.ndarray[float]
            the responses of the TN2 neurons
        """
        return np.clip(flow, 0, 1)

    @property
    def w_tl22cl1(self):
        """
        The TL2 to CL1 synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_tl22cl1

    @w_tl22cl1.setter
    def w_tl22cl1(self, v):
        self._w_tl22cl1[:] = v[:]

    @property
    def w_cl12tb1(self):
        """
        The CL1 to TB1 synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_cl12tb1

    @w_cl12tb1.setter
    def w_cl12tb1(self, v):
        self._w_cl12tb1[:] = v[:]

    @property
    def w_tb12tb1(self):
        """
        The TB1 to TB1 synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_c2c

    @w_tb12tb1.setter
    def w_tb12tb1(self, v):
        self._w_c2c[:] = v[:]

    @property
    def w_tb12cpu4(self):
        """
        The TB1 to CPU4 synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_c2m

    @w_tb12cpu4.setter
    def w_tb12cpu4(self, v):
        self._w_c2m[:] = v[:]

    @property
    def w_tn22cpu4(self):
        """
        The TN2 to CPU4 synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_tn22cpu4

    @w_tn22cpu4.setter
    def w_tn22cpu4(self, v):
        self._w_tn22cpu4[:] = v[:]

    @property
    def w_tb12cpu1a(self):
        """
        The TB1 to CPU1a synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_c2s[:, 1:-1]

    @w_tb12cpu1a.setter
    def w_tb12cpu1a(self, v):
        self._w_c2s[:, 1:-1] = v[:]

    @property
    def w_tb12cpu1b(self):
        """
        The TB1 to CPU1b synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        np.hstack([self._w_c2s[:, -1:], self._w_c2s[:, :1]])
        return np.hstack([self._w_c2s[:, -1:], self._w_c2s[:, :1]])

    @w_tb12cpu1b.setter
    def w_tb12cpu1b(self, v):
        self._w_c2s[:, -1:] = v[:, :1]
        self._w_c2s[:, :1] = v[:, -1:]

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
    def w_cpu42cpu1a(self):
        """
        The CPU4 to CPU1a synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return np.hstack([self.w_m2s[:, :self.nb_cpu1a//2], self.w_m2s[:, -self.nb_cpu1a//2:]])

    @w_cpu42cpu1a.setter
    def w_cpu42cpu1a(self, v):
        self.w_m2s[:, :self.nb_cpu1a // 2] = v[:, :self.nb_cpu1a // 2]
        self.w_m2s[:, -self.nb_cpu1a // 2:] = v[:, -self.nb_cpu1a // 2:]

    @property
    def w_cpu42cpu1b(self):
        """
        The CPU4 to CPU1b synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self.w_m2s[:, [-self.nb_cpu1a//2-1, self.nb_cpu1a//2]]

    @w_cpu42cpu1b.setter
    def w_cpu42cpu1b(self, v):
        self.w_m2s[:, [-self.nb_cpu1a//2-1, self.nb_cpu1a//2]] = v[:]

    @property
    def w_cpu1a2motor(self):
        """
        Matrix transforming the CPU1a responses to their contribution to the motor commands.

        Returns
        -------
        np.ndarray[float]
        """
        return np.vstack([self.w_s2o[:self.nb_cpu1a//2], self.w_s2o[-self.nb_cpu1a//2:]])

    @w_cpu1a2motor.setter
    def w_cpu1a2motor(self, v):
        self.w_s2o[:self.nb_cpu1a // 2, :] = v[:self.nb_cpu1a // 2, :]
        self.w_s2o[-self.nb_cpu1a // 2:, :] = v[-self.nb_cpu1a // 2:, :]

    @property
    def w_cpu1b2motor(self):
        """
        Matrix transforming the CPU1b responses to their contribution to the motor commands.

        Returns
        -------
        np.ndarray[float]
        """
        return self.w_s2o[[-self.nb_cpu1a//2-1, self.nb_cpu1a//2]]

    @w_cpu1b2motor.setter
    def w_cpu1b2motor(self, v):
        self.w_s2o[[-self.nb_cpu1a//2-1, self.nb_cpu1a//2], :] = v[:]

    @property
    def w_pontin2cpu1a(self):
        """
        The pontin to CPU1a synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_pontin2cpu1a

    @w_pontin2cpu1a.setter
    def w_pontin2cpu1a(self, v):
        self._w_pontin2cpu1a[:] = v[:]

    @property
    def w_pontin2cpu1b(self):
        """
        The pontin to CPU1b synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_pontin2cpu1b

    @w_pontin2cpu1b.setter
    def w_pontin2cpu1b(self, v):
        self._w_pontin2cpu1b[:] = v[:]

    @property
    def w_cpu42pontin(self):
        """
        The CPU4 to pontin synaptic weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_cpu42pontin

    @w_cpu42pontin.setter
    def w_cpu42pontin(self, v):
        self._w_cpu42pontin[:] = v[:]

    @property
    def b_tl2(self):
        """
        The TL2 rest response rate (bias).

        Returns
        -------
        float
        """
        return self._b_tl2

    @property
    def b_cl1(self):
        """
        The CL1 rest response rate (bias).

        Returns
        -------
        float
        """
        return self._b_cl1

    @property
    def b_tb1(self):
        """
        The TB1 rest response rate (bias).

        Returns
        -------
        float
        """
        return self._b_com

    @property
    def b_cpu4(self):
        """
        The CPU4 rest response rate (bias).

        Returns
        -------
        float
        """
        return self._b_mem

    @property
    def b_cpu1(self):
        """
        The CPU1 rest response rate (bias).

        Returns
        -------
        float
        """
        return self._b_ste

    @property
    def b_motor(self):
        """
        The motor bias.

        Returns
        -------
        float
        """
        return self._b_out

    @property
    def b_pontin(self):
        """
        The pontin rest response rate (bias).

        Returns
        -------
        float
        """
        return self._b_pontin

    @property
    def r_tb1(self):
        """
        The TB1 response rate.

        Returns
        -------
        np.ndarray[float]
        """
        return self._com

    @r_tb1.setter
    def r_tb1(self, v):
        self._com[:] = v[:]

    @property
    def nb_tb1(self):
        """
        The number TB1 neurons.

        Returns
        -------
        int
        """
        return self._nb_compass

    @property
    def r_tl2(self):
        """
        The TL2 response rate.

        Returns
        -------
        np.ndarray[float]
        """
        return self._tl2

    @r_tl2.setter
    def r_tl2(self, v):
        self._tl2[:] = v[:]

    @property
    def nb_tl2(self):
        """
        The number TL2 neurons.

        Returns
        -------
        int
        """
        return self._nb_tl2

    @property
    def r_cl1(self):
        """
        The CL1 response rate.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cl1

    @r_cl1.setter
    def r_cl1(self, v):
        self._cl1[:] = v[:]

    @property
    def nb_cl1(self):
        """
        The number CL1 neurons.

        Returns
        -------
        int
        """
        return self._nb_cl1

    @property
    def r_tn1(self):
        """
        The TN1 response rate.

        Returns
        -------
        np.ndarray[float]
        """
        return self._tn1

    @r_tn1.setter
    def r_tn1(self, v):
        self._tn1[:] = v[:]

    @property
    def nb_tn1(self):
        """
        The number TN1 neurons.

        Returns
        -------
        int
        """
        return self._nb_tn1

    @property
    def r_tn2(self):
        """
        The TN2 response rate.

        Returns
        -------
        np.ndarray[float]
        """
        return self._tn2

    @r_tn2.setter
    def r_tn2(self, v):
        self._tn2[:] = v[:]

    @property
    def nb_tn2(self):
        """
        The number TN2 neurons.

        Returns
        -------
        int
        """
        return self._nb_tn2

    @property
    def r_cpu4(self):
        """
        The CPU4 response rate.

        Returns
        -------
        np.ndarray[float]
        """
        return self._mem

    @r_cpu4.setter
    def r_cpu4(self, v):
        self._mem[:] = v[:]

    @property
    def cpu4_mem(self):
        """
        The CPU4 memory.

        Returns
        -------
        np.ndarray[float]
        """
        if self._multi:
            return np.dot(self._vec, self.w_vec2cpu4)
        else:
            return self.w_vec2cpu4[0] + np.dot(self._vec[1:], self.w_vec2cpu4[1:] - .5)

    @property
    def nb_cpu4(self):
        """
        The number CPU4 neurons.

        Returns
        -------
        int
        """
        return self._nb_memory

    @property
    def nb_cpu1a(self):
        """
        The number CPU1a neurons.

        Returns
        -------
        int
        """
        return self._nb_cpu1a

    @property
    def nb_cpu1b(self):
        """
        The number CPU1b neurons.

        Returns
        -------
        int
        """
        return self._nb_cpu1b

    @property
    def r_cpu1(self):
        """
        The CPU1 response rate.

        Returns
        -------
        np.ndarray[float]
        """
        return self._ste

    @r_cpu1.setter
    def r_cpu1(self, v):
        self._ste[:] = v[:]

    @property
    def nb_cpu1(self):
        """
        The number CPU1 neurons.

        Returns
        -------
        int
        """
        return self._nb_steering

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
