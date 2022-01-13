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
from invertpy.brain.activation import sigmoid

from .centralcomplex import CentralComplexBase
from ._helpers import tn_axes

import numpy as np
import os

# get path of the script
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

N_COLUMNS = 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class StoneCX(CentralComplexBase):

    def __init__(self, nb_tb1=8, nb_tn1=2, nb_tn2=2, nb_cl1=16, nb_tl2=16, nb_cpu4=16, nb_cpu1a=14, nb_cpu1b=2,
                 tn_prefs=np.pi/4, gain=0.05, holonomic=True, pontine=True, *args, **kwargs):
        """
        The Central Complex model of [1]_ as a component of the locust brain.

        Parameters
        ----------
        nb_tb1: int, optional
            the number of TB1 neurons.
        nb_tn1: int, optional
            the number of TN1 neurons.
        nb_tn2: int, optional
            the number of TN2 neurons.
        nb_cl1: int, optional
            the number of CL1 neurons.
        nb_tl2: int, optional
            the number of TL2 neurons.
        nb_cpu4: int, optional
            the number of CPU4 neurons.
        nb_cpu1a: int, optional
            the number of CPU1a neurons.
        nb_cpu1b: int, optional
            the number of CPU1b neurons.
        tn_prefs: float, optional
            the angular offset of preference of the TN neurons from the front direction.
        gain: float, optional
            the gain if used as charging speed for the memory.
        pontine: bool, optional
            whether to include the Pontine neurons in the circuit or not. Default is True.
        holonomic : bool, optional
            whether to use the holonomic version of the circuit or not. Default is True.

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
        if pontine:
            gain *= 5e-01
        self._gain = gain

        # set-up parameters
        self.tn_prefs = tn_prefs
        self.smoothed_flow = 0.
        self._pontine = pontine
        self._holonomic = holonomic

        self._nb_tl2 = nb_tl2
        self._nb_cl1 = nb_cl1
        self._nb_tn1 = nb_tn1
        self._nb_tn2 = nb_tn2
        self._nb_cpu1a = nb_cpu1a
        self._nb_cpu1b = nb_cpu1b

        # initialise the responses of the neurons
        self._tl2 = np.zeros(self.nb_tl2)
        self._cl1 = np.zeros(self.nb_cl1)
        self._tn1 = np.zeros(self.nb_tn1)
        self._tn2 = np.zeros(self.nb_tn2)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory

        # Weight matrices based on anatomy (These are not changeable!)
        self._w_tl22cl1 = uniform_synapses(self.nb_tl2, self.nb_cl1, fill_value=0, dtype=self.dtype)
        self._w_cl12tb1 = uniform_synapses(self.nb_cl1, self.nb_tb1, fill_value=0, dtype=self.dtype)
        self._w_tn22cpu4 = uniform_synapses(self.nb_tn2, self.nb_cpu4, fill_value=0, dtype=self.dtype)

        self._w_pontine2cpu1a = uniform_synapses(self.nb_cpu1, self.nb_cpu1a, fill_value=0, dtype=self.dtype)
        self._w_pontine2cpu1b = uniform_synapses(self.nb_cpu1, self.nb_cpu1b, fill_value=0, dtype=self.dtype)
        self._w_cpu42pontine = uniform_synapses(self.nb_cpu4, self.nb_cpu4, fill_value=0, dtype=self.dtype)

        # The cell properties (for sigmoid function)
        self._tl2_slope = 6.8
        self._cl1_slope = 3.0
        self._ste_slope = 7.5 if pontine else 5.0
        self._motor_slope = 1.0
        self._pontine_slope = 5.0

        self._b_tl2 = 3.0
        self._b_cl1 = -0.5
        self._b_ste = -1.0 if pontine else 2.5
        self._b_pontine = 2.5

        self.params.extend([
            self._w_tl22cl1,
            self._w_cl12tb1,
            self._w_cpu42pontine,
            self._w_pontine2cpu1a,
            self._w_pontine2cpu1b,
            self._b_tl2,
            self._b_cl1,
            self._b_pontine
        ])

        self.tl2_prefs = np.tile(np.linspace(0, 2 * np.pi, self.nb_tb1, endpoint=False), 2)
        # self.tl2_prefs = np.tile(np.linspace(-np.pi, np.pi, self.nb_tb1, endpoint=False), 2)

        self.f_tl2 = lambda v: sigmoid(v * self._tl2_slope - self.b_tl2, noise=self._noise, rng=self.rng)
        self.f_cl1 = lambda v: sigmoid(v * self._cl1_slope - self.b_cl1, noise=self._noise, rng=self.rng)
        self.f_pontine = lambda v: sigmoid(v * self._pontine_slope - self.b_pontine, noise=self._noise, rng=self.rng)
        self.f_cpu1 = lambda v: sigmoid(v * self._ste_slope - self.b_cpu1, noise=self._noise, rng=self.rng)

        if self.__class__ == StoneCX:
            self.reset()

    def reset(self):
        # Weight matrices based on anatomy (These are not changeable!)
        self.w_tl22cl1 = diagonal_synapses(self.nb_tl2, self.nb_cl1, fill_value=-1, dtype=self.dtype)
        self.w_cl12tb1 = diagonal_synapses(self.nb_cl1, self.nb_tb1, fill_value=1, tile=True, dtype=self.dtype)
        self.w_tb12tb1 = sinusoidal_synapses(self.nb_tb1, self.nb_tb1, fill_value=1, dtype=self.dtype) - 1

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

        w_pontine2cpu1 = pattern_synapses(pattern=diagonal_synapses(4, 4, 1, dtype=self.dtype)[:, ::-1],
                                         patch=diagonal_synapses(self.nb_cpu1//4, self.nb_cpu1//4, 1, dtype=self.dtype),
                                         dtype=self.dtype) * (-1.)

        self.w_pontine2cpu1a = np.hstack([w_pontine2cpu1[:, :self.nb_cpu1a // 2], w_pontine2cpu1[:, -self.nb_cpu1a // 2:]])
        self.w_pontine2cpu1b = w_pontine2cpu1[:, [-self.nb_cpu1a // 2 - 1, self.nb_cpu1a // 2]]

        self.w_cpu42pontine = diagonal_synapses(self.nb_cpu4, self.nb_cpu4, fill_value=1, dtype=self.dtype)

        self.r_tl2 = np.zeros(self.nb_tl2)
        self.r_cl1 = np.zeros(self.nb_cl1)
        self.r_tb1 = np.zeros(self.nb_tb1)
        self.r_tn1 = np.zeros(self.nb_tn1)
        self.r_tn2 = np.zeros(self.nb_tn2)
        self.r_cpu4 = np.zeros(self.nb_cpu4)  # cpu4 output
        self.r_cpu1 = np.zeros(self.nb_cpu1)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory
        #
        # import matplotlib.pyplot as plt
        #
        # plt.figure(1, figsize=(4, 4))
        # plt.imshow(self.w_cl12tb1, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(2, figsize=(4, 4))
        # plt.imshow(self.w_tb12tb1, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(3, figsize=(4, 4))
        # plt.imshow(self.w_tb12cpu1a, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(4, figsize=(4, 4))
        # plt.imshow(self.w_tb12cpu1b, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(5, figsize=(4, 4))
        # plt.imshow(self.w_tb12cpu4, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(6, figsize=(4, 4))
        # plt.imshow(self.w_tn22cpu4, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(7, figsize=(4, 4))
        # plt.imshow(self.w_cpu42cpu1a, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(8, figsize=(4, 4))
        # plt.imshow(self.w_cpu42cpu1b, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(9, figsize=(4, 4))
        # plt.imshow(self.w_cpu1a2motor, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(10, figsize=(4, 4))
        # plt.imshow(self.w_cpu1b2motor, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(11, figsize=(4, 4))
        # plt.imshow(self.w_pontine2cpu1a, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.figure(12, figsize=(4, 4))
        # plt.imshow(self.w_pontine2cpu1b, cmap="coolwarm", vmin=-1, vmax=1)
        #
        # plt.show()

        self.update = True

    def _fprop(self, phi, flow, tl2=None, cl1=None):

        self._tl2, self._cl1, self._com = self.update_compass(phi, tl2=tl2, cl1=cl1)
        self._tn1 = self.flow2tn1(flow)
        self._tn2 = self.flow2tn2(flow)

        self._mem = self.update_memory()

        self._ste = a_cpu1 = self.update_steering()

        return a_cpu1

    def update_compass(self, phi, tl2=None, cl1=None):
        if isinstance(phi, np.ndarray) and phi.size == 8:
            if tl2 is None:
                tl2 = np.tile(phi, 2)
            if cl1 is None:
                cl1 = np.tile(phi, 2)
            a_tl2 = self.f_tl2(tl2[::-1])
            a_cl1 = self.f_cl1(cl1[::-1])
            a_tb1 = self.f_com(5. * phi[::-1])
        else:
            a_tl2 = self.f_tl2(self.phi2tl2(phi))
            a_cl1 = self.f_cl1(a_tl2.dot(self.w_tl22cl1))
            if self._com is None:
                a_tb1 = self.f_com(a_cl1)
            else:
                p = .667  # proportion of input from CL1 to TB1
                a_tb1 = self.f_com(p * a_cl1.dot(self.w_cl12tb1) + (1 - p) * self._com.dot(self.w_tb12tb1))

        return a_tl2, a_cl1, a_tb1

    def update_memory(self, tb1=None, tn1=None, tn2=None):

        if tb1 is None:
            tb1 = self._com
        if tn1 is None:
            tn1 = self._tn1
        if tn2 is None:
            tn2 = self._tn2

        if self._pontine and not self._holonomic:
            mem = self._gain * np.clip(tn2.dot(self.w_tn22cpu4) + tb1.dot(self.w_tb12cpu4), 0, 1)
            cpu4_mem = self.__cpu4 + mem - 0.125 * self._gain
        else:
            # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
            mem_tn1 = (.5 - tn1).dot(self.w_tn22cpu4)
            mem_tb1 = (tb1 - 1).dot(self.w_tb12cpu4)

            # Both CPU4 waves must have same average
            # If we don't normalise get drift and weird steering
            mem_tn2 = 0.25 * tn2.dot(self.w_tn22cpu4)

            mem = mem_tn1 * mem_tb1 - mem_tn2

            cpu4_mem = self.__cpu4 + self._gain * mem
        # this creates a problem with vector memories
        # cpu4_mem = np.clip(cpu4_mem, 0., 1.)

        if self.update:
            self.__cpu4 = cpu4_mem

        a_cpu4 = self.f_mem(cpu4_mem)

        return a_cpu4

    def update_steering(self, cpu4=None, tb1=None):
        if cpu4 is None:
            cpu4 = self._mem
        if tb1 is None:
            tb1 = self._com

        if self.pontine:
            a_pontine = self.f_pontine(cpu4.dot(self.w_cpu42pontine))
            cpu1a = (.5 * cpu4.dot(self.w_cpu42cpu1a) +
                     .5 * a_pontine.dot(self.w_pontine2cpu1a) +
                     tb1.dot(self.w_tb12cpu1a))
            cpu1b = (.5 * cpu4.dot(self.w_cpu42cpu1b) +
                     .5 * a_pontine.dot(self.w_pontine2cpu1b) +
                     tb1.dot(self.w_tb12cpu1b))
        else:
            cpu1a = cpu4.dot(self.w_cpu42cpu1a) + tb1.dot(self.w_tb12cpu1a)
            cpu1b = cpu4.dot(self.w_cpu42cpu1b) + tb1.dot(self.w_tb12cpu1b)

        a_cpu1 = self.f_cpu1(np.hstack([cpu1b[-1], cpu1a, cpu1b[0]]))

        return a_cpu1

    def __repr__(self):
        return f"StoneCX(TB1={self.nb_tb1:d}, TN1={self.nb_tn1:d}, TN2={self.nb_tn2:d}," \
               f"CL1={self.nb_cl1:d}, TL2={self.nb_tl2:d}, CPU4={self.nb_cpu4:d}, CPU1={self.nb_cpu1:d}" \
               f"{', Pontine=True' if self.pontine else ''}{', holonomic=True' if self.holonomic else ''})"

    def reset_integrator(self):
        self.__cpu4[:] = .5

    def get_flow(self, heading, velocity, filter_steps=0):
        """
        Calculate optic flow depending on preference angles. [L, R]

        Parameters
        ----------
        heading: float
            the heading direction in radians.
        velocity: np.ndarray[float]
            the 2D linear velocity.
        filter_steps: int, optional
            the number of steps as a smoothing parameter for the filter

        Returns
        -------
        flow: np.ndarray[float]
            the estimated optic flow from both eyes [L, R]
        """
        A = tn_axes(heading, self.tn_prefs)
        flow = A.T.dot(velocity)

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
        r_tl2: np.ndarray
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
        r_tn1: np.ndarray
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
        r_tn2: np.ndarray
            the responses of the TN2 neurons
        """
        return np.clip(flow + self.rng.normal(scale=self._noise, size=flow.shape), 0, 1)

    def decode_vector(self):
        """
        Transforms the CPU4 vector memory to a vector in the Cartesian coordinate system.

        Returns
        -------
        complex
        """
        vec_reshaped = self.__cpu4.reshape((2, -1))
        vec_shifted = np.array([np.roll(vec_reshaped[0], 1, axis=-1),
                                np.roll(vec_reshaped[1], -1, axis=-1)])
        signal = np.sum(vec_shifted, axis=0)

        fund_freq = np.fft.fft(signal)[1]
        angle = -np.angle(np.conj(fund_freq))
        distance = np.absolute(fund_freq) / self._gain

        return distance * np.exp(1j * angle)

    @property
    def w_tl22cl1(self):
        """
        The TL2 to CL1 synaptic weights.
        """
        return self._w_tl22cl1

    @w_tl22cl1.setter
    def w_tl22cl1(self, v):
        self._w_tl22cl1[:] = v[:]

    @property
    def w_cl12tb1(self):
        """
        The CL1 to TB1 synaptic weights.
        """
        return self._w_cl12tb1

    @w_cl12tb1.setter
    def w_cl12tb1(self, v):
        self._w_cl12tb1[:] = v[:]

    @property
    def w_tb12tb1(self):
        """
        The TB1 to TB1 synaptic weights.
        """
        return self._w_c2c

    @w_tb12tb1.setter
    def w_tb12tb1(self, v):
        self._w_c2c[:] = v[:]

    @property
    def w_tb12cpu4(self):
        """
        The TB1 to CPU4 synaptic weights.
        """
        return self._w_c2m

    @w_tb12cpu4.setter
    def w_tb12cpu4(self, v):
        self._w_c2m[:] = v[:]

    @property
    def w_tn22cpu4(self):
        """
        The TN2 to CPU4 synaptic weights.
        """
        return self._w_tn22cpu4

    @w_tn22cpu4.setter
    def w_tn22cpu4(self, v):
        self._w_tn22cpu4[:] = v[:]

    @property
    def w_tb12cpu1a(self):
        """
        The TB1 to CPU1a synaptic weights.
        """
        return self._w_c2s[:, 1:-1]

    @w_tb12cpu1a.setter
    def w_tb12cpu1a(self, v):
        self._w_c2s[:, 1:-1] = v[:]

    @property
    def w_tb12cpu1b(self):
        """
        The TB1 to CPU1b synaptic weights.
        """
        np.hstack([self._w_c2s[:, -1:], self._w_c2s[:, :1]])
        return np.hstack([self._w_c2s[:, -1:], self._w_c2s[:, :1]])

    @w_tb12cpu1b.setter
    def w_tb12cpu1b(self, v):
        self._w_c2s[:, -1:] = v[:, :1]
        self._w_c2s[:, :1] = v[:, -1:]

    @property
    def w_cpu42cpu1a(self):
        """
        The CPU4 to CPU1a synaptic weights.
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
        """
        return self.w_m2s[:, [-self.nb_cpu1a//2-1, self.nb_cpu1a//2]]

    @w_cpu42cpu1b.setter
    def w_cpu42cpu1b(self, v):
        self.w_m2s[:, [-self.nb_cpu1a//2-1, self.nb_cpu1a//2]] = v[:]

    @property
    def w_cpu1a2motor(self):
        """
        Matrix transforming the CPU1a responses to their contribution to the motor commands.
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
        """
        return self.w_s2o[[-self.nb_cpu1a//2-1, self.nb_cpu1a//2]]

    @w_cpu1b2motor.setter
    def w_cpu1b2motor(self, v):
        self.w_s2o[[-self.nb_cpu1a//2-1, self.nb_cpu1a//2], :] = v[:]

    @property
    def w_pontine2cpu1a(self):
        """
        The Pontine to CPU1a synaptic weights.
        """
        return self._w_pontine2cpu1a

    @w_pontine2cpu1a.setter
    def w_pontine2cpu1a(self, v):
        self._w_pontine2cpu1a[:] = v[:]

    @property
    def w_pontine2cpu1b(self):
        """
        The Pontine to CPU1b synaptic weights.
        """
        return self._w_pontine2cpu1b

    @w_pontine2cpu1b.setter
    def w_pontine2cpu1b(self, v):
        self._w_pontine2cpu1b[:] = v[:]

    @property
    def w_cpu42pontine(self):
        """
        The CPU4 to Pontine synaptic weights.
        """
        return self._w_cpu42pontine

    @w_cpu42pontine.setter
    def w_cpu42pontine(self, v):
        self._w_cpu42pontine[:] = v[:]

    @property
    def b_tl2(self):
        """
        The TL2 rest response rate (bias).
        """
        return self._b_tl2

    @property
    def b_cl1(self):
        """
        The CL1 rest response rate (bias).
        """
        return self._b_cl1

    @property
    def b_tb1(self):
        """
        The TB1 rest response rate (bias).
        """
        return self._b_com

    @property
    def b_cpu4(self):
        """
        The CPU4 rest response rate (bias).
        """
        return self._b_mem

    @property
    def b_cpu1(self):
        """
        The CPU1 rest response rate (bias).
        """
        return self._b_ste

    @property
    def b_motor(self):
        """
        The motor bias.
        """
        return self._b_out

    @property
    def b_pontine(self):
        """
        The Pontine rest response rate (bias).
        """
        return self._b_pontine

    @property
    def r_tb1(self):
        """
        The TB1 response rate.
        """
        return self._com

    @r_tb1.setter
    def r_tb1(self, v):
        self._com[:] = v[:]

    @property
    def nb_tb1(self):
        """
        The number TB1 neurons.
        """
        return self._nb_compass

    @property
    def r_tl2(self):
        """
        The TL2 response rate.
        """
        return self._tl2

    @r_tl2.setter
    def r_tl2(self, v):
        self._tl2[:] = v[:]

    @property
    def nb_tl2(self):
        """
        The number TL2 neurons.
        """
        return self._nb_tl2

    @property
    def r_cl1(self):
        """
        The CL1 response rate.
        """
        return self._cl1

    @r_cl1.setter
    def r_cl1(self, v):
        self._cl1[:] = v[:]

    @property
    def nb_cl1(self):
        """
        The number CL1 neurons.
        """
        return self._nb_cl1

    @property
    def r_tn1(self):
        """
        The TN1 response rate.
        """
        return self._tn1

    @r_tn1.setter
    def r_tn1(self, v):
        self._tn1[:] = v[:]

    @property
    def nb_tn1(self):
        """
        The number TN1 neurons.
        """
        return self._nb_tn1

    @property
    def r_tn2(self):
        """
        The TN2 response rate.
        """
        return self._tn2

    @r_tn2.setter
    def r_tn2(self, v):
        self._tn2[:] = v[:]

    @property
    def nb_tn2(self):
        """
        The number TN2 neurons.
        """
        return self._nb_tn2

    @property
    def r_cpu4(self):
        """
        The CPU4 response rate.
        """
        return self._mem

    @r_cpu4.setter
    def r_cpu4(self, v):
        self._mem[:] = v[:]

    @property
    def cpu4_mem(self):
        """
        The CPU4 memory.
        """
        return self.__cpu4

    @property
    def nb_cpu4(self):
        """
        The number CPU4 neurons.
        """
        return self._nb_memory

    @property
    def nb_cpu1a(self):
        """
        The number CPU1a neurons.
        """
        return self._nb_cpu1a

    @property
    def nb_cpu1b(self):
        """
        The number CPU1b neurons.
        """
        return self._nb_cpu1b

    @property
    def r_cpu1(self):
        """
        The CPU1 response rate.
        """
        return self._ste

    @r_cpu1.setter
    def r_cpu1(self, v):
        self._ste[:] = v[:]

    @property
    def nb_cpu1(self):
        """
        The number CPU1 neurons.
        """
        return self._nb_steering

    @property
    def pontine(self):
        """
        Whether the Pontine neurons are included in the circuit.

        Returns
        -------
        bool
        """
        return self._pontine

    @property
    def holonomic(self):
        """
        Whether the holonomic version of the circuit is used.

        Returns
        -------
        bool
        """
        return self._holonomic
