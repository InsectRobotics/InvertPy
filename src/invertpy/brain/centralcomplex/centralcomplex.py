"""
Central Complex (CX) models of the insect brain.

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
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from invertpy.brain.component import Component
from invertpy.brain.synapses import *
from invertpy.brain.activation import sigmoid
from ._helpers import tn_axes

import numpy as np
import os

# get path of the script
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

N_COLUMNS = 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class CentralComplex(Component):

    def __init__(self, nb_tb1=8, nb_tn1=2, nb_tn2=2, nb_cl1=16, nb_tl2=16, nb_cpu4=16, nb_cpu1a=14, nb_cpu1b=2,
                 tn_prefs=np.pi/4, gain=0.05, pontin=False, *args, **kwargs):
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
        pontin: bool, optional
            whether to include a pontin neuron in the circuit or not. Default is False.

        Notes
        -----
        .. [1] Stone, T. et al. An Anatomically Constrained Model for Path Integration in the Bee Brain.
           Curr Biol 27, 3069-3085.e11 (2017).
        """

        kwargs.setdefault('nb_input', nb_tb1 + nb_tn1 + nb_tn2)
        kwargs.setdefault('nb_output', nb_cpu1a + nb_cpu1b)
        kwargs.setdefault('learning_rule', None)
        super(CentralComplex, self).__init__(*args, **kwargs)

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
        self._nb_tb1 = nb_tb1
        self._nb_tn1 = nb_tn1
        self._nb_tn2 = nb_tn2
        self._nb_cpu4 = nb_cpu4
        self._nb_cpu1a = nb_cpu1a
        self._nb_cpu1b = nb_cpu1b

        # initialise the responses of the neurons
        self._tl2 = np.zeros(self.nb_tl2)
        self._cl1 = np.zeros(self.nb_cl1)
        self._tb1 = np.zeros(self.nb_tb1)
        self._tn1 = np.zeros(self.nb_tn1)
        self._tn2 = np.zeros(self.nb_tn2)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory
        self._cpu4 = np.zeros(self.nb_cpu4)  # cpu4 output
        self._cpu1 = np.zeros(self.nb_cpu1)

        # Weight matrices based on anatomy (These are not changeable!)
        self._w_tl22cl1 = uniform_synapses(self.nb_tl2, self.nb_cl1, fill_value=0, dtype=self.dtype)
        self._w_cl12tb1 = uniform_synapses(self.nb_cl1, self.nb_tb1, fill_value=0, dtype=self.dtype)
        self._w_tb12tb1 = uniform_synapses(self.nb_tb1, self.nb_tb1, fill_value=0, dtype=self.dtype)
        self._w_tb12cpu4 = uniform_synapses(self.nb_tb1, self.nb_cpu4, fill_value=0, dtype=self.dtype)
        self._w_tn22cpu4 = uniform_synapses(self.nb_tn2, self.nb_cpu4, fill_value=0, dtype=self.dtype)
        self._w_tb12cpu1a = uniform_synapses(self.nb_tb1, self.nb_cpu1a, fill_value=0, dtype=self.dtype)
        self._w_tb12cpu1b = uniform_synapses(self.nb_tb1, self.nb_cpu1b, fill_value=0, dtype=self.dtype)
        self._w_cpu42cpu1a = uniform_synapses(self.nb_cpu4, self.nb_cpu1a, fill_value=0, dtype=self.dtype)
        self._w_cpu42cpu1b = uniform_synapses(self.nb_cpu4, self.nb_cpu1b, fill_value=0, dtype=self.dtype)
        self._w_cpu1a2motor = uniform_synapses(self.nb_cpu1a, 2, fill_value=0, dtype=self.dtype)
        self._w_cpu1b2motor = uniform_synapses(self.nb_cpu1b, 2, fill_value=0, dtype=self.dtype)

        self._w_pontin2cpu1a = uniform_synapses(self.nb_cpu1, self.nb_cpu1a, fill_value=0, dtype=self.dtype)
        self._w_pontin2cpu1b = uniform_synapses(self.nb_cpu1, self.nb_cpu1b, fill_value=0, dtype=self.dtype)
        self._w_cpu42pontin = uniform_synapses(self.nb_cpu4, self.nb_cpu4, fill_value=0, dtype=self.dtype)

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
        self._b_tb1 = 0.0
        self._b_cpu4 = 2.5
        self._b_cpu1 = 2.5  # -1.0
        self._b_motor = 3.0
        self._b_pontin = 2.5

        self.params.extend([
            self._w_tl22cl1,
            self._w_cl12tb1,
            self._w_tb12tb1,
            self._w_tb12cpu4,
            self._w_tb12cpu1a,
            self._w_tb12cpu1b,
            self._w_cpu42cpu1a,
            self._w_cpu42cpu1b,
            self._w_cpu1a2motor,
            self._w_cpu1b2motor,
            self._w_cpu42pontin,
            self._w_pontin2cpu1a,
            self._w_pontin2cpu1b,
            self._b_tl2,
            self._b_cl1,
            self._b_tb1,
            self._b_cpu4,
            self._b_cpu1,
            self._b_motor,
            self._b_pontin
        ])

        self.tl2_prefs = np.tile(np.linspace(0, 2 * np.pi, self.nb_tb1, endpoint=False), 2)
        # self.tl2_prefs = np.tile(np.linspace(-np.pi, np.pi, self.nb_tb1, endpoint=False), 2)

        self.f_tl2 = lambda v: sigmoid(v * self._tl2_slope - self.b_tl2, noise=self._noise, rng=self.rng)
        self.f_cl1 = lambda v: sigmoid(v * self._cl1_slope - self.b_cl1, noise=self._noise, rng=self.rng)
        self.f_tb1 = lambda v: sigmoid(v * self._tb1_slope - self.b_tb1, noise=self._noise, rng=self.rng)
        self.f_cpu4 = lambda v: sigmoid(v * self._cpu4_slope - self.b_cpu4, noise=self._noise, rng=self.rng)
        self.f_pontin = lambda v: sigmoid(v * self._pontin_slope - self.b_pontin, noise=self._noise, rng=self.rng)
        self.f_cpu1 = lambda v: sigmoid(v * self._cpu1_slope - self.b_cpu1, noise=self._noise, rng=self.rng)

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

        self.r_tl2 = np.zeros(self.nb_tl2)
        self.r_cl1 = np.zeros(self.nb_cl1)
        self.r_tb1 = np.zeros(self.nb_tb1)
        self.r_tn1 = np.zeros(self.nb_tn1)
        self.r_tn2 = np.zeros(self.nb_tn2)
        self.r_cpu4 = np.zeros(self.nb_cpu4)  # cpu4 output
        self.r_cpu1 = np.zeros(self.nb_cpu1)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory

        self.update = True

    def _fprop(self, phi, flow, tl2=None, cl1=None):
        if isinstance(phi, np.ndarray) and phi.size == 8:
            if tl2 is None:
                tl2 = np.tile(phi, 2)
            if cl1 is None:
                cl1 = np.tile(phi, 2)
            self._tl2 = a_tl2 = self.f_tl2(tl2[::-1])
            self._cl1 = a_cl1 = self.f_cl1(cl1[::-1])
            self._tb1 = a_tb1 = self.f_tb1(5. * phi[::-1])
        else:
            self._tl2 = a_tl2 = self.f_tl2(self.phi2tl2(phi))
            self._cl1 = a_cl1 = self.f_cl1(a_tl2 @ self.w_tl22cl1)
            if self._tb1 is None:
                self._tb1 = a_tb1 = self.f_tb1(a_cl1)
            else:
                p = .667  # proportion of input from CL1 to TB1
                self._tb1 = a_tb1 = self.f_tb1(p * a_cl1 @ self.w_cl12tb1 + (1 - p) * self._tb1 @ self.w_tb12tb1)
        self._tn1 = a_tn1 = self.flow2tn1(flow)
        self._tn2 = a_tn2 = self.flow2tn2(flow)

        if self.pontin:
            mem = .5 * self._gain * (np.clip(a_tn2 @ self.w_tn22cpu4 - a_tb1 @ self.w_tb12cpu4, 0, 1) - .25)
        else:
            # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
            # update = np.clip((.5 - tn1).dot(self.w_tn2cpu4), 0., 1.)  # normal
            mem_tn1 = (.5 - a_tn1) @ self.w_tn22cpu4  # holonomic

            mem_tb1 = self._gain * (a_tb1 - 1.) @ self.w_tb12cpu4
            # update *= self.gain * (1. - tb1).dot(self.w_tb12cpu4)

            # Both CPU4 waves must have same average
            # If we don't normalise get drift and weird steering
            mem_tn2 = self._gain * .25 * a_tn2.dot(self.w_tn22cpu4)

            mem = mem_tn1 * mem_tb1 - mem_tn2

        # Constant purely to visualise same as rate-based model
        cpu4_mem = np.clip(self.__cpu4 + mem, 0., 1.)

        if self.update:
            self.__cpu4 = cpu4_mem

        self._cpu4 = a_cpu4 = self.f_cpu4(cpu4_mem)

        if self.pontin:
            a_pontin = self.f_pontin(a_cpu4 @ self.w_cpu42pontin)
            cpu1a = .5 * a_cpu4 @ self.w_cpu42cpu1a - .5 * a_pontin @ self.w_pontin2cpu1a - a_tb1 @ self.w_tb12cpu1a
            cpu1b = .5 * a_cpu4 @ self.w_cpu42cpu1b - .5 * a_pontin @ self.w_pontin2cpu1b - a_tb1 @ self.w_tb12cpu1b
        else:
            cpu1a = (a_cpu4 @ self.w_cpu42cpu1a) * ((a_tb1 - 1.) @ self.w_tb12cpu1a)
            cpu1b = (a_cpu4 @ self.w_cpu42cpu1b) * ((a_tb1 - 1.) @ self.w_tb12cpu1b)

        self._cpu1 = a_cpu1 = self.f_cpu1(np.hstack([cpu1b[-1], cpu1a, cpu1b[0]]))

        return a_cpu1

    def __repr__(self):
        return "CentralComplex(TB1=%d, TN1=%d, TN2=%d, CL1=%d, TL2=%d, CPU4=%d, CPU1=%d)" % (
            self.nb_tb1, self.nb_tn1, self.nb_tn2, self.nb_cl1, self.nb_tl2, self.nb_cpu4, self.nb_cpu1
        )

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
        flow: np.ndarray
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
        return np.clip(flow, 0, 1)

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
        return self._w_tb12tb1

    @w_tb12tb1.setter
    def w_tb12tb1(self, v):
        self._w_tb12tb1[:] = v[:]

    @property
    def w_tb12cpu4(self):
        """
        The TB1 to CPU4 synaptic weights.
        """
        return self._w_tb12cpu4

    @w_tb12cpu4.setter
    def w_tb12cpu4(self, v):
        self._w_tb12cpu4[:] = v[:]

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
        return self._w_tb12cpu1a

    @w_tb12cpu1a.setter
    def w_tb12cpu1a(self, v):
        self._w_tb12cpu1a[:] = v[:]

    @property
    def w_tb12cpu1b(self):
        """
        The TB1 to CPU1b synaptic weights.
        """
        return self._w_tb12cpu1b

    @w_tb12cpu1b.setter
    def w_tb12cpu1b(self, v):
        self._w_tb12cpu1b[:] = v[:]

    @property
    def w_cpu42cpu1a(self):
        """
        The CPU4 to CPU1a synaptic weights.
        """
        return self._w_cpu42cpu1a

    @w_cpu42cpu1a.setter
    def w_cpu42cpu1a(self, v):
        self._w_cpu42cpu1a[:] = v[:]

    @property
    def w_cpu42cpu1b(self):
        """
        The CPU4 to CPU1b synaptic weights.
        """
        return self._w_cpu42cpu1b

    @w_cpu42cpu1b.setter
    def w_cpu42cpu1b(self, v):
        self._w_cpu42cpu1b[:] = v[:]

    @property
    def w_cpu1a2motor(self):
        """
        Matrix transforming the CPU1a responses to their contribution to the motor commands.
        """
        return self._w_cpu1a2motor

    @w_cpu1a2motor.setter
    def w_cpu1a2motor(self, v):
        self._w_cpu1a2motor[:] = v[:]

    @property
    def w_cpu1b2motor(self):
        """
        Matrix transforming the CPU1b responses to their contribution to the motor commands.
        """
        return self._w_cpu1b2motor

    @w_cpu1b2motor.setter
    def w_cpu1b2motor(self, v):
        self._w_cpu1b2motor[:] = v[:]

    @property
    def w_pontin2cpu1a(self):
        """
        The pontin to CPU1a synaptic weights.
        """
        return self._w_pontin2cpu1a

    @w_pontin2cpu1a.setter
    def w_pontin2cpu1a(self, v):
        self._w_pontin2cpu1a[:] = v[:]

    @property
    def w_pontin2cpu1b(self):
        """
        The pontin to CPU1b synaptic weights.
        """
        return self._w_pontin2cpu1b

    @w_pontin2cpu1b.setter
    def w_pontin2cpu1b(self, v):
        self._w_pontin2cpu1b[:] = v[:]

    @property
    def w_cpu42pontin(self):
        """
        The CPU4 to pontin synaptic weights.
        """
        return self._w_cpu42pontin

    @w_cpu42pontin.setter
    def w_cpu42pontin(self, v):
        self._w_cpu42pontin[:] = v[:]

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
        return self._b_tb1

    @property
    def b_cpu4(self):
        """
        The CPU4 rest response rate (bias).
        """
        return self._b_cpu4

    @property
    def b_cpu1(self):
        """
        The CPU1 rest response rate (bias).
        """
        return self._b_cpu1

    @property
    def b_motor(self):
        """
        The motor bias.
        """
        return self._b_motor

    @property
    def b_pontin(self):
        """
        The pontin rest response rate (bias).
        """
        return self._b_pontin

    @property
    def r_tb1(self):
        """
        The TB1 response rate.
        """
        return self._tb1

    @r_tb1.setter
    def r_tb1(self, v):
        self._tb1[:] = v[:]

    @property
    def nb_tb1(self):
        """
        The number TB1 neurons.
        """
        return self._nb_tb1

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
        return self._cpu4

    @r_cpu4.setter
    def r_cpu4(self, v):
        self._cpu4[:] = v[:]

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
        return self._nb_cpu4

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
        return self._cpu1

    @r_cpu1.setter
    def r_cpu1(self, v):
        self._cpu1[:] = v[:]

    @property
    def nb_cpu1(self):
        """
        The number CPU1 neurons.
        """
        return self._nb_cpu1a + self._nb_cpu1b
