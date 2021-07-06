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

from .component import Component
from .synapses import *
from .activation import sigmoid, relu
from .cx_helpers import tn_axes

import numpy as np
import os

# get path of the script
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

N_COLUMNS = 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class BeeCentralComplex(Component):

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
        super(BeeCentralComplex, self).__init__(*args, **kwargs)

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
        self._nb_cpu4_view = nb_cpu4
        self._nb_cpu1a = nb_cpu1a
        self._nb_cpu1b = nb_cpu1b
        self._nb_cpu1a_view = nb_cpu1a
        self._nb_cpu1b_view = nb_cpu1b


        # initialise the responses of the neurons
        self._tl2 = np.zeros(self.nb_tl2)
        self._cl1 = np.zeros(self.nb_cl1)
        self._tb1 = np.zeros(self.nb_tb1)
        self._tn1 = np.zeros(self.nb_tn1)
        self._tn2 = np.zeros(self.nb_tn2)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory
        self.__cpu4_view = .5 * np.ones(self.nb_cpu4)  # cpu4 memory
        self._cpu4 = np.zeros(self.nb_cpu4)  # cpu4 output
        self._cpu4_view = np.zeros(self.nb_cpu4)  # cpu4 output
        self._cpu1 = np.zeros(self.nb_cpu1)
        self._cpu1_view = np.zeros(self.nb_cpu1)

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
        self.r_cpu4_view = np.zeros(self.nb_cpu4_view)  # cpu4 output
        self.r_cpu1 = np.zeros(self.nb_cpu1)
        self.r_cpu1_view = np.zeros(self.nb_cpu1_view)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory
        self.__cpu4_view = .5 * np.ones(self.nb_cpu4_view)  # cpu4 memory

        self.update = True

    def _fprop(self, phi, flow, tl2=None, cl1=None, reinforcement=None):
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
            mem_view = 0.
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

            rein_mask = np.zeros(self.nb_cpu4_view, dtype=self.dtype)
            if reinforcement is not None:
                if np.shape(reinforcement)[0] > 1:
                    rein_mask[:self.nb_cpu4_view//2] = reinforcement[0]
                    rein_mask[self.nb_cpu4_view//2:] = reinforcement[1]
                else:
                    rein_mask[:] = np.asscalar(reinforcement)

            mem_tb1_view = self._gain * rein_mask * (a_tb1 - 1.) @ self.w_tb12cpu4

            mem_view = mem_tn1 * mem_tb1_view - mem_tn2

        cpu4_mem = np.clip(self.__cpu4 + mem, 0., 1.)
        cpu4_mem_view = np.clip(self.__cpu4_view + mem_view, 0., 1.)

        if self.update:
            self.__cpu4 = cpu4_mem
            self.__cpu4_view = cpu4_mem_view

        self._cpu4 = a_cpu4 = self.f_cpu4(cpu4_mem)
        self._cpu4_view = a_cpu4_view = self.f_cpu4(cpu4_mem_view)

        if self.pontin:
            a_pontin = self.f_pontin(a_cpu4 @ self.w_cpu42pontin)
            cpu1a = .5 * a_cpu4 @ self.w_cpu42cpu1a - .5 * a_pontin @ self.w_pontin2cpu1a - a_tb1 @ self.w_tb12cpu1a
            cpu1b = .5 * a_cpu4 @ self.w_cpu42cpu1b - .5 * a_pontin @ self.w_pontin2cpu1b - a_tb1 @ self.w_tb12cpu1b
        else:
            cpu1a = (a_cpu4 @ self.w_cpu42cpu1a) * ((a_tb1 - 1.) @ self.w_tb12cpu1a)
            cpu1a_view = (a_cpu4_view @ self.w_cpu42cpu1a) * ((a_tb1 - 1.) @ self.w_tb12cpu1a)
            cpu1b = (a_cpu4 @ self.w_cpu42cpu1b) * ((a_tb1 - 1.) @ self.w_tb12cpu1b)
            cpu1b_view = (a_cpu4_view @ self.w_cpu42cpu1b) * ((a_tb1 - 1.) @ self.w_tb12cpu1b)

        self._cpu1 = a_cpu1 = self.f_cpu1(np.hstack([cpu1b[-1], cpu1a, cpu1b[0]]))
        self._cpu1_view = a_cpu1_view = self.f_cpu1(np.hstack([cpu1b_view[-1], cpu1a_view, cpu1b_view[0]]))

        return a_cpu1 + a_cpu1_view

    def __repr__(self):
        return "BeeCentralComplex(TB1=%d, TN1=%d, TN2=%d, CL1=%d, TL2=%d, CPU4=%d, CPU1=%d)" % (
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

    @property
    def r_cpu4_view(self):
        """
        The CPU4 (view) response rate.
        """
        return self._cpu4_view

    @r_cpu4_view.setter
    def r_cpu4_view(self, v):
        self._cpu4_view[:] = v[:]

    @property
    def cpu4_mem_view(self):
        """
        The CPU4 (view) memory.
        """
        return self.__cpu4_view

    @property
    def nb_cpu4_view(self):
        """
        The number CPU4 (view) neurons.
        """
        return self._nb_cpu4_view

    @property
    def nb_cpu1a_view(self):
        """
        The number CPU1a (view) neurons.
        """
        return self._nb_cpu1a_view

    @property
    def nb_cpu1b_view(self):
        """
        The number CPU1b (view) neurons.
        """
        return self._nb_cpu1b

    @property
    def r_cpu1_view(self):
        """
        The CPU1 (view) response rate.
        """
        return self._cpu1_view

    @r_cpu1_view.setter
    def r_cpu1_view(self, v):
        self._cpu1_view[:] = v[:]

    @property
    def nb_cpu1_view(self):
        """
        The number CPU1 (view) neurons.
        """
        return self._nb_cpu1a_view + self._nb_cpu1b_view


class FlyCentralComplex(Component):

    def __init__(self, nb_compass=8, nb_epg=8, nb_peg=16, nb_pen=16, nb_nod=2, nb_fbn=16, nb_pfl3=16,
                 fixed_pfl3s=True, *args, **kwargs):
        """
        The Central Complex model of [1]_ as a component of the locust brain.

        Parameters
        ----------
        nb_compass: int, optional
            the dimensions of the input from the compass.
        nb_epg: int, optional
            the number of E-PG neurons.
        nb_peg: int, optional
            the number of P-EG neurons.
        nb_pen: int, optional
            the number of P-EN neurons.
        nb_nod: int, optional
            the total number of Nod_R and Nod_L neurons.
        nb_fbn: int, optional
            the total number of FBN neurons (left and right).
        nb_pfl3: int, optional
            the total number of PFL3 neurons (left and right).

        Notes
        -----
        .. [1] Goulard, R. et al. A unified mechanism for innate and learned visual landmark guidance in the insect
           central complex. bioRxic (2021).
        """
        kwargs.setdefault('nb_input', nb_compass + nb_nod)
        kwargs.setdefault('nb_output', nb_pfl3)
        kwargs.setdefault('repeat_rate', 1e-01)
        kwargs.setdefault('learning_rule', custom_learning_rule)
        super(FlyCentralComplex, self).__init__(*args, **kwargs)

        self._fixed_pfl3s = fixed_pfl3s

        self._nb_cmp = nb_compass
        self._nb_epg = nb_epg
        self._nb_peg = nb_peg
        self._nb_pen = nb_pen
        self._nb_nod = nb_nod
        self._nb_fbn = nb_fbn
        self._nb_pfl = nb_pfl3
        self._nb_dna = 2

        # initialise the responses of the neurons
        self._cmp = np.zeros(self.nb_cmp, dtype=self.dtype)
        self._epg = np.zeros(self.nb_epg, dtype=self.dtype)
        self._peg = np.zeros(self.nb_peg, dtype=self.dtype)
        self._pen = np.zeros(self.nb_pen, dtype=self.dtype)
        self._nod = np.zeros(self.nb_nod, dtype=self.dtype)
        self.__fbn = np.zeros(self.nb_fbn, dtype=self.dtype)  # memory
        self._fbn = np.zeros(self.nb_fbn, dtype=self.dtype)
        self._pfl = np.zeros(self.nb_pfl3, dtype=self.dtype)
        self._dna = np.zeros(self.nb_dna2, dtype=self.dtype)

        # Weight matrices based on anatomy (These are not changeable!)
        self._w_cmp2epg = uniform_synapses(self.nb_cmp, self.nb_epg, fill_value=0, dtype=self.dtype)
        self._w_peg2epg = uniform_synapses(self.nb_peg, self.nb_epg, fill_value=0, dtype=self.dtype)
        self._w_pen2epg = uniform_synapses(self.nb_pen, self.nb_epg, fill_value=0, dtype=self.dtype)
        self._w_epg2epg = uniform_synapses(self.nb_epg, self.nb_epg, fill_value=0, dtype=self.dtype)
        self._w_epg2peg = uniform_synapses(self.nb_epg, self.nb_peg, fill_value=0, dtype=self.dtype)
        self._w_epg2pen = uniform_synapses(self.nb_epg, self.nb_pen, fill_value=0, dtype=self.dtype)
        self._w_nod2pen = uniform_synapses(self.nb_nod, self.nb_pen, fill_value=0, dtype=self.dtype)
        self._w_epg2pfl = uniform_synapses(self.nb_epg, self.nb_pfl3, fill_value=0, dtype=self.dtype)
        self._w_epg2fbn = uniform_synapses(self.nb_epg, self.nb_fbn, fill_value=0, dtype=self.dtype)
        self._w_nod2fbn = uniform_synapses(self.nb_nod, self.nb_fbn, fill_value=0, dtype=self.dtype)
        self._w_pfl2dna = uniform_synapses(self.nb_pfl3, 2, fill_value=0, dtype=self.dtype)
        self._w_fbn2pfl = uniform_synapses(self.nb_fbn, self.nb_pfl3, fill_value=0, dtype=self.dtype)

        self.params.extend([
            self._w_cmp2epg,
            self._w_peg2epg,
            self._w_pen2epg,
            self._w_epg2epg,
            self._w_epg2peg,
            self._w_epg2pen,
            self._w_nod2pen,
            self._w_epg2pfl,
            self._w_epg2fbn,
            self._w_nod2fbn,
            self._w_pfl2dna
        ])

        # The cell properties (for sigmoid function)
        self._cmp_slope = 5.0
        self._epg_slope = 2.0
        self._peg_slope = 5.0
        self._pen_slope = 5.0
        self._pfl_slope = 5.0
        self._fbn_slope = 5.0
        self._nod_slope = 1.0
        self._dna_slope = 1.0

        self._b_cmp = 2.5
        self._b_epg = 1.0
        self._b_peg = 2.5
        self._b_pen = 3.75
        self._b_pfl = 5.0
        self._b_fbn = 2.5
        self._b_nod = 0.0
        self._b_dna = 0.0

        self.f_cmp = lambda v: sigmoid(self._cmp_slope * (v - v.min()) / (v.max() - v.min()) - self._b_cmp,
                                       noise=self._noise, rng=self.rng)
        self.f_epg = lambda v: sigmoid(self._epg_slope * v - self._b_epg, noise=self._noise, rng=self.rng)
        self.f_peg = lambda v: sigmoid(self._peg_slope * v - self._b_peg, noise=self._noise, rng=self.rng)
        self.f_pen = lambda v: sigmoid(self._pen_slope * v - self._b_pen, noise=self._noise, rng=self.rng)
        self.f_pfl = lambda v: sigmoid(self._pfl_slope * v - self._b_pfl, noise=self._noise, rng=self.rng)
        self.f_fbn = lambda v: sigmoid(self._fbn_slope * v - self._b_fbn, noise=self._noise, rng=self.rng)
        self.f_nod = lambda v: relu(self._nod_slope * v - self._b_nod, noise=self._noise, rng=self.rng)
        self.f_dna = lambda v: sigmoid(self._dna_slope * v - self._b_dna, noise=self._noise, rng=self.rng)

        self.reset()

    def reset(self):
        # Weight matrices based on anatomy (These are not changeable!)
        self.w_cmp2epg = diagonal_synapses(self.nb_cmp, self.nb_epg, fill_value=1, dtype=self.dtype)
        self.w_peg2epg = diagonal_synapses(self.nb_peg, self.nb_epg, fill_value=.5, tile=True, dtype=self.dtype)
        self.w_pen2epg = diagonal_synapses(self.nb_pen, self.nb_epg, fill_value=.5, tile=True, dtype=self.dtype)
        self.w_pen2epg[:self.nb_pen//2] = roll_synapses(self.w_pen2epg[:self.nb_pen//2], right=1)
        self.w_pen2epg[self.nb_pen//2:] = roll_synapses(self.w_pen2epg[self.nb_pen//2:], left=1)
        self.w_epg2epg = .5 * (diagonal_synapses(self.nb_epg, self.nb_epg, fill_value=1., dtype=self.dtype) - 1.)
        self.w_epg2peg = diagonal_synapses(self.nb_epg, self.nb_peg, fill_value=1., tile=True, dtype=self.dtype)
        self.w_epg2pen = diagonal_synapses(self.nb_epg, self.nb_pen, fill_value=.75, tile=True, dtype=self.dtype)
        self.w_nod2pen = chessboard_synapses(self.nb_nod, self.nb_pen, nb_rows=2, nb_cols=2, fill_value=.75,
                                             dtype=self.dtype)
        self.w_epg2fbn = diagonal_synapses(self.nb_epg, self.nb_fbn, fill_value=1., tile=True, dtype=self.dtype)
        self.w_nod2fbn = chessboard_synapses(self.nb_nod, self.nb_fbn, nb_rows=2, nb_cols=2, fill_value=-1.5,
                                             dtype=self.dtype)
        self.w_pfl2dna2 = chessboard_synapses(self.nb_pfl3, 2, nb_rows=2, nb_cols=2, fill_value=1.,
                                              dtype=self.dtype)
        self.w_fbn2pfl3 = diagonal_synapses(self.nb_fbn, self.nb_pfl3, fill_value=-1., dtype=self.dtype)
        self.w_fbn2pfl3[:self.nb_fbn//2, :self.nb_pfl3//2] = roll_synapses(
            self.w_fbn2pfl3[:self.nb_fbn//2, :self.nb_pfl3//2], left=1)
        self.w_fbn2pfl3[self.nb_fbn//2:, self.nb_pfl3//2:] = roll_synapses(
            self.w_fbn2pfl3[self.nb_fbn//2:, self.nb_pfl3//2:], right=1)

        # These are changeable!
        if self._fixed_pfl3s:
            w_epg2pfl3 = np.square(np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False)))
            w_epg2pfl3[:self.nb_pfl3//2] = np.roll(w_epg2pfl3[:self.nb_pfl3//2], -2)
            w_epg2pfl3[self.nb_pfl3//2:] = np.roll(w_epg2pfl3[self.nb_pfl3//2:], 1)
            self.w_epg2pfl3 = (diagonal_synapses(self.nb_epg, self.nb_pfl3, fill_value=1., tile=True, dtype=self.dtype) *
                               w_epg2pfl3)
        else:
            self.w_epg2pfl3 = diagonal_synapses(self.nb_epg, self.nb_pfl3, fill_value=.5, tile=True, dtype=self.dtype)

        self.r_cmp = np.zeros(self.nb_cmp, dtype=self.dtype)
        self.r_epg = np.zeros(self.nb_epg, dtype=self.dtype)
        self.r_peg = np.zeros(self.nb_peg, dtype=self.dtype)
        self.r_pen = np.zeros(self.nb_pen, dtype=self.dtype)
        self.r_nod = np.zeros(self.nb_nod, dtype=self.dtype)
        self.__fbn = np.full(self.nb_fbn, .0, dtype=self.dtype)
        self.r_fbn = np.zeros(self.nb_fbn, dtype=self.dtype)
        self.r_pfl3 = np.zeros(self.nb_pfl3, dtype=self.dtype)
        self.r_dna2 = np.zeros(self.nb_dna2, dtype=self.dtype)

        self.update = True

    def _fprop(self, compass, nod, reinforcement=None):
        """

        Parameters
        ----------
        compass: np.ndarray[float]
            the input from the compass
        nod: np.ndarray[float]
            the left and right Noduli responses (left and right turn)
        reinforcement: np.ndarray[float], int
            reinforces the familiar directions

        Returns
        -------
        np.ndarray[float]
            the PFL3 responses that can be used as a steering command
        """

        self._cmp = a_cmp = self.f_cmp(compass)
        self._nod = a_nod = self.f_nod(nod)

        self._peg = a_peg = self.f_peg(np.dot(self.r_epg, self.w_epg2peg))
        self._pen = a_pen = self.f_pen(np.dot(self.r_epg, self.w_epg2pen) +
                                       np.dot(a_nod, self.w_nod2pen) * a_peg)

        a_epg = (np.dot(a_cmp, self.w_cmp2epg) +
                 np.dot(self.r_peg, self.w_peg2epg) +
                 np.dot(self.r_pen, self.w_pen2epg))
        # process the Delta7 feedback as a second step to increase stability
        self._epg = a_epg = self.f_epg(a_epg + np.dot(self.r_epg, self.w_epg2epg))

        # memory integration
        self.__fbn += .05 * (np.dot(a_epg, self.w_epg2fbn) +
                             np.dot(a_nod, self.w_nod2fbn))

        if self._fixed_pfl3s and not self.update and reinforcement is not None:
            self._fbn = a_fbn = self.f_fbn(self.__fbn + .05 * reinforcement * np.dot(a_nod, self.w_nod2fbn))
        else:
            self._fbn = a_fbn = self.f_fbn(self.__fbn)

        self._pfl = a_pfl = self.f_pfl(np.dot(self.r_epg, self.w_epg2pfl3))

        self._dna = a_dna = self.f_dna(np.dot(a_pfl, self.w_pfl2dna2))

        if self.update and not self._fixed_pfl3s and reinforcement is not None:
            a_rein = reinforcement * np.dot(a_fbn, self.w_fbn2pfl3)
            self.w_epg2pfl3 = self.update_weights(w_pre=self.w_epg2pfl3, r_pre=a_epg, r_post=a_pfl,
                                                  rein=a_rein, w_rest=0.)

        return a_pfl

    def __repr__(self):
        return "FlyCentralComplex(compass=%d, E-PG=%d, P-EG=%d, P-EN=%d, FsBN=%d, Noduli=%d, PFL3=%d, DNa2=%d)" % (
            self.nb_cmp, self.nb_epg, self.nb_peg, self.nb_pen, self.nb_fbn, self.nb_nod, self.nb_pfl3, self.nb_dna2
        )

    @property
    def w_cmp2epg(self):
        """
        The compass to E-PG synaptic weights.
        """
        return self._w_cmp2epg

    @w_cmp2epg.setter
    def w_cmp2epg(self, v):
        self._w_cmp2epg[:] = v[:]

    @property
    def w_peg2epg(self):
        """
        The P-EG to E-PG synaptic weights.
        """
        return self._w_peg2epg

    @w_peg2epg.setter
    def w_peg2epg(self, v):
        self._w_peg2epg[:] = v[:]

    @property
    def w_pen2epg(self):
        """
        The P-EN to E-PG synaptic weights.
        """
        return self._w_pen2epg

    @w_pen2epg.setter
    def w_pen2epg(self, v):
        self._w_pen2epg[:] = v[:]

    @property
    def w_epg2epg(self):
        """
        The E-PG to E-PG synaptic weights (through Delta 7 neurons).
        """
        return self._w_epg2epg

    @w_epg2epg.setter
    def w_epg2epg(self, v):
        self._w_epg2epg[:] = v[:]

    @property
    def w_epg2peg(self):
        """
        The E-PG to P-EG synaptic weights.
        """
        return self._w_epg2peg

    @w_epg2peg.setter
    def w_epg2peg(self, v):
        self._w_epg2peg[:] = v[:]

    @property
    def w_epg2pen(self):
        """
        The E-PG to P-EN synaptic weights.
        """
        return self._w_epg2pen

    @w_epg2pen.setter
    def w_epg2pen(self, v):
        self._w_epg2pen[:] = v[:]

    @property
    def w_nod2pen(self):
        """
        The Noduli to P-EN synaptic weights.
        """
        return self._w_nod2pen

    @w_nod2pen.setter
    def w_nod2pen(self, v):
        self._w_nod2pen[:] = v[:]

    @property
    def w_epg2pfl3(self):
        """
        The E-PG to PFL3 synaptic weights.
        """
        return self._w_epg2pfl

    @w_epg2pfl3.setter
    def w_epg2pfl3(self, v):
        self._w_epg2pfl[:] = v[:]

    @property
    def w_epg2fbn(self):
        """
        The E-PG to FsBN synaptic weights.
        """
        return self._w_epg2fbn

    @w_epg2fbn.setter
    def w_epg2fbn(self, v):
        self._w_epg2fbn[:] = v[:]

    @property
    def w_nod2fbn(self):
        """
        The Noduli to FsBN synaptic weights.
        """
        return self._w_nod2fbn

    @w_nod2fbn.setter
    def w_nod2fbn(self, v):
        self._w_nod2fbn[:] = v[:]

    @property
    def w_pfl2dna2(self):
        """
        The PFL3 to DNa2 synaptic weights.
        """
        return self._w_pfl2dna

    @w_pfl2dna2.setter
    def w_pfl2dna2(self, v):
        self._w_pfl2dna[:] = v[:]

    @property
    def w_fbn2pfl3(self):
        """
        The FsBN to PFL3 synaptic weights.
        """
        return self._w_fbn2pfl

    @w_fbn2pfl3.setter
    def w_fbn2pfl3(self, v):
        self._w_fbn2pfl[:] = v[:]

    @property
    def r_cmp(self):
        """
        The compass' response rate.
        """
        return self._cmp

    @r_cmp.setter
    def r_cmp(self, v):
        self._cmp[:] = v[:]

    @property
    def nb_cmp(self):
        """
        The number units in the compass.
        """
        return self._nb_cmp

    @property
    def r_epg(self):
        """
        The E-PG response rate.
        """
        return self._epg

    @r_epg.setter
    def r_epg(self, v):
        self._epg[:] = v[:]

    @property
    def nb_epg(self):
        """
        The number E-PG neurons.
        """
        return self._nb_epg

    @property
    def r_peg(self):
        """
        The P-EG response rate.
        """
        return self._peg

    @r_peg.setter
    def r_peg(self, v):
        self._peg[:] = v[:]

    @property
    def nb_peg(self):
        """
        The number P-EG neurons.
        """
        return self._nb_peg

    @property
    def r_pen(self):
        """
        The P-EN response rate.
        """
        return self._pen

    @r_pen.setter
    def r_pen(self, v):
        self._pen[:] = v[:]

    @property
    def nb_pen(self):
        """
        The number P-EN neurons.
        """
        return self._nb_pen

    @property
    def r_pfl3(self):
        """
        The PFL3 response rate.
        """
        return self._pfl

    @r_pfl3.setter
    def r_pfl3(self, v):
        self._pfl[:] = v[:]

    @property
    def nb_pfl3(self):
        """
        The number PFL3 neurons.
        """
        return self._nb_pfl

    @property
    def r_fbn(self):
        """
        The FsBN response rate.
        """
        return self._fbn

    @r_fbn.setter
    def r_fbn(self, v):
        self._fbn[:] = v[:]

    @property
    def nb_fbn(self):
        """
        The number FsB neurons.
        """
        return self._nb_fbn

    @property
    def r_nod(self):
        """
        The Noduli response rate.
        """
        return self._nod

    @r_nod.setter
    def r_nod(self, v):
        self._nod[:] = v[:]

    @property
    def nb_nod(self):
        """
        The number CPU1a neurons.
        """
        return self._nb_nod

    @property
    def r_dna2(self):
        """
        The DNa2 response rate.
        """
        return self._dna

    @r_dna2.setter
    def r_dna2(self, v):
        self._dna[:] = v[:]

    @property
    def nb_dna2(self):
        return self._nb_dna


def custom_learning_rule(w, r_pre, r_post, rein, learning_rate=1., w_rest=.5):
    if rein.ndim > 1:
        rein = rein[:, np.newaxis, ...]
    else:
        rein = rein[np.newaxis, ...]
    d_w = learning_rate * (rein + w_rest)
    if d_w.ndim > 2:
        d_w = d_w.sum(axis=0)
    d_w = diagonal_synapses(w.shape[0], w.shape[1], fill_value=1., tile=True, dtype=w.dtype) * d_w

    return np.clip(w + d_w, 0.2, 0.8)
