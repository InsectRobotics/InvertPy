
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

import warnings

from .centralcomplex import CentralComplexBase
from .ellipsoidbody import SimpleCompass, PontineSteering
from .fanshapedbody import PathIntegratorLayer
from ._helpers import tn_axes

import numpy as np


class StoneCX(CentralComplexBase):

    def __init__(self, nb_tb1=8, nb_tn1=2, nb_tn2=2, nb_cl1=16, nb_tl2=16, nb_cpu4=16, nb_cpu1a=14, nb_cpu1b=2,
                 tn_prefs=np.pi/4, gain=0.025, *args, **kwargs):
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
        kwargs.setdefault('nb_output', nb_cpu1a+nb_cpu1b)
        kwargs.setdefault('learning_rule', None)
        super().__init__(*args, **kwargs)

        self["compass"] = SimpleCompass(nb_tl2=nb_tl2, nb_cl1=nb_cl1, nb_tb1=nb_tb1, noise=self._noise, rng=self.rng)
        self["memory"] = PathIntegratorLayer(nb_tb1=nb_tb1, nb_cpu4=nb_cpu4, nb_tn1=nb_tn1, nb_tn2=nb_tn2, gain=gain, noise=self._noise, rng=self.rng)
        self["steering"] = PontineSteering(nb_tb1=nb_tb1, nb_cpu4=nb_cpu4, nb_cpu1=nb_cpu1a+nb_cpu1b, noise=self._noise, rng=self.rng)

        self._tn_prefs = tn_prefs
        self._smoothed_flow = 0.
        self._nb_ste = nb_cpu1a + nb_cpu1b
        self._nb_com = nb_tb1

        if self.__class__ == StoneCX:
            self.reset()

    def __repr__(self):
        return f"StoneCX(CL1={self.nb_cl1}, TB1={self.nb_tb1}, CPU4={self.nb_cpu4}, CPU1={self.nb_cpu1}, " \
               f"TN1={self.nb_tn1}, TN2={self.nb_tn2})"

    def _fprop(self, phi, flow, tl2=None, cl1=None):
        a_tb1 = self.compass(phi=phi, tl2=tl2, cl1=cl1)
        a_tn1 = self.flow2tn1(flow)
        a_tn2 = self.flow2tn2(flow)

        a_cpu4 = self.memory(tb1=a_tb1, tn1=a_tn1, tn2=a_tn2)

        a_cpu1 = self.steering(cpu4=a_cpu4, tb1=a_tb1)

        return a_cpu1

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
        A = tn_axes(heading, self._tn_prefs)
        flow = A.T.dot(velocity)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self._smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                                                1.0 / filter_steps) * self._smoothed_flow)
            flow = self._smoothed_flow
        return flow

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

    def reset_integrator(self):
        if hasattr(self.memory, 'reset_integrator'):
            self.memory.reset_integrator()
        else:
            warnings.warn("There is no integrator to reset.")

    @property
    def w_cpu1a2motor(self):
        """
        Matrix transforming the CPU1a responses to their contribution to the motor commands.
        """
        return np.vstack([self._w_s2o[:self.steering.nb_cpu1a//2], self._w_s2o[-self.steering.nb_cpu1a//2:]])

    @property
    def w_cpu1b2motor(self):
        """
        Matrix transforming the CPU1b responses to their contribution to the motor commands.
        """
        return self._w_s2o[[-self.steering.nb_cpu1a//2-1, self.steering.nb_cpu1a//2]]

    @property
    def steering(self):
        """

        Returns
        -------
        PontineSteering
        """
        return self["steering"]

    @property
    def compass(self):
        """

        Returns
        -------
        SimpleCompass
        """
        return self["compass"]

    @property
    def memory(self):
        """

        Returns
        -------
        PathIntegratorLayer
        """
        return self["memory"]

    @property
    def r_compass(self):
        return self.r_tb1

    @property
    def r_steering(self):
        return self.r_cpu1

    @property
    def nb_steering(self):
        return self.nb_cpu1

    @property
    def nb_compass(self):
        return self.nb_tb1

    @property
    def r_cl1(self):
        return self.compass.r_cl1

    @property
    def nb_cl1(self):
        return self.compass.nb_cl1

    @property
    def r_tb1(self):
        return self.compass.r_tb1

    @property
    def nb_tb1(self):
        return self.compass.nb_tb1

    @property
    def r_tn1(self):
        return self.memory.r_tn1

    @property
    def nb_tn1(self):
        return self.memory.nb_tn1

    @property
    def r_tn2(self):
        return self.memory.r_tn2

    @property
    def nb_tn2(self):
        return self.memory.nb_tn2

    @property
    def r_cpu4(self):
        return self.memory.r_cpu4

    @property
    def cpu4_mem(self):
        return self.memory.cpu4_mem

    @property
    def nb_cpu4(self):
        return self.memory.nb_cpu4

    @property
    def r_cpu1(self):
        return self.steering.r_cpu1

    @property
    def nb_cpu1(self):
        return self.steering.nb_cpu1
