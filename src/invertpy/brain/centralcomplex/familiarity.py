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

from .stone import StoneCX
from .fanshapedbody import FamiliarityIntegratorLayer, WindGuidedLayer

import numpy as np


class FamiliarityIntegratorCX(StoneCX):

    def __init__(self, nb_mbon=6, *args, **kwargs):
        """
        A Central Complex model that follows familiar views along with integrating its path.

        Parameters
        ----------
        nb_mbon: int, optional
            the number of MB output neurons. Default is 6
        """
        nb_tb1 = kwargs.get('nb_tb1', kwargs.get('nb_delta7', 8))
        nb_tn1 = kwargs.get('nb_tn1', 2)
        nb_tn2 = kwargs.get('nb_tn2', 2)
        kwargs.setdefault('nb_input', nb_tb1 + nb_tn1 + nb_tn2 + nb_mbon)
        # kwargs.setdefault('gain', 0.025)
        super().__init__(*args, **kwargs)

        self._nb_mbon = nb_mbon

        self["memory"] = FamiliarityIntegratorLayer(nb_mbon=nb_mbon, gain=self.memory.gain,
                                                    nb_tb1=nb_tb1, nb_cpu4=self.nb_cpu4, nb_tn1=nb_tn1, nb_tn2=nb_tn2)

        if self.__class__ == FamiliarityIntegratorCX:
            self.reset()

    def _fprop(self, phi, flow, tl2=None, cl1=None, mbon=None):
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
            the responses from the output neurons of the mushroom body coming using the FB tangential neurons
        vec: np.ndarray[float]
            the target vector

        Returns
        -------
        np.ndarray[float]
            the CPU1 responses that are used for steering
        """
        a_tb1 = self.compass(phi=phi, tl2=tl2, cl1=cl1)
        a_tn1 = self.flow2tn1(flow)
        a_tn2 = self.flow2tn2(flow)

        a_cpu4 = self.memory(tb1=a_tb1, tn1=a_tn1, tn2=a_tn2, mbon=mbon)

        a_cpu1 = self.steering(cpu4=a_cpu4, tb1=a_tb1)

        return a_cpu1

    @property
    def memory(self):
        """
        Returns
        -------
        FamiliarityIntegratorLayer
        """
        return self["memory"]

    @property
    def r_mbon(self):
        return self.memory.r_mbon

    @property
    def nb_mbon(self):
        """
        The number of MB output neurons.

        Returns
        -------
        int
        """
        return self._nb_mbon


class FamiliarityCX(StoneCX):

    def __init__(self, nb_mbon=6, *args, **kwargs):
        """
        A Central Complex model that follows familiar views along with integrating its path.

        Parameters
        ----------
        nb_mbon: int, optional
            the number of MB output neurons. Default is 6
        """
        nb_tb1 = kwargs.get('nb_tb1', kwargs.get('nb_delta7', 8))
        nb_nod = kwargs.get('nb_nod', kwargs.get('nb_tn1', 2))
        kwargs.setdefault('nb_input', nb_tb1 + nb_nod + nb_mbon)
        super().__init__(*args, **kwargs)

        self._nb_mbon = nb_mbon

        self["memory"] = WindGuidedLayer(nb_mbon=nb_mbon,
                                         nb_epg=self.nb_cl1, nb_cpu4=self.nb_cpu4, nb_nod=nb_nod)

        self._nod_momentum = np.ones((1, 2), dtype=self.dtype) / np.sqrt(2)

        if self.__class__ == FamiliarityIntegratorCX:
            self.reset()

    def reset(self):
        StoneCX.reset(self)

        self._nod_momentum = np.ones((1, 2), dtype=self.dtype) / np.sqrt(2)

    def _fprop(self, phi, flow, tl2=None, cl1=None, mbon=None):
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
            the responses from the output neurons of the mushroom body coming using the FB tangential neurons
        vec: np.ndarray[float]
            the target vector

        Returns
        -------
        np.ndarray[float]
            the CPU1 responses that are used for steering
        """
        a_tb1 = self.compass(phi=phi, tl2=tl2, cl1=cl1)
        # a_nod = self.flow2tn1(flow)

        # the previous Noduli response + noise
        p_nod = self._nod_momentum.copy() + self.rng.rand(*self._nod_momentum.shape) * 0.001

        # the current positive - negative MBON response
        c_mbon = np.power(mbon[..., 0::2], 8) - np.power(mbon[..., 1::2], 8)

        # the previous positive - negative MBON response
        p_mbon = self.memory.r_mbon[..., 0::2] - self.memory.r_mbon[..., 1::2]

        # the difference between the two responses
        d_mbon = c_mbon - p_mbon

        # the sign of the change: 1 for >= 0 change, -1 for < 0 change
        d_sign = np.sign(np.mean(d_mbon) + np.finfo(float).eps)

        # a_nod = (d_mbon_sign * p_nod + self.rng.rand(*p_nod.shape) * 0.01)
        a_nod = d_sign * p_nod
        # print(a_nod, end=" ")
        a_nod = (a_nod - a_nod.min()) / (a_nod.max() - a_nod.min())
        # if np.greater_equal(np.mean(d_mbon - p_mbon), 0):
        # if np.greater(d_sign, 0):
        #     a_nod = d_sign * p_nod
        #     print("dM >= 0 |", end=" ")
        # # elif np.all(np.isclose([p_nod[..., 0] - p_nod[..., 1]], 0)):
        # elif np.all(np.less(np.absolute(p_nod[..., 0] - p_nod[..., 1]), 0.1)):
        #     a_nod = np.eye(p_nod.shape[-1])[self.rng.randint(0, p_nod.shape[-1], size=1, dtype=int)]
        #     print("dM < 0, NOD_L == NOD_R |", end=" ")
        # else:
        #     print("dM < 0, NOD_L != NOD_R |", end=" ")
        #     i_nod = np.argmax(p_nod)
        #     z_nod = np.zeros_like(p_nod)
        #     z_nod[i_nod] = 1
        #     a_nod = 1 - z_nod
        self._nod_momentum += .05 * a_nod
        # self._nod_momentum[:] = [1, 0]
        self._nod_momentum /= np.linalg.norm(self._nod_momentum)

        a_pfl3 = self.memory(epg=self.compass.r_epg, nod=self._nod_momentum, mbon=mbon)

        self.steering.r_cpu1 = a_pfl3
        self.steering.r_tb1 = a_tb1
        self.steering.r_cpu4 = self.memory.r_pfn

        return a_pfl3

    @property
    def memory(self):
        """
        Returns
        -------
        FamiliarityIntegratorLayer
        """
        return self["memory"]

    @property
    def r_mbon(self):
        return self.memory.r_mbon

    @property
    def nb_mbon(self):
        """
        The number of MB output neurons.

        Returns
        -------
        int
        """
        return self._nb_mbon

    @property
    def r_motor(self):
        r_motor = np.maximum(self.r_cpu1 - .5, 0).reshape((2, -1)).mean(axis=1)
        r_motor = r_motor / np.linalg.norm(r_motor)
        return r_motor

