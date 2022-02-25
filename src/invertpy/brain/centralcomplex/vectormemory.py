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

from .fanshapedbody import VectorMemoryLayer
from .stone import StoneCX

import numpy as np
import os

# get path of the script
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

N_COLUMNS = 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class VectorMemoryCX(StoneCX):

    def __init__(self, nb_vectors=4, *args, **kwargs):
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
        nb_vectors: int, optional
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

        super().__init__(*args, **kwargs)

        self["vectors"] = VectorMemoryLayer(nb_cpu4=self.nb_cpu4, nb_vec=nb_vectors)

        if self.__class__ == VectorMemoryCX:
            self.reset()

    def _fprop(self, phi, flow, tl2=None, cl1=None, vec=None):
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

        self.memory(tb1=a_tb1, tn1=a_tn1, tn2=a_tn2)
        a_cpu4 = self.vectors(cpu4=self.memory.cpu4_mem, vec=vec)
        self.vectors.update = False

        a_cpu1 = self.steering(cpu4=a_cpu4, tb1=a_tb1)

        return a_cpu1

    def reset_current_memory(self):
        self.vectors.reset()

    def reset_memory(self, id):
        if id > 0:
            print(f"STORE VECTOR AT VEC_{id+1}!")
            self.vectors.reset_memory(id)
        elif id == 0:
            self.reset_integrator()

    def get_vectors_distance(self):
        w = self.vectors.w_vec2cpu4.copy() + self.vectors.w_vec2cpu4[0] - 0.5
        return np.absolute(mem2vector(w, self.memory.gain))

    def mem2vector(self):
        """
        Transforms the internal vector memories to actual vectors in the Cartesian coordinate system.

        Returns
        -------
        np.ndarray[complex]
        """
        return mem2vector(self.vectors.w_vec2cpu4, self.memory.gain)

    def __repr__(self):
        return f"VectorMemoryCX(TB1={self.nb_tb1:d}, TN1={self.nb_tn1:d}, TN2={self.nb_tn2:d}," \
               f" CL1={self.nb_cl1:d}, TL2={self.compass.nb_tl2}, CPU4={self.nb_cpu4:d}," \
               f" vectors={self.nb_vectors:d}, CPU1={self.nb_cpu1:d})"

    @property
    def vectors(self):
        """

        Returns
        -------
        VectorMemoryLayer
        """
        return self["vectors"]

    @property
    def nb_vectors(self):
        """
        The number of visual familiarity rings.

        Returns
        -------
        int
        """
        return self.vectors.nb_vec

    @property
    def r_vec(self):
        return self.vectors.r_vec


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
