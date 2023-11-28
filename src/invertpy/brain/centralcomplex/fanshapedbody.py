
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

import numpy as np

from invertpy.brain.synapses import *
from invertpy.brain.activation import sigmoid, relu

from .centralcomplex import CentralComplexLayer
from ._helpers import decode_vector


class FanShapedBodyLayer(CentralComplexLayer):

    def __init__(self, nb_pfn, nb_tangential, nb_fbn, *args, **kwargs):
        kwargs.setdefault('nb_input', nb_pfn + nb_tangential)
        kwargs.setdefault('nb_output', nb_fbn)
        super().__init__(*args, **kwargs)

        self._w_t2f = uniform_synapses(nb_tangential, nb_fbn, fill_value=1, dtype=self.dtype)
        self._w_p2f = sinusoidal_synapses(nb_pfn, nb_fbn, dtype=self.dtype)

        self._slope = 5.0
        self._b = 2.5

        self.params.extend([
            self._w_t2f,
            self._w_p2f,
            self._slope,
            self._b
        ])

        self._pfn = np.zeros(self.nb_pfn, dtype=self.dtype)
        self._tan = np.zeros(self.nb_tangential, dtype=self.dtype)
        self._fbn = np.zeros(self.nb_fbn, dtype=self.dtype)

        self.f_fbn = lambda v: sigmoid(v * self._slope - self._b, noise=self._noise, rng=self.rng)

    def reset(self):
        self._pfn = np.zeros(self.nb_pfn, dtype=self.dtype)
        self._tan = np.zeros(self.nb_tangential, dtype=self.dtype)
        self._fbn = np.zeros(self.nb_fbn, dtype=self.dtype)

        self.update = True

    def _fprop(self, pfn=None, tangential=None):

        fbn = np.zeros(self.nb_fbn, dtype=self.dtype)
        if pfn is not None:
            self._pfn = pfn
            fbn += pfn.dot(self.w_p2f)
        if tangential is not None:
            self._tan = tangential
            fbn += tangential.dot(self.w_t2f)

        a_fbn = self._fbn = self.f_fbn(fbn)

        return a_fbn

    @property
    def w_t2f(self):
        """
        The tangential-to-fanshaped body synapses

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_t2f

    @w_t2f.setter
    def w_t2f(self, v):
        self._w_t2f[:] = v

    @property
    def w_p2f(self):
        """
        The protocerebral bridge-to-fanshaped body synapses

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_p2f

    @w_p2f.setter
    def w_p2f(self, v):
        self._w_p2f[:] = v

    @property
    def r_pfn(self):
        """
        The latest responses of the PFN (protocerebral bridge-to-fan-shaped body) neurons.

        Returns
        -------
        np.ndarray[float]
        """
        return self._pfn

    @r_pfn.setter
    def r_pfn(self, v):
        self._pfn[:] = v

    @property
    def r_tan(self):
        return self._tan

    @r_tan.setter
    def r_tan(self, value):
        self._tan[:] = value

    @property
    def r_tangential(self):
        """
        The latest responses of the tangential fan-shaped body neurons.

        Returns
        -------
        np.ndarray[float]
        """
        return self._tan

    @r_tangential.setter
    def r_tangential(self, v):
        self._tan[:] = v

    @property
    def r_fbn(self):
        """
        The last responses of the fan-shaped body neurons.

        Returns
        -------
        np.ndarray[float]
        """
        return self._fbn

    @r_fbn.setter
    def r_fbn(self, v):
        self._fbn[:] = v

    @property
    def nb_pfn(self):
        """
        The number of PFNs (input neurons from the protocerebral bridge).

        Returns
        -------
        int
        """
        return self._w_p2f.shape[0]

    @property
    def nb_tangential(self):
        """
        The number of tangential neurons.

        Returns
        -------
        int
        """
        return self._w_t2f.shape[0]

    @property
    def nb_fbn(self):
        """
        The number of fan-shaped body output neurons.

        Returns
        -------
        int
        """
        return self._nb_output


class PathIntegratorLayer(FanShapedBodyLayer):
    def __init__(self, nb_delta7=8, nb_fbn=16, nb_tn1=2, nb_tn2=2, gain=0.025, *args, **kwargs):
        """
        The path integration model of [1]_ as a component of the locust brain that lives in the Fan-shaped Body.


        Parameters
        ----------
        nb_delta7: int, optional
            the number of Delta7 neurons (fruit fly name of TB1 neuron). These are connected in the Fan-shaped Body
            through the columnar neurons (PFN).
        nb_tn1: int, optional
            the number of TN1 neurons (locust literature).
        nb_tn2: int, optional
            the number of TN2 neurons (locust literature).
        nb_fbn: int, optional
            the number of Fan-shaped Body neurons (CPU4 in locust literature).
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

        nb_delta7 = kwargs.pop('nb_tb1', nb_delta7)
        nb_fbn = kwargs.pop('nb_cpu4', nb_fbn)
        kwargs.setdefault('nb_pfn', nb_delta7)
        kwargs.setdefault('nb_tangential', nb_tn1 + nb_tn2)
        super().__init__(*args, nb_fbn=nb_fbn, **kwargs)

        self._nb_tn1 = nb_tn1
        self._nb_tn2 = nb_tn2

        self._gain = gain

        self.__mem = .5 * np.ones(self.nb_fbn, dtype=self.dtype)  # cpu4 memory

    def reset(self):
        super().reset()

        self.w_p2f = diagonal_synapses(self.nb_tb1, self.nb_cpu4, fill_value=-1, tile=True, dtype=self.dtype)
        self.w_tn12cpu4 = chessboard_synapses(self.nb_tn1, self.nb_cpu4, nb_rows=2, nb_cols=2, fill_value=1,
                                              dtype=self.dtype)
        self.w_tn22cpu4 = chessboard_synapses(self.nb_tn2, self.nb_cpu4, nb_rows=2, nb_cols=2, fill_value=1,
                                              dtype=self.dtype)

        self.__mem[:] = .5

    def _fprop(self, delta7=None, tb1=None, tn1=None, tn2=None):
        if delta7 is not None:
            tb1 = delta7
        if tb1 is None:
            tb1 = self.r_tb1
        if tn1 is None:
            tn1 = self.r_tn1
        if tn2 is None:
            tn2 = self.r_tn2

        # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
        mem_tn1 = (.5 - tn1).dot(self.w_tn12cpu4)
        mem_tb1 = (tb1 - 1).dot(self.w_p2f)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_tn2 = 0.25 * tn2.dot(self.w_tn22cpu4)

        mem = mem_tn1 * mem_tb1 - mem_tn2

        if self.update:
            cpu4_mem = self.mem_update(mem)
        else:
            cpu4_mem = mem

        # this creates a problem with vector memories
        # cpu4_mem = np.clip(cpu4_mem, 0., 1.)

        self.r_tb1 = tb1
        self.r_tn1 = tn1
        self.r_tn2 = tn2
        self.r_cpu4 = a_cpu4 = self.f_cpu4(cpu4_mem)

        return a_cpu4

    def mem_update(self, mem):
        self.__mem[:] = self.__mem + self.gain * mem
        return self.__mem

    def reset_integrator(self):
        self.__mem[:] = .5

    def decode_vector(self):
        """
        Transforms the CPU4 vector memory to a vector in the Cartesian coordinate system.

        Returns
        -------
        complex
        """
        return decode_vector(self.__mem, self._gain)

    @property
    def w_tn12cpu4(self):
        return self._w_t2f[:self.nb_tn1]

    @w_tn12cpu4.setter
    def w_tn12cpu4(self, v):
        self._w_t2f[:self.nb_tn1] = v

    @property
    def w_tn22cpu4(self):
        return self._w_t2f[self.nb_tn1:self.nb_tn1+self.nb_tn2]

    @w_tn22cpu4.setter
    def w_tn22cpu4(self, v):
        self._w_t2f[self.nb_tn1:self.nb_tn1+self.nb_tn2] = v

    @property
    def f_cpu4(self):
        return self.f_fbn

    @property
    def r_tn1(self):
        """
        The latest responses of the TN1 (part of the tangential) neurons.

        Returns
        -------
        np.ndarray[float]
        """
        return self.r_tangential[:self.nb_tn1]

    @r_tn1.setter
    def r_tn1(self, v):
        self.r_tangential[:self.nb_tn1] = v

    @property
    def r_tn2(self):
        """
        The latest responses of the TN2 (part of the tangential) neurons.

        Returns
        -------
        np.ndarray[float]
        """
        return self.r_tangential[self.nb_tn1:self.nb_tn1+self.nb_tn2]

    @r_tn2.setter
    def r_tn2(self, v):
        self.r_tangential[self.nb_tn1:self.nb_tn1+self.nb_tn2] = v

    @property
    def r_tb1(self):
        """
        The latest responses of the TB1 neurons (same as Delta7).

        Returns
        -------
        np.ndarray[float]
        """
        return self.r_pfn

    @r_tb1.setter
    def r_tb1(self, v):
        self._pfn[:] = v

    @property
    def r_delta7(self):
        """
        The latest responses of the Delta7 neurons (same as TB1).

        Returns
        -------
        np.ndarray[float]
        """
        return self.r_pfn

    @r_delta7.setter
    def r_delta7(self, v):
        self._pfn[:] = v

    @property
    def r_cpu4(self):
        """
        The latest responses of the CPU4 neurons.

        Returns
        -------
        np.ndarray[float]
        """
        return self.r_fbn

    @r_cpu4.setter
    def r_cpu4(self, v):
        self.r_fbn[:] = v

    @property
    def cpu4_mem(self):
        return self.__mem

    @cpu4_mem.setter
    def cpu4_mem(self, v):
        self.__mem[:] = v

    @property
    def nb_delta7(self):
        """
        The number of Delta7 neurons (same as TB1).

        Returns
        -------
        int
        """
        return self.nb_pfn

    @property
    def nb_tb1(self):
        """
        The number of TB1 neurons (same as Delta7).

        Returns
        -------
        int
        """
        return self.nb_pfn

    @property
    def nb_tn1(self):
        """
        The number of TN1 (part of the tangential) neurons.

        Returns
        -------
        int
        """
        return self._nb_tn1

    @property
    def nb_tn2(self):
        """
        The number of TN2 (part of the tangential) neurons.

        Returns
        -------
        int
        """
        return self._nb_tn2

    @property
    def nb_cpu4(self):
        """
        The number of CPU4 units.

        Returns
        -------
        int
        """
        return self.nb_fbn

    @property
    def gain(self):
        return self._gain


class FamiliarityIntegratorLayer(PathIntegratorLayer):
    def __init__(self, nb_mbon=6, *args, **kwargs):

        nb_tn1 = kwargs.get('nb_tn1', 2)
        nb_tn2 = kwargs.get('nb_tn2', 2)
        kwargs.setdefault('nb_tangential', nb_mbon + nb_tn1 + nb_tn2)
        kwargs.setdefault('gain', 1.)
        super().__init__(*args, **kwargs)

        self._nb_mbon = nb_mbon

        self.r_mbon = np.zeros(nb_mbon, dtype=self.dtype)
        self.__mem_vis = np.full_like(self.cpu4_mem, .5)

    def reset(self):
        super().reset()

        self.w_mbon2fbn[0::2] = 2. / self.nb_tangential
        self.w_mbon2fbn[1::2] = -2. / self.nb_tangential

        self.__mem_vis[:] = .5

    def _fprop(self, delta7=None, tb1=None, mbon=None, tn1=None, tn2=None):

        if delta7 is not None:
            tb1 = delta7
        if tb1 is None:
            tb1 = self.r_tb1
        if mbon is None:
            mbon = self.r_mbon
        if tn1 is None:
            tn1 = self.r_tn1
        if tn2 is None:
            tn2 = self.r_tn2

        # decay = 0.25
        decay = 0.7

        # PathIntegratorLayer._fprop(self, tb1=tb1, tn1=tn1, tn2=tn2)

        # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
        mem_tn1 = (.5 - tn1).dot(self.w_tn12cpu4)
        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_tn2 = decay * tn2.dot(self.w_tn22cpu4)

        # MBON contribution
        a_mbon = np.power(mbon, 8)
        mem_mbon = a_mbon.dot(self.w_mbon2fbn)

        # differential integrator
        mem_tb1 = (-tb1).dot(self.w_p2f)

        # mem_pre = self.__mem_vis - 0.5
        mem = mem_tn1 * mem_tb1 - mem_tn2

        # vis_mem = 0.5 + 0.5 * mem_pre + self._gain * mem * mem_mbon
        # vis_mem = self.__mem_vis + 2 * self._gain * mem * mem_mbon
        cpu4_mem = self.cpu4_mem + self._gain * mem_mbon * mem

        if self.update or True:
            # self.__mem_vis = vis_mem
            self.cpu4_mem = cpu4_mem

        self.r_mbon = mbon
        # self.r_cpu4 = a_cpu4 = self.f_cpu4(self.cpu4_mem + self.__mem_vis - 0.5)
        # self.r_cpu4 = a_cpu4 = self.f_cpu4(self.__mem_vis)
        self.r_cpu4 = a_cpu4 = self.f_cpu4(cpu4_mem)

        return a_cpu4

    def reset_integrator(self):
        super().reset_integrator()
        self.__mem_vis[:] = .5

    @property
    def w_mbon2fbn(self):
        return self.w_t2f[self.nb_tn1+self.nb_tn2:self.nb_tn1+self.nb_tn2+self.nb_mbon]

    @w_mbon2fbn.setter
    def w_mbon2fbn(self, v):
        self.w_t2f[self.nb_tn1+self.nb_tn2:self.nb_tn1+self.nb_tn2+self.nb_mbon] = v

    @property
    def r_mbon(self):
        return self._tan[self.nb_tn1+self.nb_tn2:self.nb_tn1+self.nb_tn2+self.nb_mbon]

    @r_mbon.setter
    def r_mbon(self, v):
        self._tan[self.nb_tn1+self.nb_tn2:self.nb_tn1+self.nb_tn2+self.nb_mbon] = v

    @property
    def nb_mbon(self):
        return self._nb_mbon


class VectorMemoryLayer(FanShapedBodyLayer):

    def __init__(self, nb_cpu4=16, nb_vec=1, *args, **kwargs):
        kwargs.setdefault('nb_pfn', nb_cpu4)
        kwargs.setdefault('nb_fbn', nb_cpu4)
        kwargs.setdefault('nb_tangential', nb_vec)
        kwargs.setdefault('learning_rule', custom_learning_rule)
        super().__init__(*args, **kwargs)

        self._w_mem2cpu4 = sinusoidal_synapses(self.nb_cpu4, self.nb_cpu4, dtype=self.dtype,
                                               in_period=self.nb_cpu4//2, out_period=self.nb_cpu4//2)

        # ensures a smooth sinusoidal pattern in the CPU4 neurons
        self._w_mem2cpu4 = self._w_mem2cpu4 / np.sum(self._w_mem2cpu4, axis=1)

        self.f_fbn = lambda x: x

    def reset(self):
        super().reset()

        self.w_vec2cpu4 = .5
        self.w_p2f = np.eye(self.nb_pfn, self.nb_fbn, dtype=self.dtype)

    def _fprop(self, cpu4=None, vec=None, **kwargs):
        cpu4 = kwargs.pop('pfn', cpu4)
        vec = kwargs.pop('tangential', vec)
        if cpu4 is None:
            cpu4 = self.r_cpu4
        if vec is None:
            vec = self.r_vec

        mem = 0.5 - vec.dot(self.w_vec2cpu4) + cpu4.dot(self.w_p2f)

        self.r_fbn = a_fbn = self.f_fbn(mem.dot(self._w_mem2cpu4))
        self.r_vec = vec

        if self.update:
            self.reset_memory()

        return a_fbn

    def reset_memory(self, id=None):
        if id is None:
            vec = self.r_vec
        else:
            vec = np.eye(self.nb_vec)[id]
        self.w_vec2cpu4 = self.update_weights(w_pre=self.w_vec2cpu4, r_pre=vec, r_post=self.r_cpu4, rein=1)

    @property
    def w_vec2cpu4(self):
        return self._w_t2f

    @w_vec2cpu4.setter
    def w_vec2cpu4(self, v):
        self._w_t2f[:] = v

    @property
    def w_mem2cpu4(self):
        """
        The weights that are used in order to load the stored memory.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_mem2cpu4

    @w_mem2cpu4.setter
    def w_mem2cpu4(self, v):
        self._w_mem2cpu4[:] = v[:]

    @property
    def r_cpu4(self):
        return self._fbn

    @r_cpu4.setter
    def r_cpu4(self, v):
        self._fbn[:] = v

    @property
    def r_vec(self):
        return self._tan

    @r_vec.setter
    def r_vec(self, v):
        self._tan[:] = v

    @property
    def nb_vec(self):
        return self.nb_tangential

    @property
    def nb_cpu4(self):
        return self.nb_fbn


class WindGuidedLayer(FanShapedBodyLayer):
    def __init__(self, nb_epg=16, nb_hdc=16, nb_pfl2=16, nb_pfl3=16, nb_nod=2, nb_mbon=6, *args, **kwargs):
        nb_hdc = kwargs.pop('nb_cpu4', nb_hdc)
        kwargs.setdefault('nb_pfn', nb_hdc)
        kwargs.setdefault('nb_fbn', nb_hdc)
        kwargs.setdefault('nb_tangential', 2)
        FanShapedBodyLayer.__init__(self, *args, **kwargs)

        self._w_nod2pfn = chessboard_synapses(nb_nod, self.nb_pfn, nb_rows=2, nb_cols=2, fill_value=1, dtype=self.dtype)
        self._w_nod2pfn = self._w_nod2pfn[::-1]
        self._w_epg2pfn = uniform_synapses(nb_epg, self.nb_pfn, fill_value=0, dtype=self.dtype)
        w_epg2pfn = sinusoidal_synapses(nb_epg, self.nb_pfn, fill_value=1, dtype=self.dtype,
                                        in_period=self.nb_pfn // 2, out_period=self.nb_pfn // 2)
        w_epg2pfn /= np.absolute(w_epg2pfn).sum(axis=1) / 2.
        self._w_epg2pfn[:, :self.nb_pfn//2] = roll_synapses(w_epg2pfn, right=self.nb_pfn // 16)[:, :self.nb_pfn//2]
        self._w_epg2pfn[:, self.nb_pfn//2:] = roll_synapses(w_epg2pfn, left=self.nb_pfn // 16)[:, self.nb_pfn//2:]

        self._w_mbon2tan = uniform_synapses(nb_mbon, self.nb_tangential, fill_value=0, dtype=self.dtype)
        self._w_mbon2tan[0::2, 0::2] = self.nb_tangential / nb_mbon
        self._w_mbon2tan[1::2, 1::2] = self.nb_tangential / nb_mbon

        self.w_tan2hdc = uniform_synapses(self.nb_tangential, nb_hdc, fill_value=1, dtype=self.dtype)
        self.w_tan2hdc[1::2] *= -1

        self.w_pfn2hdc = diagonal_synapses(self.nb_pfn, nb_hdc, fill_value=1, dtype=self.dtype)
        self.w_pfn2hdc = roll_synapses(self.w_pfn2hdc, right=self.nb_pfn//4)

        self._w_epg2pfl3 = uniform_synapses(nb_epg, nb_pfl3, fill_value=0, dtype=self.dtype)
        w_epg2pfl3 = sinusoidal_synapses(nb_epg, nb_pfl3, fill_value=1, dtype=self.dtype,
                                         in_period=nb_pfl3 // 2, out_period=nb_pfl3 // 2)
        self._w_epg2pfl3[:, :nb_pfl3//2] = roll_synapses(w_epg2pfl3, left=nb_pfl3 // 8)[:, :nb_pfl3//2]
        self._w_epg2pfl3[:, nb_pfl3//2:] = roll_synapses(w_epg2pfl3, right=nb_pfl3 // 8)[:, nb_pfl3//2:]

        self._w_epg2pfl2 = uniform_synapses(nb_epg, nb_pfl2, fill_value=0, dtype=self.dtype)
        w_epg2pfl2 = sinusoidal_synapses(nb_epg, nb_pfl2, fill_value=1, dtype=self.dtype,
                                         in_period=nb_pfl2 // 2, out_period=nb_pfl2 // 2)
        self._w_epg2pfl2[:, :nb_pfl2//2] = roll_synapses(w_epg2pfl2, left=nb_pfl2 // 4)[:, :nb_pfl2//2]
        self._w_epg2pfl2[:, nb_pfl2//2:] = roll_synapses(w_epg2pfl2, right=nb_pfl2 // 4)[:, nb_pfl2//2:]

        self._w_pfn2pfl3 = diagonal_synapses(self.nb_pfn, nb_pfl3, fill_value=1, tile=True, dtype=self.dtype)
        self._w_pfn2pfl3 += roll_synapses(self._w_pfn2pfl3, left=self.nb_pfn // 2)
        self._w_pfn2pfl2 = diagonal_synapses(self.nb_pfn, nb_pfl2, fill_value=1, tile=True, dtype=self.dtype)
        self._w_pfn2pfl2 += roll_synapses(self._w_pfn2pfl2, left=self.nb_pfn // 2)

        self._w_hdc2pfl3 = diagonal_synapses(nb_hdc, nb_pfl3, fill_value=1, tile=True, dtype=self.dtype)
        self._w_hdc2pfl3 += roll_synapses(self._w_hdc2pfl3, left=self.nb_pfn // 2)
        self._w_hdc2pfl2 = diagonal_synapses(nb_hdc, nb_pfl2, fill_value=1, tile=True, dtype=self.dtype)
        self._w_hdc2pfl2 += roll_synapses(self._w_hdc2pfl2, left=self.nb_pfn // 2)

        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure("wind-guided-weights", figsize=(8, 8))
        # axes_dict = fig.subplot_mosaic(
        #     """
        #     ABCD
        #     EFGH
        #     IJKL
        #     """
        # )
        #
        # axes_dict["A"].set_title("E-PG>P-FN")
        # axes_dict["A"].imshow(self.w_epg2pfn.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["B"].set_title("E-PG>P-FL3")
        # axes_dict["B"].imshow(self.w_epg2pfl3.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["C"].set_title("E-PG>P-FL2")
        # axes_dict["C"].imshow(self.w_epg2pfl2.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        # axes_dict["C"].sharey(axes_dict["B"])
        #
        # axes_dict["D"].set_title("P-FN>h-ΔC")
        # axes_dict["D"].imshow(self.w_pfn2hdc.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["E"].set_title("TAN>h-ΔC")
        # axes_dict["E"].imshow(self.w_tan2hdc.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["F"].set_title("MBON>TAN")
        # axes_dict["F"].imshow(self.w_mbon2tan.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["G"].set_title("TN1>P-FN")
        # axes_dict["G"].imshow(self.w_tn12pfn.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["H"].set_title("TN2>P-FN")
        # axes_dict["H"].imshow(self.w_tn22pfn.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["I"].set_title("P-FN>P-FL3")
        # axes_dict["I"].imshow(self.w_pfn2pfl3.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["J"].set_title("P-FN>P-FL2")
        # axes_dict["J"].imshow(self.w_pfn2pfl2.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["K"].set_title("h-ΔC>P-FL3")
        # axes_dict["K"].imshow(self.w_hdc2pfl3.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # axes_dict["L"].set_title("h-ΔC>P-FL2")
        # axes_dict["L"].imshow(self.w_hdc2pfl2.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        #
        # plt.tight_layout()
        # plt.show()

        self._nb_epg = nb_epg
        self._nb_pfl2 = nb_pfl2
        self._nb_pfl3 = nb_pfl3
        self._nb_nod = nb_nod
        self._nb_mbon = nb_mbon

        self._b = 0.5
        self._slope = 0.5

        self.params.extend([
            self._w_t2f,
            self._w_p2f
        ])

        self._epg = np.zeros(self.nb_epg, dtype=self.dtype)
        self._nod = np.zeros(self.nb_nod, dtype=self.dtype)
        self._mbon = np.zeros(self.nb_mbon, dtype=self.dtype)
        self._pfl3 = np.zeros(self.nb_pfl3, dtype=self.dtype)
        self._pfl2 = np.zeros(self.nb_pfl2, dtype=self.dtype)

        self.f_epg = lambda v: relu(v, cmin=0, cmax=1, noise=self._noise, rng=self.rng)
        self.f_pfn = lambda v: relu(v, cmin=0, cmax=1, noise=self._noise, rng=self.rng)
        self.f_hdc = lambda v: relu(v, cmin=0, cmax=1, noise=self._noise, rng=self.rng)
        self.f_pfl3 = lambda v: relu(v, cmin=0, cmax=1, noise=self._noise, rng=self.rng)
        self.f_pfl2 = lambda v: relu(v, cmin=0, cmax=1, noise=self._noise, rng=self.rng)
        self.f_nod = lambda v: relu(v / np.linalg.norm(v), noise=self._noise, rng=self.rng)
        self.f_tan = lambda v: relu(v, cmin=0, cmax=1, noise=self._noise, rng=self.rng)
        self.f_mbon = lambda v: relu(v, cmin=0, cmax=1, noise=self._noise, rng=self.rng)

        self._front = lambda x: self._slope * x + self._b  # transform into [0, 1] signal
        self._back = lambda x: (x - self._b) / self._slope  # transform into [-1, 1] signal
        self._norm = lambda x: x / np.max(np.absolute(x))  # normalise the signal

    def _fprop(self, epg=None, nod=None, mbon=None):

        f, b, n = self._front, self._back, self._norm
        fn = lambda x: f(n(x))

        self.r_nod = a_nod = self.f_nod(nod)

        self.r_mbon = a_mbon = self.f_mbon(mbon)
        self.r_tan = a_tan = self.f_tan(a_mbon.dot(self.w_mbon2tan))

        self.r_epg = a_epg = epg
        b_epg = b(a_epg)

        b_pfn = a_nod.dot(self.w_nod2pfn) * b_epg.dot(self.w_epg2pfn)
        self.r_pfn = a_pfn = self.f_pfn(f(b_pfn))

        b_hdc = b_pfn.dot(self.w_pfn2hdc)
        self.r_hdc = a_hdc = self.f_hdc(f(b_hdc))

        p = np.clip((a_tan.dot(self.w_tan2hdc) + 1) / 2, 0, 1)

        self.r_pfl3 = a_pfl3 = self.f_pfl3(fn(
            b_epg.dot(self.w_epg2pfl3) +
            b_pfn.dot(self.w_pfn2pfl3) * (1 - p) +
            b_hdc.dot(self.w_hdc2pfl3) * p))
        self.r_pfl2 = a_pfl2 = self.f_pfl2(fn(
            b_epg.dot(self.w_epg2pfl2) +
            b_pfn.dot(self.w_pfn2pfl2) * (1 - p) +
            b_hdc.dot(self.w_hdc2pfl2) * p))

        return a_pfl3

    @property
    def w_epg2pfn(self):
        return self._w_epg2pfn

    @w_epg2pfn.setter
    def w_epg2pfn(self, value):
        self._w_epg2pfn[:] = value

    @property
    def w_epg2pfl3(self):
        return self._w_epg2pfl3

    @w_epg2pfl3.setter
    def w_epg2pfl3(self, value):
        self._w_epg2pfl3[:] = value

    @property
    def w_epg2pfl2(self):
        return self._w_epg2pfl2

    @w_epg2pfl2.setter
    def w_epg2pfl2(self, value):
        self._w_epg2pfl2[:] = value

    @property
    def w_mbon2tan(self):
        return self._w_mbon2tan

    @w_mbon2tan.setter
    def w_mbon2tan(self, value):
        self._w_mbon2tan[:] = value

    @property
    def w_tan2hdc(self):
        return self._w_t2f

    @w_tan2hdc.setter
    def w_tan2hdc(self, value):
        self._w_t2f[:] = value

    @property
    def w_nod2pfn(self):
        return self._w_nod2pfn

    @w_nod2pfn.setter
    def w_nod2pfn(self, value):
        self._w_nod2pfn[:] = value

    @property
    def w_pfn2hdc(self):
        return self._w_p2f

    @w_pfn2hdc.setter
    def w_pfn2hdc(self, value):
        self._w_p2f[:] = value

    @property
    def w_pfn2pfl3(self):
        return self._w_pfn2pfl3

    @w_pfn2pfl3.setter
    def w_pfn2pfl3(self, value):
        self._w_pfn2pfl3[:] = value

    @property
    def w_pfn2pfl2(self):
        return self._w_pfn2pfl2

    @w_pfn2pfl2.setter
    def w_pfn2pfl2(self, value):
        self._w_pfn2pfl2[:] = value

    @property
    def w_hdc2pfl3(self):
        return self._w_hdc2pfl3

    @w_hdc2pfl3.setter
    def w_hdc2pfl3(self, value):
        self._w_hdc2pfl3[:] = value

    @property
    def w_hdc2pfl2(self):
        return self._w_hdc2pfl2

    @w_hdc2pfl2.setter
    def w_hdc2pfl2(self, value):
        self._w_hdc2pfl2[:] = value

    @property
    def r_hdc(self):
        return self._fbn

    @r_hdc.setter
    def r_hdc(self, value):
        self._fbn[:] = value

    @property
    def r_epg(self):
        return self._epg

    @r_epg.setter
    def r_epg(self, value):
        self._epg[:] = value

    @property
    def r_cpu4(self):
        return self.r_hdc

    @property
    def r_nod(self):
        return self._nod

    @r_nod.setter
    def r_nod(self, value):
        self._nod[:] = value

    @property
    def r_mbon(self):
        return self._mbon

    @r_mbon.setter
    def r_mbon(self, value):
        self._mbon[:] = value

    @property
    def r_pfl3(self):
        return self._pfl3

    @r_pfl3.setter
    def r_pfl3(self, value):
        self._pfl3[:] = value

    @property
    def r_pfl2(self):
        return self._pfl2

    @r_pfl2.setter
    def r_pfl2(self, value):
        self._pfl2[:] = value

    @property
    def r_tn1(self):
        return self.r_nod

    @property
    def r_tn2(self):
        return self.r_nod

    @property
    def cpu4_mem(self):
        return self.r_pfn

    @property
    def nb_epg(self):
        return self._nb_epg

    @property
    def nb_nod(self):
        return self._nb_nod

    @property
    def nb_mbon(self):
        return self._nb_mbon

    @property
    def nb_cpu4(self):
        return self._nb_epg

    @property
    def nb_pfl3(self):
        return self._nb_pfl3

    @property
    def nb_pfl2(self):
        return self._nb_pfl2


def custom_learning_rule(w_pre, r_pre, r_post, rein, learning_rate, w_rest=0.):
    i = np.argmax(rein)

    d_w = learning_rate * (r_post - r_pre[i] * w_pre[i])
    w_pre[i] += d_w

    return w_pre
