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

from .stone import StoneCX

import numpy as np


class PisokasCX(StoneCX):

    def __init__(self, nb_delta7=None, nb_epg=None, nb_peg=None, nb_pen=None, structure="fly", *args, **kwargs):
        """
        The Central Complex model of [1]_ as a component of the fruit fly or locust brain.

        Note: This code has not been tested!

        Parameters
        ----------
        nb_delta7: int, optional
            the number of Delta7 neurons. Default is 8 for locust, 9 for fruit fly
        nb_epg: int, optional
            the number of E-PG neurons. Default is 16 for locust, 18 for fruit fly
        nb_peg: int, optional
            the number of P-EG neurons. Default is 16 for locust, 18 for fruit fly
        nb_pen: int, optional
            the number of P-EN neurons. Default is 16 for locust, 18 for fruit fly
        structure: {"fly", "locust}
            the species of the animal. Default is "fly"

        Notes
        -----
        .. [1] Pisokas, I., Heinze, S. & Webb, B. The head direction circuit of two insect species.
               Elife 9, e53985 (2020).
        """

        if nb_delta7 is None:
            nb_delta7 = 8
        if nb_pen is None:
            nb_pen = 16
        if "fly" in structure:
            if nb_epg is None:
                nb_epg = 18
            if nb_peg is None:
                nb_peg = 18
        else:
            if nb_epg is None:
                nb_epg = 16
            if nb_peg is None:
                nb_peg = 16

        kwargs.setdefault('nb_tb1', nb_delta7)
        kwargs.setdefault('nb_cl1', nb_epg)
        kwargs.setdefault('nb_tl2', nb_epg)
        kwargs.setdefault('nb_cpu4', nb_epg)
        super().__init__(*args, **kwargs)

        self._nb_peg = nb_peg
        self._nb_pen = nb_pen

        # initialise the responses of the neurons
        self._peg = np.zeros(self.nb_peg)
        self._pen = np.zeros(self.nb_pen)

        # Weight matrices based on anatomy (These are not changeable!)
        self._w_delta72peg = uniform_synapses(self.nb_delta7, self.nb_peg, fill_value=0, dtype=self.dtype)
        self._w_delta72pen = uniform_synapses(self.nb_delta7, self.nb_pen, fill_value=0, dtype=self.dtype)

        self._w_epg2peg = uniform_synapses(self.nb_epg, self.nb_peg, fill_value=0, dtype=self.dtype)
        self._w_epg2pen = uniform_synapses(self.nb_epg, self.nb_pen, fill_value=0, dtype=self.dtype)

        self._w_peg2epg = uniform_synapses(self.nb_peg, self.nb_epg, fill_value=0, dtype=self.dtype)

        self._w_pen2epg = uniform_synapses(self.nb_pen, self.nb_epg, fill_value=0, dtype=self.dtype)

        # The cell properties (for sigmoid function)
        self._epg_slope = 5.0
        self._peg_slope = 5.0
        self._pen_slope = 5.0

        self._b_epg = 3.0
        self._b_peg = 3.0
        self._b_pen = 3.0

        self.params.extend([
            self._w_delta72peg,
            self._w_delta72pen,
            self._w_epg2peg,
            self._w_epg2pen,
            self._w_peg2epg,
            self._w_pen2epg,
            self._b_peg,
            self._b_pen,
        ])

        self._structure = structure

        self.f_epg = self.f_cl1
        self.f_peg = lambda v: sigmoid(v * self._peg_slope - self.b_peg, noise=self._noise, rng=self.rng)
        self.f_pen = lambda v: sigmoid(v * self._pen_slope - self.b_pen, noise=self._noise, rng=self.rng)

        if self.__class__ == PisokasCX:
            self.reset()

    def reset(self):
        super().reset()

        # Weight matrices based on anatomy (These are not changeable!)
        self.w_delta72delta7 = sinusoidal_synapses(self.nb_delta7, self.nb_delta7, fill_value=1, dtype=self.dtype) - 1
        self.w_delta72peg = diagonal_synapses(self.nb_delta7, self.nb_peg, fill_value=-1, tile=True, dtype=self.dtype)
        self.w_delta72pen = diagonal_synapses(self.nb_delta7, self.nb_pen, fill_value=-1, tile=True, dtype=self.dtype)

        self.w_epg2delta7 = 1 - sinusoidal_synapses(self.nb_epg, self.nb_delta7, in_period=self.nb_delta7,
                                                    fill_value=1, dtype=self.dtype)
        self.w_epg2peg = diagonal_synapses(self.nb_epg, self.nb_peg, fill_value=1., dtype=self.dtype)
        self.w_epg2pen = diagonal_synapses(self.nb_epg, self.nb_pen, fill_value=1., dtype=self.dtype)

        self.w_peg2epg = diagonal_synapses(self.nb_peg, self.nb_epg, fill_value=0.2, dtype=self.dtype)

        if "locust" in self._structure:
            self.w_pen2epg = diagonal_synapses(self.nb_pen, self.nb_epg, fill_value=0.2, dtype=self.dtype)
            self.w_pen2epg[self.nb_pen//2:, :self.nb_epg//2] += roll_synapses(
                self.w_pen2epg, right=self.nb_epg // 2-1)[self.nb_pen//2:, :self.nb_epg//2]
            self.w_pen2epg[:self.nb_pen//2, self.nb_epg//2:] += roll_synapses(
                self.w_pen2epg, left=self.nb_epg // 2-1)[:self.nb_pen//2, self.nb_epg//2:]

            if "global" in self._structure:
                self.w_delta72delta7 = -0.5 * np.sqrt(-self.w_delta72delta7)
                self.w_epg2delta7 = np.sqrt(self.w_epg2delta7)

                self.w_epg2peg *= 0.2
                self.w_epg2pen *= 0.2
                self.w_delta72peg *= 0.5
                self.w_delta72pen *= 0.5

                self.w_peg2epg[self.nb_peg//2:, :self.nb_epg//2] += roll_synapses(
                    self.w_peg2epg, right=self.nb_epg // 2-1)[self.nb_peg//2:, :self.nb_epg//2]
                self.w_peg2epg[:self.nb_peg//2, self.nb_epg//2:] += roll_synapses(
                    self.w_peg2epg, left=self.nb_epg // 2-1)[:self.nb_peg//2, self.nb_epg//2:]
            else:
                self.w_delta72delta7 = -0.5 * np.sqrt(-np.minimum(self.w_delta72delta7 + .2, 0))
                self.w_epg2delta7 = np.square(self.w_epg2delta7)

                self.w_epg2peg[self.nb_epg//2-1, self.nb_peg//2] = 1
                self.w_epg2peg[self.nb_epg//2, self.nb_peg//2-1] = 1
                self.w_epg2pen[self.nb_epg//2-1, self.nb_pen//2] = 1
                self.w_epg2pen[self.nb_epg//2, self.nb_pen//2-1] = 1

            # self.w_delta72delta7 = -(self.w_delta72delta7.T / np.sum(self.w_delta72delta7, axis=1)).T
            # self.w_epg2delta7 = (self.w_epg2delta7.T / np.sum(self.w_epg2delta7, axis=1)).T

            # self.w_delta72delta7[self.w_delta72delta7 < 0.5] = 0
        elif "fly" in self._structure:
            if "global" in self._structure:
                self.w_delta72delta7[~np.eye(self.nb_delta7, self.nb_delta7, dtype=bool)] = -.5
                self.w_epg2delta7[~np.isclose(self.w_epg2delta7, 0)] = .5
            else:
                self.w_delta72delta7 = np.sqrt(np.maximum(-self.w_delta72delta7 - .2, 0))
                self.w_epg2delta7 = np.power(self.w_epg2delta7, 4)

            self.w_delta72delta7 = -(self.w_delta72delta7.T / np.sum(self.w_delta72delta7, axis=1)).T
            self.w_epg2delta7 = (self.w_epg2delta7.T / np.sum(self.w_epg2delta7, axis=1)).T

            self.w_delta72delta7[np.eye(self.nb_delta7, self.nb_delta7, dtype=bool)] = 0
            self.w_delta72pen[:, self.nb_pen//2:] = roll_synapses(self.w_delta72pen[:, self.nb_pen//2:], left=2)

            self.w_epg2pen[:, self.nb_pen//2:] = roll_synapses(self.w_epg2pen[:, self.nb_pen//2:], down=2)

            self.w_peg2epg += roll_synapses(self.w_peg2epg, left=self.nb_epg // 2)
            self.w_peg2epg[self.nb_epg//2:, 0] = np.roll(self.w_peg2epg[self.nb_epg//2:, 0], shift=-1)
            self.w_peg2epg[:self.nb_epg//2, -1] = np.roll(self.w_peg2epg[:self.nb_epg//2, -1], shift=1)
            self.w_peg2epg[[0, self.nb_peg//2], [self.nb_epg//2-1, self.nb_epg//2-1]] = 0.2
            self.w_peg2epg[[-1, self.nb_peg//2-1], [self.nb_epg//2, self.nb_epg//2]] = 0.2

            self.w_pen2epg[:, 1:-1] = diagonal_synapses(self.nb_pen, self.nb_epg - 2, fill_value=0.2, dtype=self.dtype)
            self.w_pen2epg += roll_synapses(self.w_pen2epg, left=self.nb_epg // 2)
            self.w_pen2epg[[self.nb_pen//2-1, self.nb_pen//2-1], [0, self.nb_epg//2]] = 0.2
            self.w_pen2epg[[self.nb_pen//2, self.nb_pen//2], [-1, self.nb_epg//2-1]] = 0.2

        import matplotlib.pyplot as plt

        fig = plt.figure("pisokas-weights", figsize=(8, 8))
        axes_dict = fig.subplot_mosaic(
            """
            ....BBC
            ....BBC
            ....EEF
            ....EEF
            AADD...
            AADD...
            ....GGH
            """
        )

        axes_dict["A"].set_title("P-EN>E-PG")
        axes_dict["A"].imshow(self.w_pen2epg.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

        axes_dict["B"].set_title("E-PG>P-EN")
        axes_dict["B"].imshow(self.w_epg2pen.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

        axes_dict["C"].set_title("Δ7>P-EN")
        axes_dict["C"].imshow(self.w_delta72pen.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes_dict["C"].sharey(axes_dict["B"])

        axes_dict["D"].set_title("P-EG>E-PG")
        axes_dict["D"].imshow(self.w_peg2epg.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes_dict["D"].sharey(axes_dict["A"])

        axes_dict["E"].set_title("E-PG>P-EG")
        axes_dict["E"].imshow(self.w_epg2peg.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes_dict["E"].sharex(axes_dict["B"])

        axes_dict["F"].set_title("Δ7>P-EG")
        axes_dict["F"].imshow(self.w_delta72peg.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes_dict["F"].sharey(axes_dict["E"])
        axes_dict["F"].sharex(axes_dict["C"])

        axes_dict["G"].set_title("E-PG>Δ7")
        axes_dict["G"].imshow(self.w_epg2delta7.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes_dict["G"].sharex(axes_dict["E"])

        axes_dict["H"].set_title("Δ7>Δ7")
        axes_dict["H"].imshow(self.w_delta72delta7.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes_dict["H"].sharey(axes_dict["G"])
        axes_dict["H"].sharex(axes_dict["F"])

        plt.tight_layout()
        plt.show()

        self.r_delta7 = np.zeros(self.nb_delta7)
        self.r_epg = np.zeros(self.nb_epg)
        self.r_peg = np.zeros(self.nb_peg)
        self.r_pen = np.zeros(self.nb_pen)

    def update_compass(self, phi, tl2=None, epg=None):
        if isinstance(phi, np.ndarray) and phi.size == 8:
            if tl2 is None:
                tl2 = np.tile(phi, 2)
            if epg is None:
                epg = np.tile(phi, 2)
            a_tl2 = self.f_tl2(tl2[::-1])
            a_epg = self.f_epg(epg[::-1])
            a_delta7 = self.f_com(5. * phi[::-1])
        else:
            a_tl2 = self.f_tl2(self.phi2tl2(phi))
            a_epg = self.f_epg(a_tl2.dot(self.w_tl22cl1))
            if self._com is None:
                a_delta7 = self.f_com(a_epg.dot(self.w_epg2delta7))
            else:
                a_delta7 = self.f_com(a_epg.dot(self.w_epg2delta7) + self.r_delta7.dot(self.w_delta72delta7))

        a_peg = self.f_peg(a_epg.dot(self.w_epg2peg) + a_delta7.dot(self.w_delta72peg))
        a_pen = self.f_pen(a_epg.dot(self.w_epg2pen) + a_delta7.dot(self.w_delta72pen))

        return a_tl2, a_peg, a_pen, a_epg, a_delta7

    def update_memory(self, delta7=None, *args, **kwargs):
        return super().update_memory(tb1=delta7, *args, **kwargs)

    def update_steering(self, cpu4=None, delta7=None):
        return super().update_steering(cpu4, tb1=delta7)

    def __repr__(self):
        return f"PisokasCX(" \
               f"Δ7/TB1={self.nb_delta7:d}, " \
               f"E-PG={self.nb_epg:d}, " \
               f"P-EG={self.nb_peg:d}, " \
               f"P-EN={self.nb_pen:d}, " \
               f"CPU4={self.nb_cpu4:d}, " \
               f"CPU1={self.nb_cpu1:d}, " \
               f"TN1={self.nb_tn1:d}, " \
               f"TN2={self.nb_tn2:d})"

    @property
    def w_delta72delta7(self):
        """
        The Delta7 to Delta7 synaptic weights.
        """
        return self.w_tb12tb1

    @w_delta72delta7.setter
    def w_delta72delta7(self, v):
        self.w_tb12tb1 = v

    @property
    def w_delta72peg(self):
        """
        The Delta7 to P-EG synaptic weights.
        """
        return self._w_delta72peg

    @w_delta72peg.setter
    def w_delta72peg(self, v):
        self._w_delta72peg[:] = v[:]

    @property
    def w_delta72pen(self):
        """
        The Delta7 to P-EN synaptic weights.
        """
        return self._w_delta72pen

    @w_delta72pen.setter
    def w_delta72pen(self, v):
        self._w_delta72pen[:] = v[:]

    @property
    def w_epg2delta7(self):
        """
        The E-PG to Delta7 synaptic weights.
        """
        return self.w_cl12tb1

    @w_epg2delta7.setter
    def w_epg2delta7(self, v):
        self.w_cl12tb1 = v

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
        The TB1 to CPU1b synaptic weights.
        """
        return self._w_pen2epg

    @w_pen2epg.setter
    def w_pen2epg(self, v):
        self._w_pen2epg[:] = v[:]

    @property
    def _epg_slope(self):
        return self._cl1_slope

    @_epg_slope.setter
    def _epg_slope(self, v):
        self._cl1_slope = v

    @property
    def _b_epg(self):
        return self._b_cl1

    @_b_epg.setter
    def _b_epg(self, v):
        self._b_cl1 = v

    @property
    def b_epg(self):
        """
        The E-PG rest response rate (bias).
        """
        return self._b_epg

    @property
    def b_peg(self):
        """
        The P-EG rest response rate (bias).
        """
        return self._b_peg

    @property
    def b_pen(self):
        """
        The P-EN rest response rate (bias).
        """
        return self._b_pen

    @property
    def b_delta7(self):
        """
        The Delta7 rest response rate (bias).
        """
        return self._b_com

    @property
    def r_delta7(self):
        """
        The Delta7 response rate.
        """
        return self.r_tb1

    @r_delta7.setter
    def r_delta7(self, v):
        self.r_tb1 = v

    @property
    def nb_delta7(self):
        """
        The number Delta7 neurons.
        """
        return self._nb_compass

    @property
    def r_epg(self):
        """
        The E-PG response rate.
        """
        return self.r_cl1

    @r_epg.setter
    def r_epg(self, v):
        self.r_cl1 = v

    @property
    def nb_epg(self):
        """
        The number E-PG neurons.
        """
        return self._nb_cl1

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
