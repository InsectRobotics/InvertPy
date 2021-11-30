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


class CentralComplexBase(Component):
    def __init__(self, nb_compass=8, nb_memory=16, nb_steering=16, *args, **kwargs):
        kwargs.setdefault('nb_input', nb_compass)
        kwargs.setdefault("nb_output", 2)
        kwargs.setdefault('learning_rule', None)
        super().__init__(*args, **kwargs)

        self._nb_compass = nb_compass  # TB1 / E-PG
        self._nb_memory = nb_memory  # CPU4 / FBN
        self._nb_steering = nb_steering  # CPU1 / PLF3

        self._w_c2c = uniform_synapses(self.nb_compass, self.nb_compass, fill_value=0, dtype=self.dtype)
        self._w_c2m = uniform_synapses(self.nb_compass, self.nb_memory, fill_value=0, dtype=self.dtype)
        self._w_c2s = uniform_synapses(self.nb_compass, self.nb_steering, fill_value=0, dtype=self.dtype)
        self._w_m2s = uniform_synapses(self.nb_memory, self.nb_steering, fill_value=0, dtype=self.dtype)
        self._w_s2o = uniform_synapses(self.nb_steering, self._nb_output, fill_value=0, dtype=self.dtype)

        self._com = np.zeros(self._nb_compass, dtype=self.dtype)
        self._mem = np.zeros(self._nb_memory, dtype=self.dtype)
        self._ste = np.zeros(self._nb_steering, dtype=self.dtype)
        self._out = np.zeros(self._nb_output, dtype=self.dtype)

        # The cell properties (for sigmoid function)
        self._com_slope = 5.0
        self._mem_slope = 5.0
        self._ste_slope = 5.0  # 7.5
        self._out_slope = 1.0

        self._b_com = 0.0
        self._b_mem = 2.5
        self._b_ste = 2.5  # -1.0
        self._b_out = 3.0

        self.params.extend([
            self._w_c2c,
            self._w_c2m,
            self._w_c2s,
            self._w_m2s,
            self._w_s2o,
            self._b_com,
            self._b_mem,
            self._b_ste,
            self._b_out
        ])

        self.f_com = lambda v: sigmoid(v * self._com_slope - self._b_com, noise=self._noise, rng=self.rng)
        self.f_mem = lambda v: sigmoid(v * self._mem_slope - self._b_mem, noise=self._noise, rng=self.rng)
        self.f_ste = lambda v: sigmoid(v * self._ste_slope - self._b_ste, noise=self._noise, rng=self.rng)
        self.f_out = lambda v: sigmoid(v * self._out_slope - self._b_out, noise=self._noise, rng=self.rng)

    def reset(self):

        self._w_c2c[:] = sinusoidal_synapses(self.nb_compass, self.nb_compass, fill_value=-1, dtype=self.dtype)
        self._w_c2m[:] = diagonal_synapses(self.nb_compass, self.nb_memory, fill_value=-1, tile=True, dtype=self.dtype)
        self._w_c2s[:] = diagonal_synapses(self.nb_compass, self.nb_steering, fill_value=-1, tile=True)
        self._w_m2s[:] = opposing_synapses(self.nb_memory, self.nb_steering, fill_value=1, dtype=self.dtype)

        self._w_s2o = chessboard_synapses(self.nb_steering, self._nb_output,
                                          nb_rows=2, nb_cols=2, fill_value=1, dtype=self.dtype)

        self._com[:] = 0.
        self._mem[:] = 0.
        self._ste[:] = 0.
        self._out[:] = 0.

        self.update = True

    @property
    def w_c2c(self):
        return self._w_c2c

    @property
    def w_c2m(self):
        return self._w_c2m

    @property
    def w_c2s(self):
        return self._w_c2s

    @property
    def w_m2s(self):
        return self._w_m2s

    @property
    def w_s2o(self):
        return self._w_s2o

    @property
    def r_compass(self):
        return self._com

    @property
    def r_memory(self):
        return self._mem

    @property
    def r_steering(self):
        return self._ste

    @property
    def nb_compass(self):
        return self._nb_compass

    @property
    def nb_memory(self):
        return self._nb_memory

    @property
    def nb_steering(self):
        return self._nb_steering
