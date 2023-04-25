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
from invertpy.brain.synapses import chessboard_synapses

from abc import ABC

import numpy as np
import os


# get path of the script
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


class CentralComplexBase(Component, ABC):
    def __init__(self, nb_compass=8, nb_steering=16, *args, **kwargs):
        kwargs.setdefault('nb_input', nb_compass)
        kwargs.setdefault("nb_output", nb_steering)
        kwargs.setdefault('learning_rule', None)
        super().__init__(*args, **kwargs)

        self._nb_com = nb_compass
        self._nb_ste = nb_steering

        self.__layers = dict()

        w = chessboard_synapses(self._nb_ste, 2, nb_rows=2, nb_cols=2, fill_value=1, dtype=self.dtype)
        a = np.vstack([w[:14//2], w[-14//2:]])
        b = w[[-14//2-1, 14//2]]
        self._w_s2o = np.vstack([b[-1:], a, b[:1]])

    def reset(self):
        for name, layer in self.__layers.items():
            layer.reset()

        self.update = True

    def __getitem__(self, layer_name):
        """
        Gets a layer given the name.

        Parameters
        ----------
        layer_name : str

        Returns
        -------
        CentralComplexLayer
        """
        return self.__layers[layer_name]

    def __setitem__(self, layer_name, layer):
        """
        Sets a layer with the specified name.

        Parameters
        ----------
        layer_name : str
        layer : CentralComplexLayer
        """
        self.__layers[layer_name] = layer

    def __repr__(self):
        return f"CentralComplex(compass={self.nb_compass}, steering={self.nb_steering})"

    @property
    def w_steering2motor(self):
        return self._w_s2o

    @property
    def r_compass(self):
        raise NotImplementedError()

    @property
    def r_steering(self):
        raise NotImplementedError()

    @property
    def r_motor(self):
        return self.r_steering.dot(self.w_steering2motor)

    @property
    def nb_compass(self):
        return self._nb_com

    @property
    def nb_steering(self):
        return self._nb_ste


class CentralComplexLayer(Component, ABC):
    pass
