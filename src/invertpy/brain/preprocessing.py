"""
Package that contains processing steps of the input signal before it gets into the core components of the brain, e.g.
mushroom body and central complex. These preprocessing components emulate the function of the optic or antennal lobes
or any other pathway that connects sensory input with the two complex structures of the insect brain.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from abc import ABC

from .component import Component
from .synapses import whitening_synapses, dct_synapses
from .activation import softmax
from ._helpers import whitening, pca, eps

import numpy as np


class Preprocessing(Component, ABC):
    def __init__(self, nb_input, nb_output=None, *args, **kwargs):
        """
        A preprocessing component creates simple components without repeats or online learning.

        Parameters
        ----------
        nb_input: int
            The number of input units of the preprocessing component.
        nb_output: int, optional
            The number of output units of the preprocessing component. Default is the same as the input
        """
        if nb_output is None:
            nb_output = nb_input
        super().__init__(nb_input=nb_input, nb_output=nb_output, nb_repeats=1, repeat_rate=1, *args, **kwargs)

    def __repr__(self):
        return "Preprocessing(in=%d, out=%d)" % (self._nb_input, self._nb_output)


class Whitening(Preprocessing):

    def __init__(self, *args, samples=None, w_method=pca, **kwargs):
        """
        A component that implements the whitening preprocessing. This component needs training during the reset method.
        It transform the sample data into whitening parameters using a whitening method (default is PCA).

        Parameters
        ----------
        samples: np.ndarray[float], optional
            the samples from which the whitening synaptic weights will be created. Default is None
        w_method: callable, optional
            the whitening type. Default is PCA
        """
        if samples is not None:
            kwargs.setdefault('nb_input', samples.shape[-1])
        super().__init__(*args, **kwargs)

        self._w_white = None
        """
        Whitening synaptic weights.
        """
        self._m_white = None
        """
        Whitening mean.
        """
        self._f_white = lambda x: softmax((x - x.min()) / (x.max() - x.min() + eps), tau=.2, axis=0)
        """
        Activation function after whitening.
        """
        self._w_method = w_method
        """
        The whitening method.
        """

        self._is_calibrated = None
        """
        Indicates if the calibration process has been completed.
        """
        print(self._nb_input, self._nb_output)
        self.reset(samples=samples)
        self.params.extend([self._w_white, self._m_white])

    def reset(self, samples=None):
        """
        Resets the whitening parameters. If samples are provided, it calibrates the whitening parameters
        otherwise they are the unit parameters.

        Parameters
        ----------
        samples: np.ndarray[float], optional
            the samples from which the whitening synaptic weights will be created. Default is None
        """

        if samples is None:
            w = np.eye(self._nb_input, self._nb_output, dtype=self.dtype)
            m = np.zeros(self._nb_input, dtype=self.dtype)
            self._is_calibrated = False
        else:
            w, m = whitening_synapses(samples, w_func=self._w_method, dtype=self.dtype, bias=True)
            self._is_calibrated = True

        if self._w_white is None or self._m_white is None:
            self._w_white, self._m_white = w, m
        else:
            self._w_white[:], self._m_white[:] = w[:], m[:]

    def _fprop(self, x):
        """
        Whitens the input signal.

        Parameters
        ----------
        x: np.ndarray[float]
            the raw signal that needs to be whitened

        Returns
        -------
        np.ndarray[float]
            the whitened signal
        """
        return self._f_white(whitening(x, self._w_white, self._m_white))

    @property
    def calibrated(self):
        """
        True if samples has been provided and the calibration has been done, False otherwise.

        Returns
        -------
        bool
        """
        return self._is_calibrated

    @property
    def w_white(self):
        """
        The whitening transformation parameters.

        Returns
        -------
        np.ndarray
        """
        return self._w_white

    @property
    def m_white(self):
        """
        The mean values from the samples used for calibration.

        Returns
        -------
        np.ndarray
        """
        return self._m_white

    @property
    def w_method(self):
        """
        The whitening method.

        Returns
        -------
        str
        """
        return self._w_method.__name__.upper()

    def __repr__(self):
        return "Whitening(in=%d, out=%d, method='%s', calibrated=%s)" % (
            self._nb_input, self._nb_output, self.w_method, self.calibrated
        )


class DiscreteCosineTransform(Preprocessing):

    def __init__(self, *args, **kwargs):
        """
        A preprocessing component that transforms the input into the frequency domain by using the Discrete Cosine
        Transform (DCT) method.
        """
        super().__init__(*args, **kwargs)

        self._w_dct = None
        """
        Whitening synaptic weights.
        """
        self._f_dct = lambda x: x
        """
        Activation function after the transform.
        """

        self.reset()
        self.params.extend([self._w_dct])

    def reset(self):
        """
        Resets the DCT parameters.
        """
        w = dct_synapses(self._nb_input, dtype=self.dtype)
        if self._w_dct is None:
            self._w_dct = w
        else:
            self._w_dct[:] = w

    def _fprop(self, x):
        """
        Decomposes the input signal to the the different phases of the cosine function.

        Parameters
        ----------
        x: np.ndarray[float]
            the raw signal that needs to be transformed

        Returns
        -------
        np.ndarray[float]
            the signal in the frequency domain
        """
        return self._f_dct(x @ self._w_dct)

    @property
    def calibrated(self):
        """
        True if the DCT parameters have been calculated, False otherwise.

        Returns
        -------
        bool
        """
        return self._w_dct is not None

    @property
    def w_dct(self):
        """
        The DCT parameters.

        Returns
        -------
        np.ndarray
        """
        return self._w_dct

    def __repr__(self):
        return "DiscreteCosineTransform(in=%d, out=%d, calibrated=%s)" % (
            self._nb_input, self._nb_output, self.calibrated
        )
