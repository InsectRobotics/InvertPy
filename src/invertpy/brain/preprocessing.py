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

from invertpy.sense import CompoundEye

from .component import Component
from .synapses import whitening_synapses, dct_synapses, mental_rotation_synapses
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
            number of input units of the preprocessing component.
        nb_output: int, optional
            number of output units of the preprocessing component. Default is the same as the input
        """
        if nb_output is None:
            nb_output = nb_input
        super().__init__(nb_input=nb_input, nb_output=nb_output, nb_repeats=1, repeat_rate=1, *args, **kwargs)

    def __repr__(self):
        return "Preprocessing(in=%d, out=%d)" % (self._nb_input, self._nb_output)

    @property
    def nb_input(self):
        return self._nb_input

    @property
    def nb_output(self):
        return self._nb_output


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
        return self._f_dct(np.dot(x, self._w_dct))

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


class MentalRotation(Preprocessing):
    def __init__(self, *args, nb_input=None, nb_output=8, eye=None, pref_angles=None, sigma=.02, **kwargs):
        """
        Performs mental rotation of the visual input into a fixed number of preferred angles (nb_output).
        This is done through a number of map from each ommatidium to every other ommatidium that is positioned
        close to the direction where it would be facing if the eye was rotated. This 3D map gets as input an
        array of size `nb_input` and returns a matrix of `nb_input` x `nb_output`, where each row corresponds
        to the different ommatidia of the eye and every column to the mentally rotated orientation.

        Parameters
        ----------
        nb_input: int, optional
            number of input units is the same as the number of ommatidia. Default is the number of ommatidia of the eye
            (if provided)
        nb_output: int, optional
            number of output orientations. Default is 8
        eye: CompoundEye, optional
            compound eye which will be used to compute the parameters. Default is None
        pref_angles: np.ndarray[float], optional
            the preferred angles for the rotation. Default is uniform orientations based on the nb_output
        sigma: float, optional
            mental radius of each ommatidium (percentile). Default is 0.02
        """
        assert nb_input is not None or eye is not None, (
            "You should specify the input either by the 'nb_input' or the 'eye' attribute.")
        if eye is not None:
            nb_input = eye.nb_ommatidia
        super().__init__(*args, nb_input=nb_input, nb_output=nb_output, **kwargs)

        self._w_rot = np.zeros((nb_input, nb_input, nb_output))
        """
        The weights that perform the mental rotation.
        """

        self._omm_ori = None
        """
        The orientations of the ommatidia.
        """

        self._f_rot = lambda x: x
        """
        Activation after the rotation.
        """

        self._sigma = sigma
        """
        The mental radius of each ommatidium.
        """

        self._pref_angles = pref_angles
        """
        The preferred angles of the mental rotation.
        """
        if pref_angles is None:
            self._pref_angles = np.linspace(0, 2 * np.pi, nb_output, endpoint=False)

        self.params.append(self._w_rot)
        self.reset(eye)

    def reset(self, eye=None):
        """
        Resets the synaptic weights and eye parameters.

        Parameters
        ----------
        eye: CompoundEye, optional
            the new compound eye to extract the parameters from
        """
        if eye is not None:
            self._omm_ori = eye.omm_ori
        if self._omm_ori is not None:
            self._w_rot[:] = mental_rotation_synapses(self._omm_ori, self._nb_output,
                                                      phi_out=self._pref_angles, sigma=self._sigma)

    def _fprop(self, x):
        """
        Performs the mental rotation.

        Parameters
        ----------
        x: np.ndarray[float]
            the input from the ommatidia

        Returns
        -------
        np.ndarray[float]
            N x M matrix of copies of the input from ommatidia mapped to the different preferred orientations.

            - N: number of inputs / ommatidia
            - M: number of output mental rotations / preferred angles
        """
        return self._f_rot(np.dot(x, self._w_rot).T)

    @property
    def w_rot(self):
        """
        N x N x M matrix that maps the ommatidia input to the different mental rotations.

        - N: number of input / ommatidia
        - M: number of output mental rotations / preferred angles

        Returns
        -------
        np.ndarray[float]
        """
        return self._w_rot

    @property
    def pref_angles(self):
        """
        List of the preference angles that mental rotation will be applied to.

        Returns
        -------
        np.ndarray[float]
        """
        return self._pref_angles

    @property
    def sigma(self):
        """
        The mental radius of each ommatidium (percentile).

        Returns
        -------
        float
        """
        return self._sigma
