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

from invertpy.sense import CompoundEye

from .component import Component
from .synapses import whitening_synapses, dct_synapses, mental_rotation_synapses
from ._helpers import whitening, pca, zca, eps

from abc import ABC
from math import factorial
from scipy.spatial.transform import Rotation

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


class LateralInhibition(Preprocessing):

    def __init__(self, ori, nb_neighbours=6, *args, **kwargs):
        """
        A preprocessing component that computes the edges of the input spherical 'image' using lateral inhibition.

        In lateral inhibition, each output neuron is excited by its respective ommatidium and inhibited by its
        neightbours.

        Parameters
        ----------
        ori : Rotation
            the relative orientation of the ommatidia of interest
        nb_neighbours : int
            the number of neighbours to be inhibited from. Default is 6
        """
        kwargs.setdefault("nb_input", np.shape(ori)[0])
        kwargs.setdefault("nb_output", np.shape(ori)[0])
        super().__init__(*args, **kwargs)

        self._xyz = ori.apply([1, 0, 0])
        self._xyz = self._xyz / np.linalg.norm(self._xyz, axis=-1)[:, np.newaxis]

        self._nb_neighbours = nb_neighbours

        self._w = np.zeros((self._nb_input, self._nb_output), dtype=self.dtype)

        self._f_li = lambda x: np.clip(x, 0, 1)

        self.reset()

    def reset(self, *args):
        """
        Resets the ZM parameters.
        """
        c = np.clip(np.dot(self._xyz, self._xyz.T), -1, 1)
        d = np.arccos(c)  # angular distance between vectors
        w = np.zeros_like(self._w)

        i = np.argsort(d, axis=1)[:, :self._nb_neighbours+1]

        w[i[:, 0], np.arange(w.shape[1])] = float(self._nb_neighbours)
        for j in range(self._nb_neighbours):
            w[i[:, j + 1], np.arange(w.shape[1])] = -1

        # # the synaptic weights could be calculated using the second derivative of the Gaussian function
        # # (Ricker or Mexican hat wavelet)
        # r = 2 * d[~np.isclose(d, 0)].min()
        # z = 2 / np.sqrt(3 * r) * np.power(np.pi, 1 / 4)
        # w = z * (1 - np.square(d / r)) * np.exp(-np.square(d) / (2 * np.square(r)))
        # w[w > 0] *= 10 * (-w[w < 0]).sum(axis=1) / w[w > 0].sum(axis=1)
        # w[w < 0] *= 10 * (-w[w > 0]).sum(axis=1) / w[w < 0].sum(axis=1)

        self._w = w

    def _fprop(self, x):
        """
        Transform the input signal to its edges.

        Parameters
        ----------
        x: np.ndarray[float]
            the raw signal that needs to be transformed

        Returns
        -------
        np.ndarray[float]
        """
        return self._f_li(x.dot(self._w))

    @property
    def w(self):
        """
        The transformation weights.

        Returns
        -------
        np.ndarray[float]
        """
        return self._w

    @property
    def centres(self):
        """
        The normalised 3D positions of the ommatidia.

        Returns
        -------
        np.ndarray[float]
        """
        return self._xyz

    @property
    def nb_neighbours(self):
        """
        The number of neighbours that each ommatidium is inhibited from.

        Returns
        -------
        int
        """
        return self._nb_neighbours


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
        self._f_white = lambda x: ((x.T - x.min(axis=-1)) / (x.max(axis=-1) - x.min(axis=-1) + eps)).T
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

        w, m = None, None
        if samples is None and self._is_calibrated is None:
            w = np.eye(self._nb_input, self._nb_output, dtype=self.dtype)
            m = np.zeros(self._nb_input, dtype=self.dtype)
            self._is_calibrated = False
        elif samples is not None:
            w, m = whitening_synapses(samples, nb_out=self.nb_output, w_func=self._w_method, dtype=self.dtype, bias=True)
            self._is_calibrated = True

        if self._w_white is None or self._m_white is None:
            self._w_white, self._m_white = w, m
        elif w is not None and m is not None:
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
        return self._f_white(whitening(x, w=self._w_white, m=self._m_white))

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


class ZernikeMoments(Preprocessing):
    N_MAX = 60
    M_MAX = 20

    def __init__(self, ori, order=16, out_type="amplitude", degrees=False, *args, **kwargs):
        """
        A preprocessing component that transforms the input into the frequency domain by using the Zernike Moments (ZM)
        method. The Zernike Moments are used widely for rotation invariant preprocessing and they are particularly
        convenient for the insect panoramic vision, as they work best with polar coordinates.

        Parameters
        ----------
        ori : Rotation
            the relative orientation of the ommatidia of interest
        order : int
            the maximum order of the Zernike Moments to calculate. Default is 16
        out_type : {"amplitude", "phase", "raw"}
            defines the output type of call function; one of "amplitude", "phase" or "raw". Default is "amplitude"
        degrees : bool
            defines if the input and output angles will be in degrees or not. Default is False
        """
        kwargs.setdefault("nb_input", np.shape(ori)[0])
        nb_coeff = self.get_nb_coeff(order)
        kwargs.setdefault("nb_output", nb_coeff)
        super().__init__(*args, **kwargs)

        phi, theta, _ = ori.as_euler("ZYX", degrees=False).T

        self._phi = phi
        self._rho = (np.pi/2 - theta) / np.pi
        self._phi -= np.absolute(np.where(self._rho <= 1, 0, 1) * np.pi)
        self._rho = np.absolute(np.where(self._rho <= 1, 1, -1) * self._rho)

        self._n = order
        self._degrees = degrees
        """
        Whether to get and set values in degrees.
        """

        self._z = np.zeros(self._nb_output, dtype=complex)
        self._out_type = out_type
        """
        The type of the output.
        """
        self.__cnt = -1
        """
        The number of pixels inside the unit circle
        """

        # Pre-calculate the factorial to improve the speed
        p_max = int(self.N_MAX / 2)
        q_max = int((self.N_MAX + self.M_MAX) / 2)

        self.__FAC_S = np.zeros([p_max + 1])
        for s in range(p_max + 1):
            self.__FAC_S[s] = factorial(s)

        self.__FAC_N_S = np.zeros([self.N_MAX, p_max + 1])
        for n in range(self.N_MAX):
            for s in range(int(n / 2) + 1):
                self.__FAC_N_S[n, s] = factorial(n - s)

        self.__FAC_Q_S = np.zeros([q_max, p_max + 1])
        for q in range(q_max):
            for s in range(np.min([q + 1, p_max + 1])):
                self.__FAC_Q_S[q, s] = factorial(q - s)

        self.__FAC_P_S = np.zeros([p_max, p_max + 1])
        for p in range(p_max):
            for s in range(p + 1):
                self.__FAC_P_S[p, s] = factorial(p - s)

        self.reset()

    def reset(self, *args):
        """
        Resets the ZM parameters.
        """
        self._z = np.zeros(self._nb_output, dtype=complex)

        # count the number of pixels inside the unit circle
        self.__cnt = np.count_nonzero(self._rho) + 1

    def _fprop(self, x):
        """
        Decomposes the input signal to the the different phases of the Zernike moments.

        Parameters
        ----------
        x: np.ndarray[float]
            the raw signal that needs to be transformed

        Returns
        -------
        np.ndarray[float]
            the signal in the frequency domain
        """

        i = 0
        for n in range(self.order + 1):
            for m in range(n + 1):
                if (n - np.absolute(m)) % 2 == 0:
                    self._z[i] = self.calculate_moment(x, n, m)
                    i += 1  # process the next moment
                if i >= self._nb_output:
                    break
            if i >= self._nb_output:
                break

        if "amplitude" in self._out_type:
            return self.z_amplitude
        elif "phase" in self._out_type:
            return self.z_phase
        else:
            return self.z_moments

    def calculate_moment(self, x, order, repeat):
        """
        Calculates and returns the Zernike moment for the given input.

        Parameters
        ----------
        x : np.ndarray[float]
            the input intensities of the given ommatidia orientations.
        order : int
            the order of Zernike Moments that we are interested in.
        repeat : int
            the repeat of the order that we are interested in.

        Returns
        -------
        complex
            the Zernike moment for the given input
        """
        # get the Zernike polynomials
        Z = self.zernike_poly(self._rho, self._phi, order, repeat)

        # calculate the moments
        z = np.dot(x, Z)

        # normalize the amplitude of moments
        z = (self._n + 1) * z / self.__cnt

        return z

    def zernike_poly(self, rho, phi, order, repeat):
        """
        Calculates and returns the Zernike polynomials for the given input.

        The return values are complex: their real part represents the odd function over the azimuthal angle,
        while their imaginary part represents the respective even function.

        Parameters
        ----------
        rho : np.ndarray[float]
            the radius of interest
        phi : np.ndarray[float]
            the azimuth of interest
        order : int
            the order of interest
        repeat : int
            the repeat of interest

        Returns
        -------
        np.ndarray[complex]
            the Zernike polynomials for the given input
        """
        return self.radial_poly(rho, order, repeat) * np.exp(-1j * repeat * phi)

    def radial_poly(self, rho, order, repeat):
        """
        Calculates and returns the radial polynomials for the given input.

        Parameters
        ----------
        rho : np.ndarray[float]
            the radius of interest
        order : int
            the order of interest
        repeat : int
            the repeat of interest

        Returns
        -------
        np.ndarray[float]
            the radial polynomials for the given input
        """
        rad = np.zeros(rho.shape, dtype=rho.dtype)
        p = int((order - np.absolute(repeat)) / 2)
        q = int((order + np.absolute(repeat)) / 2)
        for s in range(p + 1):
            c = np.power(-1, s) * self.__FAC_N_S[order, s]
            c /= self.__FAC_S[s] * self.__FAC_Q_S[q, s] * self.__FAC_P_S[p, s]
            rad += c * np.power(rho, order - 2 * s)
        return rad

    @property
    def calibrated(self):
        """
        True if the ZM parameters have been calculated, False otherwise.

        Returns
        -------
        bool
        """
        return self.__cnt >= 0

    @property
    def order(self):
        """
        The maximum order of the Zernike Moments.

        Returns
        -------
        int
        """
        return self._n

    @property
    def phi(self):
        """
        The azimuth of the target intensities.

        Returns
        -------
        np.ndarray[float]
        """
        return self._phi

    @property
    def rho(self):
        """
        The radius of the target intensities is the normalised zenith angle of the given ommatidia orientations.

        Returns
        -------
        np.ndarray[float]
        """
        return self._rho

    @property
    def z_moments(self):
        """
        The computed Zernike Moments for the last input.

        Returns
        -------
        np.ndarray[complex]
        """
        return self._z

    @property
    def z_amplitude(self):
        """
        The amplitude of the lastly-calculated Zernike Moments.

        Returns
        -------
        np.ndarray[float]
        """
        return np.absolute(self._z)

    @property
    def z_phase(self):
        """
        The phase of the lastly-calculated Zernike Moments.

        Returns
        -------
        np.ndarray[float]
        """
        return np.angle(self._z)

    def __repr__(self):
        return "ZernikeMoments(in=%d, out=%d, type=%s, order=%d, calibrated=%s)" % (
            self._nb_input, self._nb_output, self._out_type, self.order, self.calibrated
        )

    @staticmethod
    def get_nb_coeff(order):
        if order % 2:
            nb_coeff = int(((1 + order) / 2) * ((3 + order) / 2))
        else:
            nb_coeff = int(np.square(order / 2. + 1))
        return nb_coeff


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
        if pref_angles is not None:
            nb_output = len(pref_angles)
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
