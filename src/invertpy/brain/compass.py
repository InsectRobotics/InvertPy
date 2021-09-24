"""
The Compass models of the insect brain.

References:
    .. [1] Gkanias, E., Risse, B., Mangan, M. & Webb, B. From skylight input to behavioural output: a computational model
       of the insect polarised light compass. PLoS Comput Biol 15, e1007123 (2019).
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from .component import Component
from .synapses import uniform_synapses
from ._helpers import eps

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class Compass(Component):
    """
    Abstract class for the Compass as a Component of the insect brain.
    """

    def reset(self):
        raise NotImplementedError()

    def _fprop(self, *args, **kwargs):
        raise NotImplementedError()


class CelestialCompass(Compass):

    def __init__(self, nb_pol, loc_ori, nb_sol=8, nb_tcl=None, sigma=13, shift=40, dt=2./60, degrees=True,
                 integrated=False, has_pol=True, has_sun=True, has_circadian=False, *args, **kwargs):
        """
        The Celestial Compass integrated a polarisation compass and a sky gradient compass presented in [1]_.

        Parameters
        ----------
        nb_pol: int
            the number of POL units
        loc_ori: R
            the orientation of the ommatidia (relative to the eye orientation). They are used in order to calculate the
            synaptic weights mapping the eye view to an internal representation.
        nb_sol: int, optional
            the number of SOL (direction relative to the sun) units.
        nb_tcl: int, optional
            the number of TCL (direction relative to North) units.
        sigma: float, optional
            the angular thickness of the tilt compensation gate function's ring.
        shift: float, optional
            the angular radius of the tilt compensation gate function's ring from the zenith.
        dt: float, optional
            the how often do we do updates.
        degrees: bool, optional
            whether the input angles are in degrees or not.
        integrated: bool, optional
            whether the model produces directly the TCL output from the POL input or though the SOL neurons.
        has_pol: bool, optional
            whether the compass has a polarisation compass.
        has_sun: bool, optional
            whether the compass has a solar compass.
        has_circadian: bool, optional
            whether the compass has a circadian mechanism and compensates for the moving sun.

        Notes
        -----
        .. [1] Gkanias, E., Risse, B., Mangan, M. & Webb, B. From skylight input to behavioural output: a computational model
           of the insect polarised light compass. PLoS Comput Biol 15, e1007123 (2019).
        """
        super().__init__(nb_pol, nb_tcl, *args, **kwargs)

        if nb_tcl is None:
            nb_tcl = nb_sol

        self._phi_sun = np.linspace(-np.pi, np.pi, nb_sol, endpoint=False)  # SUN preference angles
        self._phi_sol = np.linspace(0., 2 * np.pi, nb_sol, endpoint=False)  # SOL preference angles
        self._phi_tcl = np.linspace(0., 2 * np.pi, nb_tcl, endpoint=False)  # TCL preference angles

        self._w_sun = uniform_synapses(nb_pol, nb_sol, dtype=self.dtype)
        self._w_sol = uniform_synapses(nb_pol, nb_sol, dtype=self.dtype)
        if integrated:
            self._w_tcl = uniform_synapses(nb_pol, nb_tcl, dtype=self.dtype)
        else:
            self._w_tcl = uniform_synapses(nb_sol, nb_tcl, dtype=self.dtype)

        self.params.extend([
            self._w_sol, self._w_sun, self._w_tcl, self._phi_sol, self._phi_sun, self._phi_tcl
        ])

        self._sigma = np.deg2rad(sigma) if degrees else sigma
        self._shift = np.deg2rad(shift) if degrees else shift
        self._loc_ori = copy(loc_ori)

        self._r_pol = None
        self._r_sol = None
        self._r_sun = None
        self._r_cel = None
        self._r_tcl = None

        self._nb_pol = nb_pol
        self._nb_sol = nb_sol
        self._nb_tcl = nb_tcl

        self._has_circadian = has_circadian
        self._has_pol = has_pol
        self._has_sun = has_sun
        self._is_absolute = integrated
        self._d_phi = 0.
        self._dt = dt

        self.reset()

    def reset(self):
        yaw, pitch, roll = self._loc_ori.as_euler('ZYX', degrees=False).T

        if self.has_pol:
            self.phi_sol = np.linspace(0., 2 * np.pi, self.nb_sol, endpoint=False)  # SOL preference angles
            self.w_sol = self.generate_w_sol(yaw+np.pi/2, self.phi_sol)
        if self.has_sun:
            self.phi_sun = np.linspace(-np.pi, np.pi, self.nb_sol, endpoint=False)  # SUN preference angles
            self.w_sun = self.generate_w_sol(yaw+np.pi/2, self.phi_sun)

        self.phi_tcl = np.linspace(0., 2 * np.pi, self.nb_tcl, endpoint=False)  # TCL preference angles
        self.w_tcl = self.generate_w_tcl(yaw+np.pi/2)

        self._r_pol = np.zeros(self.nb_pol)
        self._r_sol = np.zeros(self.nb_sol)
        self._r_sun = np.zeros(self.nb_sol)
        self._r_tcl = np.zeros(self.nb_tcl)

        self.update = self._has_circadian

    def _fprop(self, r_pol: np.ndarray = None, r: np.ndarray = None, glob_ori: R = None, ori: R = None):
        if self.has_pol and r_pol is None and r is not None:
            r_pol = photoreceptor2pol(r, ori=self._loc_ori).reshape(-1)
        elif r_pol is None:
            r_pol = np.zeros(self._nb_pol)
        else:
            r_pol = r_pol.reshape(-1)

        if self.has_sun and r is not None:
            r_sun = photoreceptor2pooling(r)
        else:
            r_sun = np.zeros(self._nb_sol)

        if glob_ori is None and ori is None:
            glob_ori = self._loc_ori
        elif ori is not None:
            glob_ori = ori * self._loc_ori

        g = self.gate(glob_ori)

        r_cel = np.zeros(self.nb_sol, dtype=self.dtype)
        if self.has_pol:
            w_sol = self.w_sol * g
            r_sol = r_pol @ w_sol
            r_cel += r_sol
            self._r_sol = r_sol
        if self.has_sun:
            w_sun = self.w_sun * g
            r_sun = r_sun @ w_sun
            r_cel += r_cel
            self._r_sun = r_sun
        # r_cel /= (float(self.has_pol) + float(self.has_sun))

        if self.update:
            d_phi = self.circadian(r_cel, self._dt)
            self.phi_tcl = (self._phi_tcl + d_phi) % (2 * np.pi)
            self._d_phi += d_phi
            self.w_tcl = self.generate_w_tcl()

        if self._is_absolute:
            w_tcl = self.w_tcl * g
            # print(w_tcl)
            r_tcl = r_pol @ w_tcl
        else:
            w_tcl = self.w_tcl
            r_tcl = r_cel @ w_tcl

        self._r_pol = r_pol
        self._r_cel = r_cel
        self._r_tcl = r_tcl

        return r_tcl

    def gate(self, glob_ori, order=1.):
        """
        The tilt compensation mechanism is a set of responses that gate the synaptic weights from the POl neurons to
        the SOL neurons (or directly to the TCL neurons).

        Parameters
        ----------
        glob_ori: R
            the global orientation of the ommatidia
        order: float, optional
            the order of steepness of the Gaussian

        Returns
        -------
        g: np.ndarray
            the gate that needs to be multiplied with the weights.
        """
        _, pitch, _ = glob_ori.as_euler('ZYX', degrees=False).T
        zenith = pitch - np.pi/2
        d = np.sin(self._shift - zenith)

        return np.power(np.exp(-np.square(d) / (2. * np.square(self._sigma))), order).reshape((-1, 1))

    def circadian(self, r_sol, dt):
        """
        The time-compensation mechanism of the compass model works as an internal clock.

        Parameters
        ----------
        r_sol: np.ndarray
            the responses of the SOL neurons.
        dt: float
            the time passed.

        Returns
        -------
        d_phi: float
            the shift of the sun during the dt.
        """

        r = r_sol @ np.exp(-np.arange(self.nb_tcl) * 1j * 2 * np.pi / float(self._nb_tcl))
        res = np.clip(3.5 * (np.absolute(r) - .53), 0, 2)  # certainty of prediction
        ele_pred = 26 * (1 - 2 * np.arcsin(1 - res) / np.pi) + 15
        return np.deg2rad(9 + np.exp(.1 * (54 - ele_pred))) / (60. / float(dt))

    def generate_w_sol(self, pref_in, pref_out):
        """
        Creates the POl to SOL synaptic weights.

        Parameters
        ----------
        pref_in: np.ndarray
            the preference angles of the input layer.
        pref_out: np.ndarray
            the preference angles of the output layer.

        Returns
        -------
        w_pol2sol: np.ndarray
            the synaptic weights that transform the POL responses to SOL responses.

        """
        if self._is_absolute:
            z = float(self.nb_sol) / (2. * float(self.nb_pol))
            return -z * np.sin(pref_out[np.newaxis] - pref_in[:, np.newaxis])
        else:
            z = float(self.nb_sol) / float(self.nb_pol)
            return z * np.sin(pref_in[:, np.newaxis] - pref_out[np.newaxis])

    def generate_w_tcl(self, pref_in=None):
        """
        Creates the POl to TCL synaptic weights if absolute=True, else it creates SOL to TCL synaptic weights.

        Parameters
        ----------
        pref_in: np.ndarray, optional
            the preference angles of the input layer for the absolute case.

        Returns
        -------
        w_pol2tcl: np.ndarray
            the synaptic weights that transform the POL or SOL responses to TCL responses.

        """
        if self._is_absolute:
            if pref_in is None:
                pref_in, _, _ = self._loc_ori.as_euler('ZYX', degrees=False)
                pref_in += np.pi/2

            z = float(self.nb_tcl) / (2. * float(self.nb_pol))
            return -z * np.sin(self._phi_tcl[np.newaxis] - pref_in[:, np.newaxis])
        else:
            z = float(self.nb_tcl) / float(self.nb_sol)
            return z * np.cos(self._phi_tcl[:, np.newaxis] - self._phi_sol[np.newaxis])

    @property
    def w_sol(self):
        """
        The POL to SOL synaptic weights.
        """
        return self._w_sol

    @w_sol.setter
    def w_sol(self, v):
        self._w_sol[:] = v[:]

    @property
    def w_sun(self):
        """
        The POL to SUN synaptic weights.
        """
        return self._w_sun

    @w_sun.setter
    def w_sun(self, v):
        self._w_sun[:] = v[:]

    @property
    def w_tcl(self):
        """
        The POL or SOL to TCL synaptic weights.
        """
        return self._w_tcl

    @w_tcl.setter
    def w_tcl(self, v):
        self._w_tcl[:] = v[:]

    @property
    def phi_sol(self):
        """
        The SOL preference angles.
        """
        return self._phi_sol

    @phi_sol.setter
    def phi_sol(self, v):
        self._phi_sol[:] = v[:]

    @property
    def phi_sun(self):
        """
        The SUN preference angles.
        """
        return self._phi_sun

    @phi_sun.setter
    def phi_sun(self, v):
        self._phi_sun[:] = v[:]

    @property
    def phi_tcl(self):
        """
        The TCL preference angles.
        """
        return self._phi_tcl

    @phi_tcl.setter
    def phi_tcl(self, v):
        self._phi_tcl[:] = v[:]

    @property
    def shift(self):
        """
        The angular radius of the tilt compensation gate function's ring from the zenith.
        """
        return self._shift

    @property
    def sigma(self):
        """
        The angular thickness of the tilt compensation gate function's ring.
        """
        return self._sigma

    @property
    def r_pol(self):
        """
        The responses of the POL neurons.
        """
        return self._r_pol

    @property
    def r_sol(self):
        """
        The responses of the SOL neurons.
        """
        return self._r_sol

    @property
    def r_sun(self):
        """
        The responses of the SUN neurons.
        """
        return self._r_sun

    @property
    def r_cel(self):
        """
        The responses of the CEL neurons.
        """
        return self._r_cel

    @property
    def r_tcl(self):
        """
        The responses of the TCL neurons.
        """
        return self._r_tcl

    @property
    def nb_pol(self):
        """
        The number of POL neurons.
        """
        return self._nb_pol

    @property
    def nb_sol(self):
        """
        The number of SOL neurons.
        """
        return self._nb_sol

    @property
    def nb_tcl(self):
        """
        The number of TCL neurons.
        """
        return self._nb_tcl

    @property
    def d_phi(self):
        """
        The angular change rate of the preference angles of TCL neurons.
        """
        return self._d_phi

    @property
    def has_pol(self):
        """
        Whether it includes a Polarised light compass.
        """
        return self._has_pol

    @property
    def has_sun(self):
        """
        Whether it includes a Sky gradient compass.
        """
        return self._has_sun

    @property
    def has_circadian(self):
        """
        Whether it includes a circadian mechanism.
        """
        return self._has_circadian


class PolarisationCompass(CelestialCompass):
    def __init__(self, *args, **kwargs):
        """
        The Polarisation Compass is a special case of the Celestial Compass that does not include a sky gradient
        compass.
        """
        kwargs.setdefault('has_sun', False)
        kwargs.setdefault('has_pol', True)
        super().__init__(*args, **kwargs)


class SolarCompass(CelestialCompass):
    def __init__(self, *args, **kwargs):
        """
        The Solar Compass is a special case of the Celestial Compass that does not include a polarisation compass.
        """
        kwargs.setdefault('has_sun', True)
        kwargs.setdefault('has_pol', False)
        super().__init__(*args, **kwargs)


def photoreceptor2pol(r, ori=None, ori_cross=None, dtype='float32'):
    """
    Transforms the input from the photo-receptors into POL neurons responses.

    Parameters
    ----------
    r: np.ndarray
        the input from the photo-receptors.
    ori: R, optional
        the orientation of the ommatidia.
    ori_cross: np.ndarray, optional
        the angle of preference for each photo-receptor with respect to the orientation of each ommatidium
    dtype: np.dtype, optional
        the type of the data

    Returns
    -------
    r_pol: np.ndarray
        the responses of the POL units.
    """
    r_op = photoreceptor2opponent(r, ori=ori, ori_cross=ori_cross, dtype=dtype)
    r_po = photoreceptor2pooling(r)
    return r_op / (r_po + eps)


def photoreceptor2opponent(r, ori=None, ori_cross=None, dtype='float32'):
    """
    Transforms the input from the photo-receptors into opponent (OP) neurons responses.

    Parameters
    ----------
    r: np.ndarray
        the input from the photo-receptors.
    ori: R, optional
        the orientation of the ommatidia.
    ori_cross: np.ndarray, optional
        the angle of preference for each photo-receptor with respect to the orientation of each ommatidium
    dtype: np.dtype, optional
        the type of the data

    Returns
    -------
    r_op: np.ndarray
        the responses of the OP units.
    """
    if ori is None and ori_cross is None:
        return np.sum(r, axis=1)
    elif ori_cross is None:
        ori_cross = ori2cross(np.shape(ori)[0], nb_receptors=2, dtype=dtype)
    return np.sum(np.cos(2 * ori_cross) * r, axis=1)


def photoreceptor2pooling(r):
    """
    Transforms the input from the photo-receptors into pooling (PO) neurons responses.

    Parameters
    ----------
    r: np.ndarray
        the input from the photo-receptors.

    Returns
    -------
    r_po: np.ndarray
        the responses of the PO units.
    """
    return np.sum(r, axis=1)


def ori2cross(nb_ommatidia, nb_receptors=2, dtype='float32'):
    """
    Creates the cross directions for the different photo-receptors of each ommatidium.

    Parameters
    ----------
    nb_ommatidia: int
        the number of ommatidia.
    nb_receptors: int, optional
        the number of photo-receptors per ommatidium.
    dtype: np.dtype, optional
        the data type.

    Returns
    -------
    ori_cross: np.ndarray
        the cross directions for the different photo-receptors of each ommatidium.
    """
    ori_cross = np.zeros((nb_ommatidia, nb_receptors), dtype=dtype)
    for i in range(1, nb_receptors):
        ori_cross[..., i] = np.pi / float(nb_receptors)

    return ori_cross


def sph2ring(theta, phi=None, length=8, axis=-1):
    """
    Creates an array of responses (population code) representing the spherical coordinates.

    Parameters
    ----------
    theta: float | np.ndarray
        the zenith angle of the point(s).
    phi: float | np.ndarray
        the azimuth of the point(s).
    length: int, optional
        the size of the output population (array).
    axis: int
        the axis to apply the calculations on.

    Returns
    -------
    np.ndarray[float]
        I - the array of responses representing the given spherical coordinates.
    """
    if phi is None:
        if not isinstance(theta, float) and theta.shape[0] > 1:
            phi = theta[1]
            theta = theta[0]
        else:
            phi = theta
            theta = np.pi / 2
    theta = np.absolute(theta)
    alpha = np.linspace(0, 2 * np.pi, length, endpoint=False)
    return np.sin(alpha + phi + np.pi / 2) * theta / (length / 2.)


def ring2sph(I, axis=-1):
    """
    Creates the spherical coordinates given an array of responses (population code).

    Parameters
    ----------
    I: np.ndarray[float]
        the array of responses representing spherical coordinates.
    axis: int
        the axis to apply the calculations on.

    Returns
    -------
    np.ndarray[float]
        theta, phi - the spherical coordinates calculated using the input population codes.
    """
    fund_freq = np.fft.fft(I, axis=axis)[1]
    phi = (np.pi - np.angle(np.conj(fund_freq))) % (2 * np.pi) - np.pi
    theta = np.absolute(fund_freq)
    return np.array([theta, phi])


def ring2complex(I, axis=-1):
    """
    Creates 2D vectors (complex numbers) showing the direction of the vectors represented by given arrays of responses
    (population codes).

    Parameters
    ----------
    I: np.ndarray[float]
        the array of responses representing 2D vector.
    axis: int
        the axis to apply the calculations on.

    Returns
    -------
    np.ndarray[complex]
        (x + j * y) - the 2D vectors as complex numbers calculated using the input population codes.
    """
    length = I.shape[axis]
    alpha = np.linspace(0, 2 * np.pi, length, endpoint=False)
    vectors = np.cos(alpha) + 1j * np.sin(alpha)
    z = np.sum(I * vectors, axis=axis)

    return z
