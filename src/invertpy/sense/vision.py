"""
The CompoundEye package. Contains the basic functions of the compound eyes and a function that creates a mental rotation
matrix.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from invertpy.brain.activation import softmax
from .sensor import Sensor
from ._helpers import fibonacci_sphere, eps

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class CompoundEye(Sensor):
    def __init__(self, omm_xyz=None, omm_ori=None, omm_rho=None, omm_pol_op=None, omm_res=None, c_sensitive=None,
                 omm_photoreceptor_angle=2, *args, **kwargs):
        """
        The CompoundEye class is a representation of the insect compound eye as a simple sensor. It can have multiple
        ommatidia (multi-dimensional photo-receptors) that are distributed in eye, are pointing in different directions
        and have different properties, such as the acceptance angle, the polarisation sensitivity, the responsiveness
        and the spectral sensitivity.

        Parameters
        ----------
        omm_xyz: np.ndarray, float
            Relative 3-D position of each of the ommatidia on the eye. If None, it is automatically calculated to be
            1 unit away from the centre of the eye and towards the direction of the ommatidium.
        omm_ori: R, optional
            The relative direction of each of the ommatidia on the eye. If None, it is automatically calculated from the
            position of the ommatidia, assumming that it is phasing in the direction from the centre to the position of
            the ommatidium.
        omm_rho: np.ndarray, float
            The acceptance angle of each ommatidium in rad. If it is a single value, it is assigned
            to all the ommatidia. Default is 5 deg.
        omm_res: np.ndarray, float
            The responsivity of each ommatidium. If it is a single value, it is assigned to all the ommatidia.
            Default is 1.
        omm_pol_op: np.ndarray, float
            The polarisation sensitivity of every ommatidium (0 = None, 1 = very sensitive). If it is a single value, it
            is assigned to all the ommatidia. Default is 0.
        c_sensitive: tuple, list, np.ndarray
            The IRGBU colour code that the eyes are sensitive to (infrared, red, green, blue, ultraviolet)
        omm_photoreceptor_angle: int, list, np.ndarray
            The angle of each photoreceptor with respect to the direction of their respective ommatidium. If int, a
            homogeneously distributed array of the defined number of angles will be created. Default is 2 (one at 0 and
            one at 90 degrees).
        """
        if omm_pol_op is None or isinstance(omm_pol_op, float) or isinstance(omm_pol_op, int):
            nb_output = None
        else:
            nb_output = omm_pol_op.shape
        if omm_ori is None and omm_xyz is not None:
            omm_ori = np.empty(omm_xyz.shape[0], dtype=R)
            for i in range(omm_xyz.shape[0]):
                omm_ori[i] = R.align_vectors(omm_xyz[i], [[0], [0], [1]])
        if omm_ori is not None:
            kwargs.setdefault('nb_input', np.shape(omm_ori)[0])
        else:
            kwargs.setdefault('nb_input', 1000)
            omm_sph = fibonacci_sphere(kwargs.get('nb_input'), 2*np.pi)[..., :2]
            omm_euler = np.hstack([omm_sph, np.full((omm_sph.shape[0], 1), np.pi / 2)])
            omm_ori = R.from_euler('ZYX', omm_euler, degrees=False)
        nb_output = kwargs.get('nb_output', np.shape(omm_ori) if nb_output is None else nb_output)
        if isinstance(nb_output, int):
            kwargs['nb_output'] = (nb_output, 1)
        elif len(nb_output) < 2:
            kwargs['nb_output'] = (nb_output[0], 1)
        else:
            kwargs['nb_output'] = tuple(nb_output)

        if omm_xyz is not None:
            kwargs.setdefault('nb_input', omm_xyz.shape[0])
            kwargs.setdefault('nb_output', omm_xyz.shape[0])

        kwargs.setdefault('name', 'compound_eye')

        super().__init__(*args, **kwargs)

        if omm_xyz is None:
            omm_xyz = 1.
        if isinstance(omm_xyz, float) or isinstance(omm_xyz, int):
            omm_xyz = omm_ori.apply([omm_xyz, 0, 0])
        elif omm_xyz.ndim > 1 and omm_xyz.shape[1] < 3:
            radius = np.linalg.norm(omm_xyz, axis=1)
            omm_xyz = omm_ori.apply([radius, 0, 0])
        elif omm_xyz.ndim == 1:
            omm_xyz = omm_ori.apply([omm_xyz, 0, 0])

        if omm_pol_op is None:
            omm_pol_op = np.zeros(nb_output, dtype=self.dtype)
        elif isinstance(omm_pol_op, float) or isinstance(omm_pol_op, int):
            omm_pol_op = np.full(nb_output, fill_value=omm_pol_op, dtype=self.dtype)
        if omm_rho is None:
            omm_rho = np.full_like(omm_pol_op, np.deg2rad(5.))
        elif isinstance(omm_rho, float) or isinstance(omm_rho, int):
            omm_rho = np.full_like(omm_pol_op, omm_rho)
        if omm_res is None:
            omm_res = np.full_like(omm_rho, 1.)
        elif isinstance(omm_res, float) or isinstance(omm_res, int):
            omm_res = np.full_like(omm_rho, omm_res)

        if c_sensitive is None:
            c_sensitive = np.array([[0, 0, .4, .1, .5]] * self._nb_input, dtype=self.dtype)
        if not isinstance(c_sensitive, np.ndarray):
            c_sensitive = np.array(c_sensitive, dtype=self.dtype)
        if c_sensitive.ndim < 2:
            c_sensitive = c_sensitive[np.newaxis, ...]
        c_sensitive /= np.maximum(c_sensitive.sum(axis=1), eps)[..., np.newaxis]

        if isinstance(omm_photoreceptor_angle, int):
            omm_photoreceptor_angle = np.linspace(0, np.pi, omm_photoreceptor_angle, endpoint=False)
        else:
            omm_photoreceptor_angle = np.array(omm_photoreceptor_angle)

        self._omm_ori = omm_ori
        self._omm_xyz = omm_xyz
        self._omm_pol = omm_pol_op
        self._omm_rho = omm_rho
        self._omm_res = omm_res
        self._omm_area = None
        self._phot_angle = omm_photoreceptor_angle
        # contribution of each six points on the edges of the ommatidia
        # (sigma/2 distance from the centre of the Gaussian)
        # self._nb_gau = (np.nanmax(self._omm_rho) // np.rad2deg(10) + 1) * 6
        self._w_gau = np.exp(-.5)
        self._ori_init = copy(self._ori)
        self._c_sensitive = c_sensitive

        self._r = None

        self.reset()

    def reset(self):
        nb_omm = self.nb_ommatidia

        nb_samples = 6

        # create the 6 Gaussian samples for each ommatidium
        omm_ori_gau = [R.from_euler('Z', np.zeros((nb_omm, 1), dtype=self.dtype)) for _ in range(6)]
        for i in range(nb_samples):
            ori_p = R.from_euler(
                'XY', np.vstack([np.full_like(self._omm_rho, i*2*np.pi/nb_samples), self._omm_rho/2]).T)
            omm_ori_gau[i] = self.omm_ori * ori_p
        # augment the sampling points with the Gaussian samples
        self._omm_ori = R.from_quat(
            np.vstack([self.omm_ori.as_quat()] + [oog.as_quat() for oog in omm_ori_gau]))
        omm_ori_gau = [self._omm_ori[(i+1) * self.nb_ommatidia:(i+2) * self.nb_ommatidia] for i in range(nb_samples)]

        # reset to the initial orientation
        self._ori = copy(self._ori_init)

        # the small radius of each ommatidium (from the centre of the lens to the edges) in mm
        r_l = np.linalg.norm(omm_ori_gau[0].apply([1, 0, 0]) - self.omm_ori.apply([1, 0, 0]), axis=1)

        # calculate the area occupied by each ommatidium
        self._omm_area = np.pi * np.square(r_l)

        self._r = None  # initialise the responses

    def _sense(self, sky=None, scene=None):
        # the spectral sensitivity code
        w_c = self._c_sensitive
        # calculate the global orientation of the ommatidia
        omm_ori_glob = self._ori * self._omm_ori
        # the number of samples for each Gaussian
        nb_gau = len(omm_ori_glob) // self.nb_ommatidia - 1

        # initialise the luminance, degree and angle of polarisation, and the contribution from the world
        y = np.full(len(self._omm_ori), 1., dtype=self.dtype)
        p = np.full(len(self._omm_ori), 0., dtype=self.dtype)
        a = np.full(len(self._omm_ori), 0., dtype=self.dtype)
        c = np.full(len(self._omm_ori), np.nan, dtype=self.dtype)

        if sky is not None:
            # # the brightness of the environment can be calculated given the sun position
            # br = np.clip(np.sin(sky.theta_s), 0.1, 1)
            # get the sky contribution
            y[:], p[:], a[:] = sky(omm_ori_glob, irgbu=w_c, noise=self.noise, rng=self.rng)

        if scene is not None:
            # get the global positions of the ommatidia
            omm_pos_glob = self.xyz.reshape((1, -1))
            # get the rgb values from the scene and transform them into grey scale given the spectral sensitivity
            rgb = scene(omm_pos_glob, ori=omm_ori_glob, noise=self.noise)
            # transform the rgb into grey scale given the spectral sensitivity
            if w_c.shape[0] != rgb.shape[0]:
                w_c = np.vstack([w_c] * (rgb.shape[0] // w_c.shape[0]))
            c[:] = np.sum(rgb * w_c[..., 1:4], axis=1)

        # add the contribution of the scene to the input from the sky
        if np.all(np.isnan(y)) or np.all(~np.isnan(c)):
            max_brightness = 1.
        else:
            max_brightness = np.sqrt(np.nanmin(y[np.isnan(c)]))
        y[~np.isnan(c)] = c[~np.isnan(c)] * max_brightness
        p[~np.isnan(c)] = 0.
        a[~np.isnan(c)] = 0.

        # mix the values of each ommatidium with Gaussians
        if isinstance(self._w_gau, float):
            self._w_gau = [1.] + [self._w_gau] * nb_gau
        w_gau = self._w_gau

        # control the brightness due to wider acceptance angle
        # area * responsivity * radiation
        yy = y.reshape((nb_gau+1, -1))
        y_masked = np.ma.masked_array(yy, np.isnan(yy))
        y0 = np.ma.average(y_masked, axis=0, weights=w_gau) * self._omm_area * self._omm_res

        # update the degree of polarisation based on the acceptance angle
        pp = p.reshape((nb_gau+1, -1))
        p_masked = np.ma.masked_array(pp, np.isnan(pp))
        p0 = np.ma.average(p_masked, axis=0, weights=w_gau)

        aa = a.reshape((nb_gau+1, -1))
        # transform angle of polarisation into a complex number to allow calculation of means
        a_masked = np.ma.masked_array(np.exp(1j * aa), np.isnan(aa))
        a0 = np.ma.average(a_masked, axis=0, weights=w_gau)
        # transform the complex number back to an angle
        a0 = (np.angle(a0) + np.pi) % (2 * np.pi) - np.pi
        a0[a0 == np.ma.masked] = 0.

        y0 = y0.reshape((-1, 1))
        p0 = p0.reshape((-1, 1))
        a0 = a0.reshape((-1, 1))

        # apply the opponent polarisation filters
        op = self._omm_pol.reshape((-1, 1))
        pol_main = (self._ori * self.omm_ori).as_euler('ZYX', degrees=False)[..., 0].reshape((-1, 1))
        ori_op = (np.zeros((np.shape(self.omm_ori)[0], len(self._phot_angle)), dtype=self.dtype) +
                  self._phot_angle[np.newaxis, :])
        # ori_op[..., 1] = np.pi/2

        # calculate the responses for the 2 opponent photo-receptors
        s = y0 * ((np.square(np.sin(a0 - pol_main + ori_op)) +
                   np.square(np.cos(a0 - pol_main + ori_op)) * np.square(1. - p0)) * op +
                  (1. - op))

        # clip the responses to 1 as the incoming radiation gets saturated
        self._r = np.clip(np.sqrt(s), 0, 1)

        return self._r

    def __repr__(self):
        return ("CompoundEye(ommatidia=%d, responses=(%d, %d), "
                "pos=(%.2f, %.2f, %.2f), ori=(%.0f, %.0f, %.0f), name='%s')") % (
            self.nb_ommatidia, self._nb_output[0], self._nb_output[1],
            self.x, self.y, self.z, self.yaw_deg, self.pitch_deg, self.roll_deg, self.name
        )

    @staticmethod
    def flip(eye, horizontally=True, vertically=False, name=None):
        """
        Flips the eye horizontally, vertically or both.

        Parameters
        ----------
        eye: CompoundEye
            the eye to flip
        horizontally: bool, optional
            whether to flip it horizontally. Default is True
        vertically: bool, optional
            whether to flip it vertically. Default is False
        name: str, optional
            the name of the flipped eye. Default is None

        Returns
        -------
        CompoundEye
        """
        eye_copy = eye.copy()

        euler = eye_copy._omm_ori.as_euler('ZYX')
        xyz = eye_copy._omm_xyz
        if horizontally:
            euler[:, 0] *= -1
            xyz[:, 1] *= -1
        if vertically:
            euler[:, 1] *= -1
            xyz[:, 0] *= -1

        eye_copy._omm_ori = R.from_euler('ZYX', euler)
        eye_copy._omm_xyz = xyz
        if name is not None:
            eye_copy.name = name

        return eye_copy

    @property
    def omm_xyz(self):
        """
        The 3D positions of the ommatidia.
        """
        return self._omm_xyz

    @property
    def omm_ori(self):
        """
        The 3D orientations of the ommatidia.
        """
        return self._omm_ori[:self.nb_ommatidia]

    @property
    def omm_rho(self):
        """
        The acceptance angles of the ommatidia in rads.
        """
        return self._omm_rho

    @property
    def omm_pol(self):
        """
        The polarisation sensitivity of the ommatidia.
        """
        return self._omm_pol

    @property
    def omm_area(self):
        """
        The area occupied by each ommatidium.
        """
        return self._omm_area

    @property
    def omm_responsiveness(self):
        """
        The responsiveness of the ommatidia.
        """
        return self._omm_res

    @property
    def hue_sensitive(self):
        """
        The spectral sensitivity code of the ommatidia.
        """
        return self._c_sensitive

    @property
    def nb_ommatidia(self):
        """
        The number of ommatidia.

        Returns
        -------
        int
        """
        return self._omm_xyz.shape[0]

    @property
    def responses(self):
        """
        The latest responses generated by the eye.

        Returns
        -------
        np.ndarray[float]
        """
        return self._r


def mental_rotation_matrix(eye, nb_rotates=8, dtype='float32'):
    """
    Builds a matrix (nb_om x nb_om x nb_out) that performs mental rotation of the visual input.
    In practice, it builds a maps for each of the uniformly distributed nb_out view directions,
    that allow internal rotation of the visual input for different orientations of interest (preference angles).

    Parameters
    ----------
    eye: CompoundEye
        The compound eye structure.
    nb_rotates: int, optional
        The number of different tuning points (preference angles).
    dtype: np.dtype, optional
        The type of the elements in the matrix
    Returns
    -------
    M: np.ndarray
        A matrix that maps the input space of the eye to nb_out uniformly distributed
    """
    nb_omm = eye.nb_ommatidia
    m = np.zeros((nb_omm, nb_omm, nb_rotates), dtype=dtype)
    phi_rot = np.linspace(0, 2*np.pi, nb_rotates, endpoint=False)

    for i in range(nb_rotates):
        omm_i = R.from_euler('Z', phi_rot[i], degrees=False) * eye.omm_ori
        for j in range(nb_omm):
            omm_j = eye.omm_ori[j]
            d = np.linalg.norm((omm_i.inv() * omm_j).apply([1, 0, 0]) - np.array([1, 0, 0])) / 2
            m[j, :, i] = softmax(1. - d, tau=.01)

    return m
