__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"


from invertbrain.activation import softmax
from .sensor import Sensor
from ._helpers import fibonacci_sphere, eps

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class CompoundEye(Sensor):

    def __init__(self, omm_xyz=None, omm_ori=None, omm_rho=None, omm_pol_op=None, c_sensitive=None, *args, **kwargs):
        """

        Parameters
        ----------
        omm_xyz: np.ndarray, float
            Relative 3-D position of each of the ommatidia on the eye. If None, it is automatically calculated to be
            1 unit away from the centre of the eye and towards the direction of the ommatidium.
        omm_ori: R
            The relative direction of each of the ommatidia on the eye. If None, it is automatically calculated from the
            position of the ommatidia, assumming that it is phasing in the direction from the centre to the position of
            the ommatidium.
        omm_rho: np.ndarray, float
            The acceptance angle of each ommatidium. If it is a single value, it is assigned to all the ommatidia.
            Default is 5 deg.
        omm_pol_op: np.ndarray, float
            The polarisation sensitivity of every ommatidium (0 = None, 1 = very sensitive). If it is a single value, it
            is assigned to all the ommatidia. Default is 0.
        c_sensitive: tuple, list, np.ndarray
            The IRGBU colour that the eyes are sensitive to (infrared, red, green, blue, ultraviolet)
        """
        if omm_pol_op is None or isinstance(omm_pol_op, float) or isinstance(omm_pol_op, int):
            nb_output = None
        else:
            nb_output = omm_pol_op.shape
        if omm_ori is None and omm_xyz is not None:
            omm_ori = np.empty(omm_xyz.shape[0], dtype=R)
            for i in range(omm_xyz.shape[0]):
                omm_ori = R.align_vectors(omm_xyz[i], [[0], [0], [1]])
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
        if c_sensitive is None:
            c_sensitive = np.array([[0, 0, .4, .1, .5]] * self._nb_input, dtype=self.dtype)
        if not isinstance(c_sensitive, np.ndarray):
            c_sensitive = np.array(c_sensitive, dtype=self.dtype)
        if c_sensitive.ndim < 2:
            c_sensitive = c_sensitive[np.newaxis, ...]
        c_sensitive /= np.maximum(c_sensitive.sum(axis=1), eps)[..., np.newaxis]

        self._omm_ori = omm_ori
        self._omm_ori_gau = None
        self._omm_xyz = omm_xyz
        self._omm_pol = omm_pol_op
        self._omm_rho = omm_rho
        self._w_o2r = None  # ommatidia to responses
        # contribution of each six points on the edges of the ommatidia
        # (sigma/2 distance from the centre of the Gaussian)
        self._w_gau = np.exp(-.5)
        self._ori_init = copy(self._ori)
        self._c_sensitive = c_sensitive

        self.reset()

    def reset(self):
        self._w_o2r = np.eye(self._nb_input, self._nb_output[0], dtype=self.dtype)

        nb_omm = self.omm_xyz.shape[0]
        self._omm_ori_gau = [R.from_euler('Z', np.zeros((nb_omm, 1), dtype=self.dtype)) for _ in range(6)]
        for i in range(6):
            ori_p = R.from_euler(
                'XY', np.vstack([np.full_like(self._omm_rho, i*np.pi/3), self._omm_rho/2]).T)
            self._omm_ori_gau[i] = self._omm_ori * ori_p

        # import matplotlib.pyplot as plt
        #
        # xyz_gau = np.zeros((6, nb_omm, 3), dtype=self.dtype)
        # for i in range(6):
        #     xyz_gau[i] = self._omm_ori_gau[i].apply([1, 0, 0])
        #
        # plt.plot(xyz_gau[..., 0], xyz_gau[..., 1])
        # plt.show()

        self._ori = copy(self._ori_init)

    def _sense(self, sky=None, scene=None):
        w_c = self._c_sensitive
        omm_ori_glob = self._ori * self._omm_ori
        omm_oris_gau_glob = []
        for i, omm_ori_gau in enumerate(self._omm_ori_gau):
            omm_oris_gau_glob.append(self._ori * omm_ori_gau)

        y = np.full((len(self._omm_ori_gau) + 1, np.shape(omm_ori_glob)[0]), np.nan, dtype=self.dtype)
        p = np.full((len(self._omm_ori_gau) + 1, np.shape(omm_ori_glob)[0]), np.nan, dtype=self.dtype)
        a = np.full((len(self._omm_ori_gau) + 1, np.shape(omm_ori_glob)[0]), np.nan, dtype=self.dtype)
        c = np.full((len(self._omm_ori_gau) + 1, np.shape(omm_ori_glob)[0]), np.nan, dtype=self.dtype)
        br = 1.

        if sky is not None:
            br = np.clip(np.sin(sky.theta_s), 0.1, 1)
            y[0], p[0], a[0] = sky(omm_ori_glob, irgbu=w_c, noise=self.noise)
            # initialise weights for the gaussian blur of each ommatidium

            # add contributions of each of the six edges of the ommatidium to the received information
            for i, omm_ori_gau_glob in enumerate(omm_oris_gau_glob):
                y[i+1], p[i+1], a[i+1] = sky(omm_ori_gau_glob, irgbu=w_c, noise=self.noise)
        else:
            y[:] = 1.
            p[:] = 0.
            a[:] = 0.

        if scene is not None:
            omm_pos_glob = self.xyz + self._ori.apply(self._omm_xyz)
            c[0] = 1 - np.sum(scene(omm_pos_glob, ori=omm_ori_glob,
                                    brightness=br, noise=self.noise) * w_c[..., 1:4], axis=1)

            for i, omm_ori_gau_glob in enumerate(omm_oris_gau_glob):
                c[i+1] = 1 - np.sum(scene(omm_pos_glob, ori=omm_ori_gau_glob,
                                          brightness=br, noise=self.noise) * w_c[..., 1:4], axis=1)

        # add the contribution of the scene to the input from the sky
        y[~np.isnan(c)] = c[~np.isnan(c)]
        p[~np.isnan(c)] = 0.
        a[~np.isnan(c)] = 0.

        # mix the values of each ommatidium with Gaussians
        nb_gau = len(self._omm_ori_gau)
        w_gau = [1.] + [self._w_gau] * nb_gau

        # increase brightness due to wider acceptance angle
        brightness = np.sqrt(.5 + 4 * self._omm_rho / np.pi)

        y_masked = np.ma.masked_array(y, np.isnan(y))
        y0 = np.ma.average(y_masked, axis=0, weights=w_gau) * brightness

        p_masked = np.ma.masked_array(p, np.isnan(p))
        p0 = np.ma.average(p_masked, axis=0, weights=w_gau)

        # transform angle of polarisation into a complex number to allow calculation of means
        a_masked = np.ma.masked_array(np.exp(1j * a), np.isnan(a))
        a0 = np.ma.average(a_masked, axis=0, weights=w_gau)
        # transform the complex number back to an angle
        a0 = (np.angle(a0) + np.pi) % (2 * np.pi) - np.pi

        y0 = y0.reshape((-1, 1))
        p0 = p0.reshape((-1, 1))
        a0 = a0.reshape((-1, 1))

        # apply the opponent polarisation filters
        op = self._omm_pol.reshape((-1, 1))
        pol_main = (self._ori * self._omm_ori).as_euler('ZYX', degrees=False)[..., 0].reshape((-1, 1))
        ori_op = np.zeros((np.shape(self._omm_ori)[0], 2), dtype=self.dtype)
        ori_op[..., 1] = np.pi/2

        # calculate the responses for the 2 opponent photo-receptors
        s = y0 * ((np.square(np.sin(a0 - pol_main + ori_op)) +
                   np.square(np.cos(a0 - pol_main + ori_op)) * np.square(1. - p0)) * op +
                  (1. - op))
        r = np.sqrt(s)

        return r

    def __repr__(self):
        return ("CompoundEye(ommatidia=%d, responses=(%d, %d), "
                "pos=(%.2f, %.2f, %.2f), ori=(%.2f, %.2f, %.2f), name='%s')") % (
            self.nb_ommatidia, self._nb_output[0], self._nb_output[1],
            self.x, self.y, self.z, self.yaw_deg, self.pitch_deg, self.roll_deg, self.name
        )

    @property
    def omm_xyz(self):
        return self._omm_xyz

    @property
    def omm_ori(self):
        return self._omm_ori

    @property
    def omm_rho(self):
        return self._omm_rho

    @property
    def omm_pol(self):
        return self._omm_pol

    @property
    def hue_sensitive(self):
        return self._c_sensitive

    @property
    def nb_ommatidia(self):
        return self._omm_xyz.shape[0]


def mental_rotation_matrix(eye: CompoundEye, nb_rotates=8, dtype='float32'):

    """
    Builds a matrix (nb_om x nb_om x nb_out) that performs mental rotation of the visual input.
    In practice, it builds a maps for each of the uniformly distributed nb_out view directions,
    that allow internal rotation of the visual input for different orientations of interest (preference angles).
    Parameters
    ----------
    eye: CompoundEye
        The compound eye structure.
    nb_rotates: int
        The number of different tuning points (preference angles).
    dtype: np.dtype, str
        The type of the elements in the matrix
    Returns
    -------
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
