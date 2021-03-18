from .component import Component
from .synapses import init_synapses
from ._helpers import eps

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class CelestialCompass(Component):

    def __init__(self, nb_pol: int, loc_ori: R, nb_sol: int = 8, nb_tcl: int = None, sigma=13, shift=40, dt=2./60,
                 degrees=True, integrated=False, has_pol=True, has_sun=True, has_circadian=False, *args, **kwargs):
        super().__init__(nb_pol, nb_tcl, *args, **kwargs)

        if nb_tcl is None:
            nb_tcl = nb_sol

        self._phi_sun = np.linspace(-np.pi, np.pi, nb_sol, endpoint=False)  # SUN preference angles
        self._phi_sol = np.linspace(0., 2 * np.pi, nb_sol, endpoint=False)  # SOL preference angles
        self._phi_tcl = np.linspace(0., 2 * np.pi, nb_tcl, endpoint=False)  # TCL preference angles

        self._w_sun = init_synapses(nb_pol, nb_sol, dtype=self.dtype)
        self._w_sol = init_synapses(nb_pol, nb_sol, dtype=self.dtype)
        if integrated:
            self._w_tcl = init_synapses(nb_pol, nb_tcl, dtype=self.dtype)
        else:
            self._w_tcl = init_synapses(nb_sol, nb_tcl, dtype=self.dtype)

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
            self.w_sol = self.generate_w_sol(yaw+np.pi/2)
        if self.has_sun:
            self.phi_sun = np.linspace(-np.pi, np.pi, self.nb_sol, endpoint=False)  # SUN preference angles
            self.w_sun = self.generate_w_sun(yaw+np.pi/2)

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

    def gate(self, glob_ori: R, order=1.):
        _, pitch, _ = glob_ori.as_euler('ZYX', degrees=False).T
        zenith = pitch - np.pi/2
        d = np.sin(self._shift - zenith)

        return np.power(np.exp(-np.square(d) / (2. * np.square(self._sigma))), order).reshape((-1, 1))

    def circadian(self, r_sol, dt):
        r = r_sol @ np.exp(-np.arange(self.nb_tcl) * 1j * 2 * np.pi / float(self._nb_tcl))
        res = np.clip(3.5 * (np.absolute(r) - .53), 0, 2)  # certainty of prediction
        ele_pred = 26 * (1 - 2 * np.arcsin(1 - res) / np.pi) + 15
        return np.deg2rad(9 + np.exp(.1 * (54 - ele_pred))) / (60. / float(dt))

    def generate_w_sol(self, alpha):
        if self._is_absolute:
            z = float(self.nb_sol) / (2. * float(self.nb_pol))
            return -z * np.sin(self._phi_sol[np.newaxis] - alpha[:, np.newaxis])
        else:
            z = float(self.nb_sol) / float(self.nb_pol)
            return z * np.sin(alpha[:, np.newaxis] - self._phi_sol[np.newaxis])

    def generate_w_sun(self, alpha):
        if self._is_absolute:
            z = float(self.nb_sol) / (2. * float(self.nb_pol))
            return -z * np.sin(self._phi_sun[np.newaxis] - alpha[:, np.newaxis])
        else:
            z = float(self.nb_sol) / float(self.nb_pol)
            return z * np.sin(alpha[:, np.newaxis] - self._phi_sun[np.newaxis])

    def generate_w_tcl(self, alpha=None):
        if self._is_absolute:
            if alpha is None:
                alpha, _, _ = self._loc_ori.as_euler('ZYX', degrees=False)
                alpha += np.pi/2

            z = float(self.nb_tcl) / (2. * float(self.nb_pol))
            return -z * np.sin(self._phi_tcl[np.newaxis] - alpha[:, np.newaxis])
        else:
            z = float(self.nb_tcl) / float(self.nb_sol)
            return z * np.cos(self._phi_tcl[:, np.newaxis] - self._phi_sol[np.newaxis])

    @property
    def w_sol(self):
        return self._w_sol

    @w_sol.setter
    def w_sol(self, v):
        self._w_sol[:] = v[:]

    @property
    def w_sun(self):
        return self._w_sun

    @w_sun.setter
    def w_sun(self, v):
        self._w_sun[:] = v[:]

    @property
    def w_tcl(self):
        return self._w_tcl

    @w_tcl.setter
    def w_tcl(self, v):
        self._w_tcl[:] = v[:]

    @property
    def phi_sol(self):
        return self._phi_sol

    @phi_sol.setter
    def phi_sol(self, v):
        self._phi_sol[:] = v[:]

    @property
    def phi_sun(self):
        return self._phi_sun

    @phi_sun.setter
    def phi_sun(self, v):
        self._phi_sun[:] = v[:]

    @property
    def phi_tcl(self):
        return self._phi_tcl

    @phi_tcl.setter
    def phi_tcl(self, v):
        self._phi_tcl[:] = v[:]

    @property
    def shift(self):
        return self._shift

    @property
    def sigma(self):
        return self._sigma

    @property
    def r_pol(self):
        return self._r_pol

    @property
    def r_sol(self):
        return self._r_sol

    @property
    def r_sun(self):
        return self._r_sun

    @property
    def r_cel(self):
        return self._r_cel

    @property
    def r_tcl(self):
        return self._r_tcl

    @property
    def nb_pol(self):
        return self._nb_pol

    @property
    def nb_sol(self):
        return self._nb_sol

    @property
    def nb_tcl(self):
        return self._nb_tcl

    @property
    def d_phi(self):
        return self._d_phi

    @property
    def has_pol(self):
        return self._has_pol

    @property
    def has_sun(self):
        return self._has_sun

    @property
    def has_circadian(self):
        return self._has_circadian


class PolarisationCompass(CelestialCompass):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('has_sun', False)
        kwargs.setdefault('has_pol', True)
        super().__init__(*args, **kwargs)


class SolarCompass(CelestialCompass):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('has_sun', True)
        kwargs.setdefault('has_pol', False)
        super().__init__(*args, **kwargs)


def photoreceptor2pol(r, ori: R = None, ori_cross: np.ndarray = None, dtype='float32'):
    r_op = photoreceptor2opponent(r, ori=ori, ori_cross=ori_cross, dtype=dtype)
    r_po = photoreceptor2pooling(r)
    return r_op / (r_po + eps)


def photoreceptor2opponent(r, ori: R = None, ori_cross: np.ndarray = None, dtype='float32'):
    if ori is None and ori_cross is None:
        return np.sum(r, axis=1)
    elif ori_cross is None:
        ori_cross = ori2cross(ori, dtype=dtype)
    return np.sum(np.cos(2 * ori_cross) * r, axis=1)


def photoreceptor2pooling(r):
    return np.sum(r, axis=1)


def ori2cross(ori: R, dtype='float32'):
    ori_cross = np.zeros((np.shape(ori)[0], 2), dtype=dtype)
    ori_cross[..., 1] = np.pi / 2

    return ori_cross


def encode_sph(theta, phi=None, length=8):
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


def decode_sph(I):
    fund_freq = np.fft.fft(I)[1]
    phi = (np.pi - np.angle(np.conj(fund_freq))) % (2 * np.pi) - np.pi
    theta = np.absolute(fund_freq)
    return np.array([theta, phi])


def decode_xy(I):
    length = I.shape[-1]
    alpha = np.linspace(0, 2 * np.pi, length, endpoint=False)
    x = np.sum(I * np.cos(alpha), axis=-1)[..., np.newaxis]
    y = np.sum(I * np.sin(alpha), axis=-1)[..., np.newaxis]
    return np.concatenate((x, y), axis=-1)
