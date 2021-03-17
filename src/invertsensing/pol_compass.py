from invertbrain.compass import photoreceptor2pol
from .comoundeye import CompoundEye

from scipy.spatial.transform import Rotation as R

import numpy as np


class PolarisationCompass(CompoundEye):
    def __init__(self, field_of_view=56, *args, **kwargs):

        kwargs.setdefault('nb_input', 60)
        nb_inputs = kwargs['nb_input']
        if nb_inputs <= 8:
            nb_samples = [nb_inputs]
        elif nb_inputs <= 12:
            nb_samples = [4, nb_inputs - 4]
        elif nb_inputs <= 24:
            nb_samples = [4, 8, nb_inputs - 12]
        elif nb_inputs <= 40:
            nb_samples = [4, 8, 12, nb_inputs - 24]
        elif nb_inputs <= 60:
            nb_samples = [4, 8, 12, 16, nb_inputs - 40]
        else:
            nb_samples = None
        if nb_samples is not None:
            omm_sph = generate_rings(nb_samples=[4, 8, 12, 16, 20], fov=field_of_view)[..., :2]
            omm_euler = np.hstack([omm_sph, np.full((omm_sph.shape[0], 1), np.pi / 2)])
            kwargs.setdefault('omm_ori', R.from_euler('ZYX', omm_euler, degrees=False))
        kwargs.setdefault('omm_rho', np.deg2rad(5.4))
        kwargs.setdefault('omm_pol_op', 1)
        kwargs.setdefault('c_sensitive', [0, 0, 0, 0, 1])
        kwargs.setdefault('name', 'pol_compass')
        kwargs.setdefault('nb_output', (nb_inputs,))
        super().__init__(*args, **kwargs)

        self._field_of_view = np.deg2rad(field_of_view)

    def _sense(self, sky=None, scene=None):
        r = super()._sense(sky=sky, scene=scene)
        return photoreceptor2pol(r, ori=self._omm_ori, dtype=self.dtype).reshape((-1, 1))

    def __repr__(self):
        print(self._nb_output[1])
        return ("PolarisationCompass(ommatidia=%d, FOV=%.0f, responses=(%d, %d), "
                "pos=(%.2f, %.2f, %.2f), ori=(%.2f, %.2f, %.2f), name='%s')") % (
            self.nb_ommatidia, np.rad2deg(self.field_of_view), self._nb_output[0], self._nb_output[1],
            self.x, self.y, self.z, self.yaw_deg, self.pitch_deg, self.roll_deg, self.name
        )

    @property
    def field_of_view(self):
        return self._field_of_view


def generate_rings(nb_samples, fov):
    nb_rings = len(nb_samples)
    nb_samples_total = np.sum(nb_samples)
    v_angles = fov / float(2 * nb_rings + 1)

    phis = np.zeros(nb_samples_total, dtype='float32')
    thetas = np.zeros(nb_samples_total, dtype='float32')
    i = 0
    for r, samples in enumerate(nb_samples):
        theta = 90 + r * v_angles + v_angles / 2
        h_angles = 360. / samples
        for c in range(samples):
            phi = c * h_angles + h_angles / 2
            phis[i] = np.deg2rad(phi)
            thetas[i] = np.deg2rad(theta)
            i += 1

    return np.vstack([phis, thetas, np.ones_like(phis)]).T

