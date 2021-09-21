from invertpy.brain.preprocessing import ZernikeMoments
from invertpy.sense.vision import CompoundEye

import numpy as np
import matplotlib.pyplot as plt


def main(*args):

    eye = CompoundEye(nb_input=2000)
    zer = ZernikeMoments(ori=eye.omm_ori, order=16, out_type="raw")

    r = np.zeros((zer.z_moments.shape[0], 2000), dtype=float)

    i = 0
    for n in range(zer.order + 1):
        for m in range(n + 1):
            if (n - np.absolute(m)) % 2 == 0:
                r_complex = zer.zernike_poly(zer.rho, zer.phi, n, m)
                r[i] = (np.real(r_complex) + np.imag(r_complex)) / 2
                i += 1  # process the next moment

    # plt.figure("zernike-polynomials", figsize=(10, 10))
    # for i in range(r.shape[0]):
    #     ax = plt.subplot(9, 9, i + 1, polar=True)
    #     ax.scatter(zer.phi, zer.rho, s=20, c=r[i], cmap="coolwarm", vmin=-1, vmax=1)
    #     ax.set_ylim(0, 1)
    #     ax.set_axis_off()
    # plt.tight_layout()

    z = zer(np.ones(2000))

    plt.figure("zernike-moments", figsize=(5, 5))

    plt.subplot(211, polar=False)
    mag = np.absolute(z)
    ang = np.angle(z)
    x = np.arange(z.shape[0])
    y = np.zeros_like(x)
    u = mag * np.sin(ang)
    v = mag * np.cos(ang)
    print(np.min(v), np.max(v))
    plt.quiver(x, y, u, v, scale=.2)
    plt.xlim(-1, 82)
    plt.ylim(-1, 1)

    plt.subplot(212, polar=True)
    mag = np.absolute(z.sum())
    ang = np.angle(z.sum())
    print(mag, ang)
    plt.quiver([0], [0], [mag], [ang], scale=40)
    plt.ylim(0, 1)
    plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
