from invertbrain._helpers import RNG

import numpy as np

eps = np.finfo(float).eps


def fibonacci_sphere(samples, fov):

    samples = int(samples)
    theta_max = fov / 2  # angular distance of the outline from the zenith
    phi = (1. + np.sqrt(5)) * np.pi

    r_l = 1.  # the small radius of a hexagon (mm)
    R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
    S_l = 3 * r_l * R_l  # area of a lens (mm^2)

    S_a = samples * S_l  # area of the dome surface (mm^2)
    R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(theta_max))))  # radius of the curvature (mm)
    S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)

    total_samples = np.maximum(int(samples * S_c / (1.2 * S_a)), samples)

    indices = np.arange(0, samples, dtype=float)

    thetas = np.pi/2 - np.arccos(2 * indices / (total_samples - .5) - 1)
    phis = (phi * indices) % (2 * np.pi)

    return np.vstack([phis, thetas, np.ones_like(phis)]).T


def add_noise(v=None, noise=0., fill_value=0, rng=RNG):
    if isinstance(noise, np.ndarray):
        if noise.size == v.size:
            eta = np.array(noise, dtype=bool)
        else:
            eta = np.zeros_like(v, dtype=bool)
            eta[:noise.size] = noise
    elif noise > 0:
        eta = np.argsort(np.absolute(rng.random.randn(*v.shape)))[:int(noise * v.shape[0])]
    else:
        eta = np.zeros_like(v, dtype=bool)

    if v is not None:
        v[eta] = fill_value

    return eta

