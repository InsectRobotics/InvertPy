__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

import numpy as np

RNG = np.random.RandomState(2021)
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
