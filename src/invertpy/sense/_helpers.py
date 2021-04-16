"""
Helpers for the invertpy.sense package. Contains functions for spherical distributions and random generators.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from invertpy.__helpers import *

import numpy as np


def fibonacci_sphere(nb_samples, fov=None, degrees=False):
    """
    Distributes samples in a sphere using the Fibonacci series. It is possible to create a proportion of the sphere that
    contains the same number of samples by setting the field of view to be less than 360 degrees.

    Parameters
    ----------
    nb_samples: int
        the number of samples to create
    fov: float, optional
        the field of view sets the proportion of the sphere to be used. Default is 360.
    degrees: bool, optional
        whether the field of view is given in degrees or not. Default is False.

    Returns
    -------
    sph: np.ndarray
        a nb_samples x 3 matrix that contains the spherical coordinates (azimuth, elevation, distance) of each sample.
    """

    nb_samples = int(nb_samples)
    if fov is None:
        fov = 2 * np.pi
    elif degrees:
        fov = np.deg2rad(fov)
    theta_max = fov / 2  # angular distance of the outline from the zenith
    phi = (1. + np.sqrt(5)) * np.pi

    r_l = 1.  # the small radius of a hexagon (mm)
    R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
    S_l = 3 * r_l * R_l  # area of a lens (mm^2)

    S_a = nb_samples * S_l  # area of the dome surface (mm^2)
    R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(theta_max))))  # radius of the curvature (mm)
    S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)

    total_samples = np.maximum(int(nb_samples * S_c / (1.2 * S_a)), nb_samples)

    indices = np.arange(0, nb_samples, dtype=float)

    thetas = np.pi/2 - np.arccos(2 * indices / (total_samples - .5) - 1)
    phis = (phi * indices) % (2 * np.pi)

    return np.vstack([phis, thetas, np.ones_like(phis)]).T
