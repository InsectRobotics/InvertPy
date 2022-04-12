##! /usr/bin/env python
# -*- coding: utf-8 -*-

from invertpy.sense.polarisation import PolarisationSensor
from invertpy.io.sensor_params import save_eye2csv, load_csv2eye

import numpy as np
import sys


def main(*args):
    compass = PolarisationSensor()
    print(compass)
    save_eye2csv(compass, 'pol_compass')

    import matplotlib.pyplot as plt

    hue = compass.hue_sensitive
    if hue.shape[0] == 1:
        hue = np.vstack([hue] * compass.omm_xyz.shape[0])
    rgb = hue[..., 1:4]
    rgb[:, [0, 2]] += hue[..., 4:5] / 2
    rgb[:, 0] += hue[..., 0]
    plt.subplot(111, polar=False)
    mask = compass.omm_xyz[:, 2] > 0
    plt.scatter(compass.omm_xyz[mask, 0],
                compass.omm_xyz[mask, 1],
                s=20,
                c=np.clip(rgb[mask, :], 0, 1))
    # plt.ylim([-1.1, 1.1])
    # plt.xlim([-1.1, 1.1])
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
