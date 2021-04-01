#! /usr/bin/env python
# -*- coding: utf-8 -*-

from invertpy.sense.vision import CompoundEye
from invertpy.io.sensor_params import save_eye2csv, load_csv2eye

import numpy as np
import sys


def main(*args):
    eye = CompoundEye(nb_input=5000)
    print(eye)
    save_eye2csv(eye, 'fibonacci')

    import matplotlib.pyplot as plt

    hue = eye.hue_sensitive
    rgb = hue[..., 1:4]
    rgb[:, [0, 2]] += hue[..., 4:5] / 2
    rgb[:, 0] += hue[..., 0]
    plt.subplot(111, polar=False)
    plt.scatter(eye.omm_xyz[eye.omm_xyz[:, 2] > 0, 0],
                eye.omm_xyz[eye.omm_xyz[:, 2] > 0, 1],
                s=20,
                c=np.clip(rgb[eye.omm_xyz[:, 2] > 0, :], 0, 1))
    plt.ylim([-1.1, 1.1])
    plt.xlim([-1.1, 1.1])
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
