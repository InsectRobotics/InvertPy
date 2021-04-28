# #! /usr/bin/env python
# -*- coding: utf-8 -*-

from invertpy.brain.mushroombody import IncentiveCircuit

import matplotlib.pyplot as plt
import numpy as np
import sys


def main(*args):
    ic = IncentiveCircuit(cs_magnitude=2., nb_apl=0, ltm_charging_speed=.05, repeat_rate=.2, noise=0.)

    history = []
    for i in range(1):
        p11 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        p11 = np.append(p11, ic.r_dan[0].copy())
        p12 = ic(cs=[1., 0.], us=[0., 0.]).copy()
        p12 = np.append(p12, ic.r_dan[0].copy())
        p13 = ic(cs=[1., 0.], us=[0., 0.]).copy()
        p13 = np.append(p13, ic.r_dan[0].copy())
        p21 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        p21 = np.append(p21, ic.r_dan[0].copy())
        p22 = ic(cs=[0., 1.], us=[0., 0.]).copy()
        p22 = np.append(p22, ic.r_dan[0].copy())
        p23 = ic(cs=[0., 1.], us=[0., 0.]).copy()
        p23 = np.append(p23, ic.r_dan[0].copy())
        history.append([p11, p12, p13])
        history.append([p21, p22, p23])

    for i in range(5):
        a11 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        a11 = np.append(a11, ic.r_dan[0].copy())
        a12 = ic(cs=[1., 0.], us=[0., 0.]).copy()
        a12 = np.append(a12, ic.r_dan[0].copy())
        a13 = ic(cs=[1., 0.], us=[0., 0.]).copy()
        a13 = np.append(a13, ic.r_dan[0].copy())
        a21 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        a21 = np.append(a21, ic.r_dan[0].copy())
        a22 = ic(cs=[0., 1.], us=[0., 0.]).copy()
        a22 = np.append(a22, ic.r_dan[0].copy())
        a23 = ic(cs=[0., 1.], us=[1., 0.]).copy()
        a23 = np.append(a23, ic.r_dan[0].copy())
        history.append([a11, a12, a13])
        history.append([a21, a22, a23])

    for i in range(1):
        p11 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        p11 = np.append(p11, ic.r_dan[0].copy())
        p12 = ic(cs=[1., 0.], us=[0., 0.]).copy()
        p12 = np.append(p12, ic.r_dan[0].copy())
        p13 = ic(cs=[1., 0.], us=[0., 0.]).copy()
        p13 = np.append(p13, ic.r_dan[0].copy())
        p21 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        p21 = np.append(p21, ic.r_dan[0].copy())
        p22 = ic(cs=[0., 1.], us=[0., 0.]).copy()
        p22 = np.append(p22, ic.r_dan[0].copy())
        p23 = ic(cs=[0., 1.], us=[0., 0.]).copy()
        p23 = np.append(p23, ic.r_dan[0].copy())
        history.append([p11, p12, p13])
        history.append([p21, p22, p23])

    for i in range(5):
        r11 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        r11 = np.append(r11, ic.r_dan[0].copy())
        r12 = ic(cs=[1., 0.], us=[0., 0.]).copy()
        r12 = np.append(r12, ic.r_dan[0].copy())
        r13 = ic(cs=[1., 0.], us=[1., 0.]).copy()
        r13 = np.append(r13, ic.r_dan[0].copy())
        r21 = ic(cs=[0., 0.], us=[0., 0.]).copy()
        r21 = np.append(r21, ic.r_dan[0].copy())
        r22 = ic(cs=[0., 1.], us=[0., 0.]).copy()
        r22 = np.append(r22, ic.r_dan[0].copy())
        r23 = ic(cs=[0., 1.], us=[0., 0.]).copy()
        r23 = np.append(r23, ic.r_dan[0].copy())
        history.append([r11, r12, r13])
        history.append([r21, r22, r23])

    history = np.array(history)
    # print(history)

    print(ic)
    plt.figure("responses", figsize=(12, 7))
    for i in range(6):
        plt.subplot(6, 2, 1 + 2 * i)
        plt.plot(history[0::2, 1:, i].reshape((-1)))
        plt.plot(history[1::2, 1:, i].reshape((-1)))
        plt.ylim([-.1, 2.1])
        plt.ylabel("$%s$" % ic.mbon_names[i])

        plt.subplot(6, 2, 2 + 2 * i)
        plt.plot(history[0::2, 1:, 6 + i].reshape((-1)))
        plt.plot(history[1::2, 1:, 6 + i].reshape((-1)))
        plt.ylim([-.1, 2.1])
        plt.ylabel("$%s$" % ic.dan_names[i])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
